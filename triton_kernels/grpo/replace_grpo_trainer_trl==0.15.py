
# from replace_grpo_trainer import trigger

import torch
import triton
import triton.language as tl
from trl import GRPOTrainer

import importlib
module = importlib.import_module('trl.trainer.grpo_trainer')

# @triton.autotune([triton.Config({'BLOCK_SIZE': bsz}, num_stages=ns, num_warps=nw)
#                  for bsz in [2048*(2**i) for i in range(5)]
#                  for ns in [1,2,4]
#                  for nw in [8, 16, 32]
#                  ], key=['N']
#                  )
@triton.jit
def _grpo_loss_fwd(LOGITS, REF_LOGP, INPUT_IDS, ADVANTAGES, MASK, BETA,
                    LOSS, LSE, SAVE_KL,
                    M, N, L, INPUT_IDS_START_INDEX,
                    BLOCK_SIZE: tl.constexpr
                    ):
    row_idx = tl.program_id(0)
    # 因为最后一个位置不需要计算，实际上Logits是一个B*(L+1)行的向量，而我们只启动了B*L个程序
    # 比如3*4*vocab_size，每第4个位置不需要计算
    # row_idx从0开始，如果到第2行第一个为止，row_id为3，而真实的行id应该是4。
    # 因此用off_b去记录一个偏移量
    off_b = row_idx // L    
    N = tl.cast(N, tl.int64)

    LOGITS += N * (row_idx + off_b) # 加上偏移量
    REF_LOGP += row_idx
    # 同样input_ids前面介绍时也有多余的prompt部分
    # 比如prompt长度为64，第1行的起始位置应该从64开始
    INPUT_IDS += row_idx + (off_b+1) * INPUT_IDS_START_INDEX
    LOSS += row_idx
    LSE += row_idx
    ADVANTAGES += off_b
    
    MASK += row_idx
    not_skip = tl.load(MASK)# 跳过padding的部分，节约时间
    if not_skip == 1:       # 尤其是output长短不一时，都会pad到最长的那个，会浪费很多计算资源
        base_cols = tl.arange(0, BLOCK_SIZE)
        # 没啥好说的，计算两个lse，online-softmax那一套
        m_i = -float("inf")
        l_i = 0.0
        for start_n in tl.range(0, N, BLOCK_SIZE):
            cols = start_n + base_cols
            mask = cols < N
            logits = tl.load(LOGITS+cols, mask=mask, other=-float('inf')).to(tl.float32)
            m_ij = tl.max(logits)
            new_m_i = tl.maximum(m_i, m_ij)
            l_i = l_i * tl.exp(m_i - new_m_i) + tl.sum(tl.exp(logits - new_m_i))
            m_i = new_m_i
        lse = tl.log(l_i) + m_i

        # 有了lse，直接读取input_ids对应的logits即可，一个标量
        idx = tl.load(INPUT_IDS)
        x = tl.load(LOGITS+idx).to(tl.float32)
        advantage = tl.load(ADVANTAGES).to(tl.float32)
        ref_logp = tl.load(REF_LOGP)
        logp = x - lse
        diff = ref_logp - logp
        kl = tl.exp(diff) - diff - 1
        # 因为我们知道 torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        # 实际上等于 1 * advantages.unsqueeze(1)
        # loss我们直接减去一个advantage
        loss = kl * BETA - advantage
        tl.store(LOSS, loss)
        tl.store(LSE, lse)
        if SAVE_KL:
            tl.store(LOSS+M, kl)


# @triton.autotune([triton.Config({'BLOCK_SIZE': bsz}, num_stages=ns, num_warps=nw)
#                  for bsz in [2048*(2**i) for i in range(5)]
#                  for ns in [1,2,4]
#                  for nw in [8, 16, 32]
#                  ], key=['N']
#                  )
@triton.jit
def _grpo_loss_bwd(DLOSS, DLOGITS, 
                   LOGITS, REF_LOGP, INPUT_IDS, ADVANTAGES, MASK, BETA,
                    LSE, 
                    N, L, INPUT_IDS_START_INDEX,
                    BLOCK_SIZE: tl.constexpr
                    ):
    # 与forward部分如出一辙
    row_idx = tl.program_id(0)
    off_b = row_idx // L
    N = tl.cast(N, tl.int64)

    DLOSS += row_idx
    DLOGITS += N * (row_idx + off_b)
    LOGITS += N * (row_idx + off_b)
    REF_LOGP += row_idx
    INPUT_IDS += row_idx + (off_b+1) * INPUT_IDS_START_INDEX
    LSE += row_idx
    ADVANTAGES += off_b
    base_cols = tl.arange(0, BLOCK_SIZE)

    MASK += row_idx
    not_skip = tl.load(MASK)
    if not_skip == 1:
        dloss = tl.load(DLOSS).to(tl.float32)
        lse = tl.load(LSE)
        idx = tl.load(INPUT_IDS)
        x = tl.load(LOGITS+idx).to(tl.float32)
        advantage = tl.load(ADVANTAGES).to(tl.float32)
        ref_logp = tl.load(REF_LOGP)
        logp = x - lse

        # 算dlogp
        dlogp = (BETA * (-1.0 * tl.exp(ref_logp - logp) + 1) \
                        - advantage) \
                        * dloss

        # 用dlogp再去算dlogits
        for start_n in tl.range(0, N, BLOCK_SIZE):
            cols = start_n + base_cols
            mask = cols < N
            logits = tl.load(LOGITS+cols, mask=mask, other=-float('inf')).to(tl.float32)
            probs = tl.exp(logits - lse)
            dlogits = tl.where(cols==idx, 1-probs, -probs) * dlogp
            # DLOGITS的内存就对应REF_LOGITS，废物再利用
            tl.store(DLOGITS+cols, dlogits, mask=mask)
    else:
        dlogits = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for start_n in tl.range(0, N, BLOCK_SIZE):
            cols = start_n + base_cols
            mask = cols < N
            # DLOGITS的内存就对应REF_LOGITS，废物再利用
            tl.store(DLOGITS+cols, dlogits, mask=mask)


class _GrpoLoss(torch.autograd.Function):

    @staticmethod
    def forward(ctx, logits, ref_logp, input_ids, advantages, beta, completion_mask, save_kl, inplace):
        # 设计思路：
        # 为什么输入是模型的原始输出，而不是logits[:, :-1]？
        # triton一般需要tensor是连续的，如果不连续，处理起来很麻烦
        # 而logits[:, :-1].contiguous() 会创建一个新的张量，增加显存开销
        # 实际上我们在内部计算时，忽略掉最后一个位置即可
        assert logits.is_contiguous() and ref_logp.is_contiguous()
        ctx.input_shape = logits.shape
        B, L_ADD_1, N = ctx.input_shape
        L = L_ADD_1 - 1 
        M = B * L # 我们实际需要计算的长度是 B * (L + 1 - 1)个行向量户即可
        # input_ids也需要是连续的， 如果是 input_ids[:, -logits_to_keep:]，这就不是连续的了
        # 当然也可以是input_ids[:, -logits_to_keep:].contiguous()，这少一个vocab_size维度，基本无开销
        # 但是我们也可以记录下output的起始位置，跳过prompt部分即可
        input_ids_start_index = input_ids.size(1) - L  
        # 下面都用fp32进行存储，因为都没有vocab_size这个维度，基本无额外显存开销，但是大大提高精度
        if not save_kl:
            loss = torch.zeros(B, L, device=logits.device, dtype=torch.float32) 
        else:
            loss = torch.zeros(B*2, L, device=logits.device, dtype=torch.float32) # 后一半存kl
        lse = torch.empty(B, L, device=logits.device, dtype=torch.float32)  # 等价 max(x) + logsumexp(x)，用于backward的快速计算

        if completion_mask is None:
            completion_mask = torch.ones(B,L, device=logits.device, dtype=torch.int32)
        else:
            loss[:B].masked_fill_(completion_mask.logical_not(), 0)
        kwargs = {'BLOCK_SIZE': 8192, 'num_warps': 8, 'num_stages':1}
        _grpo_loss_fwd[(M,)](logits, ref_logp, input_ids, advantages, completion_mask, beta,
                            loss, lse, save_kl,
                            M, N, L, input_ids_start_index,
                            **kwargs,
                            )
        ctx.beta = beta
        ctx.save_for_backward(lse, logits, input_ids, advantages, completion_mask)
        ctx.ref_logp = ref_logp
        ctx.inplace = inplace
        return loss
    
    @staticmethod
    def backward(ctx, dloss):
        # logits对应的grad来自两个部分，reward部分和kl部分
        if not dloss.is_contiguous():
            dloss = dloss.contiguous()
        lse, logits, input_ids, advantages, completion_mask = ctx.saved_tensors
        B, L_ADD_1, N = ctx.input_shape
        L = L_ADD_1 - 1
        M = B * L
        input_ids_start_index = input_ids.size(1) - L
        # 实际上当我们读取一些logits的值后，这个张量就一点用都没有了
        # 我们直接把logits的grad用logits存储，直接废物再利用，节省显存
        dlogits = logits if ctx.inplace else torch.empty_like(logits)
        kwargs = {'BLOCK_SIZE': 8192, 'num_warps': 32, 'num_stages':4}
        _grpo_loss_bwd[(M,)](dloss, dlogits, 
                            logits, ctx.ref_logp, input_ids, advantages, completion_mask, ctx.beta,
                            lse,
                            N, L, input_ids_start_index,
                            **kwargs
                                )
        # 最后一个位置的token并没有参与计算，梯度需要设置为0
        # 因为empty的初始化或者ref_logits的初始化，该位置都不是0，需要手动设置下
        dlogits[:, -1, :] = 0
        return dlogits.view(*ctx.input_shape), None, None, None, None, None, None, None

def triton_grpo_loss(logits, ref_logp, input_ids, advantages, beta=0.1, completion_mask=None, save_kl=False, inplace=True) -> torch.Tensor:
    '''
    compute grpo loss, save memory(no addition usage) and fast speed(6X for A800)

    Args:
        logtits: Tensor, [B, L+1, vocab_size], the origin output of model, it's not logits[:, :-1]
        ref_logp: Tensor, [B, L], the origin output of model, it's not ref_logits[:, :-1]
        input_ids: Tensor, [B, K+L], it's prompt_completion_id, it contains the prompt ids and output ids
        advantages: Tensor, [B], the advantages of each prompt
        beta: float, the weight of kl loss
        completion_mask: Tensor, loss mask
        save_kl: bool, if true will save kl
        inplace: bool, if true, in backward use logits to store the logits's grad, it can save memory

    Retutn:
        loss: Tensor, [B, L], the loss of grpo, it contains the advantage part and kl part

    NOTE: logits(ref_logits) is computed by these steps
        logits_to_keep = completion_ids.size(1)

        def get_per_token_logits(model, input_ids, attention_mask, logits_to_keep):
            # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
            logits = model(
                input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1
            ).logits
            return logits
            
        logits = get_per_token_logits(model, prompt_completion_ids, attention_mask, logits_to_keep)
    '''
    out = _GrpoLoss.apply(logits, ref_logp, input_ids, advantages, beta, completion_mask, save_kl, inplace)
    if not save_kl:
        return out
    else:
        return out.chunk(2, axis=0)
      
def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    if return_outputs:
        raise ValueError("The GRPOTrainer does not support returning outputs")
    # Compute the per-token log probabilities for the model

    prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
    completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
    logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

    logits = model(input_ids, attention_mask=attention_mask, num_logits_to_keep=logits_to_keep + 1).logits
    # Compute the KL divergence between the model and the reference model
    ref_per_token_logps = inputs["ref_per_token_logps"]

    # x - x.detach() allows for preserving gradients from x
    advantages = inputs["advantages"]
    per_token_loss, per_token_kl = triton_grpo_loss(logits,
                                                    ref_per_token_logps,
                                                    input_ids,
                                                    advantages,
                                                    self.beta,
                                                    completion_mask=completion_mask,
                                                    save_kl=True,
                                                    inplace=True
                                                    )
    loss = (per_token_loss.sum(dim=1) / completion_mask.sum(dim=1)).mean()

    # Log the metrics
    completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
    self._metrics["completion_length"].append(completion_length)

    mean_kl = (per_token_kl.sum(dim=1) / completion_mask.sum(dim=1)).mean()
    self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

    return loss



target_module = importlib.import_module('trl')
target_module.GRPOTrainer.compute_loss = compute_loss

trigger = None

