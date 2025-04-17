import torch
import triton
import triton.language as tl
from copy import deepcopy
import os
from tqdm import tqdm
import time
os.environ['TRITON_PRINT_AUTOTUNING'] = '1'
from transformers import Qwen2ForCausalLM
from trl import GRPOTrainer
import torch.nn.functional as F

def selective_log_softmax(logits, input_ids, temperature=0.9):
    """
    A memory-efficient implementation of the common `log_softmax -> gather` operation.

    This function is equivalent to the following naive implementation:
    ```python
    logps = torch.gather(logits.log_softmax(-1), dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
    ```

    Args:
        logits (`torch.Tensor`):
            Logits tensor of shape `(..., num_classes)`.
        index (`torch.Tensor`):
            Index tensor of shape `(...)`, specifying the positions to gather from the log-softmax output.

    Returns:
        `torch.Tensor`:
            Gathered log probabilities with the same shape as `index`.
    """
    logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
    logits_to_keep = logits.size(1)
    index = input_ids[:, -logits_to_keep:]
    # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
    # See https://github.com/huggingface/trl/issues/2770
    logits = logits[:, -logits_to_keep:]
    # Divide logits by sampling temperature.
    # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
    logits = logits / temperature
    # return torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = selected_logits - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficent approach
        per_token_logps = []
        for row_logits, row_labels in zip(logits, index):  # loop to reduce peak mem consumption
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
    return per_token_logps


# 代码是根据trl仓库改的，因此triton实现也是根据这个仓库的实现方式进行改进的
# 最主要的就是p(x)和p_old(x)是一样的
def get_log_probs(logits, input_ids):
    per_token_logps = []
    for logits_row, input_ids_row in zip(logits, input_ids[:, -logits.size(1):]):
        log_probs = logits_row.log_softmax(dim=-1)
        token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
        per_token_logps.append(token_log_prob)
    return torch.stack(per_token_logps)

def torch_grpo_loss(logits, old_logp, ref_logp, completion_ids, advantages, completion_mask, temperature, beta, eps_low, eps_high):
    # logits通过以下计算得到
    # logits_to_keep = completion_ids.size(1)
    # logits = model(input_ids=input_ids, 
    #             attention_mask=attention_mask,
    #             logits_to_keep=logits_to_keep + 1).logits
    # 传ref_logp（bs*L）而不是ref_logits的原因是，该值可以在inference_mode()下得到，
    # 无需保存中间结果，ref_logits会浪费显存
    assert logits.is_contiguous() and ref_logp.is_contiguous()
    logits = logits[:, :-1] # 错一位，对应下一个输入token的概率         
    per_token_logps = get_log_probs(logits / temperature, completion_ids) # logits是需要计算梯度，因此会保存中间结果log_probs
    ref_per_token_logps = ref_logp

    if old_logp is None:
        old_logp = per_token_logps.detach()
    coef_1 = torch.exp(per_token_logps - old_logp)
    coef_2 = torch.clamp(coef_1, 1 - eps_low, 1 + eps_high)
    per_token_loss1 = coef_1 * advantages.unsqueeze(1)
    per_token_loss2 = coef_2 * advantages.unsqueeze(1)
    per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

    per_token_kl = None
    if beta != 0.0:
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        if completion_mask is not None:
            per_token_kl *= completion_mask
        per_token_loss = per_token_loss + beta * per_token_kl
    is_clipped = per_token_loss1 < per_token_loss2
    return per_token_loss, per_token_kl, is_clipped

# @triton.autotune([triton.Config({"BLOCK_N":BLOCK_N}, num_stages=ns, num_warps=nw)
#                   for BLOCK_N in [2048, 4096, 8192]
#                   for ns in [1, 2, 4]
#                   for nw in [1, 2, 4, 8, 16]],
#                   key=['N'])
@triton.jit
def _selective_log_softmax_kernel(LOGITS,
                                  INPUT_IDS,
                                  LOG_P,
                                  MASK,
                                  TEMPERATURE,
                                  stride_input_ids_b,
                                  L: tl.constexpr,
                                  N: tl.constexpr,
                                  BLOCK_N:tl.constexpr=4096):
    off_b = tl.program_id(0).cast(tl.int64)
    off_l = tl.program_id(1).cast(tl.int64)

    LOGITS += off_b * (L+1) * N + off_l * N
    INPUT_IDS += off_b * stride_input_ids_b + off_l
    LOG_P += off_b * L + off_l

    
    if MASK is not None:
        MASK += off_b * stride_input_ids_b + off_l
        not_skip = tl.load(MASK)
        if not_skip == 0:
            return

    m_i = float('-inf')
    l_i = 0. 
    for start in range(0, N, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        logits = tl.load(LOGITS + cols, mask=cols < N, other=float('-inf')).to(tl.float32) / TEMPERATURE
        new_m_i = tl.maximum(m_i, tl.max(logits))
        alpha = tl.exp(m_i - new_m_i)
        l_i = l_i * alpha + tl.sum(tl.exp(logits - new_m_i))
        m_i = new_m_i
    lse = m_i + tl.log(l_i)

    ids = tl.load(INPUT_IDS)
    x = tl.load(LOGITS + ids).to(tl.float32) / TEMPERATURE
    logp = x - lse
    tl.store(LOG_P, logp)
    

def fused_selective_log_softmax(logits:torch.Tensor, input_ids:torch.Tensor, temperature:float=0.9, mask=None):
    assert logits.is_contiguous()
    B, L_ADD_1, N = logits.shape
    L = L_ADD_1 - 1
    input_ids = input_ids[:, -L:]
    if mask is not None:
        mask = mask[:, -L:]
    log_p = torch.zeros(B, L, dtype=torch.float32, device=logits.device)
    kwargs = {"BLOCK_N":2048, "num_stages":4, "num_warps":1}
    _selective_log_softmax_kernel[(B, L)](logits,
                                          input_ids,
                                          log_p,
                                          mask,
                                          temperature,
                                          input_ids.stride(0),
                                          L,
                                          N,
                                          **kwargs
                                          )
    return log_p


@triton.jit
def _grpo_loss_fwd_kernel(LOGITS,
                         OLD_LOGP,
                         REF_LOGP,
                        INPUT_IDS,
                        COMPLETION_MASK,
                        ADVANTAGES,
                        LOSS,
                        LSE,
                        KL,
                        IS_CLIPPED,
                        TEMPERATURE,
                        BETA,
                        EPS_LOW,
                        EPS_HIGH,
                        L: tl.constexpr,
                        N: tl.constexpr,
                        BLOCK_N:tl.constexpr=4096):
    off_b = tl.program_id(0).cast(tl.int64)
    off_l = tl.program_id(1).cast(tl.int64)

    if COMPLETION_MASK is not None:
        COMPLETION_MASK += off_b * L + off_l
        not_skip = tl.load(COMPLETION_MASK)
        if not_skip == 0:
            return
        
    LOGITS += off_b * (L+1) * N + off_l * N
    INPUT_IDS += off_b * L + off_l
    ADVANTAGES += off_b
    LOSS += off_b * L + off_l
    LSE += off_b * L + off_l
    IS_CLIPPED += off_b * L + off_l
        
    m_i = float('-inf')
    l_i = 0. 
    for start in range(0, N, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        logits = tl.load(LOGITS + cols, mask=cols < N, other=float('-inf')).to(tl.float32) / TEMPERATURE
        new_m_i = tl.maximum(m_i, tl.max(logits))
        alpha = tl.exp(m_i - new_m_i)
        l_i = l_i * alpha + tl.sum(tl.exp(logits - new_m_i))
        m_i = new_m_i
    lse = m_i + tl.log(l_i)

    ids = tl.load(INPUT_IDS)
    x = tl.load(LOGITS + ids).to(tl.float32) / TEMPERATURE
    logp = x - lse
    if OLD_LOGP is None:
        old_logp = logp
    else:
        OLD_LOGP += off_b * L + off_l
        old_logp = tl.load(OLD_LOGP).to(tl.float32)
    coef_1 = tl.exp(logp - old_logp)
    coef_2 = tl.clamp(coef_1, 1 - EPS_LOW, 1 + EPS_HIGH)
    advantage = tl.load(ADVANTAGES).to(tl.float32)
    per_token_loss1 = coef_1 * advantage
    per_token_loss2 = coef_2 * advantage
    per_token_loss = -tl.minimum(per_token_loss1, per_token_loss2)
    is_clipped = per_token_loss1 < per_token_loss2

    if BETA != 0.0:
        REF_LOGP += off_b * L + off_l
        KL += off_b * L + off_l
        ref_logp = tl.load(REF_LOGP).to(tl.float32)
        kl = tl.exp(ref_logp - logp) - (ref_logp - logp) - 1
        tl.store(KL, kl)
        per_token_loss += BETA * kl
    
    tl.store(LOSS, per_token_loss)
    tl.store(LSE, lse)
    tl.store(IS_CLIPPED, is_clipped)
    
@triton.jit
def _grpo_loss_bwd_kernel(DLOSS,
                        DLOGITS,
                         OLD_LOGP,
                         REF_LOGP,
                        INPUT_IDS,
                        COMPLETION_MASK,
                        ADVANTAGES,
                        LSE,
                        TEMPERATURE,
                        BETA,
                        EPS_LOW,
                        EPS_HIGH,
                        stride_input_ids_b,
                        L: tl.constexpr,
                        N: tl.constexpr,
                        BLOCK_N:tl.constexpr=4096):

    off_b = tl.program_id(0).cast(tl.int64)
    off_l = tl.program_id(1).cast(tl.int64)

    if COMPLETION_MASK is not None:
        COMPLETION_MASK += off_b * stride_input_ids_b + off_l
        not_skip = tl.load(COMPLETION_MASK)
        if not_skip == 0:
            for start in range(0, N, BLOCK_N):
                cols = tl.arange(0, BLOCK_N) + start
                tl.store(DLOGITS+cols, 0, mask=cols<N)
            return
        
    DLOSS += off_b * L + off_l
    DLOGITS += off_b * (L+1) * N + off_l * N
    INPUT_IDS += off_b * stride_input_ids_b + off_l
    ADVANTAGES += off_b
    LSE += off_b * L + off_l

    dloss = tl.load(DLOSS).to(tl.float32)
    lse = tl.load(LSE).to(tl.float32)

    idx = tl.load(INPUT_IDS)
    x = tl.load(DLOGITS + idx).to(tl.float32) / TEMPERATURE
    logp = x - lse
    if OLD_LOGP is None:
        old_logp = logp
    else:
        OLD_LOGP += off_b * L + off_l
        old_logp = tl.load(OLD_LOGP).to(tl.float32)
    coef_1 = tl.exp(logp - old_logp)
    coef_2 = tl.clamp(coef_1, 1 - EPS_LOW, 1 + EPS_HIGH)
    advantage = tl.load(ADVANTAGES).to(tl.float32)
    per_token_loss1 = coef_1 * advantage
    per_token_loss2 = coef_2 * advantage
    mask = per_token_loss2 >= per_token_loss1

    dlogp = -per_token_loss1 * mask
    if BETA != 0.0:
        REF_LOGP += off_b * L + off_l
        ref_logp = tl.load(REF_LOGP).to(tl.float32)
        dlogp += BETA * (1 - tl.exp(ref_logp - logp))
    
    dlogp = dlogp * dloss * TEMPERATURE
    
    for start_n in tl.range(0, N, BLOCK_N):
        cols = start_n + tl.arange(0, BLOCK_N)
        logits = tl.load(DLOGITS+cols, mask=cols < N, other=-float('inf')).to(tl.float32)
        probs = tl.exp(logits - lse)
        # dlogits = tl.where(cols==idx, 1-probs, -probs) * dlogp
        dlogits = -probs * dlogp
        tl.store(DLOGITS+cols, dlogits, mask=cols < N)
        
    dx = tl.load(DLOGITS+idx).to(tl.float32)
    dx += dlogp
    tl.store(DLOGITS+idx, dx)


class GrpoLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, old_logp, ref_logp, completion_ids, advantages, completion_mask, temperature, beta, eps_low, eps_high):
        assert logits.is_contiguous() and completion_ids.is_contiguous()
        assert old_logp is None or old_logp.is_contiguous()
        assert ref_logp is None or ref_logp.is_contiguous()
        
        B, L_ADD_1, N = logits.shape
        L = L_ADD_1 - 1

        if completion_mask is not None:
            assert completion_mask.is_contiguous()

        loss = torch.zeros(B, L, device=logits.device, dtype=torch.float32)
        lse = torch.zeros_like(loss)
        kl, is_clipped = None, None
        if beta != 0.0:
            kl = torch.zeros_like(loss)
            is_clipped = torch.zeros_like(loss, dtype=torch.int32).bool()

        _grpo_loss_fwd_kernel[(B, L)](logits,
                                     old_logp,
                                     ref_logp,
                                     completion_ids,
                                     completion_mask,
                                     advantages,
                                     loss,
                                     lse,
                                     kl,
                                     is_clipped,
                                     temperature,
                                     beta,
                                     eps_low,
                                     eps_high,
                                     L,
                                     N,
                                     )
        ctx.save_for_backward(logits, old_logp, ref_logp, completion_ids, advantages, completion_mask, lse)
        ctx.infos = (temperature, beta, eps_low, eps_high)
        return loss, kl, is_clipped
    
    @staticmethod
    def backward(ctx, *args):
        dloss = args[0].contiguous()
        logits, old_logp, ref_logp, completion_ids, advantages, completion_mask, lse = ctx.saved_tensors
        temperature, beta, eps_low, eps_high = ctx.infos
        B, L_ADD_1, N = logits.shape
        L = L_ADD_1 - 1
        dlogits = logits
        _grpo_loss_bwd_kernel[(B, L)](dloss,
                                      dlogits,
                                      old_logp,
                                      ref_logp,
                                      completion_ids,
                                      advantages,
                                      completion_mask,
                                      lse,
                                      temperature,
                                      beta,
                                      eps_low,
                                      eps_high,
                                      completion_ids.stride(0),
                                      L,
                                      N,
                                        )
        return dlogits, *[None]*9

def triton_grpo_loss(logits, old_logp, ref_logp, completion_ids, advantages, completion_mask, temperature, beta, eps_low, eps_high):
    return GrpoLoss.apply(logits, 
                          old_logp, 
                          ref_logp, 
                          completion_ids, 
                          advantages, 
                          completion_mask, 
                          temperature, 
                          beta, 
                          eps_low, 
                          eps_high)
