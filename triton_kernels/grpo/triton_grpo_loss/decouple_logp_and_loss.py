import torch
import triton
import triton.language as tl

# @triton.autotune([triton.Config({"BLOCK_N":BLOCK_N}, num_stages=ns, num_warps=nw)
#                   for BLOCK_N in [2048, 4096, 8192]
#                   for ns in [1, 2, 4]
#                   for nw in [1, 2, 4, 8, 16]],
#                   key=['N'])
@triton.jit
def _fused_logp_fwd_kernel( LOGITS,
                            INPUT_IDS,
                            LOGP,
                            LSE,
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
    LOGP += off_b * L + off_l
    LSE += off_b * L + off_l

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
    tl.store(LOGP, logp)
    tl.store(LSE, lse)
    
# @triton.autotune([triton.Config({"BLOCK_N":BLOCK_N}, num_stages=ns, num_warps=nw)
#                   for BLOCK_N in [2048, 4096, 8192]
#                   for ns in [1, 2, 4]
#                   for nw in [1, 2, 4, 8, 16]],
#                   key=['N'])
@triton.jit
def _fused_logp_bwd_kernel( DLOGP,
                            DLOGITS,
                            LOGITS,
                            INPUT_IDS,
                            LSE,
                            MASK,
                            TEMPERATURE,
                            stride_input_ids_b,
                            L: tl.constexpr,
                            N: tl.constexpr,
                            BLOCK_N:tl.constexpr=4096):
    off_b = tl.program_id(0).cast(tl.int64)
    off_l = tl.program_id(1).cast(tl.int64)

    
    DLOGITS += off_b * (L+1) * N + off_l * N
    if MASK is not None:
        MASK += off_b * stride_input_ids_b + off_l
        not_skip = tl.load(MASK)
        if not_skip == 0:
            for start in range(0, N, BLOCK_N):
                cols = tl.arange(0, BLOCK_N) + start
                tl.store(DLOGITS+cols, 0., mask=cols<N)
            return
        
    LOGITS += off_b * (L+1) * N + off_l * N
    INPUT_IDS += off_b * stride_input_ids_b + off_l
    DLOGP += off_b * L + off_l
    LSE += off_b * L + off_l

    dlogp = tl.load(DLOGP).to(tl.float32) / TEMPERATURE
    lse = tl.load(LSE)
    idx = tl.load(INPUT_IDS)
    for start_n in tl.range(0, N, BLOCK_N):
        cols = start_n + tl.arange(0, BLOCK_N)
        logits = tl.load(LOGITS+cols, mask=cols < N, other=0.).to(tl.float32) / TEMPERATURE
        probs = tl.exp(logits - lse)
        dlogits = tl.where(cols==idx, 1-probs, -probs) * dlogp
        tl.store(DLOGITS+cols, dlogits, mask=cols < N)

class _FusedLogp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, input_ids, temperature, mask, inplace, grad_enable):
        assert logits.is_contiguous()
        B, L_ADD_1, N = logits.shape
        L = L_ADD_1 - 1
        input_ids = input_ids[:, -L:]
        if mask is not None:
            mask = mask[:, -L:]
        logp = torch.zeros(B, L, dtype=torch.float32, device=logits.device)
        lse = torch.zeros(B, L, dtype=torch.float32, device=logits.device)
        kwargs = {"BLOCK_N":2048, "num_stages":4, "num_warps":1}
        _fused_logp_fwd_kernel[(B, L)]( logits,
                                        input_ids,
                                        logp,
                                        lse,
                                        mask,
                                        temperature,
                                        input_ids.stride(0),
                                        L,
                                        N,
                                        **kwargs
                                        )
        if grad_enable:
            ctx.save_for_backward(logits, input_ids, mask, lse)
            ctx.temperature = temperature
            ctx.inplace = inplace
        return logp
    
    @staticmethod
    def backward(ctx, dlogp):
        # dlogp = args[0]
        logits, input_ids, mask, lse = ctx.saved_tensors
        B, L_ADD_1, N = logits.shape
        L = L_ADD_1 - 1
        dlogits = logits if ctx.inplace else torch.empty_like(dlogits)
        kwargs = {"BLOCK_N":8192, "num_stages":4, "num_warps":16}
        _fused_logp_bwd_kernel[(B, L)]( dlogp,
                                        dlogits,
                                        logits,
                                        input_ids,
                                        lse,
                                        mask,
                                        ctx.temperature,
                                        input_ids.stride(0),
                                        L,
                                        N,
                                        **kwargs
                                        )
        dlogits[:, -1] = 0
        return dlogits, None, None, None, None, None

def fused_selective_log_softmax(logits, 
                                input_ids, 
                                temperature=0.9, 
                                mask=None, 
                                inplace=True
                                ):
    
    grad_enable = torch.is_grad_enabled()

    return _FusedLogp.apply(logits,
                            input_ids,
                            temperature,
                            mask,
                            inplace,
                            grad_enable)

@torch.compile
def compile_grpo_loss(per_token_logps, 
                       advantages, 
                       old_per_token_logps=None, 
                       ref_per_token_logps=None, 
                       beta=0.04, 
                       epsilon_low=0.2, 
                       epsilon_high=0.4,
                       ):
    per_token_kl = None
    if beta != 0.0:
        per_token_kl = (
            torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        )
    # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's computation (see
    # _generate_and_score_completions) and use per_token_logps.detach() instead.
    old_per_token_logps = old_per_token_logps if old_per_token_logps is not None else per_token_logps.detach()
    coef_1 = torch.exp(per_token_logps - old_per_token_logps)
    coef_2 = torch.clamp(coef_1, 1 - epsilon_low, 1 + epsilon_high)
    per_token_loss1 = coef_1 * advantages.unsqueeze(1)
    per_token_loss2 = coef_2 * advantages.unsqueeze(1)
    per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
    if beta != 0.0:
        per_token_loss = per_token_loss + beta * per_token_kl
    is_clipped = (per_token_loss1 < per_token_loss2).float()
    return per_token_loss, per_token_kl, is_clipped
    


# @triton.autotune([triton.Config({"BLOCK_N":BLOCK_N}, num_stages=ns, num_warps=nw)
#                   for BLOCK_N in [2048, 4096, 8192]
#                   for ns in [1, 2, 4]
#                   for nw in [1, 2, 4, 8, 16]],
#                   key=['N'])
@triton.jit
def _grpo_loss_fwd_kernel(LOGP,
                         OLD_LOGP,
                         REF_LOGP,
                        ADVANTAGES,
                        LOSS,
                        KL,
                        IS_CLIPPED,
                        BETA:tl.constexpr,
                        EPS_LOW,
                        EPS_HIGH,
                        L: tl.constexpr,
                        ):
    off_b = tl.program_id(0).cast(tl.int64)
    off_l = tl.program_id(1).cast(tl.int64)

    LOGP += off_b * L + off_l
    ADVANTAGES += off_b
    LOSS += off_b * L + off_l
    
    logp = tl.load(LOGP)
    advantage = tl.load(ADVANTAGES).to(tl.float32)
    if OLD_LOGP is None:
        per_token_loss = -advantage
    else:
        OLD_LOGP += off_b * L + off_l
        IS_CLIPPED += off_b * L + off_l
        old_logp = tl.load(OLD_LOGP).to(tl.float32)
        coef_1 = tl.exp(logp - old_logp)
        coef_2 = tl.clamp(coef_1, 1 - EPS_LOW, 1 + EPS_HIGH)
        per_token_loss1 = coef_1 * advantage
        per_token_loss2 = coef_2 * advantage
        per_token_loss = -tl.minimum(per_token_loss1, per_token_loss2)
        is_clipped = per_token_loss1 < per_token_loss2
        tl.store(IS_CLIPPED, is_clipped)

    if BETA != 0.0:
        REF_LOGP += off_b * L + off_l
        KL += off_b * L + off_l
        ref_logp = tl.load(REF_LOGP).to(tl.float32)
        kl = tl.exp(ref_logp - logp) - (ref_logp - logp) - 1
        per_token_loss += BETA * kl
        tl.store(KL, kl)
        
    tl.store(LOSS, per_token_loss)
    

# @triton.autotune([triton.Config({"BLOCK_N":BLOCK_N}, num_stages=ns, num_warps=nw)
#                   for BLOCK_N in [2048, 4096, 8192]
#                   for ns in [1, 2, 4]
#                   for nw in [1, 2, 4, 8, 16]],
#                   key=['N'])  
@triton.jit
def _grpo_loss_bwd_kernel(DLOSS,
                          DLOGP,
                          LOGP,
                         OLD_LOGP,
                         REF_LOGP,
                        ADVANTAGES,
                        BETA:tl.constexpr,
                        EPS_LOW,
                        EPS_HIGH,
                        loss_stride0,
                        loss_stride1,
                        L: tl.constexpr,
                        ):

    off_b = tl.program_id(0).cast(tl.int64)
    off_l = tl.program_id(1).cast(tl.int64)
    
    
    DLOSS += off_b * loss_stride0 + off_l * loss_stride1
    DLOGP += off_b * L + off_l
    LOGP += off_b * L + off_l
    ADVANTAGES += off_b

    dloss = tl.load(DLOSS).to(tl.float32)
    logp = tl.load(LOGP)
    advantage = tl.load(ADVANTAGES).to(tl.float32
                                       )
    if OLD_LOGP is None:
        dlogp = -advantage
    else:
        OLD_LOGP += off_b * L + off_l
        old_logp = tl.load(OLD_LOGP).to(tl.float32)
        coef_1 = tl.exp(logp - old_logp)
        coef_2 = tl.clamp(coef_1, 1 - EPS_LOW, 1 + EPS_HIGH)
        per_token_loss1 = coef_1 * advantage
        per_token_loss2 = coef_2 * advantage
        mask = per_token_loss2 >= per_token_loss1
        dlogp = -per_token_loss1 * mask

    if BETA != 0.0:
        REF_LOGP += off_b * L + off_l
        ref_logp = tl.load(REF_LOGP).to(tl.float32)
        dlogp += BETA * (1 - tl.exp(ref_logp - logp))
    
    dlogp = dlogp * dloss
    tl.store(DLOGP, dlogp)
        

class _GrpoLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logp, advantages, old_logp, ref_logp, beta, eps_low, eps_high):
        assert logp.is_contiguous()
        assert old_logp is None or old_logp.is_contiguous()
        assert (ref_logp is not None and ref_logp.is_contiguous()) if beta != 0.0 else True 
        B, L = logp.shape

        loss = torch.zeros(B, L, device=logp.device, dtype=torch.float32)
        kl = torch.zeros_like(loss) if beta != 0.0 else None
        is_clipped = torch.zeros_like(loss)
        # kwargs = {"BLOCK_N":2048, "num_stages":2, "num_warps":1}
        _grpo_loss_fwd_kernel[(B, L)](logp,
                                     old_logp,
                                     ref_logp,
                                     advantages,
                                     loss,
                                     kl,
                                     is_clipped,
                                     beta,
                                     eps_low,
                                     eps_high,
                                     L,
                                    #  **kwargs
                                     )
        ctx.save_for_backward(logp, old_logp, ref_logp, advantages)
        ctx.infos = (beta, eps_low, eps_high)
        # return loss
        return loss, kl, is_clipped
    
    @staticmethod
    def backward(ctx, *args):
        dloss = args[0]
        logp, old_logp, ref_logp, advantages = ctx.saved_tensors
        beta, eps_low, eps_high = ctx.infos
        B, L = logp.shape
        dlogp = torch.empty_like(logp)
        # kwargs = {"BLOCK_N":4096, "num_stages":1, "num_warps":16}
        _grpo_loss_bwd_kernel[(B, L)](dloss,
                                      dlogp,
                                      logp,
                                      old_logp,
                                      ref_logp,
                                      advantages,
                                      beta,
                                      eps_low,
                                      eps_high,
                                      *dloss.stride(),
                                      L,
                                        )
        return dlogp, None,None,None,None,None,None


def triton_grpo_loss(per_token_logps, 
                       advantages, 
                       old_per_token_logps=None, 
                       ref_per_token_logps=None, 
                       beta=0.04, 
                       epsilon_low=0.2, 
                       epsilon_high=0.4,
                       ):
    return _GrpoLoss.apply( per_token_logps, 
                            advantages, 
                            old_per_token_logps, 
                            ref_per_token_logps, 
                            beta, 
                            epsilon_low, 
                            epsilon_high
                            )

