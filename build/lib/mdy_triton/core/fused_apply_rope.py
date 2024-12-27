import torch
import triton
import triton.language as tl



@triton.jit
def _fused_apply_rope_fwd(Q, K, COS, SIN,
                          Q_EMBED, K_EMBED,
                          stride_qb, stride_qh, stride_ql, stride_qd,
                          stride_cb, stride_cl, stride_cd,
                          B, COS_B, L, QH, KH, D:tl.constexpr, HALF_D:tl.constexpr,
                          BLOCK_QH:tl.constexpr, BLOCK_KH:tl.constexpr,
                          ):
    pid = tl.program_id(0)
    off_b = pid // L
    off_l = pid % L
    q_offset = pid * stride_ql
    factor = QH // KH
    k_offset = pid * stride_ql // factor
    if B == COS_B:
        cos_offset = pid * stride_cl
    else:
        cos_offset = off_l * stride_cl
    # cos_offset = pid * stride_cl
    Q += q_offset
    Q_EMBED += q_offset
    K += k_offset
    K_EMBED += k_offset
    COS += cos_offset
    SIN += cos_offset
    q_block_ptrs = tl.make_block_ptr(
        base=Q,
        shape=(QH, D),
        strides=(stride_qh, stride_qd),
        offsets=(0,0),
        block_shape=(BLOCK_QH, HALF_D),
        order=(1,0)
    )
    qembed_block_ptrs = tl.make_block_ptr(
        base=Q_EMBED,
        shape=(QH, D),
        strides=(stride_qh, stride_qd),
        offsets=(0,0),
        block_shape=(BLOCK_QH, HALF_D),
        order=(1,0)
    )
    k_block_ptrs = tl.make_block_ptr(
        base=K,
        shape=(KH, D),
        strides=(stride_qh, stride_qd),
        offsets=(0,0),
        block_shape=(BLOCK_KH, HALF_D),
        order=(1,0)
    )
    kembed_block_ptrs = tl.make_block_ptr(
        base=K_EMBED,
        shape=(KH, D),
        strides=(stride_qh, stride_qd),
        offsets=(0,0),
        block_shape=(BLOCK_KH, HALF_D),
        order=(1,0)
    )

    cols1 = tl.arange(0, HALF_D)
    cols2 = tl.arange(HALF_D, D)
    cos1 = tl.load(COS + cols1)
    dtype = cos1.dtype
    cos1 = cos1.to(tl.float32)
    cos2 = tl.load(COS + cols2)
    sin1 = tl.load(SIN + cols1)
    sin2 = tl.load(SIN + cols2)

    q1 = tl.load(q_block_ptrs, boundary_check=(0,)).to(tl.float32)
    q_block_ptrs = tl.advance(q_block_ptrs, offsets=(0, HALF_D))
    q2 = tl.load(q_block_ptrs, boundary_check=(0,)).to(tl.float32)
    q_embed1 = q1 * cos1 - q2 * sin1
    q_embed2 = q2 * cos2 + q1 * sin2
    tl.store(qembed_block_ptrs, q_embed1.to(dtype), boundary_check=(0,))
    qembed_block_ptrs = tl.advance(qembed_block_ptrs, offsets=(0, HALF_D))
    tl.store(qembed_block_ptrs, q_embed2.to(dtype), boundary_check=(0,))
    
    k1 = tl.load(k_block_ptrs, boundary_check=(0,)).to(tl.float32)
    k_block_ptrs = tl.advance(k_block_ptrs, offsets=(0, HALF_D))
    k2 = tl.load(k_block_ptrs, boundary_check=(0,)).to(tl.float32)
    k_embed1 = k1 * cos1 - k2 * sin1
    k_embed2 = k2 * cos2 + k1 * sin2
    tl.store(kembed_block_ptrs, k_embed1.to(dtype), boundary_check=(0,))
    kembed_block_ptrs = tl.advance(kembed_block_ptrs, offsets=(0, HALF_D))
    tl.store(kembed_block_ptrs, k_embed2.to(dtype), boundary_check=(0,))




@triton.jit
def _fused_apply_rope_bwd(DQ_EMBED, DK_EMBED, COS, SIN,
                          DQ, DK, 
                        stride_qb, stride_qh, stride_ql, stride_qd,
                        stride_kb, stride_kh, stride_kl, stride_kd,
                        stride_cb, stride_cl, stride_cd,
                        B, COS_B, L, QH, KH, D: tl.constexpr, HALF_D: tl.constexpr,
                        BLOCK_QH:tl.constexpr, BLOCK_KH:tl.constexpr,
                        ):
    pid = tl.program_id(0)
    off_b = pid // L
    off_l = pid % L
    dq_offset = off_b * stride_qb + stride_ql * off_l
    dk_offset = off_b * stride_kb  + stride_kl * off_l
    if B == COS_B:
        cos_offset = pid * stride_cl
    else:
        cos_offset = off_l * stride_cl
    # cos_offset = pid * stride_cl
    DQ += dq_offset
    DQ_EMBED += dq_offset
    DK += dk_offset
    DK_EMBED += dk_offset
    COS += cos_offset
    SIN += cos_offset

    dq_block_ptrs = tl.make_block_ptr(
        base=DQ,
        shape=(QH, D),
        strides=(stride_qh, stride_qd),
        offsets=(0,0),
        block_shape=(BLOCK_QH, HALF_D),
        order=(1,0)
    )
    dq_embed_block_ptrs = tl.make_block_ptr(
        base=DQ_EMBED,
        shape=(QH, D),
        strides=(stride_qh, stride_qd),
        offsets=(0,0),
        block_shape=(BLOCK_QH, HALF_D),
        order=(1,0)
    )
    dk_block_ptrs = tl.make_block_ptr(
        base=DK,
        shape=(KH, D),
        strides=(stride_kh, stride_kd),
        offsets=(0,0),
        block_shape=(BLOCK_KH, HALF_D),
        order=(1,0)
    )
    dk_embed_block_ptrs = tl.make_block_ptr(
        base=DK_EMBED,
        shape=(KH, D),
        strides=(stride_kh, stride_kd),
        offsets=(0,0),
        block_shape=(BLOCK_KH, HALF_D),
        order=(1,0)
    )
    cols1 = tl.arange(0, HALF_D)
    cols2 = tl.arange(HALF_D, D)
    cos1 = tl.load(COS + cols1)
    dtype = cos1.dtype
    cos1 = cos1.to(tl.float32)
    cos2 = tl.load(COS + cols2)
    sin1 = tl.load(SIN + cols1)
    sin2 = tl.load(SIN + cols2)

    dq_embed1 = tl.load(dq_embed_block_ptrs, boundary_check=(0,)).to(tl.float32)
    dq_embed_block_ptrs = tl.advance(dq_embed_block_ptrs, offsets=(0, HALF_D))
    dq_embed2 = tl.load(dq_embed_block_ptrs, boundary_check=(0,)).to(tl.float32)
    dq1 = dq_embed1 * cos1 + sin2 * dq_embed2
    dq2 = dq_embed2 * cos2 - sin1 * dq_embed1
    tl.store(dq_block_ptrs, dq1.to(dtype), boundary_check=(0,))
    dq_block_ptrs = tl.advance(dq_block_ptrs, offsets=(0, HALF_D))
    tl.store(dq_block_ptrs, dq2.to(dtype), boundary_check=(0,))

    dk_embed1 = tl.load(dk_embed_block_ptrs, boundary_check=(0,)).to(tl.float32)
    dk_embed_block_ptrs = tl.advance(dk_embed_block_ptrs, offsets=(0, HALF_D))
    dk_embed2 = tl.load(dk_embed_block_ptrs, boundary_check=(0,)).to(tl.float32)
    dk1 = dk_embed1 * cos1 + sin2 * dk_embed2
    dk2 = dk_embed2 * cos2 - sin1 * dk_embed1
    tl.store(dk_block_ptrs, dk1.to(dtype), boundary_check=(0,))
    dk_block_ptrs = tl.advance(dk_block_ptrs, offsets=(0, HALF_D))
    tl.store(dk_block_ptrs, dk2.to(dtype), boundary_check=(0,))

class _FusedApplyRope(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin):
        '''
        input:
            q    : torch.Tensor, [bs, qh, L, D]
            k    : torch.Tensor, [bs, kh, L, D]
            cos  : torch.Tensor, [bs, L, D] or [1, L, D]
            sin  : torch.Tensor, [bs, L, D] or [1, L, D]
        output:
            q_embed : torch.tensor, [bs, qh, L, D]
            k_embed : torch.tensor, [bs, kh, L, D]

        example:
          original code:
            # the function in hf-llama or hf-qwen
            q_embed, k_embed = apply_rotary_pos_emb(q, k, cos, sin)

          new code:
            q_embed, k_embed = fused_apply_rope(q, k, cos, sin)

        note:
            the input of q and k is not contiguous
            the contiguous axis order is [bs, L, h, D]
        '''
        assert q.transpose(1,2).is_contiguous()
        # print(q.stride(), k.stride())
        B, QH, L, D = q.shape
        HALF_D = D // 2
        KH = k.size(1)
        assert (D % 32 == 0) or (D % 64 == 0) or (D % 128 == 0)
        num_stages=4
        num_warps=8

        # print(qr.stride())
        q_embed = torch.empty(B, L, QH, D, device=q.device, dtype=k.dtype)
        k_embed = torch.empty(B, L, KH, D, device=q.device, dtype=k.dtype)
        M = B*L
        COS_B = cos.shape[0]
        BLOCK_QH = triton.next_power_of_2(QH)
        BLOCK_KH = triton.next_power_of_2(KH)
        _fused_apply_rope_fwd[(M,)](q,k,cos, sin,
                                    q_embed, k_embed,
                                    *q.stride(),
                                    *cos.stride(),
                                    B, COS_B, L, QH, KH, D, HALF_D,
                                    BLOCK_QH, BLOCK_KH,
                                    num_warps=num_warps, num_stages=num_stages

        )

        ctx.save_for_backward(cos, sin)
        ctx.infos = (B, QH, KH, L, D, HALF_D, M, COS_B, BLOCK_QH, BLOCK_KH)
        ctx.num_warps = num_warps
        ctx.num_stages = num_stages
        return q_embed.transpose(1,2), k_embed.transpose(1,2)

    @staticmethod
    def backward(ctx, dq_embed, dk_embed):
        B, QH, KH, L, D, HALF_D, M, COS_B, BLOCK_QH, BLOCK_KH = ctx.infos
        cos,sin = ctx.saved_tensors
        dq = torch.empty_like(dq_embed)
        dk = torch.empty_like(dk_embed)
        _fused_apply_rope_bwd[(M,)](dq_embed, dk_embed, cos, sin,
                                    dq, dk,
                                    *dq.stride(),
                                    *dk.stride(),
                                    *cos.stride(),
                                    B, COS_B, L, QH, KH, D, HALF_D, BLOCK_QH, BLOCK_KH,
                                    num_warps=ctx.num_warps, num_stages=ctx.num_stages
                                    )
        return dq, dk, None, None

fused_apply_rope = _FusedApplyRope.apply