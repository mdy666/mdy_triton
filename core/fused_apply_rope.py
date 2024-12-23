import torch
import triton
import triton.language as tl



def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def rotate_half_v2(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((x2, -x1), dim=-1)

def rotate_half_v3(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

@triton.jit
def _fused_apply_rope_fwd(Q, K, QR, KR, COS, SIN,
                          Q_EMBED, K_EMBED,
                          stride_qb, stride_qh, stride_ql, stride_qd,
                          stride_qrb, stride_qrh, stride_qrl, stride_qrd,
                          stride_cb, stride_cl, stride_cd,
                          B, COS_B, L, QH, KH, D:tl.constexpr,BLOCK_QH:tl.constexpr, BLOCK_KH:tl.constexpr,
                          ):
    pid = tl.program_id(0)
    off_b = pid // L
    off_l = pid % L
    q_offset = pid * stride_ql
    qr_offset = off_b * stride_qrb + off_l * stride_qrl
    factor = QH // KH
    k_offset = pid * stride_ql // factor
    kr_offset = off_b * stride_qrb // factor + off_l * stride_qrl
    if B == COS_B:
        cos_offset = pid * stride_cl
    else:
        cos_offset = off_l * stride_cl
    # cos_offset = pid * stride_cl
    Q += q_offset
    QR += qr_offset
    Q_EMBED += q_offset
    K += k_offset
    KR += kr_offset
    K_EMBED += k_offset
    COS += cos_offset
    SIN += cos_offset
    q_block_ptrs = tl.make_block_ptr(
        base=Q,
        shape=(QH, D),
        strides=(stride_qh, stride_qd),
        offsets=(0,0),
        block_shape=(BLOCK_QH, D),
        order=(1,0)
    )
    qr_block_ptrs = tl.make_block_ptr(
        base=QR,
        shape=(QH, D),
        strides=(stride_qrh, stride_qrd),
        offsets=(0,0),
        block_shape=(BLOCK_QH, D),
        order=(1,0)
    )
    qembed_block_ptrs = tl.make_block_ptr(
        base=Q_EMBED,
        shape=(QH, D),
        strides=(stride_qh, stride_qd),
        offsets=(0,0),
        block_shape=(BLOCK_QH, D),
        order=(1,0)
    )
    k_block_ptrs = tl.make_block_ptr(
        base=K,
        shape=(KH, D),
        strides=(stride_qh, stride_qd),
        offsets=(0,0),
        block_shape=(BLOCK_KH, D),
        order=(1,0)
    )
    kr_block_ptrs = tl.make_block_ptr(
        base=KR,
        shape=(KH, D),
        strides=(stride_qrh, stride_qrd),
        offsets=(0,0),
        block_shape=(BLOCK_KH, D),
        order=(1,0)
    )
    kembed_block_ptrs = tl.make_block_ptr(
        base=K_EMBED,
        shape=(KH, D),
        strides=(stride_qh, stride_qd),
        offsets=(0,0),
        block_shape=(BLOCK_KH, D),
        order=(1,0)
    )

    cols = tl.arange(0, D)
    cos_ptrs = COS + cols
    sin_ptrs = SIN + cols
    cos = tl.load(cos_ptrs)
    sin = tl.load(sin_ptrs)

    q = tl.load(q_block_ptrs, boundary_check=(0,))
    qr = tl.load(qr_block_ptrs, boundary_check=(0,))
    q_embed = q * cos + qr * sin
    tl.store(qembed_block_ptrs, q_embed, boundary_check=(0,))
    k = tl.load(k_block_ptrs, boundary_check=(0,))
    kr = tl.load(kr_block_ptrs, boundary_check=(0,))
    k_embed = k * cos + kr * sin
    tl.store(kembed_block_ptrs, k_embed, boundary_check=(0,))


@triton.jit
def _fused_apply_rope_bwd(DQ_EMBED, DK_EMBED, DQ_EMBED_R, DK_EMBED_R, COS, SIN,
                          DQ, DK, 
                        stride_qb, stride_qh, stride_ql, stride_qd,
                        stride_qrb, stride_qrh, stride_qrl, stride_qrd,
                        stride_kb, stride_kh, stride_kl, stride_kd,
                        stride_cb, stride_cl, stride_cd,
                        B, COS_B, L, QH, KH, D: tl.constexpr, BLOCK_QH:tl.constexpr, BLOCK_KH:tl.constexpr,
                        ):
    pid = tl.program_id(0)
    off_b = pid // L
    off_l = pid % L
    dq_offset = off_b * stride_qb + stride_ql * off_l
    dqr_offset = off_b * stride_qrb + stride_qrl * off_l
    dk_offset = off_b * stride_kb  + stride_kl * off_l
    if B == COS_B:
        cos_offset = pid * stride_cl
    else:
        cos_offset = off_l * stride_cl
    # cos_offset = pid * stride_cl
    DQ += dq_offset
    DQ_EMBED += dq_offset
    DQ_EMBED_R += dqr_offset
    DK += dk_offset
    DK_EMBED += dk_offset
    DK_EMBED_R += dk_offset
    COS += cos_offset
    SIN += cos_offset

    dq_block_ptrs = tl.make_block_ptr(
        base=DQ,
        shape=(QH, D),
        strides=(stride_qh, stride_qd),
        offsets=(0,0),
        block_shape=(BLOCK_QH, D),
        order=(1,0)
    )
    dq_embed_block_ptrs = tl.make_block_ptr(
        base=DQ_EMBED,
        shape=(QH, D),
        strides=(stride_qh, stride_qd),
        offsets=(0,0),
        block_shape=(BLOCK_QH, D),
        order=(1,0)
    )
    dq_embed_r_block_ptrs = tl.make_block_ptr(
        base=DQ_EMBED_R,
        shape=(QH, D),
        strides=(stride_qrh, stride_qrd),
        offsets=(0,0),
        block_shape=(BLOCK_QH, D),
        order=(1,0)
    )
    dk_block_ptrs = tl.make_block_ptr(
        base=DK,
        shape=(KH, D),
        strides=(stride_kh, stride_kd),
        offsets=(0,0),
        block_shape=(BLOCK_KH, D),
        order=(1,0)
    )
    dk_embed_block_ptrs = tl.make_block_ptr(
        base=DK_EMBED,
        shape=(KH, D),
        strides=(stride_kh, stride_kd),
        offsets=(0,0),
        block_shape=(BLOCK_KH, D),
        order=(1,0)
    )
    dk_embed_r_block_ptrs = tl.make_block_ptr(
        base=DK_EMBED_R,
        shape=(KH, D),
        strides=(stride_kh, stride_kd),
        offsets=(0,0),
        block_shape=(BLOCK_KH, D),
        order=(1,0)
    )
    cols = tl.arange(0, D)
    cos_ptrs = COS + cols
    sin_ptrs = SIN + cols

    cos = tl.load(cos_ptrs)
    sin = tl.load(sin_ptrs)

    dq_embed = tl.load(dq_embed_block_ptrs, boundary_check=(0,))
    dq_embed_r = tl.load(dq_embed_r_block_ptrs, boundary_check=(0,))
    dq = dq_embed * cos + dq_embed_r * sin
    tl.store(dq_block_ptrs, dq, boundary_check=(0,))
    dk_embed = tl.load(dk_embed_block_ptrs, boundary_check=(0,))
    dk_embed_r = tl.load(dk_embed_r_block_ptrs, boundary_check=(0,))
    dk = dk_embed * cos + dk_embed_r * sin
    tl.store(dk_block_ptrs, dk, boundary_check=(0,))

class _FusedApplyRope(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin):
        assert q.transpose(1,2).is_contiguous()
        # print(q.stride(), k.stride())
        B, QH, L, D = q.shape
        KH = k.size(1)
        assert (D % 32 == 0) or (D % 64 == 0) or (D % 128 == 0)
        num_stages=1
        num_warps=8

        qr = rotate_half(q)
        kr = rotate_half(k)
        # print(qr.stride())
        q_embed = torch.empty(B, L, QH, D, device=q.device, dtype=k.dtype)
        k_embed = torch.empty(B, L, KH, D, device=q.device, dtype=k.dtype)
        M = B*L
        COS_B = cos.shape[0]
        BLOCK_QH = triton.next_power_of_2(QH)
        BLOCK_KH = triton.next_power_of_2(KH)
        _fused_apply_rope_fwd[(M,)](q,k,qr,kr,cos, sin,
                                    q_embed, k_embed,
                                    *q.stride(),
                                    *qr.stride(),
                                    *cos.stride(),
                                    B, COS_B, L, QH, KH, D, BLOCK_QH, BLOCK_KH,
                                    num_warps=num_warps, num_stages=num_stages

        )

        ctx.save_for_backward(cos, sin)
        ctx.infos = (B, QH, KH, L, D, M, COS_B, BLOCK_QH, BLOCK_KH)
        ctx.num_warps = num_warps
        ctx.num_stages = num_stages
        return q_embed.transpose(1,2), k_embed.transpose(1,2)

    @staticmethod
    def backward(ctx, dq_embed, dk_embed):
        B, QH, KH, L, D, M, COS_B ,BLOCK_QH, BLOCK_KH= ctx.infos
        cos,sin = ctx.saved_tensors
        dq = torch.empty_like(dq_embed)
        dk = torch.empty_like(dk_embed)
        dq_embed_r = rotate_half_v3(dq_embed)
        dk_embed_r = rotate_half_v3(dk_embed)
        sin = rotate_half_v2(sin)
        _fused_apply_rope_bwd[(M,)](dq_embed, dk_embed, dq_embed_r, dk_embed_r, cos, sin,
                                    dq, dk,
                                    *dq.stride(),
                                    *dq_embed_r.stride(),
                                    *dk.stride(),
                                    *cos.stride(),
                                    B, COS_B, L, QH, KH, D, BLOCK_QH, BLOCK_KH,
                                    num_warps=ctx.num_warps, num_stages=ctx.num_stages
                                    )
        # print(dq.stride(), dq_embed.stride(), dq_embed_r.stride())
        # print(dk.stride(), dk_embed.stride(), dk_embed_r.stride())
        # 在模型中是：
        # 不连续，不连续，连续
        # 连续，连续，连续
        # 在这个测试代码中不太一样，都是连续的
        return dq, dk, None, None

fused_apply_rope = _FusedApplyRope.apply