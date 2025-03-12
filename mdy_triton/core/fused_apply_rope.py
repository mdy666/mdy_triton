import torch
import triton
import triton.language as tl



@triton.jit
def _fused_apply_rope_fwd(Q, K, COS, SIN,
                          Q_EMBED, K_EMBED,
                          B, COS_B, L, QH, KH, D:tl.constexpr, HALF_D:tl.constexpr,
                          BLOCK_QH:tl.constexpr, BLOCK_KH:tl.constexpr, BLOCK_D:tl.constexpr,
                          CHUNK_N:tl.constexpr=64
                          ):
    start_n = tl.cast(tl.program_id(0), tl.int64) * CHUNK_N + tl.program_id(1)
    if start_n >= B*L:
        return
    off_b = start_n // L
    off_l = start_n % L

    Q += start_n * QH * D
    Q_EMBED += start_n * QH * D
    K += start_n * KH * D
    K_EMBED +=  start_n * KH * D
    COS += off_b * (COS_B // B) * L * D + off_l * D
    SIN += off_b * (COS_B // B) * L * D + off_l * D

    cols = tl.arange(0, BLOCK_D)
    cos1 = tl.load(COS + cols, mask=cols<HALF_D, other=0.).to(tl.float32)[None, :]
    cos2 = tl.load(COS + cols + HALF_D, mask=(cols + HALF_D)<D, other=0.).to(tl.float32)[None, :]
    sin1 = tl.load(SIN + cols, mask=cols<HALF_D, other=0.).to(tl.float32)[None, :]
    sin2 = tl.load(SIN + cols + HALF_D, mask=(cols + HALF_D)<D, other=0.).to(tl.float32)[None, :]

    k1 = tl.load(K + tl.arange(0, BLOCK_KH)[:, None] * D + cols[None, :], 
                 mask=(cols[None, :]<HALF_D) & (tl.arange(0, BLOCK_KH)[:, None] < KH), other=0.).to(tl.float32)
    k2 = tl.load(K + tl.arange(0, BLOCK_KH)[:, None] * D + cols[None, :] + HALF_D, 
                 mask=((HALF_D+cols[None, :])<D) & (tl.arange(0, BLOCK_KH)[:, None] < KH), other=0.).to(tl.float32)
    k_embed1 = k1 * cos1 - k2 * sin1
    k_embed2 = k2 * cos2 + k1 * sin2
    tl.store(K_EMBED + tl.arange(0, BLOCK_KH)[:, None] * D + cols[None, :], k_embed1, mask=(cols[None, :]<HALF_D)&(tl.arange(0, BLOCK_KH)[:, None] < KH))
    tl.store(K_EMBED + tl.arange(0, BLOCK_KH)[:, None] * D + cols[None, :] + HALF_D, k_embed2, mask=((HALF_D+cols[None, :])<D) & (tl.arange(0, BLOCK_KH)[:, None] < KH))
    
    q1 = tl.load(Q + tl.arange(0, BLOCK_QH)[:, None] * D + cols[None, :], 
                 mask=(cols[None, :]<HALF_D) & (tl.arange(0, BLOCK_QH)[:, None] < QH), other=0.).to(tl.float32)
    q2 = tl.load(Q + tl.arange(0, BLOCK_QH)[:, None] * D + cols[None, :] + HALF_D, 
                 mask=((HALF_D+cols[None, :])<D) & (tl.arange(0, BLOCK_QH)[:, None] < QH), other=0.).to(tl.float32)
    q_embed1 = q1 * cos1 - q2 * sin1
    q_embed2 = q2 * cos2 + q1 * sin2
    tl.store(Q_EMBED + tl.arange(0, BLOCK_QH)[:, None] * D + cols[None, :], q_embed1, mask=(cols[None, :]<HALF_D)&(tl.arange(0, BLOCK_QH)[:, None] < QH))
    tl.store(Q_EMBED + tl.arange(0, BLOCK_QH)[:, None] * D + cols[None, :] + HALF_D, q_embed2, mask=((HALF_D+cols[None, :])<D) & (tl.arange(0, BLOCK_QH)[:, None] < QH))
    

@triton.jit
def _fused_apply_rope_bwd(DQ_EMBED, DK_EMBED, COS, SIN,
                          DQ, DK, 
                          dq_stride_b, dq_stride_h, dq_stride_l, dq_stride_d,
                          dk_stride_b, dk_stride_h, dk_stride_l, dk_stride_d,
                        B, COS_B, L, QH, KH, D: tl.constexpr, HALF_D: tl.constexpr,
                        BLOCK_QH:tl.constexpr, BLOCK_KH:tl.constexpr, BLOCK_D:tl.constexpr,
                        CHUNK_N:tl.constexpr=64
                        ):
    start_n = tl.cast(tl.program_id(0), tl.int64) * CHUNK_N + tl.program_id(1)
    if start_n >= B*L:
        return
    off_b = start_n // L
    off_l = start_n % L

    DQ += off_b * dq_stride_b + off_l * dq_stride_l
    DQ_EMBED += off_b * dq_stride_b + off_l * dq_stride_l
    DK += off_b * dk_stride_b + off_l * dk_stride_l
    DK_EMBED += off_b * dk_stride_b + off_l * dk_stride_l
    COS += off_b * (COS_B // B) * L * D + off_l * D
    SIN += off_b * (COS_B // B) * L * D + off_l * D

    cols = tl.arange(0, BLOCK_D)
    cos1 = tl.load(COS + cols, mask=cols<HALF_D, other=0.).to(tl.float32)[None, :]
    cos2 = tl.load(COS + cols + HALF_D, mask=(cols + HALF_D)<D, other=0.).to(tl.float32)[None, :]
    sin1 = tl.load(SIN + cols, mask=cols<HALF_D, other=0.).to(tl.float32)[None, :]
    sin2 = tl.load(SIN + cols + HALF_D, mask=(cols + HALF_D)<D, other=0.).to(tl.float32)[None, :]

    dk_embed1 = tl.load(DK_EMBED + tl.arange(0, BLOCK_KH)[:, None] * dk_stride_h + cols[None, :], 
                 mask=(cols[None, :]<HALF_D) & (tl.arange(0, BLOCK_KH)[:, None] < KH), other=0.).to(tl.float32)
    dk_embed2 = tl.load(DK_EMBED + tl.arange(0, BLOCK_KH)[:, None] * dk_stride_h + cols[None, :] + HALF_D, 
                 mask=((HALF_D+cols[None, :])<D) & (tl.arange(0, BLOCK_KH)[:, None] < KH), other=0.).to(tl.float32)
    dk1 = dk_embed1 * cos1 + sin2 * dk_embed2
    dk2 = dk_embed2 * cos2 - sin1 * dk_embed1
    tl.store(DK + tl.arange(0, BLOCK_KH)[:, None] * dk_stride_h + cols[None, :], dk1, mask=(cols[None, :]<HALF_D)&(tl.arange(0, BLOCK_KH)[:, None] < KH))
    tl.store(DK + tl.arange(0, BLOCK_KH)[:, None] * dk_stride_h + cols[None, :] + HALF_D, dk2, mask=((HALF_D+cols[None, :])<D) & (tl.arange(0, BLOCK_KH)[:, None] < KH))

    dq_embed1 = tl.load(DQ_EMBED + tl.arange(0, BLOCK_QH)[:, None] * dq_stride_h + cols[None, :], 
                 mask=(cols[None, :]<HALF_D) & (tl.arange(0, BLOCK_QH)[:, None] < QH), other=0.).to(tl.float32)
    dq_embed2 = tl.load(DQ_EMBED + tl.arange(0, BLOCK_QH)[:, None] * dq_stride_h + cols[None, :] + HALF_D, 
                 mask=((HALF_D+cols[None, :])<D) & (tl.arange(0, BLOCK_QH)[:, None] < QH), other=0.).to(tl.float32)
    dq1 = dq_embed1 * cos1 + sin2 * dq_embed2
    dq2 = dq_embed2 * cos2 - sin1 * dq_embed1
    tl.store(DQ + tl.arange(0, BLOCK_QH)[:, None] * dq_stride_h + cols[None, :], dq1, mask=(cols[None, :]<HALF_D)&(tl.arange(0, BLOCK_QH)[:, None] < QH))
    tl.store(DQ + tl.arange(0, BLOCK_QH)[:, None] * dq_stride_h + cols[None, :] + HALF_D, dq2, mask=((HALF_D+cols[None, :])<D) & (tl.arange(0, BLOCK_QH)[:, None] < QH))

class _FusedApplyRope(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin):
        assert q.transpose(1,2).is_contiguous()
        # print(q.stride(), k.stride())
        B, QH, L, D = q.shape
        HALF_D = D // 2
        KH = k.size(1)
        # assert (D % 32 == 0) or (D % 64 == 0) or (D % 128 == 0)
        BLOCK_D = triton.next_power_of_2(HALF_D)
        num_stages=4
        num_warps=8

        q_embed = torch.empty(B, L, QH, D, device=q.device, dtype=k.dtype)
        k_embed = torch.empty(B, L, KH, D, device=q.device, dtype=k.dtype)
        
        N = B * L 
        COS_B = cos.shape[0]
        BLOCK_QH = triton.next_power_of_2(QH)
        BLOCK_KH = triton.next_power_of_2(KH)
        grid = lambda meta: (triton.cdiv(N, meta['CHUNK_N']), meta['CHUNK_N'])
        _fused_apply_rope_fwd[(grid)](q,k,cos, sin,
                                    q_embed, k_embed,
                                    B, COS_B, L, QH, KH, D, HALF_D,
                                    BLOCK_QH, BLOCK_KH, BLOCK_D,
                                    num_warps=num_warps, num_stages=num_stages

        )

        ctx.save_for_backward(cos, sin)
        ctx.infos = (B, QH, KH, L, D, HALF_D, N, COS_B, BLOCK_QH, BLOCK_KH, BLOCK_D)
        ctx.num_warps = num_warps
        ctx.num_stages = num_stages
        return q_embed.transpose(1,2), k_embed.transpose(1,2)

    @staticmethod
    def backward(ctx, dq_embed, dk_embed):
        print(dq_embed.stride(), dk_embed.shape)
        B, QH, KH, L, D, HALF_D, N, COS_B, BLOCK_QH, BLOCK_KH, BLOCK_D = ctx.infos
        cos,sin = ctx.saved_tensors
        dq = torch.empty_like(dq_embed)
        dk = torch.empty_like(dk_embed)
        grid = lambda meta: (triton.cdiv(N, meta['CHUNK_N']), meta['CHUNK_N'])
        _fused_apply_rope_bwd[grid](dq_embed, dk_embed, cos, sin,
                                    dq, dk,
                                    *dq.stride(),
                                    *dk.stride(),
                                    B, COS_B, L, QH, KH, D, HALF_D, BLOCK_QH, BLOCK_KH, BLOCK_D,
                                    num_warps=ctx.num_warps, num_stages=ctx.num_stages
                                    )
        return dq, dk, None, None

fused_apply_rope = _FusedApplyRope.apply