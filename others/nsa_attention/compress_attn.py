# [2022-10-23] Downloaded from https://github.com/openai/triton/blob/master/python/tutorials/06-fused-attention.py
# for benchmarking.
# We fixed a few dtype cast to make it work for bf16

"""
Fused Attention
===============
This is a Triton implementation of the Flash Attention algorithm
(see: Dao et al., https://arxiv.org/pdf/2205.14135v2.pdf; Rabe and Staats https://arxiv.org/pdf/2112.05682v2.pdf)
"""

import torch
import triton
import triton.language as tl
import math

# @triton.autotune([triton.Config({}, num_stages=ns, num_warps=nw)
#                   for ns in [1,2,4]
#                   for nw in [4, 8]
#                   ],
#                   key=['D1','D2','BLOCK_SIZE_N'])
@triton.jit
def _block_compress_fwd(X, W, PE, Y, 
                        x_stride_b, x_stride_n, x_stride_h, x_stride_d,
                        y_stride_b, y_stride_m, y_stride_h, y_stride_d,
                        stride, kernel_size, 
                        D,
                        D1:tl.constexpr, D2:tl.constexpr, BLOCK_SIZE_N:tl.constexpr):
    off_b = tl.cast(tl.program_id(0), tl.int64)
    off_h = tl.cast(tl.program_id(1), tl.int64)
    off_m = tl.cast(tl.program_id(2), tl.int64)
    
    X += off_b * x_stride_b + off_h * x_stride_h + stride * off_m * x_stride_n
    Y += off_b * y_stride_b + off_h * y_stride_h + off_m * y_stride_m

    rows = tl.arange(0, BLOCK_SIZE_N)
    mask = rows < kernel_size

    w = tl.load(W + rows, mask=mask, other=0.).to(tl.float32)

    x_ptrs = X + rows[:, None] * x_stride_n + tl.arange(0, D1)[None, :]
    x = tl.load(x_ptrs, mask=mask[:, None], other=0.).to(tl.float32)
    pe_ptrs = PE + rows[:, None] * D + tl.arange(0, D1)[None, :]
    pe = tl.load(pe_ptrs, mask=mask[:, None], other=0.).to(tl.float32)
    y = tl.sum((x + pe) * w[:, None], axis=0)
    y_ptrs = Y + tl.arange(0, D1)
    tl.store(y_ptrs, y)
    
    if D2 > 0:
        x_ptrs = X + rows[:, None] * x_stride_n + tl.arange(0, D2)[None, :] + D1
        x = tl.load(x_ptrs, mask=mask[:, None], other=0.).to(tl.float32)
        pe_ptrs = PE + rows[:, None] * D + tl.arange(0, D2)[None, :] + D1
        pe = tl.load(pe_ptrs, mask=mask[:, None], other=0.).to(tl.float32)
        y = tl.sum((x + pe) * w[:, None], axis=0)
        y_ptrs = Y + tl.arange(0, D2) + D1
        tl.store(y_ptrs, y)

# @triton.autotune([triton.Config({}, num_stages=ns, num_warps=nw)
#                   for ns in [1,2,4]
#                   for nw in [4, 8]
#                   ],
#                   key=['D1','D2', 'BLOCK_SIZE_N'])
@triton.jit
def _block_compress_dwdpe(DY, DW, DPE,
                          X, W, PE,
                          dy_stride_b, dy_stride_m, dy_stride_h, dy_stride_d,
                          x_stride_b, x_stride_n, x_stride_h, x_stride_d,
                          stride, kernel_size, num_blocks, NUM_SMS, 
                          B, H, D,
                          D1:tl.constexpr, D2:tl.constexpr, BLOCK_SIZE_N:tl.constexpr
                          ):
    pid = tl.cast(tl.program_id(0), tl.int64)
    current_id = pid
    total = B * H * num_blocks

    rows = tl.arange(0, BLOCK_SIZE_N)
    mask = rows < kernel_size
    cols = tl.arange(0, D1)

    w = tl.load(W+rows, mask=mask, other=0.).to(tl.float32)
    pe_ptrs = PE + rows[:, None] * D + cols[None, :]
    pe = tl.load(pe_ptrs, mask=mask[:, None], other=0.).to(tl.float32)

    dpe = tl.zeros((BLOCK_SIZE_N, D1), dtype=tl.float32)
    dw = tl.zeros((BLOCK_SIZE_N, ), dtype=tl.float32)
    if D2 > 0:
        cols2 = tl.arange(0, D2) + D1
        pe_ptrs2 = PE + rows[:, None] * D + cols2[None, :]
        pe2 = tl.load(pe_ptrs2, mask=mask[:, None], other=0.).to(tl.float32)
        dpe2 = tl.zeros((BLOCK_SIZE_N, D2), dtype=tl.float32)

    while current_id < total:
        off_m = current_id % num_blocks
        off_bh = current_id // num_blocks
        off_b = off_bh // H
        off_h = off_bh % H

        dy_ptrs = DY + off_b * dy_stride_b + off_h * dy_stride_h + off_m * dy_stride_m + cols
        x_ptrs = X + off_b * x_stride_b + off_h * x_stride_h \
                + (stride * off_m + rows[:, None]) * x_stride_n \
                + cols[None, :]
        dy = tl.load(dy_ptrs).to(tl.float32)
        x = tl.load(x_ptrs, mask=mask[:, None], other=0.).to(tl.float32)
        x_pe = x + pe
        dw += tl.sum(x_pe * dy[None, :], axis=1)
        dpe += w[:, None] * dy[None, :]

        if D2 > 0:
            dy_ptrs2 = DY + off_b * dy_stride_b + off_h * dy_stride_h + off_m * dy_stride_m + cols2
            x_ptrs2 = X + off_b * x_stride_b + off_h * x_stride_h \
                    + (stride * off_m + rows[:, None]) * x_stride_n \
                    + cols2[None, :]
            dy2 = tl.load(dy_ptrs2).to(tl.float32)
            x2 = tl.load(x_ptrs2, mask=mask[:, None], other=0.).to(tl.float32)
            x_pe2 = x2 + pe2
            dw += tl.sum(x_pe2 * dy2[None, :], axis=1)
            dpe2 += w[:, None] * dy2[None, :]

        current_id += NUM_SMS

    dw_ptrs = DW + pid * kernel_size + rows
    dpe_ptrs = DPE + pid * kernel_size * D + rows[:, None] * D + cols[None, :]
    tl.store(dw_ptrs, dw)
    tl.store(dpe_ptrs, dpe, mask=mask[:, None])
    if D2 > 0:
        dpe_ptrs2 = DPE + pid * kernel_size * D + rows[:, None] * D + cols2[None, :]
        tl.store(dpe_ptrs2, dpe2, mask=mask[:, None])


# @triton.autotune([triton.Config({}, num_stages=ns, num_warps=nw)
#                   for ns in [1,2,4]
#                   for nw in [4, 8]
#                   ],
#                   key=['D1', 'D2'])
@triton.jit
def _block_compress_dx(DY, DX,
                        W, 
                        dy_stride_b, dy_stride_m, dy_stride_h, dy_stride_d,
                        dx_stride_b, dx_stride_n, dx_stride_h, dx_stride_d,
                        stride:tl.constexpr, kernel_size, num_blocks, 
                        D1:tl.constexpr, D2:tl.constexpr,
                          ):
    off_b = tl.cast(tl.program_id(0), tl.int64)
    off_h = tl.cast(tl.program_id(1), tl.int64)
    pid_k = tl.cast(tl.program_id(2), tl.int64)

    DY += off_b * dy_stride_b + off_h * dy_stride_h
    DX += off_b * dx_stride_b + off_h * dx_stride_h

    rows = tl.arange(0, stride)
    cols = tl.arange(0, D1)
    # tl.static_print()
    dx = tl.zeros((stride, D1), dtype=tl.float32)
    if D2>0:
        cols2 = tl.arange(0, D2) + D1
        dx2 = tl.zeros((stride, D2), dtype=tl.float32)
    for idx in range(0, (kernel_size/stride).to(tl.int32)):
        block_idx = pid_k - idx # 大小为stride的mini block在block中的位置从开头到末尾
        if block_idx >=0 and block_idx < num_blocks:
            dy_ptrs = DY + block_idx * dy_stride_m + cols
            dy = tl.load(dy_ptrs).to(tl.float32)
            w = tl.load(W + idx*stride + rows).to(tl.float32)
            dx += dy[None, :] * w[:, None]
            if D2 > 0:
                dy_ptrs2 = DY + block_idx * dy_stride_m + cols2
                dy2 = tl.load(dy_ptrs2).to(tl.float32)
                dx2 += dy2[None, :] * w[:, None]
    dx_ptrs = DX + (pid_k * stride + rows[:, None]) * dx_stride_n + cols[None, :]
    tl.store(dx_ptrs, dx)
    if D2 > 0:
        dx_ptrs2 = DX + (pid_k * stride + rows[:, None]) * dx_stride_n + cols2[None, :]
        tl.store(dx_ptrs2, dx2)


class _BlockCompress(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, pe, stride):
        B, N, H, D = x.shape
        kernel_size = len(weight)
        assert kernel_size % stride == 0
        assert math.log2(kernel_size).is_integer()
        assert N >= kernel_size
        num_blocks = (N - kernel_size) // stride + 1
        assert num_blocks > 0

        BLOCK_SIZE_N = triton.next_power_of_2(kernel_size)
        
        if math.log2(D).is_integer():
            D1 = D
            D2 = 0
        else:
            D1 = 2**int(math.log2(D-1))
            D2 = D - D1
            assert math.log2(D2).is_integer()
        y = torch.empty(B, num_blocks, H, D, device=x.device, dtype=x.dtype)
        grids = (B, H, num_blocks)
        kwargs = {'num_warps':4, 'num_stages': 1}
        _block_compress_fwd[grids](x, weight, pe, y,
                                   *x.stride(),
                                   *y.stride(),
                                    stride, kernel_size,
                                    D, D1, D2, BLOCK_SIZE_N,
                                    **kwargs
                                   )
        ctx.save_for_backward(x, weight, pe)
        ctx.infos = (B, H, N, D, kernel_size, stride, num_blocks, D1, D2, BLOCK_SIZE_N)
        return y
    
    @staticmethod
    def backward(ctx, dy):
        x, weight, pe = ctx.saved_tensors
        B, H, N, D, kernel_size, stride, num_blocks, D1, D2, BLOCK_SIZE_N = ctx.infos

        NUM_SMS = torch.cuda.get_device_properties('cuda').multi_processor_count
        dw = torch.empty(NUM_SMS, kernel_size, device=x.device, dtype=torch.float32)
        dpe = torch.empty(NUM_SMS, kernel_size, D, device=x.device, dtype=torch.float32)
        kwargs = {'num_warps':8, 'num_stages': 1}
        _block_compress_dwdpe[(NUM_SMS,)](dy, dw, dpe,
                                         x, weight, pe,
                                         *dy.stride(),
                                         *x.stride(),
                                         stride, kernel_size, num_blocks, NUM_SMS,
                                         B, H, D, 
                                         D1, D2, BLOCK_SIZE_N,
                                         **kwargs
                                         )
        dw = dw.sum(0).to(weight.dtype)
        dpe = dpe.sum(0).to(dpe.dtype)

        K = (stride * num_blocks + (kernel_size - stride)) // stride
        dx = torch.empty_like(x)
        dx[:, :, num_blocks * stride + kernel_size - stride:] = 0
        kwargs = {'num_warps':4, 'num_stages': 2}
        _block_compress_dx[(B,H, K)](dy, dx,
                                         weight,
                                         *dy.stride(),
                                         *dx.stride(),
                                         stride, kernel_size, num_blocks, 
                                         D1, D2, 
                                         **kwargs
                                         )
        return dx, dw, dpe, None

def blcok_compress(x, weight, pe, stride):
    return _BlockCompress.apply(x, weight, pe, stride)


# @triton.autotune([triton.Config({'BLOCK_SIZE_N': bsn, 'BLOCK_SIZE_M': bsm}, num_stages=ns, num_warps=nw)
#                  for bsm in [32, 64, 128]
#                  for bsn in [32, 64, 128]
#                  for ns in [1, 2, 4]
#                  for nw in [4, 8]
#                  ], key=['N', "M"])
@triton.jit
def _fwd_kernel(Q, K, V, O, LSE, 
                q_stride_b, q_stride_n, q_stride_h, q_stride_d,
                k_stride_b, k_stride_m, k_stride_h, k_stride_d,
                v_stride_b, v_stride_m, v_stride_h, v_stride_d,
                o_stride_b, o_stride_n, o_stride_h, o_stride_d,
                lse_stride_b, lse_stride_h, lse_stride_n,
                sm_scale, kernel_size, stride,
                B, N, M, QH, KH, 
                D1: tl.constexpr, D2: tl.constexpr, VD: tl.constexpr, 
                BLOCK_SIZE_N: tl.constexpr=64, BLOCK_SIZE_M: tl.constexpr=128):

    off_b = tl.cast(tl.program_id(0), tl.int64)
    off_qh = tl.cast(tl.program_id(1), tl.int64)
    off_n = tl.cast(tl.program_id(2), tl.int64) * BLOCK_SIZE_N
    off_kh = off_qh // (QH // KH)

    Q += off_b * q_stride_b + off_qh * q_stride_h
    K += off_b * k_stride_b + off_kh * k_stride_h 
    V += off_b * v_stride_b + off_kh * v_stride_h
    O += off_b * o_stride_b + off_qh * o_stride_h

    rows = tl.arange(0, BLOCK_SIZE_N) + off_n
    cols = tl.arange(0, BLOCK_SIZE_M)
    d1_cols = tl.arange(0, D1)
    vd_cols = tl.arange(0, VD)
    row_mask = rows < N

    q_ptrs = Q + rows[:, None] * q_stride_n + d1_cols[None, :]
    k_ptrs = K + cols[None, :] * k_stride_m + d1_cols[:, None]
    v_ptrs = V + cols[:, None] * v_stride_m + vd_cols[None, :]

    q = tl.load(q_ptrs, mask=row_mask[:, None], other=0.)
    if D2 > 0:
        d2_cols = tl.arange(0, D2) + D1
        q_ptrs2 = Q + rows[:, None] * q_stride_n + d2_cols[None, :]
        k_ptrs2 = K + cols[None, :] * k_stride_m + d2_cols[:, None]
        q2 = tl.load(q_ptrs2, mask=row_mask[:, None], other=0.)

    m_i = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)
    acc = tl.zeros([BLOCK_SIZE_N, VD], dtype=tl.float32)

    start_m = 0 
    # end_q_idx = tl.minimum(off_n + BLOCK_SIZE_N, N) - 1
    # start_kv_idx = kernel_size - 1
    for start_kv_idx in range(kernel_size-1, off_n + BLOCK_SIZE_N, BLOCK_SIZE_M * stride):
        block_idx =  start_m + cols
        col_mask = block_idx < M
        k = tl.load(k_ptrs, mask=col_mask[None, :], other=0.)
        attn_score = tl.dot(q, k)
        if D2>0:
            k2 = tl.load(k_ptrs2, mask=col_mask[None, :], other=0.)
            attn_score = tl.dot(q2, k2, attn_score)
        attn_score *= sm_scale

        k_idx = block_idx * stride + kernel_size - 1
        causal_mask = rows[:, None] >= k_idx[None, :]
        attn_score = tl.where(causal_mask, attn_score, float('-inf'))

        m_ij = tl.max(attn_score, axis=1)
        new_m_i = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - new_m_i)

        exp_attn_score = tl.exp(attn_score - new_m_i[:, None])

        l_i = l_i * alpha + tl.sum(exp_attn_score, axis=-1)
        acc = acc * alpha[:, None]

        v = tl.load(v_ptrs, mask=col_mask[:, None], other=0.)
        acc = tl.dot(exp_attn_score.to(v.dtype), v, acc=acc)

        m_i = new_m_i
        start_m += BLOCK_SIZE_M
        k_ptrs += BLOCK_SIZE_M * k_stride_m
        v_ptrs += BLOCK_SIZE_M * v_stride_m
        if D2 > 0:
            k_ptrs2 += BLOCK_SIZE_M * k_stride_m

    acc /= l_i[:, None]
    if off_n == 0:
        acc = tl.where(rows[:, None]>=(kernel_size-1), acc, 0)
    o_ptrs = O + rows[:, None] * o_stride_n + vd_cols[None, :]
    tl.store(o_ptrs, acc, mask=row_mask[:, None])

    lse = m_i + tl.log(l_i)
    if off_n == 0:
        lse = tl.where(rows>=(kernel_size-1), lse, 0)
    tl.store(LSE + off_b * lse_stride_b + off_qh * lse_stride_h + rows, lse, mask=row_mask)


# @triton.autotune([triton.Config({'BLOCK_SIZE_N': bsn}, num_stages=ns, num_warps=nw)
#                  for bsn in [16, 32, 64, 128]
#                  for ns in [1, 2, 4]
#                  for nw in [4, 8]
#                  ], key=['N'])
@triton.jit
def _bwd_preprocess(O,DO,Delta,
                    o_stride_b, o_stride_n, o_stride_h, o_stride_d,
                    delta_stride_b, delta_stride_h, delta_stride_n,
                    N, VD: tl.constexpr,
                    BLOCK_SIZE_N: tl.constexpr=16
                    ):
    off_b = tl.cast(tl.program_id(0), tl.int64)
    off_h = tl.cast(tl.program_id(1), tl.int64)
    off_n = tl.cast(tl.program_id(2), tl.int64) * BLOCK_SIZE_N

    O += off_b * o_stride_b + off_h * o_stride_h
    DO += off_b * o_stride_b + off_h * o_stride_h
    Delta += off_b * delta_stride_b + off_h * delta_stride_h
    
    rows = tl.arange(0, BLOCK_SIZE_N) + off_n
    row_mask = rows < N
    cols = tl.arange(0, VD)
    
    o = tl.load(O + rows[:, None] * o_stride_n + cols[None, :], mask=row_mask[:, None], other=0.).to(tl.float32)
    do = tl.load(DO + rows[:, None] * o_stride_n + cols[None, :], mask=row_mask[:, None], other=0.).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    tl.store(Delta + rows, delta, mask=row_mask)

# @triton.autotune([triton.Config({'BLOCK_SIZE_N': bsn, 'BLOCK_SIZE_M': bsm}, num_stages=ns, num_warps=nw)
#                  for bsm in [16, 32, 64]
#                  for bsn in [64, 128, 256]
#                  for ns in [1, 2]
#                  for nw in [8]
#                  ], key=['N', "M"])
@triton.jit
def _bwd_kernel(DQ, DK, DV, DO, 
                Q, K, V, 
                Lse, Delta,
                q_stride_b, q_stride_n, q_stride_h, q_stride_d,
                k_stride_b, k_stride_m, k_stride_h, k_stride_d,
                v_stride_b, v_stride_m, v_stride_h, v_stride_d,
                dk_stride_b, dk_stride_m, dk_stride_h, dk_stride_d,
                dv_stride_b, dv_stride_m, dv_stride_h, dv_stride_d,
                do_stride_b, do_stride_n, do_stride_h, do_stride_d,
                lse_stride_b, lse_stride_h, lse_stride_n,
                sm_scale, kernel_size, stride,
                B, N, M, QH, KH, 
                D1: tl.constexpr, D2: tl.constexpr, VD: tl.constexpr, 
                BLOCK_SIZE_N: tl.constexpr=64, BLOCK_SIZE_M: tl.constexpr=64
                ):
    off_m = tl.cast(tl.program_id(0), tl.int64) * BLOCK_SIZE_M
    off_b = tl.cast(tl.program_id(1), tl.int64)
    off_qh = tl.cast(tl.program_id(2), tl.int64)
    
    off_kh = off_qh // (QH // KH)

    Q += off_b * q_stride_b + off_qh * q_stride_h
    K += off_b * k_stride_b + off_kh * k_stride_h 
    V += off_b * v_stride_b + off_kh * v_stride_h
    DQ += off_b * q_stride_b + off_qh * q_stride_h
    DK += off_b * dk_stride_b + off_qh * dk_stride_h 
    DV += off_b * dv_stride_b + off_qh * dv_stride_h
    DO += off_b * do_stride_b + off_qh * do_stride_h
    Lse += off_b * lse_stride_b + off_qh * lse_stride_h
    Delta += off_b * lse_stride_b + off_qh * lse_stride_h

    rows = tl.arange(0, BLOCK_SIZE_N)
    cols = tl.arange(0, BLOCK_SIZE_M) + off_m
    col_mask = cols < M

    k_ptrs = K + cols[None, :] * k_stride_m + tl.arange(0, D1)[:, None]
    v_ptrs = V + cols[None, :] * v_stride_m + tl.arange(0, VD)[:, None]
    start_n = off_m * stride + kernel_size - 1
    q_ptrs = Q + (rows[:, None] + start_n) * q_stride_n + tl.arange(0, D1)[None, :]
    dq_ptrs = DQ + (rows[:, None] + start_n) * q_stride_n + tl.arange(0, D1)[None, :]
    do_ptrs = DO + (rows[:, None] + start_n) * do_stride_n + tl.arange(0, VD)[None, :]
    lse_ptrs = Lse + rows + start_n
    delta_ptrs = Delta + rows + start_n

    k = tl.load(k_ptrs, mask=col_mask[None, :], other=0.)
    v = tl.load(v_ptrs, mask=col_mask[None, :], other=0.)
    
    acc_dk = tl.zeros((BLOCK_SIZE_M, D1), dtype=tl.float32)
    acc_dv = tl.zeros((BLOCK_SIZE_M, VD), dtype=tl.float32)
    k_idx = cols * stride + kernel_size - 1

    if D2 > 0:
        k2_ptrs = K + cols[None, :] * k_stride_m + tl.arange(0, D2)[:, None] + D1
        q2_ptrs = Q + (rows[:, None] + start_n) * q_stride_n + tl.arange(0, D2)[None, :] + D1
        dq2_ptrs = DQ + (rows[:, None] + start_n) * q_stride_n + tl.arange(0, D2)[None, :] + D1
        k2 = tl.load(k2_ptrs, mask=col_mask[None, :], other=0.)
        acc_dk2 = tl.zeros((BLOCK_SIZE_M, D2), dtype=tl.float32)
    for start_q_idx in range(off_m * stride + kernel_size - 1, N, BLOCK_SIZE_N):
        q_idx = rows + start_q_idx
        row_mask = q_idx < N
        q = tl.load(q_ptrs, mask=row_mask[:, None], other=0.)
        do = tl.load(do_ptrs, mask=row_mask[:, None], other=0.)
        lse = tl.load(lse_ptrs, mask=row_mask, other=0.)
        delta = tl.load(delta_ptrs, mask=row_mask, other=0.)

        attn_score = tl.dot(q, k) 

        if D2 > 0:
            q2 = tl.load(q2_ptrs, mask=row_mask[:, None], other=0.)
            attn_score = tl.dot(q2, k2, attn_score)

        causal_mask = q_idx[:, None] >= k_idx[None, :]
        attn_score = tl.where(causal_mask, attn_score, float('-inf'))
        p = tl.exp(attn_score * sm_scale - lse[:, None])
        
        acc_dv = tl.dot(tl.trans(p, 1, 0).to(do.dtype), do, acc_dv)

        dp = tl.dot(do, v)
        ds = p * (dp - delta[:, None]) * sm_scale

        # dq = tl.load(dq_ptrs, mask=row_mask[:, None], other=0., eviction_policy="evict_last")
        
        dq = tl.dot(ds.to(k.dtype), tl.trans(k, 1, 0))
        tl.atomic_add(dq_ptrs, dq, mask=row_mask[:, None])
        
        acc_dk = tl.dot(tl.trans(ds, 1, 0).to(q.dtype), q, acc_dk)
        if D2 > 0:
            dq2 = tl.dot(ds.to(k.dtype), tl.trans(k2, 1, 0))
            tl.atomic_add(dq2_ptrs, dq2, mask=row_mask[:, None])
            acc_dk2 = tl.dot(tl.trans(ds, 1, 0).to(q.dtype), q2, acc_dk2)
            q2_ptrs += BLOCK_SIZE_N * q_stride_n
            dq2_ptrs += BLOCK_SIZE_N * q_stride_n

        q_ptrs += BLOCK_SIZE_N * q_stride_n
        dq_ptrs += BLOCK_SIZE_N * q_stride_n
        do_ptrs += BLOCK_SIZE_N * do_stride_n
        lse_ptrs += BLOCK_SIZE_N
        delta_ptrs += BLOCK_SIZE_N
    
    dk_ptrs = DK + cols[:, None] * dk_stride_m + tl.arange(0, D1)[None, :]
    dv_ptrs = DV + cols[:, None] * dv_stride_m + tl.arange(0, VD)[None, :]
    tl.store(dk_ptrs, acc_dk, mask=col_mask[:, None])
    tl.store(dv_ptrs, acc_dv, mask=col_mask[:, None])
    if D2 > 0:
        dk2_ptrs = DK + cols[:, None] * dk_stride_m + tl.arange(0, D2)[None, :] + D1
        tl.store(dk2_ptrs, acc_dk2, mask=col_mask[:, None])


class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, kernel_size, stride, sm_scale):
        B, N, QH, D = q.shape
        B2, M, KH, D2 = k.shape
        B3, M2, KH2, VD = v.shape
        assert B == B2 and B == B3 and M == M2 and D == D2 and KH == KH2
        assert QH % KH == 0
        assert math.log2(VD).is_integer()

        if math.log2(D).is_integer():
            D1 = D
            D2 = 0
        else:
            D1 = 2**int(math.log2(D-1))
            D2 = D - D1
            assert math.log2(D2).is_integer()
        if sm_scale is None:
            sm_scale = D**-0.5
        o = torch.empty(B, N, QH, VD, device=q.device, dtype=q.dtype)
        o[:, :kernel_size-1] = 0.
        lse = torch.empty(B, QH, N, dtype=torch.float32, device=q.device,)
        grid = lambda meta: (B, QH, triton.cdiv(N, meta['BLOCK_SIZE_N']))
        kwargs = {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_M": 32, "num_warps": 4, "num_stages": 2}
        _fwd_kernel[grid](q, k, v, o, lse,
                          *q.stride(),
                          *k.stride(),
                          *v.stride(),
                          *o.stride(),
                          *lse.stride(),
                          sm_scale, kernel_size, stride,
                          B, N, M, QH, KH, 
                          D1, D2, VD,
                          **kwargs
                          )
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.infos = (B, N, M, QH, KH, D1, D2, VD, sm_scale, kernel_size, stride)
        return o, lse

    @staticmethod
    def backward(ctx, do, *args):
        assert do.is_contiguous()
        B, N, M, QH, KH, D1, D2, VD, sm_scale, kernel_size, stride = ctx.infos
        q, k, v, o, lse = ctx.saved_tensors
        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.empty(B, M, QH, D1+D2, device=q.device, dtype=q.dtype)
        dv = torch.empty(B, M, QH, VD, device=q.device, dtype=q.dtype)

        delta = torch.empty_like(lse)
        # delta[:, :, :kernel_size-1] = 0
        grid = lambda meta: (B, QH, triton.cdiv(N, meta["BLOCK_SIZE_N"]))
        kwargs = {"BLOCK_SIZE_N": 64, "num_warps": 8, "num_stages": 1}
        _bwd_preprocess[grid](o, do, delta,
                              *o.stride(), 
                              *delta.stride(),
                              N, VD,
                              **kwargs
                              )

        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE_M"]), B, QH)
        kwargs = {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_M": 64, "num_warps": 8, "num_stages": 2}
        _bwd_kernel[grid](dq, dk, dv, do, 
                          q, k, v,
                          lse, delta,
                          *q.stride(), # dq和q一样
                          *k.stride(),
                          *v.stride(),
                          *dk.stride(),
                          *dv.stride(),
                          *do.stride(), # do和o一样
                          *lse.stride(), # lse和delta一样
                          sm_scale, kernel_size, stride,
                          B, N, M, QH, KH, 
                          D1, D2, VD,
                          **kwargs
                          )
        dq = dq.to(q.dtype)
        dk = dk.view(B, M, KH, -1, D1+D2).sum(3)
        dv = dv.view(B, M, KH, -1, VD).sum(3)
        return dq, dk, dv, None, None, None


def compress_attn(q, k, v, kernel_size, stride, sm_scale=None):
    return _attention.apply(q, k, v, kernel_size, stride, sm_scale)


class CompressKV(torch.nn.Module):
    def __init__(self, head_dim, kernel_size, stride):
        super().__init__()
        self.head_dim = head_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.pe = torch.nn.Parameter(torch.randn(kernel_size, head_dim))
        self.weight = torch.nn.Parameter(torch.randn(kernel_size,))

    def forward(self, x):
        return blcok_compress(x, self.weight, self.pe, self.stride)
    
class CompressAttn(torch.nn.Module):
    def __init__(self, qk_head_dim, v_head_dim, kernel_size, stride):
        super().__init__()
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.compress_key = CompressKV(self.qk_head_dim, kernel_size, stride)
        self.compress_value = CompressKV(self.v_head_dim, kernel_size, stride)
        self.sm_scale = qk_head_dim ** -0.5

    def forward(self, q, k, v):
        cmp_k = self.compress_key(k)
        cmp_v = self.compress_value(v)
        o, lse = compress_attn(q, cmp_k, cmp_v, self.kernel_size, self.stride, self.sm_scale)
        return o, lse, cmp_k
    
这是分割线下面是实验的中间结果 = 1
# 2是把192扩大256，不拆成128 + 64
# 3是不在外部对dk和dv进行sum


# @triton.autotune([triton.Config({'BLOCK_SIZE_N': bsn, 'BLOCK_SIZE_M': bsm}, num_stages=ns, num_warps=nw)
#                  for bsm in [32, 64, 128]
#                  for bsn in [32, 64, 128]
#                  for ns in [1, 2, 4]
#                  for nw in [4, 8]
#                  ], key=['N', "M"])
@triton.jit
def _fwd_kernel2(Q, K, V, O, LSE, 
                q_stride_b, q_stride_n, q_stride_h, q_stride_d,
                k_stride_b, k_stride_m, k_stride_h, k_stride_d,
                v_stride_b, v_stride_m, v_stride_h, v_stride_d,
                o_stride_b, o_stride_n, o_stride_h, o_stride_d,
                lse_stride_b, lse_stride_h, lse_stride_n,
                sm_scale, kernel_size, stride,
                B, N, M, QH, KH, 
                D, VD: tl.constexpr, BLOCK_SIZE_D: tl.constexpr,
                BLOCK_SIZE_N: tl.constexpr=64, BLOCK_SIZE_M: tl.constexpr=128):

    off_b = tl.cast(tl.program_id(0), tl.int64)
    off_qh = tl.cast(tl.program_id(1), tl.int64)
    off_n = tl.cast(tl.program_id(2), tl.int64) * BLOCK_SIZE_N
    off_kh = off_qh // (QH // KH)

    Q += off_b * q_stride_b + off_qh * q_stride_h
    K += off_b * k_stride_b + off_kh * k_stride_h 
    V += off_b * v_stride_b + off_kh * v_stride_h
    O += off_b * o_stride_b + off_qh * o_stride_h

    rows = tl.arange(0, BLOCK_SIZE_N) + off_n
    cols = tl.arange(0, BLOCK_SIZE_M)
    d_cols = tl.arange(0, BLOCK_SIZE_D)
    vd_cols = tl.arange(0, VD)
    row_mask = rows < N
    d_mask = d_cols < D

    q_ptrs = Q + rows[:, None] * q_stride_n + d_cols[None, :]
    k_ptrs = K + cols[None, :] * k_stride_m + d_cols[:, None]
    v_ptrs = V + cols[:, None] * v_stride_m + vd_cols[None, :]

    q = tl.load(q_ptrs, mask=row_mask[:, None] & d_mask[None, :], other=0.)

    m_i = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)
    acc = tl.zeros([BLOCK_SIZE_N, VD], dtype=tl.float32)

    start_m = 0 
    # end_q_idx = tl.minimum(off_n + BLOCK_SIZE_N, N) - 1
    # start_kv_idx = kernel_size - 1
    for start_kv_idx in range(kernel_size-1, off_n + BLOCK_SIZE_N, BLOCK_SIZE_M * stride):
        block_idx =  start_m + cols
        col_mask = block_idx < M
        k = tl.load(k_ptrs, mask=col_mask[None, :] & d_mask[:, None], other=0.)
        attn_score = tl.dot(q, k)
        attn_score *= sm_scale

        k_idx = block_idx * stride + kernel_size - 1
        causal_mask = rows[:, None] >= k_idx[None, :]
        attn_score = tl.where(causal_mask, attn_score, float('-inf'))

        m_ij = tl.max(attn_score, axis=1)
        new_m_i = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - new_m_i)

        exp_attn_score = tl.exp(attn_score - new_m_i[:, None])

        l_i = l_i * alpha + tl.sum(exp_attn_score, axis=-1)
        acc = acc * alpha[:, None]

        v = tl.load(v_ptrs, mask=col_mask[:, None], other=0.)
        acc = tl.dot(exp_attn_score.to(v.dtype), v, acc=acc)

        m_i = new_m_i
        start_m += BLOCK_SIZE_M
        k_ptrs += BLOCK_SIZE_M * k_stride_m
        v_ptrs += BLOCK_SIZE_M * v_stride_m

    acc /= l_i[:, None]
    if off_n == 0:
        acc = tl.where(rows[:, None]>=(kernel_size-1), acc, 0)
    o_ptrs = O + rows[:, None] * o_stride_n + vd_cols[None, :]
    tl.store(o_ptrs, acc, mask=row_mask[:, None])

    lse = m_i + tl.log(l_i)
    if off_n == 0:
        lse = tl.where(rows>=(kernel_size-1), lse, 0)
    tl.store(LSE + off_b * lse_stride_b + off_qh * lse_stride_h + rows, lse, mask=row_mask)



# @triton.autotune([triton.Config({'BLOCK_SIZE_N': bsn, 'BLOCK_SIZE_M': bsm}, num_stages=ns, num_warps=nw)
#                  for bsm in [16, 32, 64]
#                  for bsn in [64, 128, 256]
#                  for ns in [1, 2]
#                  for nw in [8]
#                  ], key=['N', "M"])
@triton.jit
def _bwd_kernel2(DQ, DK, DV, DO, 
                Q, K, V, 
                Lse, Delta,
                q_stride_b, q_stride_n, q_stride_h, q_stride_d,
                k_stride_b, k_stride_m, k_stride_h, k_stride_d,
                v_stride_b, v_stride_m, v_stride_h, v_stride_d,
                dk_stride_b, dk_stride_m, dk_stride_h, dk_stride_d,
                dv_stride_b, dv_stride_m, dv_stride_h, dv_stride_d,
                do_stride_b, do_stride_n, do_stride_h, do_stride_d,
                lse_stride_b, lse_stride_h, lse_stride_n,
                sm_scale, kernel_size, stride,
                B, N, M, QH, KH, 
                D, VD: tl.constexpr, BLOCK_SIZE_D: tl.constexpr,
                BLOCK_SIZE_N: tl.constexpr=64, BLOCK_SIZE_M: tl.constexpr=64
                ):
    off_m = tl.cast(tl.program_id(0), tl.int64) * BLOCK_SIZE_M
    off_b = tl.cast(tl.program_id(1), tl.int64)
    off_qh = tl.cast(tl.program_id(2), tl.int64)
    
    off_kh = off_qh // (QH // KH)

    Q += off_b * q_stride_b + off_qh * q_stride_h
    K += off_b * k_stride_b + off_kh * k_stride_h 
    V += off_b * v_stride_b + off_kh * v_stride_h
    DQ += off_b * q_stride_b + off_qh * q_stride_h
    DK += off_b * dk_stride_b + off_qh * dk_stride_h 
    DV += off_b * dv_stride_b + off_qh * dv_stride_h
    DO += off_b * do_stride_b + off_qh * do_stride_h
    Lse += off_b * lse_stride_b + off_qh * lse_stride_h
    Delta += off_b * lse_stride_b + off_qh * lse_stride_h

    rows = tl.arange(0, BLOCK_SIZE_N)
    cols = tl.arange(0, BLOCK_SIZE_M) + off_m
    col_mask = cols < M

    k_ptrs = K + cols[None, :] * k_stride_m + tl.arange(0, BLOCK_SIZE_D)[:, None]
    v_ptrs = V + cols[None, :] * v_stride_m + tl.arange(0, VD)[:, None]
    start_n = off_m * stride + kernel_size - 1
    q_ptrs = Q + (rows[:, None] + start_n) * q_stride_n + tl.arange(0, BLOCK_SIZE_D)[None, :]
    dq_ptrs = DQ + (rows[:, None] + start_n) * q_stride_n + tl.arange(0, BLOCK_SIZE_D)[None, :]
    do_ptrs = DO + (rows[:, None] + start_n) * do_stride_n + tl.arange(0, VD)[None, :]
    lse_ptrs = Lse + rows + start_n
    delta_ptrs = Delta + rows + start_n

    k = tl.load(k_ptrs, mask=col_mask[None, :] & (tl.arange(0, BLOCK_SIZE_D)[:, None] < D), other=0.)
    v = tl.load(v_ptrs, mask=col_mask[None, :], other=0.)
    
    acc_dk = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_D), dtype=tl.float32)
    acc_dv = tl.zeros((BLOCK_SIZE_M, VD), dtype=tl.float32)
    k_idx = cols * stride + kernel_size - 1

    for start_q_idx in range(off_m * stride + kernel_size - 1, N, BLOCK_SIZE_N):
        q_idx = rows + start_q_idx
        row_mask = q_idx < N
        q = tl.load(q_ptrs, mask=row_mask[:, None] & (tl.arange(0, BLOCK_SIZE_D)[None, :] < D), other=0.)
        do = tl.load(do_ptrs, mask=row_mask[:, None], other=0.)
        lse = tl.load(lse_ptrs, mask=row_mask, other=0.)
        delta = tl.load(delta_ptrs, mask=row_mask, other=0.)

        attn_score = tl.dot(q, k) 

        causal_mask = q_idx[:, None] >= k_idx[None, :]
        attn_score = tl.where(causal_mask, attn_score, float('-inf'))
        p = tl.exp(attn_score * sm_scale - lse[:, None])
        
        acc_dv = tl.dot(tl.trans(p, 1, 0).to(do.dtype), do, acc_dv)

        dp = tl.dot(do, v)
        ds = p * (dp - delta[:, None]) * sm_scale

        # dq = tl.load(dq_ptrs, mask=row_mask[:, None], other=0., eviction_policy="evict_last")
        
        dq = tl.dot(ds.to(k.dtype), tl.trans(k, 1, 0))
        tl.atomic_add(dq_ptrs, dq, mask=row_mask[:, None] & (tl.arange(0, BLOCK_SIZE_D)[None, :] < D))
        acc_dk = tl.dot(tl.trans(ds, 1, 0).to(q.dtype), q, acc_dk)

        q_ptrs += BLOCK_SIZE_N * q_stride_n
        dq_ptrs += BLOCK_SIZE_N * q_stride_n
        do_ptrs += BLOCK_SIZE_N * do_stride_n
        lse_ptrs += BLOCK_SIZE_N
        delta_ptrs += BLOCK_SIZE_N
    
    dk_ptrs = DK + cols[:, None] * dk_stride_m + tl.arange(0, BLOCK_SIZE_D)[None, :]
    dv_ptrs = DV + cols[:, None] * dv_stride_m + tl.arange(0, VD)[None, :]
    tl.store(dk_ptrs, acc_dk, mask=col_mask[:, None] & (tl.arange(0, BLOCK_SIZE_D)[None, :] < D))
    tl.store(dv_ptrs, acc_dv, mask=col_mask[:, None])



class _attention2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, kernel_size, stride, sm_scale):
        B, N, QH, D = q.shape
        B2, M, KH, D2 = k.shape
        B3, M2, KH2, VD = v.shape
        assert B == B2 and B == B3 and M == M2 and D == D2 and KH == KH2
        assert QH % KH == 0
        assert math.log2(VD).is_integer()

        BLOCK_SIZE_D = triton.next_power_of_2(D)
        if sm_scale is None:
            sm_scale = D**-0.5
        o = torch.empty(B, N, QH, VD, device=q.device, dtype=q.dtype)
        o[:, :kernel_size-1] = 0.
        lse = torch.empty(B, QH, N, dtype=torch.float32, device=q.device,)
        grid = lambda meta: (B, QH, triton.cdiv(N, meta['BLOCK_SIZE_N']))
        kwargs = {"BLOCK_SIZE_N": 128, "BLOCK_SIZE_M": 64, "num_warps": 8, "num_stages": 2}
        _fwd_kernel2[grid](q, k, v, o, lse,
                          *q.stride(),
                          *k.stride(),
                          *v.stride(),
                          *o.stride(),
                          *lse.stride(),
                          sm_scale, kernel_size, stride,
                          B, N, M, QH, KH, 
                          D, VD, BLOCK_SIZE_D,
                          **kwargs
                          )
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.infos = (B, N, M, QH, KH, D, BLOCK_SIZE_D, VD, sm_scale, kernel_size, stride)
        return o, lse

    @staticmethod
    def backward(ctx, do, *args):
        assert do.is_contiguous()
        B, N, M, QH, KH, D, BLOCK_SIZE_D, VD,  sm_scale, kernel_size, stride = ctx.infos
        q, k, v, o, lse = ctx.saved_tensors
        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.empty(B, M, QH, D, device=q.device, dtype=q.dtype)
        dv = torch.empty(B, M, QH, VD, device=q.device, dtype=q.dtype)

        delta = torch.empty_like(lse)
        # delta[:, :, :kernel_size-1] = 0
        grid = lambda meta: (B, QH, triton.cdiv(N, meta["BLOCK_SIZE_N"]))
        kwargs = {"BLOCK_SIZE_N": 64, "num_warps": 8, "num_stages": 1}
        _bwd_preprocess[grid](o, do, delta,
                              *o.stride(), 
                              *delta.stride(),
                              N, VD,
                              **kwargs
                              )

        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE_M"]), B, QH)
        kwargs = {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_M": 64, "num_warps": 8, "num_stages": 2}
        _bwd_kernel2[grid](dq, dk, dv, do, 
                          q, k, v,
                          lse, delta,
                          *q.stride(), # dq和q一样
                          *k.stride(),
                          *v.stride(),
                          *dk.stride(),
                          *dv.stride(),
                          *do.stride(), # do和o一样
                          *lse.stride(), # lse和delta一样
                          sm_scale, kernel_size, stride,
                          B, N, M, QH, KH, 
                          D, VD, BLOCK_SIZE_D,
                          **kwargs
                          )
        dq = dq.to(q.dtype)
        dk = dk.view(B, M, KH, -1, D).sum(3)
        dv = dv.view(B, M, KH, -1, VD).sum(3)
        return dq, dk, dv, None, None, None
    
def compress_attn2(q, k, v, kernel_size, stride, sm_scale=None):
    return _attention2.apply(q, k, v, kernel_size, stride, sm_scale)


# @triton.autotune([triton.Config({'BLOCK_SIZE_N': bsn, 'BLOCK_SIZE_M': bsm}, num_stages=ns, num_warps=nw)
#                  for bsm in [32, 64, 128]
#                  for bsn in [32, 64, 128]
#                  for ns in [1, 2, 4]
#                  for nw in [4, 8]
#                  ], key=['N', "M"])
@triton.jit
def _fwd_kernel3(Q, K, V, O, LSE, 
                q_stride_b, q_stride_n, q_stride_h, q_stride_d,
                k_stride_b, k_stride_m, k_stride_h, k_stride_d,
                v_stride_b, v_stride_m, v_stride_h, v_stride_d,
                o_stride_b, o_stride_n, o_stride_h, o_stride_d,
                lse_stride_b, lse_stride_h, lse_stride_n,
                sm_scale, kernel_size, stride,
                B, N, M, QH, KH, 
                D1: tl.constexpr, D2: tl.constexpr, VD: tl.constexpr, 
                BLOCK_SIZE_N: tl.constexpr=64, BLOCK_SIZE_M: tl.constexpr=64):

    off_b = tl.cast(tl.program_id(0), tl.int64)
    off_qh = tl.cast(tl.program_id(1), tl.int64)
    off_n = tl.cast(tl.program_id(2), tl.int64) * BLOCK_SIZE_N
    off_kh = off_qh // (QH // KH)

    Q += off_b * q_stride_b + off_qh * q_stride_h
    K += off_b * k_stride_b + off_kh * k_stride_h 
    V += off_b * v_stride_b + off_kh * v_stride_h
    O += off_b * o_stride_b + off_qh * o_stride_h

    rows = tl.arange(0, BLOCK_SIZE_N) + off_n
    cols = tl.arange(0, BLOCK_SIZE_M)
    d1_cols = tl.arange(0, D1)
    vd_cols = tl.arange(0, VD)
    row_mask = rows < N

    q_ptrs = Q + rows[:, None] * q_stride_n + d1_cols[None, :]
    k_ptrs = K + cols[None, :] * k_stride_m + d1_cols[:, None]
    v_ptrs = V + cols[:, None] * v_stride_m + vd_cols[None, :]

    q = tl.load(q_ptrs, mask=row_mask[:, None], other=0.)
    if D2 > 0:
        d2_cols = tl.arange(0, D2) + D1
        q_ptrs2 = Q + rows[:, None] * q_stride_n + d2_cols[None, :]
        k_ptrs2 = K + cols[None, :] * k_stride_m + d2_cols[:, None]
        q2 = tl.load(q_ptrs2, mask=row_mask[:, None], other=0.)

    m_i = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)
    acc = tl.zeros([BLOCK_SIZE_N, VD], dtype=tl.float32)

    start_m = 0 
    # end_q_idx = tl.minimum(off_n + BLOCK_SIZE_N, N) - 1
    # start_kv_idx = kernel_size - 1
    for start_kv_idx in range(kernel_size-1, off_n + BLOCK_SIZE_N, BLOCK_SIZE_M * stride):
        block_idx =  start_m + cols
        col_mask = block_idx < M
        k = tl.load(k_ptrs, mask=col_mask[None, :], other=0.)
        attn_score = tl.dot(q, k)
        if D2>0:
            k2 = tl.load(k_ptrs2, mask=col_mask[None, :], other=0.)
            attn_score = tl.dot(q2, k2, attn_score)
        attn_score *= sm_scale

        k_idx = block_idx * stride + kernel_size - 1
        causal_mask = rows[:, None] >= k_idx[None, :]
        attn_score = tl.where(causal_mask, attn_score, float('-inf'))

        m_ij = tl.max(attn_score, axis=1)
        new_m_i = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - new_m_i)

        exp_attn_score = tl.exp(attn_score - new_m_i[:, None])

        l_i = l_i * alpha + tl.sum(exp_attn_score, axis=-1)
        acc = acc * alpha[:, None]

        v = tl.load(v_ptrs, mask=col_mask[:, None], other=0.)
        acc = tl.dot(exp_attn_score.to(v.dtype), v, acc=acc)

        m_i = new_m_i
        start_m += BLOCK_SIZE_M
        k_ptrs += BLOCK_SIZE_M * k_stride_m
        v_ptrs += BLOCK_SIZE_M * v_stride_m
        if D2 > 0:
            k_ptrs2 += BLOCK_SIZE_M * k_stride_m

    acc /= l_i[:, None]
    if off_n == 0:
        acc = tl.where(rows[:, None]>=(kernel_size-1), acc, 0)
    o_ptrs = O + rows[:, None] * o_stride_n + vd_cols[None, :]
    tl.store(o_ptrs, acc, mask=row_mask[:, None])

    lse = m_i + tl.log(l_i)
    if off_n == 0:
        lse = tl.where(rows>=(kernel_size-1), lse, 0)
    tl.store(LSE + off_b * lse_stride_b + off_qh * lse_stride_h + rows, lse, mask=row_mask)



# @triton.autotune([triton.Config({'BLOCK_SIZE_N': bsn, 'BLOCK_SIZE_M': bsm}, num_stages=ns, num_warps=nw)
#                  for bsm in [16, 32, 64]
#                  for bsn in [64, 128, 256]
#                  for ns in [1, 2]
#                  for nw in [8]
#                  ], key=['N', "M"])
@triton.jit
def _bwd_kernel3(DQ, DK, DV, DO, 
                Q, K, V, 
                Lse, Delta,
                q_stride_b, q_stride_n, q_stride_h, q_stride_d,
                k_stride_b, k_stride_m, k_stride_h, k_stride_d,
                v_stride_b, v_stride_m, v_stride_h, v_stride_d,
                do_stride_b, do_stride_n, do_stride_h, do_stride_d,
                lse_stride_b, lse_stride_h, lse_stride_n,
                sm_scale, kernel_size, stride,
                B, N, M, QH, KH, 
                D1: tl.constexpr, D2: tl.constexpr, VD: tl.constexpr, 
                BLOCK_SIZE_N: tl.constexpr=64, BLOCK_SIZE_M: tl.constexpr=64
                ):
    pid0 = tl.cast(tl.program_id(0), tl.int64)
    off_m = pid0 * BLOCK_SIZE_M
    off_b = tl.cast(tl.program_id(1), tl.int64)
    off_kh = tl.cast(tl.program_id(2), tl.int64)
    nrep = QH // KH
    off_qh =  off_kh * nrep

    Q += off_b * q_stride_b + off_qh * q_stride_h
    K += off_b * k_stride_b + off_kh * k_stride_h 
    V += off_b * v_stride_b + off_kh * v_stride_h
    DQ += off_b * q_stride_b + off_qh * q_stride_h
    DK += off_b * k_stride_b + off_kh * k_stride_h 
    DV += off_b * v_stride_b + off_kh * v_stride_h
    DO += off_b * do_stride_b + off_qh * do_stride_h
    Lse += off_b * lse_stride_b + off_qh * lse_stride_h
    Delta += off_b * lse_stride_b + off_qh * lse_stride_h

    rows = tl.arange(0, BLOCK_SIZE_N)
    cols = tl.arange(0, BLOCK_SIZE_M) + off_m
    col_mask = cols < M
    
    k_ptrs = K + cols[None, :] * k_stride_m + tl.arange(0, D1)[:, None]
    v_ptrs = V + cols[None, :] * v_stride_m + tl.arange(0, VD)[:, None]
    k = tl.load(k_ptrs, mask=col_mask[None, :], other=0.)
    v = tl.load(v_ptrs, mask=col_mask[None, :], other=0.)
    acc_dk = tl.zeros((BLOCK_SIZE_M, D1), dtype=tl.float32)
    acc_dv = tl.zeros((BLOCK_SIZE_M, VD), dtype=tl.float32)
    k_idx = cols * stride + kernel_size - 1
    start_n = off_m * stride + kernel_size - 1
    if D2 > 0:
        k2_ptrs = K + cols[None, :] * k_stride_m + tl.arange(0, D2)[:, None] + D1
        k2 = tl.load(k2_ptrs, mask=col_mask[None, :], other=0.)
        acc_dk2 = tl.zeros((BLOCK_SIZE_M, D2), dtype=tl.float32)

    for idx in range(pid0, pid0+nrep):
        h_idx = idx % nrep
        q_ptrs = Q + (rows[:, None] + start_n) * q_stride_n + tl.arange(0, D1)[None, :] + h_idx * q_stride_h
        dq_ptrs = DQ + (rows[:, None] + start_n) * q_stride_n + tl.arange(0, D1)[None, :] + h_idx * q_stride_h
        do_ptrs = DO + (rows[:, None] + start_n) * do_stride_n + tl.arange(0, VD)[None, :] + h_idx * do_stride_h
        lse_ptrs = Lse + rows + start_n + h_idx * lse_stride_h
        delta_ptrs = Delta + rows + start_n + h_idx * lse_stride_h

        if D2 > 0:
            q2_ptrs = Q + (rows[:, None] + start_n) * q_stride_n + tl.arange(0, D2)[None, :] + D1 + h_idx * q_stride_h
            dq2_ptrs = DQ + (rows[:, None] + start_n) * q_stride_n + tl.arange(0, D2)[None, :] + D1 + h_idx * q_stride_h

        for start_q_idx in range(off_m * stride + kernel_size - 1, N, BLOCK_SIZE_N):
            q_idx = rows + start_q_idx
            row_mask = q_idx < N
            q = tl.load(q_ptrs, mask=row_mask[:, None], other=0.)
            do = tl.load(do_ptrs, mask=row_mask[:, None], other=0.)
            lse = tl.load(lse_ptrs, mask=row_mask, other=0.)
            delta = tl.load(delta_ptrs, mask=row_mask, other=0.)

            attn_score = tl.dot(q, k) 

            if D2 > 0:
                q2 = tl.load(q2_ptrs, mask=row_mask[:, None], other=0.)
                attn_score = tl.dot(q2, k2, attn_score)

            causal_mask = q_idx[:, None] >= k_idx[None, :]
            attn_score = tl.where(causal_mask, attn_score, float('-inf'))
            p = tl.exp(attn_score * sm_scale - lse[:, None])
            
            acc_dv = tl.dot(tl.trans(p, 1, 0).to(do.dtype), do, acc_dv)

            dp = tl.dot(do, v)
            ds = p * (dp - delta[:, None]) * sm_scale

            # dq = tl.load(dq_ptrs, mask=row_mask[:, None], other=0., eviction_policy="evict_last")
            
            dq = tl.dot(ds.to(k.dtype), tl.trans(k, 1, 0))
            tl.atomic_add(dq_ptrs, dq, mask=row_mask[:, None])
            acc_dk = tl.dot(tl.trans(ds, 1, 0).to(q.dtype), q, acc_dk)
            if D2 > 0:
                dq2 = tl.dot(ds.to(k.dtype), tl.trans(k2, 1, 0))
                tl.atomic_add(dq2_ptrs, dq2, mask=row_mask[:, None])
                acc_dk2 = tl.dot(tl.trans(ds, 1, 0).to(q.dtype), q2, acc_dk2)
                q2_ptrs += BLOCK_SIZE_N * q_stride_n
                dq2_ptrs += BLOCK_SIZE_N * q_stride_n

            q_ptrs += BLOCK_SIZE_N * q_stride_n
            dq_ptrs += BLOCK_SIZE_N * q_stride_n
            do_ptrs += BLOCK_SIZE_N * do_stride_n
            lse_ptrs += BLOCK_SIZE_N
            delta_ptrs += BLOCK_SIZE_N
    
    dk_ptrs = DK + cols[:, None] * k_stride_m + tl.arange(0, D1)[None, :]
    dv_ptrs = DV + cols[:, None] * v_stride_m + tl.arange(0, VD)[None, :]
    tl.store(dk_ptrs, acc_dk, mask=col_mask[:, None])
    tl.store(dv_ptrs, acc_dv, mask=col_mask[:, None])
    if D2 > 0:
        dk2_ptrs = DK + cols[:, None] * k_stride_m + tl.arange(0, D2)[None, :] + D1
        tl.store(dk2_ptrs, acc_dk2, mask=col_mask[:, None])


class _attention3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, kernel_size, stride, sm_scale):
        B, N, QH, D = q.shape
        B2, M, KH, D2 = k.shape
        B3, M2, KH2, VD = v.shape
        assert B == B2 and B == B3 and M == M2 and D == D2 and KH == KH2
        assert QH % KH == 0
        assert math.log2(VD).is_integer()

        if math.log2(D).is_integer():
            D1 = D
            D2 = 0
        else:
            D1 = 2**int(math.log2(D-1))
            D2 = D - D1
            assert math.log2(D2).is_integer()
        if sm_scale is None:
            sm_scale = D**-0.5
        o = torch.empty(B, N, QH, VD, device=q.device, dtype=q.dtype)
        o[:, :kernel_size-1] = 0.
        lse = torch.empty(B, QH, N, dtype=torch.float32, device=q.device,)
        grid = lambda meta: (B, QH, triton.cdiv(N, meta['BLOCK_SIZE_N']))
        kwargs = {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_M": 32, "num_warps": 4, "num_stages": 2}
        _fwd_kernel3[grid](q, k, v, o, lse,
                          *q.stride(),
                          *k.stride(),
                          *v.stride(),
                          *o.stride(),
                          *lse.stride(),
                          sm_scale, kernel_size, stride,
                          B, N, M, QH, KH, 
                          D1, D2, VD,
                          **kwargs
                          )
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.infos = (B, N, M, QH, KH, D1, D2, VD, sm_scale, kernel_size, stride)
        return o, lse

    @staticmethod
    def backward(ctx, do, *args):
        assert do.is_contiguous()
        B, N, M, QH, KH, D1, D2, VD, sm_scale, kernel_size, stride = ctx.infos
        q, k, v, o, lse = ctx.saved_tensors
        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.empty(B, M, KH, D1+D2, device=q.device, dtype=q.dtype)
        dv = torch.empty(B, M, KH, VD, device=q.device, dtype=q.dtype)

        delta = torch.empty_like(lse)
        # delta[:, :, :kernel_size-1] = 0
        grid = lambda meta: (B, QH, triton.cdiv(N, meta["BLOCK_SIZE_N"]))
        kwargs = {"BLOCK_SIZE_N": 64, "num_warps": 8, "num_stages": 1}
        _bwd_preprocess[grid](o, do, delta,
                              *o.stride(), 
                              *delta.stride(),
                              N, VD,
                              **kwargs
                              )

        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE_M"]), B, KH)
        kwargs = {"BLOCK_SIZE_N": 128, "BLOCK_SIZE_M": 16, "num_warps": 8, "num_stages": 1}
        _bwd_kernel3[grid](dq, dk, dv, do, 
                          q, k, v,
                          lse, delta,
                          *q.stride(), # dq和q一样
                          *k.stride(),
                          *v.stride(),
                          *do.stride(), # do和o一样
                          *lse.stride(), # lse和delta一样
                          sm_scale, kernel_size, stride,
                          B, N, M, QH, KH, 
                          D1, D2, VD,
                          **kwargs
                          )
        dq = dq.to(q.dtype)
        return dq, dk, dv, None, None, None


def compress_attn3(q, k, v, kernel_size, stride, sm_scale=None):
    return _attention3.apply(q, k, v, kernel_size, stride, sm_scale)

