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

@triton.jit
def _compute_attn_probs(Q, K, Lse, P,
                q_stride_b, q_stride_n, q_stride_h, q_stride_d,
                k_stride_b, k_stride_m, k_stride_h, k_stride_d,
                lse_stride_b, lse_stride_h, lse_stride_n,
                p_stride_b, p_stride_h, p_stride_n, p_stride_m,
                sm_scale, kernel_size, stride,
                B, N, M, QH, KH, 
                D1: tl.constexpr, D2: tl.constexpr, 
                BLOCK_SIZE_N: tl.constexpr=64, BLOCK_SIZE_M: tl.constexpr=64):


    off_b = tl.cast(tl.program_id(0), tl.int64)
    off_qh = tl.cast(tl.program_id(1), tl.int64)
    off_n = tl.cast(tl.program_id(2), tl.int64) * BLOCK_SIZE_N
    off_kh = off_qh // (QH // KH)

    Q += off_b * q_stride_b + off_qh * q_stride_h
    K += off_b * k_stride_b + off_kh * k_stride_h 
    Lse += off_b * lse_stride_b + off_qh * lse_stride_h
    P += off_b * p_stride_b + off_qh * p_stride_h

    rows = tl.arange(0, BLOCK_SIZE_N) + off_n
    cols = tl.arange(0, BLOCK_SIZE_M)
    d1_cols = tl.arange(0, D1)
    row_mask = rows < N

    q_ptrs = Q + rows[:, None] * q_stride_n + d1_cols[None, :]
    k_ptrs = K + cols[None, :] * k_stride_m + d1_cols[:, None]
    p_ptrs = P + rows[:, None] * p_stride_n + cols[None, :]

    q = tl.load(q_ptrs, mask=row_mask[:, None], other=0.)
    lse = tl.load(Lse + rows, mask=row_mask, other=0.)
    if D2 > 0:
        d2_cols = tl.arange(0, D2) + D1
        q_ptrs2 = Q + rows[:, None] * q_stride_n + d2_cols[None, :]
        k_ptrs2 = K + cols[None, :] * k_stride_m + d2_cols[:, None]
        q2 = tl.load(q_ptrs2, mask=row_mask[:, None], other=0.)



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

        k_idx = block_idx * stride + kernel_size - 1
        causal_mask = rows[:, None] >= k_idx[None, :]
        attn_score = tl.where(causal_mask, attn_score, float('-inf'))

        p = tl.exp(attn_score * sm_scale - lse[:, None])
        tl.store(p_ptrs, p, mask=row_mask[:, None] & col_mask[None, :])
        start_m += BLOCK_SIZE_M
        k_ptrs += BLOCK_SIZE_M * k_stride_m
        p_ptrs += BLOCK_SIZE_M
        if D2 > 0:
            k_ptrs2 += BLOCK_SIZE_M * k_stride_m

# @triton.autotune([triton.Config({'BLOCK_SIZE_M': bsm}, num_stages=ns, num_warps=nw)
#                  for bsm in [32, 64, 128, 256]
#                  for ns in [1, 2, 4]
#                  for nw in [4, 8]
#                  ], key=['N', "M"])
@triton.jit
def _compute_attn_probs2(Q, K, Lse, P,
                q_stride_b, q_stride_n, q_stride_h, q_stride_d,
                k_stride_b, k_stride_m, k_stride_h, k_stride_d,
                lse_stride_b, lse_stride_h, lse_stride_n,
                p_stride_b, p_stride_h, p_stride_n, p_stride_m,
                sm_scale, kernel_size, stride,
                B, N, M, nrep,
                D1: tl.constexpr, D2: tl.constexpr, 
                BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_M: tl.constexpr=64):
    off_b = tl.cast(tl.program_id(0), tl.int64)
    off_kh = tl.cast(tl.program_id(1), tl.int64)
    off_n = tl.cast(tl.program_id(2), tl.int64)
    off_qh = tl.cast(off_kh * nrep, tl.int64)
    k_stride_m = tl.cast(k_stride_m, tl.int64)
    q_stride_h = tl.cast(q_stride_h, tl.int64)
    lse_stride_h = tl.cast(lse_stride_h, tl.int64)

    Q += off_b * q_stride_b + off_qh * q_stride_h + off_n * q_stride_n
    K += off_b * k_stride_b + off_kh * k_stride_h
    Lse += off_b * lse_stride_b + off_qh * lse_stride_h + off_n
    P += off_b * p_stride_b + off_kh * p_stride_h + off_n * p_stride_n

    rows = tl.arange(0, BLOCK_SIZE_H)
    cols = tl.arange(0, BLOCK_SIZE_M)
    d1_cols = tl.arange(0, D1)
    row_mask = rows < nrep

    q_ptrs = Q + rows[:, None] * q_stride_h + d1_cols[None, :]
    k_ptrs = K + cols[None, :] * k_stride_m + d1_cols[:, None]
    p_ptrs = P + cols

    q = tl.load(q_ptrs, mask=row_mask[:, None], other=0.)
    lse = tl.load(Lse + rows * lse_stride_h, mask=row_mask, other=0.)
    if D2 > 0:
        d2_cols = tl.arange(0, D2) + D1
        q_ptrs2 = Q + rows[:, None] * q_stride_h + d2_cols[None, :]
        k_ptrs2 = K + cols[None, :] * k_stride_m + d2_cols[:, None]
        q2 = tl.load(q_ptrs2, mask=row_mask[:, None], other=0.)

    start_m = 0 
    for start_kv_idx in range(kernel_size-1, off_n+1, BLOCK_SIZE_M * stride):
        block_idx =  start_m + cols
        col_mask = block_idx < M
        k = tl.load(k_ptrs, mask=col_mask[None, :], other=0.)
        attn_score = tl.dot(q, k)
        if D2>0:
            k2 = tl.load(k_ptrs2, mask=col_mask[None, :], other=0.)
            attn_score = tl.dot(q2, k2, attn_score)

        k_idx = block_idx * stride + kernel_size - 1
        causal_mask = off_n >= k_idx[None, :]
        attn_score = tl.where(causal_mask, attn_score, float('-inf'))

        p = tl.sum(tl.exp(attn_score * sm_scale - lse[:, None]), 0)
        tl.store(p_ptrs, p, mask=col_mask)
        start_m += BLOCK_SIZE_M
        k_ptrs += BLOCK_SIZE_M * k_stride_m
        p_ptrs += BLOCK_SIZE_M
        if D2 > 0:
            k_ptrs2 += BLOCK_SIZE_M * k_stride_m

# @triton.autotune([triton.Config({'BLOCK_SIZE_M': bsm}, num_stages=ns, num_warps=nw)
#                  for bsm in [32, 64, 128, 256]
#                  for ns in [1, 2, 4]
#                  for nw in [4, 8]
#                  ], key=['N', "M"])
@triton.jit
def _compute_attn_probs3(Q, K, Lse, P,
                q_stride_b, q_stride_n, q_stride_h, q_stride_d,
                k_stride_b, k_stride_m, k_stride_h, k_stride_d,
                lse_stride_b, lse_stride_h, lse_stride_n,
                p_stride_b, p_stride_h, p_stride_n, p_stride_m,
                sm_scale, kernel_size, stride,
                B, N, M, nrep,
                D1: tl.constexpr, D2: tl.constexpr, 
                BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_N: tl.constexpr=4, BLOCK_SIZE_M: tl.constexpr=64):
    # pass

    off_b = tl.cast(tl.program_id(0), tl.int64)
    off_kh = tl.cast(tl.program_id(1), tl.int64)
    off_n = tl.cast(tl.program_id(2), tl.int64) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    stop_n = tl.cast(tl.program_id(2), tl.int64) * BLOCK_SIZE_N + BLOCK_SIZE_N 
    off_qh = off_kh * nrep

    Q += off_b * q_stride_b + off_qh * q_stride_h
    K += off_b * k_stride_b + off_kh * k_stride_h
    Lse += off_b * lse_stride_b + off_qh * lse_stride_h
    P += off_b * p_stride_b + off_kh * p_stride_h

    rows = tl.arange(0, BLOCK_SIZE_H)
    cols = tl.arange(0, BLOCK_SIZE_M)
    d1_cols = tl.arange(0, D1)
    row_mask = rows < nrep

    q_ptrs = Q + off_n[:, None, None] * q_stride_n + rows[None, :, None] * q_stride_h + d1_cols[None, None, :]
    # k_ptrs = K + cols[None, None, :] * k_stride_m + d1_cols[None, :, None] + tl.arange(0, BLOCK_SIZE_N)[:, None, None] // N 
    k_ptrs = K + cols[None, :] * k_stride_m + d1_cols[:, None]
    p_ptrs = P + cols[None, :] + off_n[:, None] * p_stride_n

    q = tl.load(q_ptrs, mask=row_mask[None, :, None] & (off_n < N)[:, None, None], other=0.)
    # tl.static_print(q.shape)
    lse = tl.load(Lse + off_n[:, None] * lse_stride_n + rows[None, :] * lse_stride_h, mask=row_mask[None, :] & (off_n < N)[:, None], other=0.)
    if D2 > 0:
        d2_cols = tl.arange(0, D2) + D1
        q_ptrs2 = Q + off_n[:, None, None] * q_stride_n + rows[None, :, None] * q_stride_h + d2_cols[None, None, :]
        k_ptrs2 = K + cols[None, :] * k_stride_m + d2_cols[:, None]
        q2 = tl.load(q_ptrs2, mask=row_mask[None, :, None] & (off_n < N)[:, None, None], other=0.)

    start_m = 0 
    for start_kv_idx in range(kernel_size-1, stop_n, BLOCK_SIZE_M * stride):
        block_idx =  start_m + cols
        col_mask = block_idx < M
        k = tl.load(k_ptrs, mask=col_mask[None, :], other=0.)
        # tl.static_print(k.shape)
        attn_score = tl.dot(q, k[None, :, :])
        # tl.static_print(attn_score.shape)
        if D2>0:
            k2 = tl.load(k_ptrs2, mask=col_mask[None, :], other=0.)
            attn_score = tl.dot(q2, k2[None, :, :], attn_score)

        k_idx = block_idx * stride + kernel_size - 1
        causal_mask = off_n[:, None, None] >= k_idx[None, None, :]
        attn_score = tl.where(causal_mask, attn_score, float('-inf'))
        p = tl.exp(attn_score * sm_scale - lse[:, :, None])
        tl.static_print(p.shape)
        # sum_p = tl.sum(p, axis=-1)
        # tl.static_print(sum_p.shape)
        # p = tl.sum(tl.exp(attn_score * sm_scale - lse[:, :, None]), 1)
        # tl.static_print(p.shape)
        # tl.store(p_ptrs, p, mask=col_mask[None, :] & (off_n < N)[:, None])
        # start_m += BLOCK_SIZE_M
        # k_ptrs += BLOCK_SIZE_M * k_stride_m
        # p_ptrs += BLOCK_SIZE_M
        # if D2 > 0:
        #     k_ptrs2 += BLOCK_SIZE_M * k_stride_m


# @triton.autotune([triton.Config({'BLOCK_SIZE': bs,}, num_stages=ns, num_warps=nw)
#                  for bs in [1024, 2048, 4096, 8192]
#                  for ns in [1, 2, 4]
#                  for nw in [4, 8]
#                  ], key=['N', "M"])
@triton.jit
def _compute_select_probs(AP, SP,
                          ap_stride_b, ap_stride_h, ap_stride_n, ap_stride_m,
                          sp_stride_b, sp_stride_h, sp_stride_n, sp_stride_m,
                          kernel_size, stride, 
                          select_size, num_selcct_blocks,
                          N, M, QH, KH,
                          BLOCK_SIZE: tl.constexpr=2048,
                            ):
    off_bh = tl.program_id(0)
    off_n = tl.cast(tl.program_id(1), tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    off_m = tl.program_id(2)
    off_b = off_bh // KH
    off_kh = off_bh % KH
    nrep = QH // KH
    off_qh = off_kh * nrep

    mask = off_n < N
    

    ap_ptrs = AP + off_b * ap_stride_b + off_qh * ap_stride_h + off_n * ap_stride_n
    acc_p = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    select_idx = off_m
    select_start = select_idx * select_size
    select_end = tl.minimum(select_start+select_size, N)

    for h_idx in range(nrep):
        compress_start_idx = tl.maximum((select_start-kernel_size) // stride + 1, 0)
        compress_start = compress_start_idx * stride
        while (compress_start < select_end) & (compress_start + kernel_size <= N):
            compress_end = compress_start + kernel_size
            area = tl.minimum(compress_end, select_end) - tl.maximum(compress_start, select_start)
            acc_p += tl.load(ap_ptrs+compress_start_idx, mask=mask, other=0.) * area / stride
            compress_start_idx += 1
            compress_start += stride
        ap_ptrs += ap_stride_h
    acc_p = tl.where(off_n//select_size == off_m, 9999, acc_p)
    tl.store(SP + off_b * sp_stride_b + off_kh * sp_stride_h + off_n * sp_stride_n + off_m, acc_p, mask=mask)


# @triton.autotune([triton.Config({'BLOCK_SIZE': bs,}, num_stages=ns, num_warps=nw)
#                  for bs in [4096, 8192, 9]
#                  for ns in [1, 2, 4]
#                  for nw in [4, 8]
#                  ], key=['N', "M"])
@triton.jit
def _compute_select_probs2(AP, SP,
                          ap_stride_b, ap_stride_h, ap_stride_n, ap_stride_m,
                          sp_stride_b, sp_stride_h, sp_stride_n, sp_stride_m,
                          kernel_size, stride, 
                          select_size, num_selcct_blocks,
                          N, M, QH, KH,
                          BLOCK_SIZE: tl.constexpr=2048,
                            ):
    off_bh = tl.cast(tl.program_id(0), tl.int64)
    off_n = tl.cast(tl.program_id(1), tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    off_m = tl.cast(tl.program_id(2), tl.int64)
    off_b = off_bh // KH
    off_h = off_bh % KH
    mask = off_n < N
    

    ap_ptrs = AP + off_b * ap_stride_b + off_h * ap_stride_h + off_n * ap_stride_n
    acc_p = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    select_idx = off_m
    select_start = select_idx * select_size
    select_end = tl.minimum(select_start+select_size, N)

    compress_start_idx = tl.maximum((select_start-kernel_size) // stride + 1, 0)
    compress_start = compress_start_idx * stride
    while (compress_start < select_end) & (compress_start + kernel_size <= N):
        compress_end = compress_start + kernel_size
        area = tl.minimum(compress_end, select_end) - tl.maximum(compress_start, select_start)
        acc_p += tl.load(ap_ptrs+compress_start_idx, mask=mask, other=0.) * area / stride
        compress_start_idx += 1
        compress_start += stride
    acc_p = tl.where(off_n//select_size == off_m, 9999, acc_p)
    tl.store(SP + off_b * sp_stride_b + off_h * sp_stride_h + off_n * sp_stride_n + off_m, acc_p, mask=mask)

# @triton.autotune([triton.Config({'BLOCK_SIZE': bs,}, num_stages=ns, num_warps=nw)
#                  for bs in [4096, 8192, 8192 * 2, 8192 * 4]
#                  for ns in [1, 2, 4]
#                  for nw in [4, 8]
#                  ], key=['N', "M"])
@triton.jit
def _compute_select_probs3(AP, SP,
                          ap_stride,
                          sp_stride,
                          kernel_size, stride, 
                          select_size, num_selcct_blocks,
                          B, N, KH,
                          BLOCK_SIZE: tl.constexpr=2048,
                            ):
    off_n = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    off_m = tl.program_id(1)
    mask = off_n < (B * N * KH)
    

    ap_ptrs = AP + off_n * ap_stride
    acc_p = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    select_idx = off_m
    select_start = select_idx * select_size
    select_end = tl.minimum(select_start+select_size, N)

    compress_start_idx = tl.maximum((select_start-kernel_size) // stride + 1, 0)
    compress_start = compress_start_idx * stride
    while (compress_start < select_end) & (compress_start + kernel_size <= N):
        compress_end = compress_start + kernel_size
        area = tl.minimum(compress_end, select_end) - tl.maximum(compress_start, select_start)
        acc_p += tl.load(ap_ptrs+compress_start_idx, mask=mask, other=0.) * area / stride
        compress_start_idx += 1
        compress_start += stride
    acc_p = tl.where((off_n % N) // select_size == off_m, 9999, acc_p)
    tl.store(SP + off_n * sp_stride + off_m, acc_p, mask=mask)


# @triton.autotune([triton.Config({'BLOCK_SIZE_N': bsn}, num_stages=ns, num_warps=nw)
#                  for bsn in [16, 32, 64, 128, 256]
#                  for ns in [1, 2, 4]
#                  for nw in [4, 8]
#                  ], key=['N', "M"]) 
triton.heuristics   
@triton.jit
def _fix_indices(Ind,
                ind_stride_b, ind_stride_h, ind_stride_n, ind_stride_m,
                select_size, num_selcct_blocks, top_n,
                N, M, 
                BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr=64, 
                ):
    
    off_b = tl.cast(tl.program_id(0), tl.int64)
    off_h = tl.cast(tl.program_id(1), tl.int64)
    off_n = tl.cast(tl.program_id(2), tl.int64) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    row_mask = off_n < N
    col_mask = tl.arange(0, BLOCK_SIZE_M) < M

    Ind += off_b * ind_stride_b + off_h * ind_stride_h

    ind = tl.load(Ind + off_n[:, None] * ind_stride_n + tl.arange(0, BLOCK_SIZE_M)[None, :], 
                 mask=row_mask[:, None] & col_mask[None, :],
                 other=0.)
    fill_mask = (off_n // select_size)[:, None] >= tl.arange(0, BLOCK_SIZE_M)[None, :]
    ind = tl.where(fill_mask, ind, num_selcct_blocks)
    tl.store(Ind + off_n[:, None] * ind_stride_n + tl.arange(0, BLOCK_SIZE_M)[None, :], 
            ind,
            mask=row_mask[:, None] & col_mask[None, :])

@torch.inference_mode()
def select_for_fwd(q, k, lse, kernel_size, stride, select_size, top_n, sm_scale=None):
    B, N, QH, D = q.shape
    B2, M, KH, D2 = k.shape
    assert QH % KH == 0

    if math.log2(D).is_integer():
        D1 = D
        D2 = 0
    else:
        D1 = 2**int(math.log2(D-1))
        D2 = D - D1
        assert math.log2(D2).is_integer()
    if sm_scale is None:
        sm_scale = D**-0.5

    num_selcct_blocks = triton.cdiv(N, select_size)
    top_n = min(num_selcct_blocks, top_n)

    nrep = QH // KH
    if nrep <= 8:
        probs = torch.zeros(B, QH, N, M, device=q.device, dtype=torch.float32)
        grid = lambda meta: (B, QH, triton.cdiv(N, meta['BLOCK_SIZE_N']))
        kwargs = {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_M": 32, "num_warps": 4, "num_stages": 2}
        _compute_attn_probs[grid](q, k, lse, probs,
                            *q.stride(),
                            *k.stride(),
                            *lse.stride(),
                            *probs.stride(),
                            sm_scale, kernel_size, stride,
                            B, N, M, QH, KH, 
                            D1, D2,
                            # **kwargs
                            )
        probs = probs.view(B, KH, -1, N, M).sum(2)
    else:
        probs = torch.zeros(B, KH, N, M, device=q.device, dtype=torch.float32)
        grid = (B, KH, N)
        kwargs = { "BLOCK_SIZE_M": 64, "num_warps": 4, "num_stages": 2}
        BLOCK_SIZE_H = triton.next_power_of_2(nrep)
        _compute_attn_probs2[grid](q, k, lse, probs,
                            *q.stride(),
                            *k.stride(),
                            *lse.stride(),
                            *probs.stride(),
                            sm_scale, kernel_size, stride,
                            B, N, M, nrep,
                            D1, D2,
                            BLOCK_SIZE_H,
                            **kwargs
                            )
        
        # grid = lambda meta: (B, KH, triton.cdiv(N, meta['BLOCK_SIZE_N']))
        # kwargs = { "BLOCK_SIZE_M": 64, "num_warps": 4, "num_stages": 2}
        # BLOCK_SIZE_H = triton.next_power_of_2(nrep)
        # _compute_attn_probs3[grid](q, k, lse, probs,
        #                     *q.stride(),
        #                     *k.stride(),
        #                     *lse.stride(),
        #                     *probs.stride(),
        #                     sm_scale, kernel_size, stride,
        #                     B, N, M, nrep,
        #                     D1, D2,
        #                     BLOCK_SIZE_H,
        #                     **kwargs
        #                     )
    
    # select_probs = torch.zeros(B, KH, N, num_selcct_blocks, device=q.device, dtype=torch.float32)
    # grid=lambda meta: (B * KH, triton.cdiv(N, meta['BLOCK_SIZE']), num_selcct_blocks)
    # kwargs = {"BLOCK_SIZE": 4096, "num_warps": 8, "num_stages": 4}
    # _compute_select_probs2[grid](probs, select_probs,
    #                             *probs.stride(),
    #                             *select_probs.stride(),
    #                             kernel_size, stride, 
    #                             select_size,num_selcct_blocks,
    #                             N, M, QH, KH,
    #                             **kwargs
    #                             )
    
    select_probs = torch.zeros(B * KH * N, num_selcct_blocks, device=q.device, dtype=torch.float32)
    probs = probs.view(-1, M)
    grid=lambda meta: (B * KH * triton.cdiv(N, meta['BLOCK_SIZE']), num_selcct_blocks)
    kwargs = {"BLOCK_SIZE": 8192, "num_warps": 8, "num_stages": 4}
    _compute_select_probs3[grid](probs, select_probs,
                                probs.stride(0),
                                select_probs.stride(0),
                                kernel_size, stride, 
                                select_size,num_selcct_blocks,
                                B, N, KH,
                                **kwargs
                                )
    select_probs = select_probs.view(B, KH, N, num_selcct_blocks)

    _, indices = torch.topk(select_probs, k=top_n, dim=-1)
    grid = lambda meta: (B, KH, triton.cdiv(N, meta['BLOCK_SIZE_N']))
    BLOCK_SIZE_M = triton.next_power_of_2(top_n)
    kwargs = {"BLOCK_SIZE_N": 32, "num_warps": 4, "num_stages": 2}
    _fix_indices[grid](indices,
                    *indices.stride(),
                    select_size, num_selcct_blocks, top_n, 
                    N, M, 
                    BLOCK_SIZE_M,
                    **kwargs
                    )
    return select_probs, indices

@torch.inference_mode()
def select_for_bwd(ind):
    b, kh, n, top_n = ind.shape
    ignore_index = ind[0][0][0][-1]
    bwd_ind = torch.zeros(b, kh, n, ignore_index + 1, dtype=torch.int64, device=ind.device)
    bwd_ind.scatter_(-1, ind, 1)

    bwd_ind = bwd_ind.transpose(-1, -2).contiguous()
    count = bwd_ind.sum(-1)
    _, bwd_ind = bwd_ind.sort(-1, descending=True)
    assert bwd_ind.is_contiguous()
    assert count.is_contiguous()
    return bwd_ind, count

# @triton.autotune([triton.Config({'BLOCK_SIZE_M': bsm}, num_stages=ns, num_warps=nw)
#                  for bsm in [64, 128, 256]
#                  for ns in [1, 2, 4]
#                  for nw in [4, 8]
#                  ], key=['N', "M"])
@triton.jit
def _fwd_kernel(Q, K, V, O, LSE, Ind,
                q_stride_b, q_stride_n, q_stride_h, q_stride_d,
                k_stride_b, k_stride_m, k_stride_h, k_stride_d,
                v_stride_b, v_stride_m, v_stride_h, v_stride_d,
                o_stride_b, o_stride_n, o_stride_h, o_stride_d,
                lse_stride_b, lse_stride_h, lse_stride_n,
                ind_stride_b, ind_stride_h, ind_stride_n, ind_stride_k,
                sm_scale, top_n,
                B, N, M, nrep,
                D1: tl.constexpr, D2: tl.constexpr, VD: tl.constexpr, 
                BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_M: tl.constexpr=128):

    off_b = tl.cast(tl.program_id(0), tl.int64)
    off_kh = tl.cast(tl.program_id(1), tl.int64)
    off_n = tl.cast(tl.program_id(2), tl.int64)
    off_qh = tl.cast(off_kh * nrep, tl.int64)
    k_stride_m = tl.cast(k_stride_m, tl.int64)
    v_stride_m = tl.cast(v_stride_m, tl.int64)

    Q += off_b * q_stride_b + off_qh * q_stride_h + off_n * q_stride_n
    K += off_b * k_stride_b + off_kh * k_stride_h
    V += off_b * v_stride_b + off_kh * v_stride_h
    O += off_b * o_stride_b + off_qh * o_stride_h + off_n * o_stride_n
    LSE += off_b * lse_stride_b + off_qh * lse_stride_h + off_n
    Ind += off_b * ind_stride_b + off_kh * ind_stride_h + off_n * ind_stride_n
    rows = tl.arange(0, BLOCK_SIZE_H)
    cols = tl.arange(0, BLOCK_SIZE_M)
    d1_cols = tl.arange(0, D1)
    vd_cols = tl.arange(0, VD)
    row_mask = rows < nrep

    q_ptrs = Q + rows[:, None] * q_stride_h + d1_cols[None, :]
    k_ptrs = K + cols[None, :] * k_stride_m + d1_cols[:, None]
    v_ptrs = V + cols[:, None] * v_stride_m + vd_cols[None, :]

    q = tl.load(q_ptrs, mask=row_mask[:, None], other=0.)
    if D2 > 0:
        d2_cols = tl.arange(0, D2) + D1
        q_ptrs2 = Q + rows[:, None] * q_stride_h + d2_cols[None, :]
        k_ptrs2 = K + cols[None, :] * k_stride_m + d2_cols[:, None]
        q2 = tl.load(q_ptrs2, mask=row_mask[:, None], other=0.)

    m_i = tl.zeros([BLOCK_SIZE_H], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_SIZE_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_SIZE_H, VD], dtype=tl.float32)

    i = 0
    stop_n = tl.minimum(top_n, tl.cdiv(off_n+1, BLOCK_SIZE_M))

    while i < stop_n:
        select_idx = tl.load(Ind + i)
        start_m = tl.cast(select_idx * BLOCK_SIZE_M, tl.int64)
        k_idx = start_m + cols
        col_mask = k_idx < M
        k = tl.load(k_ptrs + start_m * k_stride_m, mask=col_mask[None, :], other=0.)
        attn_score = tl.dot(q, k)
        if D2>0:
            k2 = tl.load(k_ptrs2 + start_m * k_stride_m, mask=col_mask[None, :], other=0.)
            attn_score = tl.dot(q2, k2, attn_score)
        attn_score *= sm_scale

        
        causal_mask = off_n >= k_idx
        attn_score = tl.where(causal_mask[None, :], attn_score, float('-inf'))

        m_ij = tl.max(attn_score, axis=1)
        new_m_i = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - new_m_i)

        exp_attn_score = tl.exp(attn_score - new_m_i[:, None])

        l_i = l_i * alpha + tl.sum(exp_attn_score, axis=-1)
        acc = acc * alpha[:, None]

        v = tl.load(v_ptrs + start_m * v_stride_m, mask=col_mask[:, None], other=0.)
        acc = tl.dot(exp_attn_score.to(v.dtype), v, acc=acc)
        m_i = new_m_i

        i += 1

    acc /= l_i[:, None]
    o_ptrs = O + rows[:, None] * tl.cast(o_stride_h, tl.int64) + vd_cols[None, :]
    tl.store(o_ptrs, acc, mask=row_mask[:, None])

    lse = m_i + tl.log(l_i)
    tl.store(LSE + rows * tl.cast(lse_stride_h, tl.int64), lse, mask=row_mask)


# @triton.autotune([triton.Config({'BLOCK_SIZE_N': bsn}, num_stages=ns, num_warps=nw)
#                  for bsn in [16, 32, 64, 128]
#                  for ns in [1, 2, 4]
#                  for nw in [4, 8]
#                  ], key=['N', "M"])
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
#                  for bsm in [32, 64, 128]
#                  for bsn in [32, 64, 128]
#                  for ns in [1, 2, 4]
#                  for nw in [4, 8]
#                  ], key=['N', "M"])
@triton.jit
def _bwd_kernel(DQ, DK, DV, DO, 
                Q, K, V, 
                Lse, Delta, Ind, Count,
                q_stride_b, q_stride_n, q_stride_h, q_stride_d,
                k_stride_b, k_stride_m, k_stride_h, k_stride_d,
                v_stride_b, v_stride_m, v_stride_h, v_stride_d,
                do_stride_b, do_stride_n, do_stride_h, do_stride_d,
                lse_stride_b, lse_stride_h, lse_stride_n,
                ind_stride_b, ind_stride_h, ind_stride_m, ind_stride_n,
                cnt_stride_b, cnt_stride_h, cnt_stride_m,
                sm_scale, 
                B, N, M, nrep,  
                D1: tl.constexpr, D2: tl.constexpr, VD: tl.constexpr, 
                BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_M: tl.constexpr
                ):
    pid0 = tl.cast(tl.program_id(0), tl.int64) 
    off_m = tl.cast(pid0 * BLOCK_SIZE_M, tl.int64)
    off_b = tl.cast(tl.program_id(1), tl.int64)
    off_kh = tl.cast(tl.program_id(2), tl.int64)
    off_qh = tl.cast(off_kh * nrep, tl.int64)
    # k_stride_m = tl.cast(k_stride_m, tl.int64)
    # v_stride_m = tl.cast(v_stride_m, tl.int64)
    # q_stride_n = tl.cast(q_stride_n, tl.int64)
    # q_stride_n = tl.cast(q_stride_n, tl.int64)

    Q += off_b * q_stride_b + off_qh * q_stride_h
    K += off_b * k_stride_b + off_kh * k_stride_h
    V += off_b * v_stride_b + off_kh * v_stride_h
    DQ += off_b * q_stride_b + off_qh * q_stride_h
    DK += off_b * k_stride_b + off_kh * k_stride_h
    DV += off_b * v_stride_b + off_kh * v_stride_h
    DO += off_b * do_stride_b + off_qh * do_stride_h
    Lse += off_b * lse_stride_b + off_qh * lse_stride_h 
    Delta += off_b * lse_stride_b + off_qh * lse_stride_h
    Ind += off_b * ind_stride_b + off_kh * ind_stride_h + pid0 * ind_stride_m
    Count += off_b * cnt_stride_b + off_kh * cnt_stride_h

    rows = tl.arange(0, BLOCK_SIZE_H)
    cols = tl.arange(0, BLOCK_SIZE_M) + off_m
    row_mask = rows < nrep
    col_mask = cols < M

    k_ptrs = K + cols[None, :] * k_stride_m + tl.arange(0, D1)[:, None]
    v_ptrs = V + cols[None, :] * v_stride_m + tl.arange(0, VD)[:, None]
    
    q_ptrs = Q + rows[:, None] * q_stride_h + tl.arange(0, D1)[None, :]
    dq_ptrs = DQ + rows[:, None] * q_stride_h + tl.arange(0, D1)[None, :]
    do_ptrs = DO + rows[:, None] * do_stride_h + tl.arange(0, VD)[None, :]
    lse_ptrs = Lse + rows * lse_stride_h
    delta_ptrs = Delta + rows * lse_stride_h

    k = tl.load(k_ptrs, mask=col_mask[None, :], other=0.)
    v = tl.load(v_ptrs, mask=col_mask[None, :], other=0.)
    
    acc_dk = tl.zeros((BLOCK_SIZE_M, D1), dtype=tl.float32)
    acc_dv = tl.zeros((BLOCK_SIZE_M, VD), dtype=tl.float32)

    if D2 > 0:
        k2_ptrs = K + cols[None, :] * k_stride_m + tl.arange(0, D2)[:, None] + D1
        q2_ptrs = Q + rows[:, None] * q_stride_h + tl.arange(0, D2)[None, :] + D1
        dq2_ptrs = DQ + rows[:, None] * q_stride_h + tl.arange(0, D2)[None, :] + D1
        k2 = tl.load(k2_ptrs, mask=col_mask[None, :], other=0.)
        acc_dk2 = tl.zeros((BLOCK_SIZE_M, D2), dtype=tl.float32)

    count = tl.load(Count + pid0)
    for idx in range(0, count):
        q_idx = tl.cast(tl.load(Ind + idx), tl.int64)
        q = tl.load(q_ptrs + q_idx * q_stride_n, mask=row_mask[:, None], other=0.)
        do = tl.load(do_ptrs + q_idx * do_stride_n, mask=row_mask[:, None], other=0.)
        lse = tl.load(lse_ptrs + q_idx, mask=row_mask, other=0.)
        delta = tl.load(delta_ptrs + q_idx, mask=row_mask, other=0.)

        attn_score = tl.dot(q, k) 

        if D2 > 0:
            q2 = tl.load(q2_ptrs + q_idx * q_stride_n, mask=row_mask[:, None], other=0.)
            attn_score = tl.dot(q2, k2, attn_score)

        causal_mask = (q_idx >= cols[None, :])
        attn_score = tl.where(causal_mask, attn_score, float('-inf'))
        p = tl.exp(attn_score * sm_scale - lse[:, None])
        
        acc_dv = tl.dot(tl.permute(p, 1, 0).to(do.dtype), do, acc_dv)

        dp = tl.dot(do, v)
        ds = p * (dp - delta[:, None]) * sm_scale

        # dq = tl.load(dq_ptrs, mask=row_mask[:, None], other=0., eviction_policy="evict_last")
        
        dq = tl.dot(ds.to(k.dtype), tl.trans(k, 1, 0))
        tl.atomic_add(dq_ptrs + q_idx * q_stride_n, dq, mask=row_mask[:, None])
        acc_dk = tl.dot(tl.permute(ds, 1, 0).to(q.dtype), q, acc_dk)
        if D2 > 0:
            dq2 = tl.dot(ds.to(k.dtype), tl.trans(k2, 1, 0))
            tl.atomic_add(dq2_ptrs + q_idx * q_stride_n, dq2, mask=row_mask[:, None])
            acc_dk2 = tl.dot(tl.trans(ds, 1, 0).to(q.dtype), q2, acc_dk2)
    
    dk_ptrs = DK + cols[:, None] * k_stride_m + tl.arange(0, D1)[None, :]
    dv_ptrs = DV + cols[:, None] * v_stride_m + tl.arange(0, VD)[None, :]
    tl.store(dk_ptrs, acc_dk, mask=col_mask[:, None])
    tl.store(dv_ptrs, acc_dv, mask=col_mask[:, None])
    if D2 > 0:
        dk2_ptrs = DK + cols[:, None] * k_stride_m + tl.arange(0, D2)[None, :] + D1
        tl.store(dk2_ptrs, acc_dk2, mask=col_mask[:, None])




class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, select_size, select_indices, sm_scale):
        B, N, QH, D = q.shape
        B2, M, KH, D2 = k.shape
        B3, M2, KH2, VD = v.shape
        assert B == B2 and B == B3 and M == M2 and D == D2 and KH == KH2
        assert QH % KH == 0
        assert math.log2(VD).is_integer()
        assert math.log2(select_size).is_integer()
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
        lse = torch.empty(B, QH, N, dtype=torch.float32, device=q.device,)


        grid = lambda meta: (B, KH, N)
        nrep = QH // KH
        BLOCK_SIZE_H = triton.next_power_of_2(nrep)
        BLOCK_SIZE_M = select_size
        top_n = select_indices.size(-1)
        _fwd_kernel[grid](q, k, v, o, lse, select_indices,
                          *q.stride(),
                          *k.stride(),
                          *v.stride(),
                          *o.stride(),
                          *lse.stride(),
                          *select_indices.stride(),
                          sm_scale, top_n,
                          B, N, M, nrep, 
                          D1, D2, VD,
                          BLOCK_SIZE_H, BLOCK_SIZE_M
                        #   **kwargs
                          )
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.select_indices = select_indices
        
        ctx.infos = (B, N, M, QH, KH, D1, D2, VD, sm_scale, nrep, BLOCK_SIZE_H, BLOCK_SIZE_M)
        return o

    @staticmethod
    def backward(ctx, do, *args):
        assert do.is_contiguous()
        bwd_ind, count = select_for_bwd(ctx.select_indices)
        B, N, M, QH, KH, D1, D2, VD, sm_scale, nrep, BLOCK_SIZE_H, BLOCK_SIZE_M = ctx.infos
        q, k, v, o, lse = ctx.saved_tensors
        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        delta = torch.empty_like(lse)
        grid = lambda meta: (B, QH, triton.cdiv(N, meta["BLOCK_SIZE_N"]))
        kwargs = {"BLOCK_SIZE_N": 64, "num_warps": 8, "num_stages": 2}
        _bwd_preprocess[grid](o, do, delta,
                              *o.stride(), 
                              *delta.stride(),
                              N, VD,
                              **kwargs
                              )
        grid = (triton.cdiv(M, BLOCK_SIZE_M), B, KH)
        # grid = (B, KH, triton.cdiv(M, BLOCK_SIZE_M))
        _bwd_kernel[grid](dq, dk, dv, do, 
                          q, k, v,
                          lse, delta, 
                          bwd_ind, # [b, kh, num_select_blocks + 1, n],  最后一行不用，使用到了哪些q，前面是使用的，后面是没使用的
                          count, # [b, kh, num_select_blocks+1], 记录某个kv块被多少q块使用
                          *q.stride(), # dq和q一样
                          *k.stride(),
                          *v.stride(),
                          *do.stride(), # do和o一样
                          *lse.stride(), # lse和delta一样
                          *bwd_ind.stride(),
                          *count.stride(),
                          sm_scale, 
                          B, N, M, nrep,  
                          D1, D2, VD,
                          BLOCK_SIZE_H, BLOCK_SIZE_M,
                          )
        dq = dq.to(q.dtype)
        return dq, dk, dv, None, None, None


def select_attn(q, k, v, select_size, select_indices, sm_scale=None):
    return _attention.apply(q, k, v, select_size, select_indices, sm_scale)


# class SelectAttn(torch.nn.Module):
#     def __init__(self, select_size):
#         super().__init__()
#         self.select_size = select_size

#     def forward(self, q, k, v, select_indices, s):
#         return select_attn(q, k, v, self.select_size, select_indices)




