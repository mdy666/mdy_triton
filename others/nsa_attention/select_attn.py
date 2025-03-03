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


# @triton.autotune([triton.Config({'BLOCK_SIZE_M': bsm, 'BLOCK_SIZE_N': bsn}, num_stages=ns, num_warps=nw)
#                  for bsm in [32, 64, 128]
#                  for bsn in [32, 64, 128]
#                  for ns in [1, 2, 4]
#                  for nw in [4, 8]
#                  ], key=['N', "M"])
@triton.jit
def _compute_attn_probs(Q, K, Lse, P,
                q_stride_b, q_stride_n, q_stride_h, q_stride_d,
                k_stride_b, k_stride_m, k_stride_h, k_stride_d,
                lse_stride_b, lse_stride_h, lse_stride_n,
                p_stride_b, p_stride_h, p_stride_n, p_stride_m,
                sm_scale, kernel_size, stride,
                B, N, M, KH, nrep,
                D1: tl.constexpr, D2: tl.constexpr, 
                BLOCK_SIZE_N: tl.constexpr=64, BLOCK_SIZE_M: tl.constexpr=64):
    start_n = tl.cast(tl.program_id(1), tl.int64) * BLOCK_SIZE_N
    start_m = tl.cast(tl.program_id(2), tl.int64) * BLOCK_SIZE_M
    if (start_n + BLOCK_SIZE_N) < (start_m * stride + kernel_size):
        return  
    off_bh = tl.cast(tl.program_id(0), tl.int64)
    off_kh = off_bh % KH
    off_b = off_bh // KH
    off_qh = off_kh * nrep

    off_m = start_m + tl.arange(0, BLOCK_SIZE_M)
    off_n = start_n + tl.arange(0, BLOCK_SIZE_N)

    Q += off_b * q_stride_b + off_qh * q_stride_h
    K += off_b * k_stride_b + off_kh * k_stride_h
    Lse += off_b * lse_stride_b + off_qh * lse_stride_h
    P += off_b * p_stride_b + off_kh * p_stride_h


    k = tl.load(K + off_m[None, :] * k_stride_m + tl.arange(0, D1)[:, None], mask=off_m[None, :]<M)
    if D2 > 0:
        k2 = tl.load(K + off_m[None, :] * k_stride_m + tl.arange(0, D2)[:, None] + D1, mask=off_m[None, :]<M)

    k_idx = off_m * stride + kernel_size - 1
    causal_mask = off_n[:, None] >= k_idx[None, :]
    p = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
    for h_idx in range(nrep):
        q = tl.load(Q + h_idx * q_stride_h + off_n[:, None] * q_stride_n + tl.arange(0, D1)[None, :], mask=off_n[:, None] < N, other=0.)
        lse = tl.load(Lse + h_idx * lse_stride_h + off_n * lse_stride_n, mask=off_n < N, other=0.)
        attn_score = tl.dot(q, k)
        if D2 > 0:
            q2 = tl.load(Q + h_idx * q_stride_h + off_n[:, None] * q_stride_n + tl.arange(0, D2)[None, :] + D1, mask=off_n[:, None] < N, other=0.)
            attn_score = tl.dot(q2, k2, attn_score)
        attn_score = tl.where(causal_mask, attn_score, float('-inf'))
        p += tl.exp(attn_score * sm_scale - lse[:, None])
    tl.store(P + off_n[:, None] * p_stride_n + off_m[None, :] * p_stride_m, p, mask=(off_n[:, None] < N) & (off_m[None, :] < M))

# @triton.autotune([triton.Config({'BLOCK_SIZE_N': bs,}, num_stages=ns, num_warps=nw)
#                  for bs in [1, 2, 4, 8, 16, 32]
#                  for ns in [1, 2, 4]
#                  for nw in [4, 8]
#                  ], key=['N', "M"])
@triton.jit
def _compute_select_probs(AP, SP, FInd, BInd,
                          ap_stride_b, ap_stride_h, ap_stride_n, ap_stride_m,
                          sp_stride_b, sp_stride_h, sp_stride_n, sp_stride_k,
                          find_stride_b, find_stride_h, find_stride_n, find_stride_k,
                          bind_stride_b, bind_stride_h, bind_stride_k, bind_stride_n,
                          kernel_size, stride, 
                          select_size, num_selcct_blocks, top_n, return_p: tl.constexpr,
                          B, N, M, KH, 
                          BLOCK_SIZE_K: tl.constexpr,
                          BLOCK_SIZE_N: tl.constexpr=16,
                          CHUNK_N: tl.constexpr=128
                            ):
    off_bh = tl.cast(tl.program_id(0), tl.int64)
    off_h = off_bh % KH
    off_b = off_bh // KH
    start_n = tl.cast(tl.program_id(1), tl.int64) * CHUNK_N \
            + tl.program_id(2) * BLOCK_SIZE_N
    if start_n >= N:
        return
    off_n = start_n + tl.arange(0, BLOCK_SIZE_N)


    AP += off_b * ap_stride_b + off_h * ap_stride_h
    SP += off_b * sp_stride_b + off_h * sp_stride_h
    FInd += off_b * find_stride_b + off_h * find_stride_h
    BInd += off_b * bind_stride_b + off_h * bind_stride_h

    acc_p = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K), dtype=tl.float32)
    select_idx = tl.arange(0, BLOCK_SIZE_K)

    select_start = 0
    select_end = select_size
    compress_start = stride - kernel_size 
    # num_loops = (select_size + 2 * (kernel_size - stride) - kernel_size) // stride + 1
    num_loops = (select_size + kernel_size - stride) // stride
    # tl.static_print(num_loops)
    compress_idx = (select_idx * select_size - kernel_size) // stride + 1
    for _ in range(num_loops):
        compress_end = compress_start + kernel_size
        area = tl.minimum(compress_end, select_end) - tl.maximum(compress_start, select_start)
        w = area / stride
        mask = (compress_idx >= 0) & (compress_idx < M)
        p = tl.load(AP + off_n[:, None] * ap_stride_n + compress_idx[None, :] * ap_stride_m, 
                    mask=(off_n[:, None] < N) & mask[None, :], other=0.) * w
        acc_p += p
        compress_idx += 1
        compress_start += stride
    if return_p:
        acc_p = tl.where(tl.arange(0, BLOCK_SIZE_K)[None, :] == (off_n // select_size)[:, None], 9999, acc_p)
        tl.store(SP + off_n[:, None] * sp_stride_n + select_idx[None, :] * sp_stride_k, 
                  acc_p, mask=(off_n[:, None] < N) & (select_idx[None, :] < num_selcct_blocks))
    tl.store(BInd + off_n * bind_stride_n + (off_n // select_size) * bind_stride_k, off_n + 1, mask=off_n < N)
    tl.store(FInd + off_n * find_stride_n, off_n // select_size, mask=off_n < N)
    acc_p = tl.where(tl.arange(0, BLOCK_SIZE_K)[None, :] == (off_n // select_size)[:, None],
                     -1., acc_p)
    top_n = tl.minimum(top_n, (start_n + BLOCK_SIZE_N - 1) // select_size + 1)

    for i in range(1, top_n):
        max_idx = tl.argmax(acc_p, axis=-1)
        tl.store(BInd + off_n * bind_stride_n + max_idx * bind_stride_k, off_n + 1, mask=off_n < N)
        tl.store(FInd + off_n * find_stride_n + i * find_stride_k, max_idx, mask=off_n < N)
        acc_p = tl.where(tl.arange(0, BLOCK_SIZE_K)[None, :] == max_idx[:, None],
                    -1., acc_p)


@torch.inference_mode()
def select_for_fwd_bwd(q, k, lse, kernel_size, stride, select_size, top_n, sm_scale=None, return_p=False):
    B, N, QH, D = q.shape
    B2, M, KH, D2 = k.shape
    assert QH % KH == 0
    nrep = QH // KH

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

    
    probs = torch.zeros(B, KH, N, M, device=q.device, dtype=torch.float16)
    
    if D == 192:
        kwargs = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "num_warps": 8, "num_stages": 4}
    else:
        kwargs = {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "num_warps": 8, "num_stages": 4}
    grid = lambda meta: (B*KH, triton.cdiv(N, meta['BLOCK_SIZE_N']), triton.cdiv(M, meta['BLOCK_SIZE_M']))
    _compute_attn_probs[grid](q, k, lse, probs,
                        *q.stride(),
                        *k.stride(),
                        *lse.stride(),
                        *probs.stride(),
                        sm_scale, kernel_size, stride,
                        B, N, M, KH, nrep,
                        D1, D2,
                        **kwargs
                        )
    BLOCK_SIZE_K = triton.next_power_of_2(num_selcct_blocks)
    select_probs = None
    if return_p:
        select_probs = torch.zeros(B, KH, N, num_selcct_blocks, device=probs.device, dtype=torch.float16)
    # indices = torch.empty(B, KH, N, num_selcct_blocks, dtype=torch.int32, device=probs.device,)
    fwd_ind = torch.full((B, KH, N, top_n), num_selcct_blocks, dtype=torch.int32, device=probs.device)
    bwd_ind = torch.zeros(B, KH, num_selcct_blocks, N, dtype=torch.int32, device=probs.device)
    BLOCK_SIZE_N = 32
    if N > 8192:
        BLOCK_SIZE_N = 8
    elif N >= 1024 * 32:
        BLOCK_SIZE_N = 4
    elif N >= 1024 * 64:
        BLOCK_SIZE_N = 2
    grid=lambda meta: (B * KH, triton.cdiv(N, meta['CHUNK_N']), triton.cdiv(meta['CHUNK_N'], meta['BLOCK_SIZE_N']))
    kwargs = {"BLOCK_SIZE_N": BLOCK_SIZE_N, "num_warps": 4, "num_stages": 4}
    _compute_select_probs[grid](probs, select_probs if return_p else probs, fwd_ind, bwd_ind,
                                *probs.stride(),
                                *(select_probs.stride() if return_p else probs.stride()),
                                *fwd_ind.stride(),
                                *bwd_ind.stride(),
                                kernel_size, stride, 
                                select_size, num_selcct_blocks, top_n, return_p,
                                B, N, M, KH,
                                BLOCK_SIZE_K,
                                **kwargs
                                )
    return select_probs, fwd_ind, bwd_ind

# @triton.autotune([triton.Config({'BLOCK_SIZE_N': bsn}, num_stages=ns, num_warps=nw)
#                  for bsn in [128, 256, 512, 1024]
#                  for ns in [1, 2, 4]
#                  for nw in [4, 8]
#                  ], key=['N']) 
@triton.jit
def _fix_bwd_indices(Ind, Cnt,
                ind_stride_b, ind_stride_h, ind_stride_k, ind_stride_n,
                cnt_stride_b, cnt_stride_h, cnt_stride_k,
                N,
                BLOCK_SIZE_N: tl.constexpr=1024, 
                ):
    
    off_b = tl.cast(tl.program_id(0), tl.int64)
    off_h = tl.cast(tl.program_id(1), tl.int64)
    off_k = tl.cast(tl.program_id(2), tl.int64)

    Ind += off_b * ind_stride_b + off_h * ind_stride_h + off_k * ind_stride_k
    Cnt += off_b * cnt_stride_b + off_h * cnt_stride_h + off_k * cnt_stride_k

    last_cnt = 0
    cols = tl.arange(0, BLOCK_SIZE_N)
    for start_n in range(0, N, BLOCK_SIZE_N):
        off_n = start_n + cols
        ind = tl.load(Ind + off_n, mask=off_n < N, other=0)
        this_cnt = tl.sum(ind)
        if this_cnt > 0:
            this_cnt = tl.sum(tl.where(ind == 0, 0, 1))
            ind = tl.sort(ind, descending=True)
            tl.store(Ind + last_cnt + cols, ind - 1, mask=cols < this_cnt)
            last_cnt += this_cnt
    tl.store(Cnt, last_cnt)

from copy import deepcopy

@torch.inference_mode()
def fix_bwd_ind(bwd_ind, inplace=True):
    assert bwd_ind.is_contiguous()
    if not inplace:
        bwd_ind = deepcopy(bwd_ind)
    B, KH, num_selcct_blocks, N = bwd_ind.shape
    count = torch.empty(B, KH, num_selcct_blocks, dtype=torch.int32, device=bwd_ind.device)
    kwargs = {"BLOCK_SIZE_N": 256, "num_warps": 4, "num_stages": 4}
    _fix_bwd_indices[(B, KH, num_selcct_blocks)](bwd_ind, count,
                                                 *bwd_ind.stride(),
                                                 *count.stride(),
                                                 N,
                                                 **kwargs
                                                 )
    return bwd_ind, count

# @triton.autotune([triton.Config({}, num_stages=ns, num_warps=nw)
#                  for ns in [1, 2, 4]
#                  for nw in [4, 8]
#                  ], key=['D1', 'D2'])
@triton.jit
def _fwd_kernel(Q, K, V, O, LSE, Ind,
                q_stride_b, q_stride_n, q_stride_h, q_stride_d,
                k_stride_b, k_stride_m, k_stride_h, k_stride_d,
                v_stride_b, v_stride_m, v_stride_h, v_stride_d,
                o_stride_b, o_stride_n, o_stride_h, o_stride_d,
                lse_stride_b, lse_stride_h, lse_stride_n,
                ind_stride_b, ind_stride_h, ind_stride_n, ind_stride_k,
                sm_scale, top_n,
                B, N, M, KH, nrep,
                D1: tl.constexpr, D2: tl.constexpr, VD: tl.constexpr, 
                BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_M: tl.constexpr=128,
                CHUNK_N: tl.constexpr=64):

    off_bh = tl.cast(tl.program_id(0), tl.int64)
    off_kh = off_bh % KH
    off_b = off_bh // KH
    off_n = tl.cast(tl.program_id(1), tl.int64) * CHUNK_N + tl.program_id(2)
    if off_n >= N:
        return
    off_qh = off_kh * nrep


    Q += off_b * q_stride_b + off_qh * q_stride_h + off_n * q_stride_n
    K += off_b * k_stride_b + off_kh * k_stride_h
    V += off_b * v_stride_b + off_kh * v_stride_h
    O += off_b * o_stride_b + off_qh * o_stride_h + off_n * o_stride_n
    LSE += off_b * lse_stride_b + off_qh * lse_stride_h + off_n
    Ind += off_b * ind_stride_b + off_kh * ind_stride_h + off_n * ind_stride_n

    rows = tl.arange(0, BLOCK_SIZE_H)
    q = tl.load(Q + rows[:, None] * q_stride_h + tl.arange(0, D1)[None, :], mask=rows[:, None] < nrep, other=0.)
    if D2 > 0:
        q2 = tl.load(Q + rows[:, None] * q_stride_h + tl.arange(0, D2)[None, :] + D1, mask=rows[:, None] < nrep, other=0.)

    m_i = tl.zeros([BLOCK_SIZE_H], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_SIZE_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_SIZE_H, VD], dtype=tl.float32)

    i = 0
    stop_n = tl.minimum(top_n, tl.cdiv(off_n+1, BLOCK_SIZE_M))

    while i < stop_n:
        select_idx = tl.load(Ind + i).to(tl.int64)
        start_m = select_idx * BLOCK_SIZE_M
        k_idx = start_m + tl.arange(0, BLOCK_SIZE_M)
        k = tl.load(K + k_idx[None, :] * k_stride_m + tl.arange(0, D1)[:, None], mask=k_idx[None, :] < M, other=0.)
        attn_score = tl.dot(q, k)
        if D2>0:
            k2 = tl.load(K + k_idx[None, :] * k_stride_m + tl.arange(0, D2)[:, None] + D1, mask=k_idx[None, :] < M, other=0.)
            attn_score = tl.dot(q2, k2, attn_score)
        attn_score *= sm_scale

        attn_score = tl.where(off_n >= k_idx[None, :], attn_score, float('-inf'))

        m_ij = tl.max(attn_score, axis=1)
        new_m_i = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - new_m_i)

        exp_attn_score = tl.exp(attn_score - new_m_i[:, None])

        l_i = l_i * alpha + tl.sum(exp_attn_score, axis=-1)
        acc = acc * alpha[:, None]

        v = tl.load(V + k_idx[:, None] * v_stride_m + tl.arange(0, VD)[None, :], mask=k_idx[:, None] < M, other=0.)
        acc = tl.dot(exp_attn_score.to(v.dtype), v, acc=acc)
        m_i = new_m_i

        i += 1

    acc /= l_i[:, None]
    tl.store(O + rows[:, None] * o_stride_h + tl.arange(0, VD)[None, :], acc, mask=rows[:, None] < nrep)

    lse = m_i + tl.log(l_i)
    tl.store(LSE + rows * lse_stride_h, lse, mask=rows < nrep)



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


# @triton.autotune([triton.Config({}, num_stages=ns, num_warps=nw)
#                  for ns in [1, 2, 4]
#                  for nw in [4, 8]
#                  ], key=['D1', 'D2'])
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
    off_m = pid0 * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    off_b = tl.cast(tl.program_id(1), tl.int64)
    off_kh = tl.cast(tl.program_id(2), tl.int64)
    off_qh = off_kh * nrep

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

    heads = tl.arange(0, BLOCK_SIZE_H)

    k = tl.load(K + off_m[None, :] * k_stride_m + tl.arange(0, D1)[:, None], mask=off_m[None, :] < M, other=0.)
    v = tl.load(V + off_m[None, :] * v_stride_m + tl.arange(0, VD)[:, None], mask=off_m[None, :] < M, other=0.)
    acc_dk = tl.zeros((BLOCK_SIZE_M, D1), dtype=tl.float32)
    acc_dv = tl.zeros((BLOCK_SIZE_M, VD), dtype=tl.float32)

    acc_dk = tl.zeros((BLOCK_SIZE_M, D1), dtype=tl.float32)
    acc_dv = tl.zeros((BLOCK_SIZE_M, VD), dtype=tl.float32)

    if D2 > 0:
        k2 = tl.load(K + off_m[None, :] * k_stride_m + tl.arange(0, D2)[:, None] + D1, mask=off_m[None, :] < M, other=0.)
        acc_dk2 = tl.zeros((BLOCK_SIZE_M, D2), dtype=tl.float32)

    count = tl.load(Count + pid0)
    for idx in range(0, count):
        q_idx = tl.cast(tl.load(Ind + idx), tl.int64)
        q = tl.load(Q + q_idx * q_stride_n + heads[:, None] * q_stride_h + tl.arange(0, D1)[None, :], mask=heads[:, None]<nrep, other=0.)
        do = tl.load(DO + q_idx * do_stride_n + heads[:, None] * do_stride_h + tl.arange(0, VD)[None, :], mask=heads[:, None]<nrep, other=0.)
        lse = tl.load(Lse + q_idx * lse_stride_n + heads * lse_stride_h, mask=heads<nrep, other=0.)
        delta = tl.load(Delta + q_idx * lse_stride_n + heads * lse_stride_h, mask=heads<nrep, other=0.)

        attn_score = tl.dot(q, k) 

        if D2 > 0:
            q2 = tl.load(Q + q_idx * q_stride_n + heads[:, None] * q_stride_h + tl.arange(0, D2)[None, :] + D1, mask=heads[:, None]<nrep, other=0.)
            attn_score = tl.dot(q2, k2, attn_score)

        attn_score = tl.where(q_idx >= off_m[None, :], attn_score, float('-inf'))
        p = tl.exp(attn_score * sm_scale - lse[:, None])
        
        acc_dv = tl.dot(tl.permute(p, 1, 0).to(do.dtype), do, acc_dv)

        dp = tl.dot(do, v)
        ds = p * (dp - delta[:, None]) * sm_scale

        # dq = tl.load(dq_ptrs, mask=row_mask[:, None], other=0., eviction_policy="evict_last")
        
        dq = tl.dot(ds.to(k.dtype), tl.trans(k, 1, 0))
        tl.atomic_add(DQ + q_idx * q_stride_n + heads[:, None] * q_stride_h + tl.arange(0, D1)[None, :], dq, mask=heads[:, None]<nrep)
        acc_dk = tl.dot(tl.permute(ds, 1, 0).to(q.dtype), q, acc_dk)
        if D2 > 0:
            dq2 = tl.dot(ds.to(k.dtype), tl.trans(k2, 1, 0))
            tl.atomic_add(DQ + q_idx * q_stride_n + heads[:, None] * q_stride_h + tl.arange(0, D2)[None, :] + D1, dq2, mask=heads[:, None]<nrep)
            acc_dk2 = tl.dot(tl.trans(ds, 1, 0).to(q.dtype), q2, acc_dk2)
    
    tl.store(DK + off_m[:, None] * k_stride_m + tl.arange(0, D1)[None, :], acc_dk, mask=off_m[:, None] < M)
    tl.store(DV + off_m[:, None] * v_stride_m + tl.arange(0, VD)[None, :], acc_dv, mask=off_m[:, None] < M)
    if D2 > 0:
        tl.store(DK + off_m[:, None] * k_stride_m + tl.arange(0, D2)[None, :] + D1, acc_dk2, mask=off_m[:, None] < M)



class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, select_size, fwd_ind, bwd_ind, sm_scale, inplace):
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

        nrep = QH // KH
        BLOCK_SIZE_H = triton.next_power_of_2(nrep)
        BLOCK_SIZE_M = select_size
        top_n = fwd_ind.size(-1)
        if D == 192:
            kwargs = {"num_warps": 4, "num_stages": 4}
        else:
            kwargs = {"num_warps": 4, "num_stages": 1}
        grid = lambda meta: (B*KH, triton.cdiv(N, meta['CHUNK_N']), meta['CHUNK_N'])
        _fwd_kernel[grid](q, k, v, o, lse, fwd_ind,
                        *q.stride(),
                        *k.stride(),
                        *v.stride(),
                        *o.stride(),
                        *lse.stride(),
                        *fwd_ind.stride(),
                        sm_scale, top_n,
                        B, N, M, KH, nrep, 
                        D1, D2, VD,
                        BLOCK_SIZE_H, BLOCK_SIZE_M,
                        **kwargs
                        )
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.bwd_ind = bwd_ind
        ctx.infos = (B, N, M, QH, KH, D1, D2, VD, sm_scale, nrep, BLOCK_SIZE_H, BLOCK_SIZE_M)
        ctx.inplace = inplace
        return o

    @staticmethod
    def backward(ctx, do, *args):
        assert do.is_contiguous()
        bwd_ind, count = fix_bwd_ind(ctx.bwd_ind, ctx.inplace)
        # bwd_ind, count = select_for_bwd(ctx.fwd_ind.to(torch.int64))
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
        
        if (D1 + D2) == 192:
            kwargs = {"num_warps": 8, "num_stages": 4}
        else:
            kwargs = {"num_warps": 8, "num_stages": 1}
        grid = (triton.cdiv(M, BLOCK_SIZE_M), B, KH)
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
                          **kwargs
                          )
        dq = dq.to(q.dtype)
        return dq, dk, dv, None, None, None, None, None


def select_attn(q, k, v, select_size, fwd_ind, bwd_ind, sm_scale=None, inplace=True):
    return _attention.apply(q, k, v, select_size, fwd_ind, bwd_ind, sm_scale, inplace)


# class SelectAttn(torch.nn.Module):
#     def __init__(self, select_size):
#         super().__init__()
#         self.select_size = select_size

#     def forward(self, q, k, v, select_indices, s):
#         return select_attn(q, k, v, self.select_size, select_indices)




