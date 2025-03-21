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
                sm_scale, 
                B, N, M, QH, KH, 
                D1: tl.constexpr, D2: tl.constexpr, VD: tl.constexpr, 
                BLOCK_SIZE_N: tl.constexpr=64, BLOCK_SIZE_M: tl.constexpr=128):

    off_b = tl.cast(tl.program_id(0), tl.int64)
    off_qh = tl.cast(tl.program_id(1), tl.int64)
    start_n = tl.cast(tl.program_id(2), tl.int64) * BLOCK_SIZE_N
    off_n = start_n + tl.arange(0, BLOCK_SIZE_N)
    off_kh = off_qh // (QH // KH)

    Q += off_b * q_stride_b + off_qh * q_stride_h
    K += off_b * k_stride_b + off_kh * k_stride_h 
    V += off_b * v_stride_b + off_kh * v_stride_h
    O += off_b * o_stride_b + off_qh * o_stride_h
    LSE += off_b * lse_stride_b + off_qh * lse_stride_h

    q = tl.load(Q + off_n[:, None] * q_stride_n + tl.arange(0, D1)[None, :], mask=off_n[:, None] < N, other=0.)
    if D2 > 0:
        q2 = tl.load(Q + off_n[:, None] * q_stride_n + tl.arange(0, D2)[None, :] + D1, mask=off_n[:, None] < N, other=0.)

    m_i = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)
    acc = tl.zeros([BLOCK_SIZE_N, VD], dtype=tl.float32)

    for start_m in range(0, start_n + BLOCK_SIZE_N, BLOCK_SIZE_M):
        off_m = start_m + tl.arange(0, BLOCK_SIZE_M)
        k = tl.load(K + off_m[None, :] * k_stride_m + tl.arange(0, D1)[:, None], mask=off_m[None, :] < M, other=0.)
        attn_score = tl.dot(q, k)
        if D2>0:
            k2 = tl.load(K + off_m[None, :] * k_stride_m + tl.arange(0, D2)[:, None] + D1, mask=off_m[None, :] < M, other=0.)
            attn_score = tl.dot(q2, k2, attn_score)

        attn_score = tl.where(off_n[:, None] >= off_m[None, :], attn_score * sm_scale, float('-inf'))

        m_ij = tl.max(attn_score, axis=1)
        new_m_i = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - new_m_i)

        exp_attn_score = tl.exp(attn_score - new_m_i[:, None])

        l_i = l_i * alpha + tl.sum(exp_attn_score, axis=-1)
        acc = acc * alpha[:, None]

        v = tl.load(V + off_m[:, None] * v_stride_m + tl.arange(0, VD)[None, :], mask=off_m[:, None] < M, other=0.)
        acc = tl.dot(exp_attn_score.to(v.dtype), v, acc=acc)

        m_i = new_m_i

    acc /= l_i[:, None]
    lse = m_i + tl.log(l_i)
    tl.store(O + off_n[:, None] * o_stride_n + tl.arange(0, VD)[None, :], acc, mask=off_n[:, None] < N)   
    tl.store(LSE + off_n * lse_stride_n, lse, mask=off_n < N)


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
#                  for bsm in [32, 64, 128]
#                  for bsn in [32, 64, 128]
#                  for ns in [1, 2]
#                  for nw in [4, 8]
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
                sm_scale, 
                B, N, M, QH, KH, 
                D1: tl.constexpr, D2: tl.constexpr, VD: tl.constexpr, 
                BLOCK_SIZE_N: tl.constexpr=64, BLOCK_SIZE_M: tl.constexpr=64
                ):
    start_m = tl.cast(tl.program_id(0), tl.int64) * BLOCK_SIZE_M
    off_m = start_m + tl.arange(0, BLOCK_SIZE_M)
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



    k = tl.load(K + off_m[None, :] * k_stride_m + tl.arange(0, D1)[:, None], mask=off_m[None, :] < M, other=0.)
    v = tl.load(V + off_m[None, :] * v_stride_m + tl.arange(0, VD)[:, None], mask=off_m[None, :] < M, other=0.)
    acc_dk = tl.zeros((BLOCK_SIZE_M, D1), dtype=tl.float32)
    acc_dv = tl.zeros((BLOCK_SIZE_M, VD), dtype=tl.float32)

    acc_dk = tl.zeros((BLOCK_SIZE_M, D1), dtype=tl.float32)
    acc_dv = tl.zeros((BLOCK_SIZE_M, VD), dtype=tl.float32)

    if D2 > 0:
        k2 = tl.load(K + off_m[None, :] * k_stride_m + tl.arange(0, D2)[:, None] + D1, mask=off_m[None, :] < M, other=0.)
        acc_dk2 = tl.zeros((BLOCK_SIZE_M, D2), dtype=tl.float32)
    for start_n in range(start_m, N, BLOCK_SIZE_N):
        off_n = start_n + tl.arange(0, BLOCK_SIZE_N)
        q = tl.load(Q + off_n[:, None] * q_stride_n + tl.arange(0, D1), mask=off_n[:, None] < N, other=0.)
        do = tl.load(DO + off_n[:, None] * do_stride_n + tl.arange(0, VD), mask=off_n[:, None] < N, other=0.)
        lse = tl.load(Lse + off_n, mask=off_n < N, other=0.)
        delta = tl.load(Delta + off_n, mask=off_n < N, other=0.)

        attn_score = tl.dot(q, k) 

        if D2 > 0:
            q2 = tl.load(Q + off_n[:, None] * q_stride_n + tl.arange(0, D2) + D1, mask=off_n[:, None] < N, other=0.)
            attn_score = tl.dot(q2, k2, attn_score)

        attn_score = tl.where(off_n[:, None] >= off_m[None, :], attn_score, float('-inf'))
        p = tl.exp(attn_score * sm_scale - lse[:, None])
        
        acc_dv = tl.dot(tl.trans(p, 1, 0).to(do.dtype), do, acc_dv)

        dp = tl.dot(do, v)
        ds = p * (dp - delta[:, None]) * sm_scale

        # dq = tl.load(dq_ptrs, mask=row_mask[:, None], other=0., eviction_policy="evict_last")
        
        dq = tl.dot(ds.to(k.dtype), tl.trans(k, 1, 0))
        tl.atomic_add(DQ + off_n[:, None] * q_stride_n + tl.arange(0, D1), dq, mask=off_n[:, None] < N)
        
        
        if D2 > 0:
            dq2 = tl.dot(ds.to(k.dtype), tl.trans(k2, 1, 0))
            tl.atomic_add(DQ + off_n[:, None] * q_stride_n + tl.arange(0, D2) + D1, dq2, mask=off_n[:, None] < N)
            acc_dk2 = tl.dot(tl.trans(ds, 1, 0).to(q.dtype), q2, acc_dk2)
        acc_dk = tl.dot(tl.trans(ds, 1, 0).to(q.dtype), q, acc_dk)

    tl.store(DK + off_m[:, None] * dk_stride_m + tl.arange(0, D1)[None, :], acc_dk, mask=off_m[:, None] < M)
    tl.store(DV + off_m[:, None] * dv_stride_m + tl.arange(0, VD)[None, :], acc_dv, mask=off_m[:, None] < M)
    if D2 > 0:
        tl.store(DK + off_m[:, None] * dk_stride_m + tl.arange(0, D2)[None, :] + D1, acc_dk2, mask=off_m[:, None] < M)


# @triton.autotune([triton.Config({'BLOCK_SIZE_N': bsn, 'BLOCK_SIZE_M': bsm}, num_stages=ns, num_warps=nw)
#                  for bsm in [32, 64, 128]
#                  for bsn in [32, 64, 128]
#                  for ns in [1, 2]
#                  for nw in [4, 8]
#                  ], key=['N', "M"])
@triton.jit
def _dkdv_kernel(DK, DV, DO, 
                Q, K, V, 
                Lse, Delta,
                q_stride_b, q_stride_n, q_stride_h, q_stride_d,
                k_stride_b, k_stride_m, k_stride_h, k_stride_d,
                v_stride_b, v_stride_m, v_stride_h, v_stride_d,
                dk_stride_b, dk_stride_m, dk_stride_h, dk_stride_d,
                dv_stride_b, dv_stride_m, dv_stride_h, dv_stride_d,
                do_stride_b, do_stride_n, do_stride_h, do_stride_d,
                lse_stride_b, lse_stride_h, lse_stride_n,
                sm_scale, 
                B, N, M, QH, KH, 
                D1: tl.constexpr, D2: tl.constexpr, VD: tl.constexpr, 
                BLOCK_SIZE_N: tl.constexpr=64, BLOCK_SIZE_M: tl.constexpr=64
                ):
    start_m = tl.cast(tl.program_id(2), tl.int64) * BLOCK_SIZE_M
    off_m = start_m + tl.arange(0, BLOCK_SIZE_M)
    off_b = tl.cast(tl.program_id(0), tl.int64)
    off_kh = tl.cast(tl.program_id(1), tl.int64)
    nrep = QH//KH
    nrep = tl.cast(nrep, tl.int64)
    off_qh = off_kh * nrep

    Q += off_b * q_stride_b + off_qh * q_stride_h
    K += off_b * k_stride_b + off_kh * k_stride_h 
    V += off_b * v_stride_b + off_kh * v_stride_h
    DK += off_b * dk_stride_b + off_kh * dk_stride_h 
    DV += off_b * dv_stride_b + off_kh * dv_stride_h
    DO += off_b * do_stride_b + off_qh * do_stride_h
    Lse += off_b * lse_stride_b + off_qh * lse_stride_h
    Delta += off_b * lse_stride_b + off_qh * lse_stride_h



    k = tl.load(K + off_m[None, :] * k_stride_m + tl.arange(0, D1)[:, None], mask=off_m[None, :] < M, other=0.)
    v = tl.load(V + off_m[None, :] * v_stride_m + tl.arange(0, VD)[:, None], mask=off_m[None, :] < M, other=0.)
    acc_dk = tl.zeros((BLOCK_SIZE_M, D1), dtype=tl.float32)
    acc_dv = tl.zeros((BLOCK_SIZE_M, VD), dtype=tl.float32)

    acc_dk = tl.zeros((BLOCK_SIZE_M, D1), dtype=tl.float32)
    acc_dv = tl.zeros((BLOCK_SIZE_M, VD), dtype=tl.float32)

    if D2 > 0:
        k2 = tl.load(K + off_m[None, :] * k_stride_m + tl.arange(0, D2)[:, None] + D1, mask=off_m[None, :] < M, other=0.)
        acc_dk2 = tl.zeros((BLOCK_SIZE_M, D2), dtype=tl.float32)
    for start_n in range(start_m, N, BLOCK_SIZE_N):
        off_n = start_n + tl.arange(0, BLOCK_SIZE_N)
        causal_mask = off_n[:, None] >= off_m[None, :]
        for h_idx in range(nrep):
            q = tl.load(Q + h_idx * q_stride_h + off_n[:, None] * q_stride_n + tl.arange(0, D1), mask=off_n[:, None] < N, other=0.)
            do = tl.load(DO + h_idx * do_stride_h +off_n[:, None] * do_stride_n + tl.arange(0, VD), mask=off_n[:, None] < N, other=0.)
            lse = tl.load(Lse + h_idx * lse_stride_h + off_n, mask=off_n < N, other=0.)
            delta = tl.load(Delta + h_idx * lse_stride_h + off_n, mask=off_n < N, other=0.)

            attn_score = tl.dot(q, k) 

            if D2 > 0:
                q2 = tl.load(Q + h_idx * q_stride_h + off_n[:, None] * q_stride_n + tl.arange(0, D2) + D1, mask=off_n[:, None] < N, other=0.)
                attn_score = tl.dot(q2, k2, attn_score)

            attn_score = tl.where(causal_mask, attn_score, float('-inf'))
            p = tl.exp(attn_score * sm_scale - lse[:, None])
            
            acc_dv = tl.dot(tl.trans(p, 1, 0).to(do.dtype), do, acc_dv)

            dp = tl.dot(do, v)
            ds = p * (dp - delta[:, None]) * sm_scale

            if D2 > 0:
                acc_dk2 = tl.dot(tl.trans(ds, 1, 0).to(q.dtype), q2, acc_dk2)
            acc_dk = tl.dot(tl.trans(ds, 1, 0).to(q.dtype), q, acc_dk)
    tl.store(DK + off_m[:, None] * dk_stride_m + tl.arange(0, D1)[None, :], acc_dk, mask=off_m[:, None] < M)
    tl.store(DV + off_m[:, None] * dv_stride_m + tl.arange(0, VD)[None, :], acc_dv, mask=off_m[:, None] < M)
    if D2 > 0:
        tl.store(DK + off_m[:, None] * dk_stride_m + tl.arange(0, D2)[None, :] + D1, acc_dk2, mask=off_m[:, None] < M)

# @triton.autotune([triton.Config({'BLOCK_SIZE_N': bsn, 'BLOCK_SIZE_M': bsm}, num_stages=ns, num_warps=nw)
#                  for bsm in [32, 64, 128]
#                  for bsn in [32, 64, 128]
#                  for ns in [1, 2]
#                  for nw in [4, 8]
#                  ], key=['N', "M"])
@triton.jit
def _dq_kernel(DQ, DO, 
                Q, K, V, 
                Lse, Delta,
                q_stride_b, q_stride_n, q_stride_h, q_stride_d,
                k_stride_b, k_stride_m, k_stride_h, k_stride_d,
                v_stride_b, v_stride_m, v_stride_h, v_stride_d,
                do_stride_b, do_stride_n, do_stride_h, do_stride_d,
                lse_stride_b, lse_stride_h, lse_stride_n,
                sm_scale, 
                B, N, M, QH, KH, 
                D1: tl.constexpr, D2: tl.constexpr, VD: tl.constexpr, 
                BLOCK_SIZE_N: tl.constexpr=64, BLOCK_SIZE_M: tl.constexpr=64
                ):
    start_n = tl.cast(tl.program_id(2), tl.int64) * BLOCK_SIZE_N
    off_n = start_n + tl.arange(0, BLOCK_SIZE_N)
    off_b = tl.cast(tl.program_id(0), tl.int64)
    off_qh = tl.cast(tl.program_id(1), tl.int64)
    
    off_kh = off_qh // (QH // KH)

    Q += off_b * q_stride_b + off_qh * q_stride_h
    K += off_b * k_stride_b + off_kh * k_stride_h 
    V += off_b * v_stride_b + off_kh * v_stride_h
    DQ += off_b * q_stride_b + off_qh * q_stride_h
    DO += off_b * do_stride_b + off_qh * do_stride_h
    Lse += off_b * lse_stride_b + off_qh * lse_stride_h
    Delta += off_b * lse_stride_b + off_qh * lse_stride_h

    q = tl.load(Q + off_n[:, None] * q_stride_n + tl.arange(0, D1), mask=off_n[:, None] < N, other=0.)
    acc_dq = tl.zeros((BLOCK_SIZE_N, D1), dtype=tl.float32)
    do = tl.load(DO + off_n[:, None] * do_stride_n + tl.arange(0, VD), mask=off_n[:, None] < N, other=0.)
    lse = tl.load(Lse + off_n, mask=off_n < N, other=0.)
    delta = tl.load(Delta + off_n, mask=off_n < N, other=0.)
    if D2 > 0:
        q2 = tl.load(Q + off_n[:, None] * q_stride_n + tl.arange(0, D2) + D1, mask=off_n[:, None] < N, other=0.)
        acc_dq2 = tl.zeros((BLOCK_SIZE_N, D2), dtype=tl.float32)

    for start_m in range(0, start_n + BLOCK_SIZE_N, BLOCK_SIZE_M):
        off_m = start_m + tl.arange(0, BLOCK_SIZE_M)

        k = tl.load(K + off_m[None, :] * k_stride_m + tl.arange(0, D1)[:, None], mask=off_m[None, :] < M, other=0.)
        v = tl.load(V + off_m[None, :] * v_stride_m + tl.arange(0, VD)[:, None], mask=off_m[None, :] < M, other=0.)
        attn_score = tl.dot(q, k) 
        if D2 > 0:
            k2 = tl.load(K + off_m[None, :] * k_stride_m + tl.arange(0, D2)[:, None] + D1, mask=off_m[None, :] < M, other=0.)
            attn_score = tl.dot(q2, k2, attn_score)

        attn_score = tl.where(off_n[:, None] >= off_m[None, :], attn_score, float('-inf'))
        p = tl.exp(attn_score * sm_scale - lse[:, None])

        dp = tl.dot(do, v)
        ds = p * (dp - delta[:, None]) * sm_scale
        
        acc_dq = tl.dot(ds.to(k.dtype), tl.trans(k, 1, 0), acc_dq)
        if D2 > 0:
            acc_dq2 = tl.dot(ds.to(k.dtype), tl.trans(k2, 1, 0), acc_dq2)

    tl.store(DQ + off_n[:, None] * q_stride_n + tl.arange(0, D1)[None, :], acc_dq, mask=off_n[:, None] < N)
    if D2 > 0:
        tl.store(DQ + off_n[:, None] * q_stride_n + tl.arange(0, D2)[None, :] + D1, acc_dq2, mask=off_n[:, None] < N)

class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sm_scale):
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
        # o[:, :kernel_size-1] = 0.
        lse = torch.empty(B, QH, N, dtype=torch.float32, device=q.device,)
        grid = lambda meta: (B, QH, triton.cdiv(N, meta['BLOCK_SIZE_N']))
        if D == 192:
            kwargs = {"BLOCK_SIZE_N": 128, "BLOCK_SIZE_M": 128, "num_warps": 8, "num_stages": 2}
        else:
            kwargs = {"BLOCK_SIZE_N": 128, "BLOCK_SIZE_M": 128, "num_warps": 8, "num_stages": 2}
        _fwd_kernel[grid](q, k, v, o, lse,
                          *q.stride(),
                          *k.stride(),
                          *v.stride(),
                          *o.stride(),
                          *lse.stride(),
                          sm_scale,
                          B, N, M, QH, KH, 
                          D1, D2, VD,
                          **kwargs
                          )
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.infos = (B, N, M, QH, KH, D1, D2, VD, sm_scale)
        return o, lse

    @staticmethod
    def backward(ctx, do, *args):
        assert do.is_contiguous()
        B, N, M, QH, KH, D1, D2, VD, sm_scale = ctx.infos
        q, k, v, o, lse = ctx.saved_tensors
        delta = torch.empty_like(lse)
        grid = lambda meta: (B, QH, triton.cdiv(N, meta["BLOCK_SIZE_N"]))
        kwargs = {"BLOCK_SIZE_N": 64, "num_warps": 8, "num_stages": 1}
        _bwd_preprocess[grid](o, do, delta,
                              *o.stride(), 
                              *delta.stride(),
                              N, VD,
                              **kwargs
                              )
        
        # dq = torch.zeros_like(q, dtype=torch.float32)
        # dk = torch.empty(B, M, QH, D1+D2, device=q.device, dtype=q.dtype)
        # dv = torch.empty(B, M, QH, VD, device=q.device, dtype=q.dtype)

        # grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE_M"]), B, QH)
        # if (D1 + D2) == 192:
        #     kwargs = {"BLOCK_SIZE_N": 32, "BLOCK_SIZE_M": 128, "num_warps": 8, "num_stages": 1}
        # else:
        #     kwargs = {"BLOCK_SIZE_N": 32, "BLOCK_SIZE_M": 128, "num_warps": 8, "num_stages": 1}
        # _bwd_kernel[grid](dq, dk, dv, do, 
        #                   q, k, v,
        #                   lse, delta,
        #                   *q.stride(), # dq和q一样
        #                   *k.stride(),
        #                   *v.stride(),
        #                   *dk.stride(),
        #                   *dv.stride(),
        #                   *do.stride(), # do和o一样
        #                   *lse.stride(), # lse和delta一样
        #                   sm_scale, 
        #                   B, N, M, QH, KH, 
        #                   D1, D2, VD,
        #                   **kwargs
        #                   )
        # dq = dq.to(q.dtype)
        # dk = dk.view(B, M, KH, -1, D1+D2).sum(3)
        # dv = dv.view(B, M, KH, -1, VD).sum(3)


        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        if (D1 + D2) == 192:
            kwargs = {"BLOCK_SIZE_N": 32, "BLOCK_SIZE_M": 128, "num_warps": 8, "num_stages": 1}
        else:
            kwargs = {"BLOCK_SIZE_N": 128, "BLOCK_SIZE_M": 64, "num_warps": 8, "num_stages": 2}
        grid = lambda meta: (B, KH, triton.cdiv(M, meta["BLOCK_SIZE_M"]))
        _dkdv_kernel[grid](dk, dv, do, 
                          q, k, v,
                          lse, delta,
                          *q.stride(), # dq和q一样
                          *k.stride(),
                          *v.stride(),
                          *dk.stride(),
                          *dv.stride(),
                          *do.stride(), # do和o一样
                          *lse.stride(), # lse和delta一样
                          sm_scale, 
                          B, N, M, QH, KH, 
                          D1, D2, VD,
                          **kwargs
                          )
        if (D1 + D2) == 192:
            kwargs = {"BLOCK_SIZE_N": 32, "BLOCK_SIZE_M": 128, "num_warps": 8, "num_stages": 1}
        else:
            kwargs = {"BLOCK_SIZE_N": 128, "BLOCK_SIZE_M": 128, "num_warps": 8, "num_stages": 2}
        grid = lambda meta: (B, QH, triton.cdiv(N, meta["BLOCK_SIZE_N"]), )
        _dq_kernel[grid](dq, do, 
                          q, k, v,
                          lse, delta,
                          *q.stride(), # dq和q一样
                          *k.stride(),
                          *v.stride(),
                          *do.stride(), # do和o一样
                          *lse.stride(), # lse和delta一样
                          sm_scale, 
                          B, N, M, QH, KH, 
                          D1, D2, VD,
                          **kwargs
                          )
        return dq, dk, dv, None, None, None


def triton_fa2(q, k, v, sm_scale=None):
    return _attention.apply(q, k, v, sm_scale)
