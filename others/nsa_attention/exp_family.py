import torch
import triton
import triton.language as tl
import math
from pku_nsa import argsort


# @triton.autotune([triton.Config({'BLOCK_SIZE_M': bsm}, num_stages=ns, num_warps=nw)
#                  for bsm in [32, 64, 128, 256]
#                  for ns in [1, 2, 4]
#                  for nw in [4, 8]
#                  ], key=['N', "M"])
@triton.jit
def _compute_attn_probs1(Q, K, Lse, P,
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
        # tl.store(p_ptrs, p, mask=col_mask)
        start_m += BLOCK_SIZE_M
        k_ptrs += BLOCK_SIZE_M * k_stride_m
        p_ptrs += BLOCK_SIZE_M
        if D2 > 0:
            k_ptrs2 += BLOCK_SIZE_M * k_stride_m

# @triton.autotune([triton.Config({'BLOCK_SIZE_M': bsm, 'BLOCK_SIZE_N': bsn}, num_stages=ns, num_warps=nw)
#                  for bsm in [32, 64, 128]
#                  for bsn in [32, 64, 128]
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
                # BLOCK_SIZE_H: tl.constexpr, 
                BLOCK_SIZE_N: tl.constexpr=64, BLOCK_SIZE_M: tl.constexpr=64):
    off_b = tl.cast(tl.program_id(0), tl.int64)
    off_kh = tl.cast(tl.program_id(1), tl.int64)
    start_m = tl.cast(tl.program_id(2), tl.int64) * BLOCK_SIZE_M
    off_m = start_m + tl.arange(0, BLOCK_SIZE_M)
    off_qh = tl.cast(off_kh * nrep, tl.int64)
    k_stride_m = tl.cast(k_stride_m, tl.int64)
    q_stride_h = tl.cast(q_stride_h, tl.int64)
    lse_stride_h = tl.cast(lse_stride_h, tl.int64)

    Q += off_b * q_stride_b + off_qh * q_stride_h
    K += off_b * k_stride_b + off_kh * k_stride_h
    Lse += off_b * lse_stride_b + off_qh * lse_stride_h
    P += off_b * p_stride_b + off_kh * p_stride_h


    k = tl.load(K + off_m[None, :] * k_stride_m + tl.arange(0, D1)[:, None], mask=off_m[None, :]<M)
    if D2 > 0:
        k2 = tl.load(K + off_m[None, :] * k_stride_m + tl.arange(0, D2)[:, None] + D1, mask=off_m[None, :]<M)

    k_idx = off_m * stride + kernel_size - 1
    for start_q_idx in range(start_m * stride + kernel_size - 1, N, BLOCK_SIZE_N):
        off_n = tl.cast(start_q_idx, tl.int64) + tl.arange(0, BLOCK_SIZE_N)
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


# @triton.autotune([triton.Config({'BLOCK_SIZE_M': bsm, 'BLOCK_SIZE_N': bsn}, num_stages=ns, num_warps=nw)
#                  for bsm in [32, 64, 128]
#                  for bsn in [32, 64, 128]
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

# @triton.autotune([triton.Config({'BLOCK_SIZE_M': bsm, 'BLOCK_SIZE_N': bsn}, num_stages=ns, num_warps=nw)
#                  for bsm in [32, 64, 128]
#                  for bsn in [32, 64, 128]
#                  for ns in [1, 2, 4]
#                  for nw in [4, 8]
#                  ], key=['N', "M"])
@triton.jit
def _compute_attn_probs4(Q, K, Lse, P,
                q_stride_b, q_stride_n, q_stride_h, q_stride_d,
                k_stride_b, k_stride_m, k_stride_h, k_stride_d,
                lse_stride_b, lse_stride_h, lse_stride_n,
                p_stride_b, p_stride_h, p_stride_m, p_stride_n,
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



@torch.inference_mode()
def compute_p(q, k, lse, kernel_size, stride, sm_scale=None, method=1):
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
    nrep = QH // KH
    
    
    if method == 1:
        probs = torch.zeros(B, KH, N, M, device=q.device, dtype=torch.float16)
        grid = (B, KH, N)
        kwargs = { "BLOCK_SIZE_M": 64, "num_warps": 4, "num_stages": 2}
        BLOCK_SIZE_H = triton.next_power_of_2(nrep)
        _compute_attn_probs1[grid](q, k, lse, probs,
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
    elif method == 2:
        probs = torch.zeros(B, KH, N, M, device=q.device, dtype=torch.float16)
        grid = lambda meta: (B, KH, triton.cdiv(M, meta['BLOCK_SIZE_M']))
        kwargs = {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "num_warps": 8, "num_stages": 4}
        _compute_attn_probs2[grid](q, k, lse, probs,
                            *q.stride(),
                            *k.stride(),
                            *lse.stride(),
                            *probs.stride(),
                            sm_scale, kernel_size, stride,
                            B, N, M, nrep,
                            D1, D2,
                            **kwargs
                            )
    elif method == 3:
        probs = torch.zeros(B, KH, N, M, device=q.device, dtype=torch.float16)
        grid = lambda meta: (B*KH, triton.cdiv(N, meta['BLOCK_SIZE_N']), triton.cdiv(M, meta['BLOCK_SIZE_M']))
        if D == 192:
            kwargs = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "num_warps": 8, "num_stages": 4}
        else:
            kwargs = {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "num_warps": 8, "num_stages": 4}
        _compute_attn_probs3[grid](q, k, lse, probs,
                            *q.stride(),
                            *k.stride(),
                            *lse.stride(),
                            *probs.stride(),
                            sm_scale, kernel_size, stride,
                            B, N, M, KH, nrep,
                            D1, D2,
                            **kwargs
                            )
    elif method == 4:
        probs = torch.zeros(B, KH, M, N, device=q.device, dtype=torch.float16)
        grid = lambda meta: (B*KH, triton.cdiv(N, meta['BLOCK_SIZE_N']), triton.cdiv(M, meta['BLOCK_SIZE_M']))
        if D == 192:
            kwargs = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "num_warps": 8, "num_stages": 4}
        else:
            kwargs = {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "num_warps": 8, "num_stages": 4}
        _compute_attn_probs4[grid](q, k, lse, probs,
                            *q.stride(),
                            *k.stride(),
                            *lse.stride(),
                            *probs.stride(),
                            sm_scale, kernel_size, stride,
                            B, N, M, KH, nrep,
                            D1, D2,
                            **kwargs
                            )
    return probs


# @triton.autotune([triton.Config({'BLOCK_SIZE': bs,}, num_stages=ns, num_warps=nw)
#                  for bs in [4096, 8192, 8192 * 2, 8192 * 4]
#                  for ns in [1, 2, 4]
#                  for nw in [4, 8]
#                  ], key=['N', "M"])
@triton.jit
def _compute_select_probs1(AP, SP,
                          ap_stride,
                          sp_stride,
                          kernel_size, stride, 
                          select_size, num_selcct_blocks,
                          B, N, KH,
                          BLOCK_SIZE: tl.constexpr=2048,
                            ):
    off_n = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    off_m = tl.program_id(1)
    total = B * N * KH

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
        w = area / stride
        p = tl.load(ap_ptrs+compress_start_idx, mask=off_n < total, other=0.) * w
        acc_p = tl.where((off_n % N) // select_size == off_m, 9999, acc_p + p)
        compress_start_idx += 1
        compress_start += stride
    tl.store(SP + off_n * sp_stride + off_m, acc_p, mask=off_n < total)

# @triton.autotune([triton.Config({'BLOCK_SIZE': bs,}, num_stages=ns, num_warps=nw)
#                  for bs in [4096, 8192, 8192 * 2, 8192 * 4]
#                  for ns in [1, 2, 4]
#                  for nw in [4, 8]
#                  ], key=['N', "M"])
@triton.jit
def _compute_select_probs2(AP, SP,
                          ap_stride_b, ap_stride_h, ap_stride_k, ap_stride_n,
                          sp_stride_b, sp_stride_h, sp_stride_n, ap_stride_m,
                          kernel_size, stride, 
                          select_size, num_selcct_blocks,
                          B, N, KH,
                          BLOCK_SIZE: tl.constexpr=2048,
                            ):
    off_bh = tl.cast(tl.program_id(0), tl.int64)
    off_b = off_bh // KH
    off_h = off_bh % KH
    off_n = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    off_m = tl.program_id(2)

    ap_ptrs = AP + off_b * ap_stride_b + off_h * ap_stride_h
    sp_ptrs = SP + off_b * sp_stride_b + off_h * sp_stride_h + off_m
    acc_p = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    select_idx = off_m
    select_start = select_idx * select_size
    select_end = tl.minimum(select_start+select_size, N)

    compress_start_idx = tl.maximum((select_start-kernel_size) // stride + 1, 0)
    compress_start = compress_start_idx * stride
    while (compress_start < select_end) & (compress_start + kernel_size <= N):
        compress_end = compress_start + kernel_size
        area = tl.minimum(compress_end, select_end) - tl.maximum(compress_start, select_start)
        w = area / stride
        p = tl.load(ap_ptrs + compress_start_idx * ap_stride_k + off_n, mask=off_n < N, other=0.) * w
        acc_p = tl.where(off_n // select_size == off_m, 9999, acc_p + p)
        compress_start_idx += 1
        compress_start += stride
    tl.store(sp_ptrs + off_n * sp_stride_n, acc_p, mask=off_n < N)

# @triton.autotune([triton.Config({'BLOCK_SIZE': bs,}, num_stages=ns, num_warps=nw)
#                  for bs in [4096, 8192, 8192 * 2, 8192 * 4]
#                  for ns in [1, 2, 4]
#                  for nw in [4, 8]
#                  ], key=['N', "M"])
@triton.jit
def _compute_select_probs3(AP, SP,
                          ap_stride_b, ap_stride_h, ap_stride_k, ap_stride_n,
                          sp_stride_b, sp_stride_h, sp_stride_m, sp_stride_n,
                          kernel_size, stride, 
                          select_size, num_selcct_blocks,
                          B, N, KH,
                          BLOCK_SIZE: tl.constexpr=2048,
                            ):
    off_bh = tl.cast(tl.program_id(0), tl.int64)
    off_b = off_bh // KH
    off_h = off_bh % KH
    off_n = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    off_m = tl.program_id(2)

    ap_ptrs = AP + off_b * ap_stride_b + off_h * ap_stride_h
    sp_ptrs = SP + off_b * sp_stride_b + off_h * sp_stride_h + off_m * sp_stride_m
    acc_p = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    select_idx = off_m
    select_start = select_idx * select_size
    select_end = tl.minimum(select_start+select_size, N)

    compress_start_idx = tl.maximum((select_start-kernel_size) // stride + 1, 0)
    compress_start = compress_start_idx * stride
    while (compress_start < select_end) & (compress_start + kernel_size <= N):
        compress_end = compress_start + kernel_size
        area = tl.minimum(compress_end, select_end) - tl.maximum(compress_start, select_start)
        w = area / stride
        p = tl.load(ap_ptrs + compress_start_idx * ap_stride_k + off_n, mask=off_n < N, other=0.) * w
        acc_p = tl.where(off_n // select_size == off_m, 9999, acc_p + p)
        compress_start_idx += 1
        compress_start += stride
    tl.store(sp_ptrs + off_n, acc_p, mask=off_n < N)

# @triton.autotune([triton.Config({'BLOCK_SIZE_N': bs,}, num_stages=ns, num_warps=nw)
#                  for bs in [1, 2, 4, 8, 16, 32]
#                  for ns in [1, 2, 4]
#                  for nw in [4, 8]
#                  ], key=['N', "M"])
@triton.jit
def _compute_select_probs4(AP, SP, Ind,
                          ap_stride_b, ap_stride_h, ap_stride_n, ap_stride_m,
                          sp_stride_b, sp_stride_h, sp_stride_n, sp_stride_k,
                          ind_stride_b, ind_stride_h, ind_stride_n, ind_stride_k,
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
    tmp = tl.cast(tl.program_id(1), tl.int64) * CHUNK_N \
            + tl.program_id(2) * BLOCK_SIZE_N
    if tmp > N:
        return
    off_n = tmp + tl.arange(0, BLOCK_SIZE_N)


    AP += off_b * ap_stride_b + off_h * ap_stride_h
    SP += off_b * sp_stride_b + off_h * sp_stride_h
    Ind += off_b * ind_stride_b + off_h * ind_stride_h

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
    tl.store(Ind + off_n * ind_stride_n, off_n // select_size, mask=off_n < N)
    acc_p = tl.where(tl.arange(0, BLOCK_SIZE_K)[None, :] == (off_n // select_size)[:, None],
                     -1., acc_p)
    for i in range(1, top_n):
        if (tmp // select_size) >= i:
            max_idx = tl.argmax(acc_p, axis=-1)
            tl.store(Ind + off_n * ind_stride_n + i, max_idx, mask=off_n < N)
            acc_p = tl.where(tl.arange(0, BLOCK_SIZE_K)[None, :] == max_idx[:, None],
                        -1., acc_p)
        else:
            tl.store(Ind + off_n * ind_stride_n + i, num_selcct_blocks, mask=off_n < N)


# @triton.autotune([triton.Config({'BLOCK_SIZE_N': bs,}, num_stages=ns, num_warps=nw)
#                  for bs in [1, 2, 4, 8, 16, 32]
#                  for ns in [1, 2, 4]
#                  for nw in [4, 8]
#                  ], key=['N', "M"])
@triton.jit
def _compute_select_probs5(AP, SP, FInd, BInd,
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
    top_n = tl.minimum(top_n, (start_n + BLOCK_SIZE_N - 1) // select_size + 1)
    tl.store(BInd + off_n * bind_stride_n + (off_n // select_size) * bind_stride_k, off_n + 1, mask=off_n < N)
    tl.store(FInd + off_n * find_stride_n, off_n // select_size, mask=off_n < N)
    acc_p = tl.where(tl.arange(0, BLOCK_SIZE_K)[None, :] == (off_n // select_size)[:, None],
                     -1., acc_p)

    for i in range(1, top_n):
        max_idx = tl.argmax(acc_p, axis=-1)
        tl.store(BInd + off_n * bind_stride_n + max_idx * bind_stride_k, off_n + 1, mask=off_n < N)
        tl.store(FInd + off_n * find_stride_n + i * find_stride_k, max_idx, mask=off_n < N)
        acc_p = tl.where(tl.arange(0, BLOCK_SIZE_K)[None, :] == max_idx[:, None],
                    -1., acc_p)

    # acc_p = tl.where(tl.arange(0, BLOCK_SIZE_K)[None, :] == (off_n // select_size)[:, None], 9999, acc_p)
    # ind = tl.broadcast_to(tl.arange(0, BLOCK_SIZE_K)[None, :], (BLOCK_SIZE_N, BLOCK_SIZE_K))
    # o, ind = argsort(acc_p, ind, 1, True)
    # tl.store(FInd + off_n[:, None] * find_stride_n + tl.arange(0, BLOCK_SIZE_K)[None, :], ind, mask=tl.arange(0, BLOCK_SIZE_K)[None, :]<top_n)
    # tl.store(BInd + off_n[:, None] * bind_stride_n + ind * bind_stride_k, (off_n+1)[:, None], mask=tl.arange(0, BLOCK_SIZE_K)[None, :]<top_n)


# @triton.autotune([triton.Config({'BLOCK_SIZE_N': bsn}, num_stages=ns, num_warps=nw)
#                  for bsn in [16, 32, 64, 128, 256]
#                  for ns in [1, 2, 4]
#                  for nw in [4, 8]
#                  ], key=['N', "M"]) 
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
    col_mask = tl.arange(0, BLOCK_SIZE_M) < top_n

    Ind += off_b * ind_stride_b + off_h * ind_stride_h

    ind = tl.load(Ind + off_n[:, None] * ind_stride_n + tl.arange(0, BLOCK_SIZE_M)[None, :], 
                 mask=row_mask[:, None] & col_mask[None, :],
                 other=0.)
    fill_mask = (off_n // select_size)[:, None] >= tl.arange(0, BLOCK_SIZE_M)[None, :]
    ind = tl.where(fill_mask, ind, num_selcct_blocks)
    tl.store(Ind + off_n[:, None] * ind_stride_n + tl.arange(0, BLOCK_SIZE_M)[None, :], 
            ind,
            mask=row_mask[:, None] & col_mask[None, :])
    

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

    
@torch.inference_mode()
def compute_select_p(probs, kernel_size, stride, select_size, top_n=16, method=1, return_p=False):
    if method == 1:
        B, KH, N, M = probs.shape
        probs = probs.view(-1, M)
        num_selcct_blocks = triton.cdiv(N, select_size)
        select_probs = torch.zeros(B * KH * N, num_selcct_blocks, device=probs.device, dtype=torch.float16)
        grid=lambda meta: (triton.cdiv(B * KH * N, meta['BLOCK_SIZE']), num_selcct_blocks)
        kwargs = {"BLOCK_SIZE": 4096, "num_warps": 4, "num_stages": 2}
        _compute_select_probs1[grid](probs, select_probs,
                                    probs.stride(0),
                                    select_probs.stride(0),
                                    kernel_size, stride, 
                                    select_size,num_selcct_blocks,
                                    B, N, KH,
                                    **kwargs
                                    )
        select_probs = select_probs.view(B, KH, N, num_selcct_blocks)
        _, indices = torch.topk(select_probs,top_n, -1)
    if method == 2:
        B, KH, M, N = probs.shape
        num_selcct_blocks = triton.cdiv(N, select_size)
        select_probs = torch.zeros(B, KH, N, num_selcct_blocks, device=probs.device, dtype=torch.float16)
        grid=lambda meta: (B * KH, triton.cdiv(N, meta['BLOCK_SIZE']), num_selcct_blocks)
        kwargs = {"BLOCK_SIZE": 4096, "num_warps": 4, "num_stages": 2}
        _compute_select_probs2[grid](probs, select_probs,
                                    *probs.stride(),
                                    *select_probs.stride(),
                                    kernel_size, stride, 
                                    select_size,num_selcct_blocks,
                                    B, N, KH,
                                    **kwargs
                                    )
        _, indices = torch.topk(select_probs, top_n, -1)
    if method == 3:
        B, KH, M, N = probs.shape
        # probs = probs.view(-1, M)
        num_selcct_blocks = triton.cdiv(N, select_size)
        select_probs = torch.zeros(B, KH, num_selcct_blocks, N,  device=probs.device, dtype=torch.float16)
        grid=lambda meta: (B * KH, triton.cdiv(N, meta['BLOCK_SIZE']), num_selcct_blocks)
        kwargs = {"BLOCK_SIZE": 4096, "num_warps": 8, "num_stages": 1}
        _compute_select_probs3[grid](probs, select_probs,
                                    *probs.stride(),
                                    *select_probs.stride(),
                                    kernel_size, stride, 
                                    select_size,num_selcct_blocks,
                                    B, N, KH,
                                    **kwargs
                                    )
        _, indices = torch.topk(select_probs.transpose(-1, -2), top_n, -1)
    if method == 4:
        B, KH, N, M = probs.shape
        num_selcct_blocks = triton.cdiv(N, select_size)
        BLOCK_SIZE_K = triton.next_power_of_2(num_selcct_blocks)
        select_probs = None
        if return_p:
            select_probs = torch.zeros(B, KH, N, num_selcct_blocks, device=probs.device, dtype=torch.float16)
        indices = torch.empty(B, KH, N, top_n, dtype=torch.int64, device=probs.device,)
        BLOCK_SIZE_N = 32
        if N > 8192:
            BLOCK_SIZE_N = 8
        elif N >= 1024 * 32:
            BLOCK_SIZE_N = 4
        elif N >= 1024 * 64:
            BLOCK_SIZE_N = 2
        grid=lambda meta: (B * KH, triton.cdiv(N, meta['CHUNK_N']), triton.cdiv(meta['CHUNK_N'], meta['BLOCK_SIZE_N']))
        kwargs = {"BLOCK_SIZE_N": BLOCK_SIZE_N, "num_warps": 4, "num_stages": 4}
        _compute_select_probs4[grid](probs, select_probs if return_p else probs, indices,
                                    *probs.stride(),
                                    *(select_probs.stride() if return_p else probs.stride()),
                                    *indices.stride(),
                                    kernel_size, stride, 
                                    select_size, num_selcct_blocks, top_n, return_p,
                                    B, N, M, KH,
                                    BLOCK_SIZE_K,
                                    **kwargs
                                    )
        # _, indices = torch.topk(select_probs, top_n, -1)
    if method != 4:
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

def select_for_fwd_bwd(probs, kernel_size, stride, select_size, top_n=16, return_p=False):
    B, KH, N, M = probs.shape
    num_selcct_blocks = triton.cdiv(N, select_size)
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
    _compute_select_probs5[grid](probs, select_probs if return_p else probs, fwd_ind, bwd_ind,
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

    count = torch.empty(B, KH, num_selcct_blocks, dtype=torch.int32, device=probs.device)
    kwargs = {"BLOCK_SIZE_N": 256, "num_warps": 4, "num_stages": 4}
    # _fix_bwd_indices[(B, KH, num_selcct_blocks)](bwd_ind, count,
    #                                              *bwd_ind.stride(),
    #                                              *count.stride(),
    #                                              N,
    #                                              **kwargs
    #                                              )
    return select_probs, fwd_ind, bwd_ind, count


@triton.autotune([triton.Config({'BLOCK_SIZE_M': bsm}, num_stages=ns, num_warps=nw)
                 for bsm in [32, 64, 128, 256]
                 for ns in [1, 2, 4]
                 for nw in [4, 8]
                 ], key=['N', "M"])
@triton.jit
def _fused_p(Q, K, Lse, Ind,
                q_stride_b, q_stride_n, q_stride_h, q_stride_d,
                k_stride_b, k_stride_m, k_stride_h, k_stride_d,
                lse_stride_b, lse_stride_h, lse_stride_n,
                ind_stride_b, ind_stride_h, ind_stride_n, ind_stride_k,
                sm_scale, kernel_size, stride, select_size, num_select_blocks, top_n,
                B, N, M, KH, nrep,
                D1: tl.constexpr, D2: tl.constexpr, 
                BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_M: tl.constexpr=64,
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
    Lse += off_b * lse_stride_b + off_qh * lse_stride_h + off_n
    Ind += off_b * ind_stride_b + off_kh * ind_stride_h + off_n * ind_stride_n


    q = tl.load(Q + tl.arange(0, BLOCK_SIZE_H)[:, None] * q_stride_h + tl.arange(0, D1)[None, :],
                mask=tl.arange(0, BLOCK_SIZE_H)[:, None] < nrep, other=0.)
    lse = tl.load(Lse + tl.arange(0, BLOCK_SIZE_H) * lse_stride_h, 
                  mask=tl.arange(0, BLOCK_SIZE_H)< nrep, other=0.)
    if D2 > 0:
        q2 = tl.load(Q + tl.arange(0, BLOCK_SIZE_H)[:, None] * q_stride_h + tl.arange(0, D2)[None, :] + D1,
                    mask=tl.arange(0, BLOCK_SIZE_H)[:, None] < nrep, other=0.)
    top1_values = -float('inf')
    top2_values = -float('inf')
    top1_ind = num_select_blocks
    top2_ind = num_select_blocks
    for start_m in range(0, off_n//select_size, BLOCK_SIZE_M):
        select_idx = start_m + tl.arange(0, BLOCK_SIZE_M)
        select_start = 0
        select_end = select_size
        compress_start = stride - kernel_size 
        num_loops = (select_size + kernel_size - stride) // stride
        compress_idx = (select_idx * select_size - kernel_size) // stride + 1
        acc_p = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
        for _ in range(num_loops):
            compress_end = compress_start + kernel_size
            area = tl.minimum(compress_end, select_end) - tl.maximum(compress_start, select_start)
            w = area / stride
            mask = (compress_idx >= 0) & (compress_idx < M)
            k = tl.load(K + compress_idx[None, :] * k_stride_m + tl.arange(0, D1)[:, None],
                        mask=mask[None, :], other=0.)
            attn_score = tl.dot(q, k)
            if D2>0:
                k2 = tl.load(K + compress_idx[None, :] * k_stride_m + tl.arange(0, D2)[:, None] + D1,
                    mask=mask[None, :], other=0.)
                attn_score = tl.dot(q2, k2, attn_score)
            k_idx = compress_idx * stride + kernel_size - 1
            attn_score = tl.where(off_n >= k_idx[None, :], attn_score, float('-inf'))
            p = tl.exp(attn_score * sm_scale - lse[:, None])
            acc_p += tl.sum(p, 0) * w
            compress_idx += 1
            compress_start += stride
        top1_values_i, top1_ind_i = tl.max(acc_p, -1, True)
        if top1_values_i > top1_values:
            top2_values = top1_values
            top2_ind = top1_ind
            top1_values = top1_values_i
            top1_ind = top1_ind_i

            acc_p = tl.where(tl.arange(0, BLOCK_SIZE_M) == top1_ind_i, -1., acc_p)
            top2_values_i, top2_ind_i = tl.max(acc_p, -1, True)
            if top2_values_i > top2_values:
                top2_values = top2_values_i
                top2_ind = top2_ind_i
        elif top1_values_i > top2_values:
            top2_values = top1_values_i
            top2_ind = top1_ind_i

    tl.store(Ind + 0, off_n // select_size)
    tl.store(Ind + 1, top1_ind)
    tl.store(Ind + 2, top2_ind)

@triton.autotune([triton.Config({'BLOCK_SIZE_M': bsm}, num_stages=ns, num_warps=nw)
                 for bsm in [32, 64, 128, 256]
                 for ns in [1, 2, 4]
                 for nw in [4, 8]
                 ], key=['N', "M"])
@triton.jit
def _fused_p2(Q, K, Lse, Val, Ind,
                q_stride_b, q_stride_n, q_stride_h, q_stride_d,
                k_stride_b, k_stride_m, k_stride_h, k_stride_d,
                lse_stride_b, lse_stride_h, lse_stride_n,
                ind_stride_b, ind_stride_h, ind_stride_n, ind_stride_k,
                sm_scale, kernel_size, stride, select_size, num_select_blocks, top_n,
                B, N, M, KH, nrep,
                D1: tl.constexpr, D2: tl.constexpr, 
                BLOCK_SIZE_N: tl.constexpr=64, BLOCK_SIZE_M: tl.constexpr=64,):
    off_bh = tl.cast(tl.program_id(0), tl.int64)
    off_kh = off_bh % KH
    off_b = off_bh // KH
    off_qh = off_kh * nrep
    start_n = tl.cast(tl.program_id(1), tl.int64) * BLOCK_SIZE_N
    start_m = tl.cast(tl.program_id(2), tl.int64) * BLOCK_SIZE_M
    if (start_n + BLOCK_SIZE_N - 1) // select_size < start_m:
        return  
    off_n = start_n + tl.arange(0, BLOCK_SIZE_N)
    select_idx = start_m + tl.arange(0, BLOCK_SIZE_M)
    Q += off_b * q_stride_b + off_qh * q_stride_h
    K += off_b * k_stride_b + off_kh * k_stride_h
    Lse += off_b * lse_stride_b + off_qh * lse_stride_h
    Ind += off_b * ind_stride_b + off_kh * ind_stride_h
    Val += off_b * ind_stride_b + off_kh * ind_stride_h

    compress_idx = (select_idx * select_size - kernel_size) // stride + 1
    select_start = 0
    select_end = select_size
    compress_start = stride - kernel_size 
    num_loops = (select_size + kernel_size - stride) // stride
    acc_p = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)

    for _ in range(num_loops):
        compress_end = compress_start + kernel_size
        area = tl.minimum(compress_end, select_end) - tl.maximum(compress_start, select_start)
        w = area / stride
        mask = (compress_idx >= 0) & (compress_idx < M)
        k = tl.load(K + compress_idx[None, :] * k_stride_m + tl.arange(0, D1)[:, None],
            mask=mask[None, :], other=0.)
        if D2>0:
            k2 = tl.load(K + compress_idx[None, :] * k_stride_m + tl.arange(0, D2)[:, None] + D1,
                mask=mask[None, :], other=0.)   
        for h_idx in range(nrep):
            q = tl.load(Q + h_idx * q_stride_h + off_n[:, None] * q_stride_n + tl.arange(0, D1)[None, :],
                        mask=off_n[:, None] < N, other=0.)
            lse = tl.load(Lse + h_idx * lse_stride_h + off_n * lse_stride_n, 
                        mask=off_n < N, other=0.)
            attn_score = tl.dot(q, k)
            if D2 > 0:
                q2 = tl.load(Q + h_idx * q_stride_h + off_n[:, None] * q_stride_n + tl.arange(0, D2)[None, :] + D1,
                        mask=off_n[:, None] < N, other=0.)
                attn_score = tl.dot(q2, k2, attn_score)
            k_idx = compress_idx * stride + kernel_size - 1
            attn_score = tl.where(off_n[:, None] >= k_idx[None, :], attn_score, float('-inf'))
            p = tl.exp(attn_score * sm_scale - lse[:, None])
            acc_p += p * w
        compress_idx += 1
        compress_start += stride
    if tl.program_id(2) == 0:
        tl.store(Ind + off_n * ind_stride_n, off_n // select_size, mask=off_n < N)
        tl.store(Val + off_n * ind_stride_n, 9999., mask=off_n < N)
    acc_p = tl.where(select_idx[None, :] == off_n[:, None] // select_size, -1., acc_p)
    start_k = tl.program_id(2) * (top_n - 1) + 1
    for _ in range(1, top_n):
        max_val, max_ind = tl.max(acc_p, -1, True)
        tl.store(Ind + off_n * ind_stride_n + start_k, max_ind, mask=off_n < N)
        tl.store(Val + off_n * ind_stride_n + start_k, max_val, mask=off_n < N)
        acc_p = tl.where(select_idx[None, :] == max_ind[:, None] , -1., acc_p)
        start_k += 1

@triton.autotune([triton.Config({'BLOCK_SIZE_M': bsm}, num_stages=ns, num_warps=nw)
                 for bsm in [32, 64, 128, 256]
                 for ns in [1, 2, 4]
                 for nw in [4, 8]
                 ], key=['N', "M"])
@triton.jit
def _fused_p3(Q, K, Lse, P,
                q_stride_b, q_stride_n, q_stride_h, q_stride_d,
                k_stride_b, k_stride_m, k_stride_h, k_stride_d,
                lse_stride_b, lse_stride_h, lse_stride_n,
                p_stride_b, p_stride_h, p_stride_n, p_stride_m,
                sm_scale, kernel_size, stride, select_size, num_select_blocks, top_n,
                B, N, M, KH, nrep,
                D1: tl.constexpr, D2: tl.constexpr, 
                BLOCK_SIZE_N: tl.constexpr=64, BLOCK_SIZE_M: tl.constexpr=64,):
    off_bh = tl.cast(tl.program_id(0), tl.int64)
    off_kh = off_bh % KH
    off_b = off_bh // KH
    off_qh = off_kh * nrep
    start_n = tl.cast(tl.program_id(1), tl.int64) * BLOCK_SIZE_N
    start_m = tl.cast(tl.program_id(2), tl.int64) * BLOCK_SIZE_M
    if ((start_n + BLOCK_SIZE_N - 1) // select_size) < start_m:
        return  
    off_n = start_n + tl.arange(0, BLOCK_SIZE_N)
    select_idx = start_m + tl.arange(0, BLOCK_SIZE_M)
    Q += off_b * q_stride_b + off_qh * q_stride_h
    K += off_b * k_stride_b + off_kh * k_stride_h
    Lse += off_b * lse_stride_b + off_qh * lse_stride_h
    P += off_b * p_stride_b + off_kh * p_stride_h

    compress_idx = (select_idx * select_size - kernel_size) // stride + 1
    select_start = 0
    select_end = select_size
    compress_start = stride - kernel_size 
    num_loops = (select_size + kernel_size - stride) // stride
    acc_p = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)

    for _ in range(num_loops):
        compress_end = compress_start + kernel_size
        area = tl.minimum(compress_end, select_end) - tl.maximum(compress_start, select_start)
        w = area / stride
        mask = (compress_idx >= 0) & (compress_idx < M)
        k = tl.load(K + compress_idx[None, :] * k_stride_m + tl.arange(0, D1)[:, None],
            mask=mask[None, :], other=0.)
        if D2>0:
            k2 = tl.load(K + compress_idx[None, :] * k_stride_m + tl.arange(0, D2)[:, None] + D1,
                mask=mask[None, :], other=0.)   
        for h_idx in range(nrep):
            q = tl.load(Q + h_idx * q_stride_h + off_n[:, None] * q_stride_n + tl.arange(0, D1)[None, :],
                        mask=off_n[:, None] < N, other=0.)
            lse = tl.load(Lse + h_idx * lse_stride_h + off_n * lse_stride_n, 
                        mask=off_n < N, other=0.)
            attn_score = tl.dot(q, k)
            if D2 > 0:
                q2 = tl.load(Q + h_idx * q_stride_h + off_n[:, None] * q_stride_n + tl.arange(0, D2)[None, :] + D1,
                        mask=off_n[:, None] < N, other=0.)
                attn_score = tl.dot(q2, k2, attn_score)
            k_idx = compress_idx * stride + kernel_size - 1
            attn_score = tl.where(off_n[:, None] >= k_idx[None, :], attn_score, float('-inf'))
            p = tl.exp(attn_score * sm_scale - lse[:, None])
            acc_p += p * w
        compress_idx += 1
        compress_start += stride
    tl.store(P + off_n[:, None] * p_stride_n + select_idx[None, :], acc_p, (off_n[:, None] < N) & (select_idx[None, :] < num_select_blocks))

@triton.autotune([triton.Config({'BLOCK_SIZE_M': bsm}, num_stages=ns, num_warps=nw)
                 for bsm in [32, 64, 128, 256]
                 for ns in [1, 2, 4]
                 for nw in [4, 8]
                 ], key=['N', "M"])
@triton.jit
def _fused_p4(Q, K, Lse, P,
                q_stride_b, q_stride_n, q_stride_h, q_stride_d,
                k_stride_b, k_stride_m, k_stride_h, k_stride_d,
                lse_stride_b, lse_stride_h, lse_stride_n,
                p_stride_b, p_stride_h, p_stride_n, p_stride_m,
                sm_scale, kernel_size, stride, select_size, num_select_blocks, top_n,
                B, N, M, KH, nrep,
                D1: tl.constexpr, D2: tl.constexpr, 
                BLOCK_SIZE_N: tl.constexpr=64, BLOCK_SIZE_M: tl.constexpr=64,):
    off_b = tl.cast(tl.program_id(0), tl.int64)
    off_qh = tl.cast(tl.program_id(1), tl.int64)
    off_kh = off_qh // nrep
    start_n = tl.cast(tl.program_id(2), tl.int64) * BLOCK_SIZE_N
    off_n = start_n + tl.arange(0, BLOCK_SIZE_N)

    Q += off_b * q_stride_b + off_qh * q_stride_h
    K += off_b * k_stride_b + off_kh * k_stride_h
    Lse += off_b * lse_stride_b + off_qh * lse_stride_h
    P += off_b * p_stride_b + off_qh * p_stride_h

    q = tl.load(Q + off_n[:, None] * q_stride_n + tl.arange(0, D1)[None, :],
                mask=off_n[:, None] < N, other=0.)
    lse = tl.load(Lse + off_n * lse_stride_n, 
                mask=off_n < N, other=0.)
    if D2 > 0:
        q2 = tl.load(Q + off_n[:, None] * q_stride_n + tl.arange(0, D2)[None, :] + D1,
                mask=off_n[:, None] < N, other=0.)
    num_loops = (select_size + kernel_size - stride) // stride
    for start_m in range(0, (start_n + BLOCK_SIZE_N - 1) // select_size + 1, BLOCK_SIZE_M):
        select_idx = start_m + tl.arange(0, BLOCK_SIZE_M)

        compress_idx = (select_idx * select_size - kernel_size) // stride + 1
        select_start = 0
        select_end = select_size
        compress_start = stride - kernel_size 
        acc_p = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)

        for _ in range(num_loops):
            compress_end = compress_start + kernel_size
            area = tl.minimum(compress_end, select_end) - tl.maximum(compress_start, select_start)
            w = area / stride
            mask = (compress_idx >= 0) & (compress_idx < M)
            k = tl.load(K + compress_idx[None, :] * k_stride_m + tl.arange(0, D1)[:, None],
                mask=mask[None, :], other=0.)
            attn_score = tl.dot(q, k)
            if D2 > 0:
                k2 = tl.load(K + compress_idx[None, :] * k_stride_m + tl.arange(0, D2)[:, None] + D1,
                    mask=mask[None, :], other=0.)
                attn_score = tl.dot(q2, k2, attn_score)
            k_idx = compress_idx * stride + kernel_size - 1
            attn_score = tl.where(off_n[:, None] >= k_idx[None, :], attn_score, float('-inf'))
            p = tl.exp(attn_score * sm_scale - lse[:, None])
            acc_p += p * w
        compress_idx += 1
        compress_start += stride
        tl.store(P + off_n[:, None] * p_stride_n + select_idx[None, :], acc_p, (off_n[:, None] < N) & (select_idx[None, :] < num_select_blocks))

def fused_p(q, k, lse, kernel_size, stride, select_size, top_n=3, sm_scale=None, method=1):
    assert top_n == 3
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
    nrep = QH // KH
    
    num_select_blocks = triton.cdiv(N, select_size)
    if method == 1:
        indices = torch.zeros(B, KH, N, top_n, device=q.device, dtype=torch.float16)
        grid = grid=lambda meta: (B * KH, triton.cdiv(N, meta['CHUNK_N']), meta['CHUNK_N'])
        kwargs = { "BLOCK_SIZE_M": 64, "num_warps": 4, "num_stages": 2}
        BLOCK_SIZE_H = triton.next_power_of_2(nrep)
        _fused_p[grid](q, k, lse, indices,
                            *q.stride(),
                            *k.stride(),
                            *lse.stride(),
                            *indices.stride(),
                            sm_scale, kernel_size, stride, select_size, num_select_blocks, top_n,
                            B, N, M, KH, nrep,
                            D1, D2,
                            BLOCK_SIZE_H,
                            # **kwargs
                            )
    elif method == 2:
        indices = torch.zeros(B, KH, N, 1 + 2 * triton.cdiv(num_select_blocks, 64), device=q.device, dtype=torch.int64)
        values = torch.zeros(B, KH, N, 1 + 2 * triton.cdiv(num_select_blocks, 64), device=q.device, dtype=torch.float16)
        grid = grid=lambda meta: (B * KH, triton.cdiv(N, meta['BLOCK_SIZE_N']), triton.cdiv(num_select_blocks, meta['BLOCK_SIZE_M']))
        kwargs = { "BLOCK_SIZE_M": 64, "num_warps": 4, "num_stages": 2}
        BLOCK_SIZE_N = select_size
        _fused_p2[grid](q, k, lse, values, indices,
                            *q.stride(),
                            *k.stride(),
                            *lse.stride(),
                            *indices.stride(),
                            sm_scale, kernel_size, stride, select_size, num_select_blocks, top_n,
                            B, N, M, KH, nrep,
                            D1, D2,
                            BLOCK_SIZE_N,
                            # **kwargs
                            )
    elif method == 3:
        # indices = torch.zeros(B, KH, N, 1 + 2 * triton.cdiv(num_select_blocks, 64), device=q.device, dtype=torch.int64)
        # values = torch.zeros(B, KH, N, 1 + 2 * triton.cdiv(num_select_blocks, 64), device=q.device, dtype=torch.float16)
        select_probs = torch.zeros(B, KH, N, num_select_blocks, device=q.device, dtype=torch.float16)
        grid = grid=lambda meta: (B * KH, triton.cdiv(N, meta['BLOCK_SIZE_N']), triton.cdiv(num_select_blocks, meta['BLOCK_SIZE_M']))
        kwargs = { "BLOCK_SIZE_M": 64, "num_warps": 4, "num_stages": 2}
        BLOCK_SIZE_N = select_size
        _fused_p3[grid](q, k, lse, select_probs,
                            *q.stride(),
                            *k.stride(),
                            *lse.stride(),
                            *select_probs.stride(),
                            sm_scale, kernel_size, stride, select_size, num_select_blocks, top_n,
                            B, N, M, KH, nrep,
                            D1, D2,
                            BLOCK_SIZE_N,
                            # **kwargs
                            )
    elif method == 4:
        # indices = torch.zeros(B, KH, N, 1 + 2 * triton.cdiv(num_select_blocks, 64), device=q.device, dtype=torch.int64)
        # values = torch.zeros(B, KH, N, 1 + 2 * triton.cdiv(num_select_blocks, 64), device=q.device, dtype=torch.float16)
        select_probs = torch.zeros(B, QH, N, num_select_blocks, device=q.device, dtype=torch.float16)
        grid = lambda meta: (B, QH, triton.cdiv(N, meta['BLOCK_SIZE_N']))
        kwargs = { "BLOCK_SIZE_M": 64, "num_warps": 4, "num_stages": 2}
        BLOCK_SIZE_N = select_size
        _fused_p4[grid](q, k, lse, select_probs,
                            *q.stride(),
                            *k.stride(),
                            *lse.stride(),
                            *select_probs.stride(),
                            sm_scale, kernel_size, stride, select_size, num_select_blocks, top_n,
                            B, N, M, KH, nrep,
                            D1, D2,
                            BLOCK_SIZE_N,
                            # **kwargs
                            )
        select_probs = select_probs.view(B, KH, -1, N, num_select_blocks).sum(2)
    return None

@triton.heuristics({
    'USE_OFFSETS': lambda args: args['offsets'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8, 16]
    ],
    key=['BS', 'BK', 'BV'],
)
@triton.jit
def parallel_nsa_fwd_kernel(
    q,
    k,
    v,
    o,
    lse,
    scale,
    block_indices,
    offsets,
    token_indices,
    T,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    S: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_OFFSETS: tl.constexpr
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    # tl.static_print(T, H, HQ, G, K, V, S, BS, BK, BV)
    if USE_OFFSETS:
        i_n, i_t = tl.load(token_indices + i_t * 2).to(tl.int32), tl.load(token_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    block_indices += (bos + i_t) * H*S + i_h * S

    p_q = tl.make_block_ptr(q + (bos + i_t) * HQ*K, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0))
    p_o = tl.make_block_ptr(o + (bos + i_t) * HQ*V, (HQ, V), (V, 1), (i_h * G, i_v * BV), (G, BV), (1, 0))
    p_lse = lse + (bos + i_t) * HQ + i_h * G + tl.arange(0, G)

    # the Q block is kept in the shared memory throughout the whole kernel
    # [G, BK]
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)
    # [G, BV]
    b_o = tl.zeros([G, BV], dtype=tl.float32)

    b_m = tl.full([G], float('-inf'), dtype=tl.float32)
    b_acc = tl.zeros([G], dtype=tl.float32)
    for i in range(S):
        i_s = tl.load(block_indices + i).to(tl.int32) * BS

        p_k = tl.make_block_ptr(k, (K, T), (1, H*K), (0, i_s), (BK, BS), (0, 1))
        p_v = tl.make_block_ptr(v, (T, V), (H*V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
        # [BK, BS]
        b_k = tl.load(p_k, boundary_check=(0, 1))
        # [BS, BV]
        b_v = tl.load(p_v, boundary_check=(0, 1))
        # [G, BS]
        b_s = tl.dot(b_q, b_k)
        b_s = tl.where((i_t >= (i_s + tl.arange(0, BS)))[None, :], b_s, float('-inf'))

        # [G]
        b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
        b_r = tl.exp(b_mp - b_m)
        # [G, BS]
        b_p = tl.exp(b_s - b_m[:, None])
        # [G]
        b_acc = b_acc * b_r + tl.sum(b_p, 1)
        # [G, BV]
        b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_q.dtype), b_v)

        b_mp = b_m
    b_o = b_o / b_acc[:, None]
    b_m += tl.log(b_acc)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_lse, b_m.to(p_lse.dtype.element_ty))

@triton.autotune([triton.Config({},num_warps=nw)
                 for nw in [1,2,4, 8]
                 ], key=['D1', 'D2'])
@triton.jit
def _fwd_kernel4(Q, K, V, O, LSE, Ind,
                q_stride_b, q_stride_n, q_stride_h, q_stride_d,
                k_stride_b, k_stride_m, k_stride_h, k_stride_d,
                v_stride_b, v_stride_m, v_stride_h, v_stride_d,
                o_stride_b, o_stride_n, o_stride_h, o_stride_d,
                lse_stride_b, lse_stride_h, lse_stride_n,
                ind_stride_b, ind_stride_h, ind_stride_n, ind_stride_k,
                sm_scale, top_n,
                B, N, M, QH, KH, nrep,
                D1: tl.constexpr, D2: tl.constexpr, VD: tl.constexpr, 
                BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_M: tl.constexpr=128,
                CHUNK_N: tl.constexpr=64):
    
    off_n = tl.program_id(0) * CHUNK_N + tl.program_id(1)
    if off_n >= N:
        return
    off_bh = tl.program_id(2)
    off_kh = off_bh % KH
    off_b = off_bh // KH

    off_qh = off_kh * nrep

    D = D1 + D2

    Q += off_b * q_stride_b + off_n * q_stride_n
    K += off_b * k_stride_b + off_kh * k_stride_h
    V += off_b * v_stride_b + off_kh * v_stride_h
    O += off_b * o_stride_b + off_n * o_stride_n
    
    Ind += off_b * ind_stride_b + off_kh * ind_stride_h + off_n * ind_stride_n
    q_ptrs = tl.make_block_ptr(Q,
                               (QH, D), 
                               (q_stride_h, 1), 
                               (off_qh, 0),
                               (BLOCK_SIZE_H, D1), 
                               (1,0))
    # O + off_b * o_stride_b + off_n * o_stride_n,
    o_ptrs = tl.make_block_ptr(O,
                               (QH, VD), 
                               (o_stride_h,1), 
                               (off_qh, 0),
                               (BLOCK_SIZE_H, VD), 
                               (1,0))
    q = tl.load(q_ptrs, boundary_check=(0,1))
    if D2 > 0:
        q_ptrs2 = tl.make_block_ptr(Q,
                                (QH, D), 
                                (D, 1), 
                                (off_kh * BLOCK_SIZE_H, D1),
                                (BLOCK_SIZE_H, D2), 
                                (1,0))
        q2 = tl.load(q_ptrs2, boundary_check=(0,1))

    lse_ptrs = LSE + off_b * lse_stride_b + (off_qh + tl.arange(0, BLOCK_SIZE_H)) * lse_stride_h + off_n
    m_i = tl.zeros([BLOCK_SIZE_H], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_SIZE_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_SIZE_H, VD], dtype=tl.float32)

    stop_n = tl.minimum(top_n, tl.cdiv(off_n+1, BLOCK_SIZE_M))
    for i in range(0, stop_n):
        start_m = tl.load(Ind + i).to(tl.int32) * BLOCK_SIZE_M
        k_ptrs = tl.make_block_ptr(K, (D, M), (1, k_stride_m), (0, start_m), (D1, BLOCK_SIZE_M), (0,1))
        v_ptrs = tl.make_block_ptr(V, (M, VD), (v_stride_m , 1), (start_m, 0), (BLOCK_SIZE_M, VD), (1, 0))
        k = tl.load(k_ptrs, boundary_check=(0, 1))
        v = tl.load(v_ptrs, boundary_check=(0, 1))
        attn_score = tl.dot(q, k)
        if D2>0:
            k_ptrs2 = tl.make_block_ptr(K, (D, M), (1, k_stride_m), (D1, start_m), (D2, BLOCK_SIZE_M), (0,1))
            k2 = tl.load(k_ptrs2, boundary_check=(0, 1))
            attn_score = tl.dot(q2, k2, attn_score)
        attn_score *= sm_scale

        attn_score = tl.where(off_n >= (start_m + tl.arange(0, BLOCK_SIZE_M))[None, :], attn_score, float('-inf'))

        m_ij = tl.max(attn_score, axis=1)
        new_m_i = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - new_m_i)

        exp_attn_score = tl.exp(attn_score - new_m_i[:, None])

        l_i = l_i * alpha + tl.sum(exp_attn_score, axis=-1)
        acc = acc * alpha[:, None]

        acc = tl.dot(exp_attn_score.to(v.dtype), v, acc=acc)
        m_i = new_m_i

        i += 1

    acc /= l_i[:, None]
    tl.store(o_ptrs, acc.to(o_ptrs.dtype.element_ty), boundary_check=(0,1))

    lse = m_i + tl.log(l_i)
    tl.store(lse_ptrs, lse, mask=tl.arange(0, BLOCK_SIZE_H) < nrep)

@triton.autotune([triton.Config({}, num_warps=nw)
                #  for ns in [1, 2, 4]
                 for nw in [1, 2, 4, 8, 16]
                 ], key=['D1', "D2", 'VD', 'BLOCK_SIZE_H', 'BLOCK_SIZE_M'])
@triton.jit
def _fwd_kernel1(Q, K, V, O, LSE, Ind,
                sm_scale, top_n:tl.constexpr,
                N, M, KH: tl.constexpr, QH: tl.constexpr, nrep:tl.constexpr,
                D:tl.constexpr, D1: tl.constexpr, D2: tl.constexpr, VD: tl.constexpr, 
                BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_M: tl.constexpr=64,
                CHUNK_N: tl.constexpr=64):

    # off_bh = tl.cast(tl.program_id(2), tl.int64)
    off_bh, off_n = tl.program_id(1), tl.program_id(0)
    off_kh = off_bh % KH
    off_b = off_bh // KH

    bos = off_b * N

    K += (bos * KH + off_kh) * D
    V += (bos * KH + off_kh) * VD
    # Ind += (bos * KH + off_n + off_kh * N) * top_n
    # lse_ptrs = LSE + (off_b * QH + off_qh + tl.arange(0, BLOCK_SIZE_H)) * N + off_n

    q_ptrs = tl.make_block_ptr(Q + (bos + off_n) * QH * D, 
                               (QH, D), 
                               (D, 1), 
                               (off_kh * BLOCK_SIZE_H, 0),
                               (BLOCK_SIZE_H, D1), 
                               (1,0))
    # O + off_b * o_stride_b + off_n * o_stride_n,
    o_ptrs = tl.make_block_ptr(O + (bos + off_n) * QH * VD,
                               (QH, VD), 
                               (VD,1), 
                               (off_kh * BLOCK_SIZE_H, 0),
                               (BLOCK_SIZE_H, VD), 
                               (1,0))
    q = tl.load(q_ptrs, boundary_check=(0,1))
    q = (q * sm_scale).to(q.dtype)
    # if D2 > 0:
    #     q2 = tl.load(Q + rows[:, None] * q_stride_h + tl.arange(0, D2)[None, :] + D1, mask=rows[:, None] < nrep, other=0.)

    m_i = tl.full([BLOCK_SIZE_H], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_SIZE_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_SIZE_H, VD], dtype=tl.float32)

    stop_n = tl.minimum(top_n, tl.cdiv(off_n+1, BLOCK_SIZE_M))
    for i in range(0, stop_n):
        start_m = tl.load(Ind + i).to(tl.int32) * BLOCK_SIZE_M
        k_ptrs = tl.make_block_ptr(K, (D, M), (1, KH * D), (0, start_m), (D1, BLOCK_SIZE_M), (0,1))
        v_ptrs = tl.make_block_ptr(V, (M, VD), (KH * VD , 1), (start_m, 0), (BLOCK_SIZE_M, VD), (1, 0))
        k = tl.load(k_ptrs, boundary_check=(0, 1))
        v = tl.load(v_ptrs, boundary_check=(0, 1))
        # k = tl.load(K + k_idx[None, :] * k_stride_m + tl.arange(0, D1)[:, None], mask=k_idx[None, :] < M, other=0.)
        attn_score = tl.dot(q, k)
        # if D2>0:
        #     k2 = tl.load(K + k_idx[None, :] * k_stride_m + tl.arange(0, D2)[:, None] + D1, mask=k_idx[None, :] < M, other=0.)
        #     attn_score = tl.dot(q2, k2, attn_score)
        # attn_score *= sm_scale

        attn_score = tl.where(off_n >= (start_m + tl.arange(0, BLOCK_SIZE_M))[None, :], attn_score, float('-inf'))

        new_m_i = tl.maximum(m_i, tl.max(attn_score, axis=1))
        alpha = tl.exp(m_i - new_m_i)

        exp_attn_score = tl.exp(attn_score - new_m_i[:, None])

        l_i = l_i * alpha + tl.sum(exp_attn_score, axis=-1)

        # v = tl.load(V + k_idx[:, None] * v_stride_m + tl.arange(0, VD)[None, :], mask=k_idx[:, None] < M, other=0.)
        acc = acc * alpha[:, None] + tl.dot(exp_attn_score.to(v.dtype), v)
        m_i = new_m_i


    acc /= l_i[:, None]
    # tl.store(O + rows[:, None] * o_stride_h + tl.arange(0, VD)[None, :], acc, mask=rows[:, None] < nrep)
    tl.store(o_ptrs, acc.to(o_ptrs.dtype.element_ty), boundary_check=(0,1))
    # lse = m_i + tl.log(l_i)
    # tl.store(lse_ptrs, lse, mask=tl.arange(0, BLOCK_SIZE_H) < nrep)

# @triton.autotune([triton.Config({}, num_stages=ns, num_warps=nw)
#                  for ns in [1, 2, 4]
#                  for nw in [4, 8]
#                  ], key=['N', "M"])
@triton.jit
def _fwd_kernel2(Q, K, V, O, LSE, Ind,
                q_stride_b, q_stride_n, q_stride_h, q_stride_d,
                k_stride_b, k_stride_m, k_stride_h, k_stride_d,
                v_stride_b, v_stride_m, v_stride_h, v_stride_d,
                o_stride_b, o_stride_n, o_stride_h, o_stride_d,
                lse_stride_b, lse_stride_h, lse_stride_n,
                ind_stride_b, ind_stride_h, ind_stride_n, ind_stride_k,
                sm_scale, top_n,
                B, N, M, QH: tl.constexpr, KH: tl.constexpr,
                D1: tl.constexpr, D2: tl.constexpr, VD: tl.constexpr, 
                BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_M: tl.constexpr=128,
                CHUNK_N: tl.constexpr=64):

    off_b = tl.cast(tl.program_id(0), tl.int64)
    off_n = tl.cast(tl.program_id(1), tl.int64) * CHUNK_N + tl.program_id(2)
    if off_n >= N:
        return

    Q += off_b * q_stride_b + off_n * q_stride_n
    K += off_b * k_stride_b
    V += off_b * v_stride_b
    O += off_b * o_stride_b + off_n * o_stride_n
    LSE += off_b * lse_stride_b + off_n
    Ind += off_b * ind_stride_b + off_n * ind_stride_n

    nrep = QH // KH
    q = tl.load(Q + tl.arange(0, QH)[:, None] * q_stride_h + tl.arange(0, D1)[None, :]).reshape(KH, BLOCK_SIZE_H, D1)
    if D2 > 0:
        q2 = tl.load(Q + tl.arange(0, QH)[:, None] * q_stride_h + tl.arange(0, D2)[None, :] + D1).reshape(KH, BLOCK_SIZE_H, D2)

    m_i = tl.zeros([KH, BLOCK_SIZE_H], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([KH, BLOCK_SIZE_H], dtype=tl.float32)
    acc = tl.zeros([KH, BLOCK_SIZE_H, VD], dtype=tl.float32)

    i = 0
    stop_n = tl.minimum(top_n, tl.cdiv(off_n+1, BLOCK_SIZE_M))

    ind_ptrs = Ind + tl.arange(0, KH) * ind_stride_h
    while i < stop_n:
        select_idx = tl.load(ind_ptrs + i).to(tl.int64)
        off_m = select_idx[:, None] * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)[None, :]
        k = tl.load(K + tl.arange(0, KH)[:, None, None] * k_stride_h + 
                     off_m[:, None, :] * k_stride_m + 
                     tl.arange(0, D1)[None, :, None], 
                     mask=off_m[:, None, :] < M, other=0.)
        attn_score = tl.dot(q, k)
        if D2>0:
            k2 = tl.load(K + tl.arange(0, KH)[:, None, None] * k_stride_h + 
                     off_m[:, None, :] * k_stride_m + 
                     tl.arange(0, D2)[None, :, None] + D1, 
                     mask=off_m[:, None, :] < M, other=0.)
            attn_score = tl.dot(q2, k2, attn_score)
        attn_score *= sm_scale

        attn_score = tl.where(off_n >= off_m[:, None, :], attn_score, float('-inf'))

        m_ij = tl.max(attn_score, axis=-1)
        new_m_i = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - new_m_i)

        exp_attn_score = tl.exp(attn_score - new_m_i[:, :, None])

        l_i = l_i * alpha + tl.sum(exp_attn_score, axis=-1)
        acc = acc * alpha[:, :, None]

        v = tl.load(V + tl.arange(0, KH)[:, None, None] * v_stride_h + 
                        off_m[:, :, None] * v_stride_m + tl.arange(0, VD)[None, None, :], 
                        mask=off_m[:, :, None] < M, other=0.)
        acc = tl.dot(exp_attn_score.to(v.dtype), v, acc=acc)
        m_i = new_m_i

        i += 1
    acc /= l_i[:, :, None]
    tl.store(O + tl.arange(0, QH)[:, None] * o_stride_h + tl.arange(0, VD)[None, :], tl.reshape(acc, QH, VD))

    lse = m_i + tl.log(l_i)
    tl.store(LSE + tl.arange(0, QH) * lse_stride_h, tl.reshape(lse, QH))

@triton.autotune([triton.Config({'step':step}, num_stages=ns, num_warps=nw)
                 for step in [2]
                 for ns in [1,2, 4]
                 for nw in [1,2,4,8]
                 ], key=['N', "M"])
@triton.jit
def _fwd_kernel3(Q, K, V, O, LSE, Ind,
                q_stride_b, q_stride_n, q_stride_h, q_stride_d,
                k_stride_b, k_stride_m, k_stride_h, k_stride_d,
                v_stride_b, v_stride_m, v_stride_h, v_stride_d,
                o_stride_b, o_stride_n, o_stride_h, o_stride_d,
                lse_stride_b, lse_stride_h, lse_stride_n,
                ind_stride_b, ind_stride_h, ind_stride_n, ind_stride_k,
                sm_scale, top_n,num_select_blocks,
                B, N, M, KH, QH, nrep,
                D1: tl.constexpr, D2: tl.constexpr, VD: tl.constexpr,  
                BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_M: tl.constexpr,
                step:tl.constexpr=4, CHUNK_N:tl.constexpr=64):
    
    off_bh = tl.program_id(2)
    off_kh = off_bh % KH
    off_b = off_bh // KH
    off_n = tl.program_id(0) * CHUNK_N + tl.program_id(1)
    if off_n >= N:
        return
    off_qh = off_kh * nrep

    bos = off_b * N
    D = D1 + D2


    K += off_b * k_stride_b + off_kh * k_stride_h
    V += off_b * v_stride_b + off_kh * v_stride_h
    Ind += off_b * ind_stride_b + off_kh * ind_stride_h + off_n * ind_stride_n
    lse_ptrs = LSE + off_b * lse_stride_b + (off_qh + tl.arange(0, BLOCK_SIZE_H)) * lse_stride_h + off_n

    q_ptrs = tl.make_block_ptr(Q + (bos + off_n) * QH * D, 
                               (QH, D), 
                               (D, 1), 
                               (off_kh * BLOCK_SIZE_H, 0),
                               (BLOCK_SIZE_H, D1), 
                               (1,0))
    # O + off_b * o_stride_b + off_n * o_stride_n,
    o_ptrs = tl.make_block_ptr(O + (bos + off_n) * QH * VD,
                               (QH, VD), 
                               (VD,1), 
                               (off_kh * BLOCK_SIZE_H, 0),
                               (BLOCK_SIZE_H, VD), 
                               (1,0))
    
    q = tl.load(q_ptrs, boundary_check=(0,1))
    if D2 > 0:
        q2 = tl.load(Q + tl.arange(0, BLOCK_SIZE_H)[:, None] * q_stride_h + tl.arange(0, D2)[None, :] + D1, 
                    mask=tl.arange(0, BLOCK_SIZE_H)[:, None] < nrep, other=0.)

    m_i = tl.zeros([BLOCK_SIZE_H], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_SIZE_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_SIZE_H, VD], dtype=tl.float32)

    stop_n = tl.minimum(top_n, tl.cdiv(off_n+1, BLOCK_SIZE_M))
    
    for start_k in range(0, stop_n, step):
        off_k = tl.arange(0, step) + start_k
        start_m = tl.load(Ind + off_k, mask=off_k < stop_n, other=0) * BLOCK_SIZE_M
        off_m = start_m[:, None] + tl.arange(0, BLOCK_SIZE_M)[None, :]
        off_m = tl.where(off_k[:, None]<stop_n, off_m, M)
        off_m = tl.reshape(off_m, step * BLOCK_SIZE_M)
        k = tl.load(K + off_m[None, :] * k_stride_m + tl.arange(0, D1)[:, None], 
                    mask=off_m[None, :] < M, other=0.)
        v = tl.load(V + off_m[:, None] * v_stride_m + tl.arange(0, VD)[None, :], 
                    mask=off_m[:, None] < M, other=0.)

        # k_ptrs = tl.make_block_ptr(K, (D1, M), (1, k_stride_m),(0, 0),(D1, BLOCK_SIZE_M), (0,1))
        # v_ptrs = tl.make_block_ptr(V, (M, VD), (v_stride_m, 1),(0, 0),(BLOCK_SIZE_M, D), (0,1))
        attn_score = tl.dot(q, k)
        if D2>0:
            k2 = tl.load(K + off_m[None, :] * k_stride_m + tl.arange(0, D2)[:, None] + D1, 
                    mask=off_m[None, :] < M, other=0.)
            attn_score = tl.dot(q2, k2, attn_score)
        attn_score *= sm_scale

        attn_score = tl.where(off_n >= off_m[None, :], attn_score, float('-inf'))

        m_ij = tl.max(attn_score, axis=1)
        new_m_i = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - new_m_i)

        exp_attn_score = tl.exp(attn_score - new_m_i[:, None])

        l_i = l_i * alpha + tl.sum(exp_attn_score, axis=-1)
        acc = acc * alpha[:, None]


        acc = tl.dot(exp_attn_score.to(v.dtype), v, acc=acc)
        m_i = new_m_i

    acc /= l_i[:, None]
    tl.store(o_ptrs, acc.to(o_ptrs.dtype.element_ty), boundary_check=(0,1))
    tl.store(lse_ptrs, m_i + tl.log(l_i), mask=tl.arange(0, BLOCK_SIZE_H) < nrep)


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
def _bwd_kernel1(DQ, DK, DV, DO, 
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
    off_qh = tl.cast(off_kh * nrep, tl.int64)

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


# @triton.autotune([triton.Config({'BLOCK_SIZE_N': bsn, 'BLOCK_SIZE_M': bsm}, num_stages=ns, num_warps=nw)
#                  for bsm in [32, 64, 128]
#                  for bsn in [32, 64, 128]
#                  for ns in [1, 2, 4]
#                  for nw in [4, 8]
#                  ], key=['N', "M"])
@triton.jit
def _bwd_kernel3(DQ, DK, DV, DO, 
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
                BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_M: tl.constexpr,
                step: tl.constexpr=2,
                ):
    pid0 = tl.cast(tl.program_id(0), tl.int64) 
    off_m = tl.cast(pid0 * BLOCK_SIZE_M, tl.int64) + tl.arange(0, BLOCK_SIZE_M)
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

    heads = tl.arange(0, BLOCK_SIZE_H)

    k = tl.load(K + off_m[None, :] * k_stride_m + tl.arange(0, D1)[:, None], mask=off_m[None, :] < M, other=0.)
    v = tl.load(V + off_m[None, :] * v_stride_m + tl.arange(0, VD)[:, None], mask=off_m[None, :] < M, other=0.)
    acc_dk = tl.zeros((BLOCK_SIZE_M, D1), dtype=tl.float32)
    acc_dv = tl.zeros((BLOCK_SIZE_M, VD), dtype=tl.float32)

    if D2 > 0:
        k2 = tl.load(K + off_m[None, :] * k_stride_m + tl.arange(0, D2)[:, None] + D1, mask=off_m[None, :] < M, other=0.)
        acc_dk2 = tl.zeros((BLOCK_SIZE_M, D2), dtype=tl.float32)
    zeros = tl.zeros((BLOCK_SIZE_H, ), dtype=tl.int64)
    count = tl.load(Count + pid0)
    for idx in range(0, count, step):
        off_k = idx + tl.arange(0, step)
        q_idx = tl.load(Ind + off_k, mask=off_k<count, other=N).to(tl.int64)
        # q_idx = tl.ravel(q_idx[:, None] + heads[None, :])
        q = tl.load(Q + q_idx[:, None, None] * q_stride_n + heads[None, :, None] * q_stride_h + tl.arange(0, D1)[None, None, :], 
                    mask=(q_idx[:, None, None]<N) & (heads[None, :, None] < nrep), other=0.)
        q = tl.reshape(q, step * BLOCK_SIZE_H, D1)
        
        do = tl.load(Q + q_idx[:, None, None] * do_stride_n + heads[None, :, None] * do_stride_h + tl.arange(0, VD)[None, None, :], 
                    mask=(q_idx[:, None, None]<N) & (heads[None, :, None] < nrep), other=0.)
        do = tl.reshape(do, step * BLOCK_SIZE_H, VD)
        lse = tl.load(Lse + q_idx[:, None] * lse_stride_n + heads[None, :] * lse_stride_h, 
                    mask=(q_idx[:, None]<N) & (heads[None, :] < nrep), other=0.)
        lse = tl.reshape(lse, step * BLOCK_SIZE_H)
        delta = tl.load(Delta + q_idx[:, None] * lse_stride_n + heads[None, :] * lse_stride_h, 
                    mask=(q_idx[:, None]<N) & (heads[None, :] < nrep), other=0.)
        delta = tl.reshape(delta, step * BLOCK_SIZE_H)

        attn_score = tl.dot(q, k) 

        if D2 > 0:
            q2 = tl.load(Q + q_idx[:, None, None] * q_stride_n + heads[None, :, None] * q_stride_h + tl.arange(0, D2)[None, None, :] + D1, 
                        mask=(q_idx[:, None, None]<N) & (heads[None, :, None] < nrep), other=0.)
            q2 = tl.reshape(q2, step * BLOCK_SIZE_H, D2)
            attn_score = tl.dot(q2, k2, attn_score)
        rows = tl.ravel(q_idx[:, None] + zeros[None, :])
        attn_score = tl.where(rows[:, None] >= off_m[None, :], attn_score, float('-inf'))
        p = tl.exp(attn_score * sm_scale - lse[:, None])
        
        acc_dv = tl.dot(tl.permute(p, 1, 0).to(do.dtype), do, acc_dv)

        dp = tl.dot(do, v)
        ds = p * (dp - delta[:, None]) * sm_scale

        # dq = tl.load(dq_ptrs, mask=row_mask[:, None], other=0., eviction_policy="evict_last")
        
        dq = tl.dot(ds.to(k.dtype), tl.trans(k, 1, 0))
        
        dq_ptrs = DQ + tl.ravel(q_idx[:, None] * q_stride_n + heads[None, :] * q_stride_h)[:, None] + tl.arange(0, D1)[None, :]
        tl.static_print(dq_ptrs.shape, dq.shape, rows.shape)
        tl.atomic_add(dq_ptrs,
                      dq, mask=rows[:, None]<N)
    #     acc_dk = tl.dot(tl.permute(ds, 1, 0).to(q.dtype), q, acc_dk)
    #     if D2 > 0:
    #         dq2 = tl.dot(ds.to(k.dtype), tl.trans(k2, 1, 0))
    #         tl.atomic_add(dq_ptrs[:, None] + tl.arange(0, D2)[None, :] + D1,
    #                     dq2, mask=rows[:, None]<N)
    #         acc_dk2 = tl.dot(tl.trans(ds, 1, 0).to(q.dtype), q2, acc_dk2)
    
    # tl.store(DK + off_m[:, None] * k_stride_m + tl.arange(0, D1)[None, :], acc_dk, mask=off_m[:, None] < M)
    # tl.store(DV + off_m[:, None] * v_stride_m + tl.arange(0, VD)[None, :], acc_dv, mask=off_m[:, None] < M)
    # if D2 > 0:
    #     tl.store(DK + off_m[:, None] * k_stride_m + tl.arange(0, D2)[None, :] + D1, acc_dk2, mask=off_m[:, None] < M)

# @triton.autotune([triton.Config({'BLOCK_SIZE_N': bsn, 'BLOCK_SIZE_M': bsm}, num_stages=ns, num_warps=nw)
#                  for bsm in [32, 64, 128]
#                  for bsn in [32, 64, 128]
#                  for ns in [1, 2, 4]
#                  for nw in [4, 8]
#                  ], key=['N', "M"])
@triton.jit
def _bwd_kernel2(DQ, DK, DV, DO, 
                Q, K, V, 
                Lse, Delta, Ind,
                q_stride_b, q_stride_n, q_stride_h, q_stride_d,
                k_stride_b, k_stride_m, k_stride_h, k_stride_d,
                v_stride_b, v_stride_m, v_stride_h, v_stride_d,
                do_stride_b, do_stride_n, do_stride_h, do_stride_d,
                lse_stride_b, lse_stride_h, lse_stride_n,
                ind_stride_b, ind_stride_h, ind_stride_n, ind_stride_k,
                sm_scale, top_n,
                B, N, M, KH, nrep,  
                D1: tl.constexpr, D2: tl.constexpr, VD: tl.constexpr, 
                BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_M: tl.constexpr,
                CHUNK_N:tl.constexpr=1024,
                ):
    off_n = tl.cast(tl.program_id(1), tl.int64) * CHUNK_N + tl.program_id(0)
    if off_n >= N:
        return
    off_bh = tl.cast(tl.program_id(2), tl.int64)
    off_kh = off_bh % KH
    off_b = off_bh // KH
    off_qh = off_kh * nrep


    Q += off_b * q_stride_b + off_qh * q_stride_h + off_n * q_stride_n
    K += off_b * k_stride_b + off_kh * k_stride_h
    V += off_b * v_stride_b + off_kh * v_stride_h
    DQ += off_b * q_stride_b + off_qh * q_stride_h + off_n * q_stride_n
    DK += off_b * k_stride_b + off_kh * k_stride_h
    DV += off_b * v_stride_b + off_kh * v_stride_h
    DO += off_b * do_stride_b + off_qh * do_stride_h + off_n * do_stride_n
    Lse += off_b * lse_stride_b + off_qh * lse_stride_h + off_n
    Delta += off_b * lse_stride_b + off_qh * lse_stride_h + off_n
    Ind += off_b * ind_stride_b + off_kh * ind_stride_h + off_n * ind_stride_n

    rows = tl.arange(0, BLOCK_SIZE_H)

    rows = tl.arange(0, BLOCK_SIZE_H)
    q = tl.load(Q + rows[:, None] * q_stride_h + tl.arange(0, D1)[None, :], mask=rows[:, None] < nrep, other=0.)
    if D2 > 0:
        q2 = tl.load(Q + rows[:, None] * q_stride_h + tl.arange(0, D2)[None, :] + D1, mask=rows[:, None] < nrep, other=0.)
    lse = tl.load(Lse + rows*lse_stride_h, mask=rows<nrep)
    delta = tl.load(Delta + rows*lse_stride_h, mask=rows<nrep)
    do = tl.load(DO + rows[:, None] * do_stride_h + tl.arange(0, VD)[None, :], mask=rows[:, None] < nrep, other=0.)

    acc_dq = tl.zeros((BLOCK_SIZE_H, D1), dtype=tl.float32)
    if D2 > 0:
        acc_dq2 = tl.zeros((BLOCK_SIZE_H, D2), dtype=tl.float32)

    i = 0
    stop_n = tl.minimum(top_n, tl.cdiv(off_n+1, BLOCK_SIZE_M))
    while i < stop_n:
        select_idx = tl.load(Ind + i)
        start_m = select_idx * BLOCK_SIZE_M
        k_idx = start_m + tl.arange(0, BLOCK_SIZE_M)
        k = tl.load(K + k_idx[None, :] * k_stride_m + tl.arange(0, D1)[:, None], mask=k_idx[None, :] < M, other=0.)
        attn_score = tl.dot(q, k)
        if D2>0:
            k2 = tl.load(K + k_idx[None, :] * k_stride_m + tl.arange(0, D2)[:, None] + D1, mask=k_idx[None, :] < M, other=0.)
            attn_score = tl.dot(q2, k2, attn_score)

        attn_score = tl.where(off_n >= k_idx[None, :], attn_score, float('-inf'))
        p = tl.exp(attn_score * sm_scale - lse[:, None])
        dv = tl.dot(tl.permute(p, 1, 0).to(do.dtype), do)
        tl.atomic_add(DV + k_idx[:, None] * v_stride_m + tl.arange(0, VD)[None, :], dv, mask=k_idx[:, None] < M)
        v = tl.load(V + k_idx[None, :] * v_stride_m + tl.arange(0, VD)[:, None], mask=k_idx[None, :] < M)
        dp = tl.dot(do, v)
        ds = p * (dp - delta[:, None]) * sm_scale

        acc_dq = tl.dot(ds.to(k.dtype), tl.trans(k, 1, 0), acc_dq)
        dk = tl.dot(tl.permute(ds, 1, 0).to(q.dtype), q)
        # tl.atomic_add(DK + k_idx[:, None] * k_stride_m + tl.arange(0, D1)[None, :], dk, mask=k_idx[:, None] < M)
        if D2 > 0:
            acc_dq2 = tl.dot(ds.to(k.dtype), tl.trans(k2, 1, 0), acc_dq2)
            dk2 = tl.dot(tl.trans(ds, 1, 0).to(q.dtype), q2)
            tl.atomic_add(DK + k_idx[:, None] * k_stride_m + tl.arange(0, D2)[None, :] + D1, dk2, mask=k_idx[:, None] < M)
        i += 1

    tl.store(DQ + rows[:, None] * q_stride_h + tl.arange(0, D1)[None, :], acc_dq, mask=rows[:, None] < nrep)
    if D2 > 0:
        tl.store(DQ + rows[:, None] * q_stride_h + tl.arange(0, D2)[None, :] + D1, acc_dq2, mask=rows[:, None] < nrep)



class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, select_size, select_indices, sm_scale, method):
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
        top_n = select_indices.size(-1)
        num_select_blocks = select_indices[0,0,0,-1]
        if method == 4:
            grid = lambda meta: (triton.cdiv(N, meta['CHUNK_N']), meta['CHUNK_N'], B*KH)
            _fwd_kernel4[grid](q, k, v, o, lse, select_indices,
                            *q.stride(),
                            *k.stride(),
                            *v.stride(),
                            *o.stride(),
                            *lse.stride(),
                            *select_indices.stride(),
                            sm_scale, top_n,
                            B, N, M, QH, KH, nrep,
                            D1, D2, VD,
                            BLOCK_SIZE_H, BLOCK_SIZE_M,
                            #   **kwargs
                            )

        if method == 1:
            kwargs = {"num_warps": 1, "num_stages": 2}
            # grid = lambda meta: (triton.cdiv(N, meta['CHUNK_N']), meta['CHUNK_N'], B*KH)
            _fwd_kernel1[(N, B * KH)](q, k, v, o, lse, select_indices,
                            sm_scale, top_n,
                            N, M, KH, QH, nrep, 
                            D, D1, D2, VD,
                            BLOCK_SIZE_H, BLOCK_SIZE_M,
                            # num_warps=1
                            # **kwargs
                            )
            # lse = torch.empty(B, N, QH, dtype=torch.float32, device=q.device,)
            # parallel_nsa_fwd_kernel[(1, N, B * KH)](q, k, v, o, lse, sm_scale, select_indices, None, None,
            #                                     N, KH, QH, BLOCK_SIZE_H, D, VD, top_n, BLOCK_SIZE_M,D, VD
            #                                     )
        if method == 3:
            kwargs = {"num_warps": 4, "num_stages": 1 if D == 192 else 2, 'step':2}
            grid = lambda meta: (triton.cdiv(N, meta['CHUNK_N']), meta['CHUNK_N'], B*KH)
            _fwd_kernel3[grid](q, k, v, o, lse, select_indices,
                            *q.stride(),
                            *k.stride(),
                            *v.stride(),
                            *o.stride(),
                            *lse.stride(),
                            *select_indices.stride(),
                            sm_scale, top_n,num_select_blocks,
                            B, N, M, KH, QH, nrep, 
                            D1, D2, VD,
                            BLOCK_SIZE_H, BLOCK_SIZE_M,
                            #   **kwargs
                            )

        ctx.save_for_backward(q, k, v, o, lse)
        ctx.select_indices = select_indices
        ctx.method = method
        ctx.infos = (B, N, M, QH, KH, D1, D2, VD, sm_scale, nrep, BLOCK_SIZE_H, BLOCK_SIZE_M)
        return o

    @staticmethod
    def backward(ctx, do, *args):
        assert do.is_contiguous()
        
        B, N, M, QH, KH, D1, D2, VD, sm_scale, nrep, BLOCK_SIZE_H, BLOCK_SIZE_M = ctx.infos
        q, k, v, o, lse = ctx.saved_tensors


        delta = torch.empty_like(lse)
        grid = lambda meta: (B, QH, triton.cdiv(N, meta["BLOCK_SIZE_N"]))
        kwargs = {"BLOCK_SIZE_N": 64, "num_warps": 8, "num_stages": 2}
        _bwd_preprocess[grid](o, do, delta,
                              *o.stride(), 
                              *delta.stride(),
                              N, VD,
                              **kwargs
                              )
        
        if ctx.method == 1:
            dq = torch.zeros_like(q, dtype=torch.float32)
            dk = torch.zeros_like(k)
            dv = torch.zeros_like(v)
            bwd_ind, count = select_for_bwd(ctx.select_indices)
            # grid = (B, KH, triton.cdiv(M, BLOCK_SIZE_M))
            grid = (triton.cdiv(M, BLOCK_SIZE_M), B, KH)
            _bwd_kernel1[grid](dq, dk, dv, do, 
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
        if ctx.method == 3:
            dq = torch.zeros_like(q, dtype=torch.float32)
            dk = torch.zeros_like(k)
            dv = torch.zeros_like(v)
            bwd_ind, count = select_for_bwd(ctx.select_indices)
            # grid = (B, KH, triton.cdiv(M, BLOCK_SIZE_M))
            grid = (triton.cdiv(M, BLOCK_SIZE_M), B, KH)
            _bwd_kernel3[grid](dq, dk, dv, do, 
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
                            BLOCK_SIZE_H, BLOCK_SIZE_M
                            )
            dq = dq.to(q.dtype)
        if ctx.method == 3:
            dq = torch.zeros_like(q)
            dk = torch.zeros_like(k, dtype=torch.float32)
            dv = torch.zeros_like(v, dtype=torch.float32)
            # grid = (B, KH, triton.cdiv(M, BLOCK_SIZE_M))
            grid = lambda meta: (meta['CHUNK_N'], triton.cdiv(N, meta['CHUNK_N']), B*KH)
            _bwd_kernel2[grid](dq, dk, dv, do, 
                            q, k, v,
                            lse, delta, 
                            ctx.select_indices, 
                            *q.stride(), # dq和q一样
                            *k.stride(),
                            *v.stride(),
                            *do.stride(), # do和o一样
                            *lse.stride(), # lse和delta一样
                            *ctx.select_indices.stride(),
                            sm_scale, ctx.select_indices.size(-1),
                            B, N, M, KH, nrep,  
                            D1, D2, VD,
                            BLOCK_SIZE_H, BLOCK_SIZE_M,
                            )
            dk = dk.to(q.dtype)
            dv = dv.to(q.dtype)
        return dq, dk, dv, None, None, None, None


def select_attn(q, k, v, select_size, select_indices, sm_scale=None):
    return _attention.apply(q, k, v, select_size, select_indices, sm_scale, 1)


# @triton.autotune([triton.Config({'BLOCK_SIZE_N': bsn, 'BLOCK_SIZE_M': bsm}, num_stages=ns, num_warps=nw)
#                  for bsm in [32, 64, 128]
#                  for bsn in [32, 64, 128]
#                  for ns in [1, 2, 4]
#                  for nw in [4, 8]
#                  ], key=['N', "M"])
@triton.jit
def _ca_fwd_kernel1(Q, K, V, O, LSE, 
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


# @triton.autotune([triton.Config({'BLOCK_SIZE_N': bsn, 'BLOCK_SIZE_M': bsm}, num_stages=ns, num_warps=nw)
#                  for bsm in [32, 64]
#                  for bsn in [64, 128]
#                  for ns in [1, 2, 3, 4, 5]
#                  for nw in [2, 4, 8]
#                  ], key=['N', "M"])
@triton.jit
def _ca_fwd_kernel2(Q, K, V, O, LSE, 
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
    off_qh = tl.cast(tl.program_id(2), tl.int64)
    start_n = tl.cast(tl.program_id(1), tl.int64) * BLOCK_SIZE_N
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

    block_idx = tl.arange(0, BLOCK_SIZE_M)
    for start_kv_idx in range(kernel_size-1, start_n + BLOCK_SIZE_N, BLOCK_SIZE_M * stride):
        k = tl.load(K + block_idx[None, :] * k_stride_m + tl.arange(0, D1)[:, None], mask=block_idx[None, :] < M, other=0.)
        attn_score = tl.dot(q, k)
        if D2>0:
            k2 = tl.load(K + block_idx[None, :] * k_stride_m + tl.arange(0, D2)[:, None] + D1, mask=block_idx[None, :] < M, other=0.)
            attn_score = tl.dot(q2, k2, attn_score)

        k_idx = block_idx * stride + kernel_size - 1
        attn_score = tl.where(off_n[:, None] >= k_idx[None, :], attn_score * sm_scale, float('-inf'))

        m_ij = tl.max(attn_score, axis=1)
        new_m_i = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - new_m_i)

        exp_attn_score = tl.exp(attn_score - new_m_i[:, None])

        l_i = l_i * alpha + tl.sum(exp_attn_score, axis=-1)
        acc = acc * alpha[:, None]

        v = tl.load(V + block_idx[:, None] * v_stride_m + tl.arange(0, VD)[None, :], mask=block_idx[:, None] < M, other=0.)
        acc = tl.dot(exp_attn_score.to(v.dtype), v, acc=acc)

        m_i = new_m_i
        block_idx += BLOCK_SIZE_M

    acc /= l_i[:, None]
    lse = m_i + tl.log(l_i)
    if  tl.program_id(1) == 0:
        acc = tl.where(off_n[:, None]>=(kernel_size-1), acc, 0)
        lse = tl.where(off_n>=(kernel_size-1), lse, 0)
    tl.store(O + off_n[:, None] * o_stride_n + tl.arange(0, VD)[None, :], acc, mask=off_n[:, None] < N)   
    tl.store(LSE + off_n * lse_stride_n, lse, mask=off_n < N)


# @triton.autotune([triton.Config({'BLOCK_SIZE_N': bsn}, num_stages=ns, num_warps=nw)
#                  for bsn in [16, 32, 64, 128]
#                  for ns in [1, 2, 4]
#                  for nw in [4, 8]
#                  ], key=['N'])
@triton.jit
def _ca_bwd_preprocess(O,DO,Delta,
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
def _ca_bwd_kernel(DQ, DK, DV, DO, 
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


# @triton.autotune([triton.Config({'BLOCK_SIZE_N': bsn, 'BLOCK_SIZE_M': bsm}, num_stages=ns, num_warps=nw)
#                  for bsm in [16, 32, 64]
#                  for bsn in [64, 128, 256]
#                  for ns in [1, 2]
#                  for nw in [8]
#                  ], key=['N', "M"])
@triton.jit
def _ca_bwd_kernel2(DQ, DK, DV, DO, 
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
    k_idx = off_m * stride + kernel_size - 1
    for start_q_idx in range(start_m * stride + kernel_size - 1, N, BLOCK_SIZE_N):
        off_n = start_q_idx + tl.arange(0, BLOCK_SIZE_N)
        q = tl.load(Q + off_n[:, None] * q_stride_n + tl.arange(0, D1), mask=off_n[:, None] < N, other=0.)
        do = tl.load(DO + off_n[:, None] * do_stride_n + tl.arange(0, VD), mask=off_n[:, None] < N, other=0.)
        lse = tl.load(Lse + off_n, mask=off_n < N, other=0.)
        delta = tl.load(Delta + off_n, mask=off_n < N, other=0.)

        attn_score = tl.dot(q, k) 

        if D2 > 0:
            q2 = tl.load(Q + off_n[:, None] * q_stride_n + tl.arange(0, D2) + D1, mask=off_n[:, None] < N, other=0.)
            attn_score = tl.dot(q2, k2, attn_score)

        attn_score = tl.where(off_n[:, None] >= k_idx[None, :], attn_score, float('-inf'))
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


class _cattention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, kernel_size, stride, sm_scale, method=1):
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
        if method == 1:
            grid = lambda meta: (B, QH, triton.cdiv(N, meta['BLOCK_SIZE_N']))
            kwargs = {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_M": 32, "num_warps": 4, "num_stages": 3}
            # kwargs = {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_M": 64, "num_warps": 8, "num_stages": 1}
            _ca_fwd_kernel1[grid](q, k, v, o, lse,
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
        if method == 2:
            grid = lambda meta: (B, triton.cdiv(N, meta['BLOCK_SIZE_N']), QH)
            # grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE_N']), B, QH)
            kwargs = {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_M": 32, "num_warps": 4, "num_stages": 3}
            # kwargs = {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_M": 64, "num_warps": 8, "num_stages": 1}
            _ca_fwd_kernel2[grid](q, k, v, o, lse,
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
        ctx.method = method
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
    
        grid = lambda meta: (B, QH, triton.cdiv(N, meta["BLOCK_SIZE_N"]))
        kwargs = {"BLOCK_SIZE_N": 64, "num_warps": 8, "num_stages": 1}
        _ca_bwd_preprocess[grid](o, do, delta,
                              *o.stride(), 
                              *delta.stride(),
                              N, VD,
                              **kwargs
                              )
        if ctx.method == 1:
            grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE_M"]), B, QH)
            kwargs = {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_M": 64, "num_warps": 8, "num_stages": 2}
            _ca_bwd_kernel[grid](dq, dk, dv, do, 
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
        if ctx.method == 2:
            grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE_M"]), B, QH)
            kwargs = {"BLOCK_SIZE_N": 128, "BLOCK_SIZE_M": 64, "num_warps": 8, "num_stages": 2}
            _ca_bwd_kernel2[grid](dq, dk, dv, do, 
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
        return dq, dk, dv, None, None, None, None


