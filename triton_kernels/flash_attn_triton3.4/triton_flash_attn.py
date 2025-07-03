import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

import math
def split_d(D):
    if math.log2(D).is_integer():
        D1 = D
        D2 = 0
    else:
        D1 = triton.next_power_of_2(D) // 2
        D2 = D - D1
        assert math.log2(D2).is_integer()
    return D1, D2


# @triton.autotune(configs=[triton.Config({"BLOCK_M": BM, "BLOCK_N":BN}, num_stages=ns, num_warps=nw)
#                           for BM in [64, 128]
#                           for BN in [64, 128]
#                           for ns in [1,2, 3, 4]
#                           for nw in [4, 8]], key=["D1", "D2", "VD"])
@triton.jit
def _attn_fwd(Q, 
              K,
              V, 
              O, 
              CU_SEQLENS, 
              LSE,
                q_stride_n, q_stride_h, q_stride_d,
                k_stride_n, k_stride_h, k_stride_d,
                v_stride_n, v_stride_h, v_stride_d,
                o_stride_n, o_stride_h, o_stride_d,
              sm_scale,
              WINDOW: tl.constexpr,
              window_size,
              CAUSAL: tl.constexpr,
              T: tl.constexpr,
              KH: tl.constexpr,
              G: tl.constexpr,
              D1: tl.constexpr,
              D2: tl.constexpr,
              VD: tl.constexpr,
              BLOCK_M: tl.constexpr,
              BLOCK_N: tl.constexpr,
              ):
    start_m = tl.program_id(0) * BLOCK_M
    start_m = (tl.num_programs(0) - tl.program_id(0) - 1) * BLOCK_M
    off_b = tl.program_id(1)
    off_qh = tl.program_id(2)
    off_kh = off_qh // G
    bos, eos = tl.load(CU_SEQLENS + off_b), tl.load(CU_SEQLENS + off_b + 1)
    S = eos - bos
    if start_m >= S:
        return
    
    desc_q = tl.make_tensor_descriptor(Q + bos * q_stride_n + off_qh * q_stride_h, (S, D1), (q_stride_n, q_stride_d), (BLOCK_M, D1))
    desc_k = tl.make_tensor_descriptor(K + bos * k_stride_n + off_kh * k_stride_h, (S, D1), (k_stride_n, k_stride_d), (BLOCK_N, D1))
    desc_v = tl.make_tensor_descriptor(V + bos * v_stride_n + off_kh * v_stride_h, (S, VD), (v_stride_n, v_stride_d), (BLOCK_N, VD))
    desc_o = tl.make_tensor_descriptor(O + bos * o_stride_n + off_qh * o_stride_h, (S, VD), (o_stride_n, o_stride_d), (BLOCK_M, VD))
    if D2 > 0:
        desc_q2 = tl.make_tensor_descriptor(Q + bos * q_stride_n + off_qh * q_stride_h + D1, (S, D2), (q_stride_n, q_stride_d), (BLOCK_M, D2))
        desc_k2 = tl.make_tensor_descriptor(K + bos * k_stride_n + off_kh * k_stride_h + D1, (S, D2), (k_stride_n, k_stride_d), (BLOCK_N, D2))
        
    q_idx = start_m + tl.arange(0, BLOCK_M)
    q = desc_q.load([start_m, 0])
    if D2 > 0:
        q2 = desc_q2.load([start_m, 0])
    
    sm_scale *= 1.44269504
    acc = tl.zeros((BLOCK_M, VD), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    m_i = tl.zeros((BLOCK_M,), dtype=tl.float32) - float('inf')

    if not WINDOW:
        start = 0
    else:
        start = tl.maximum(start_m - window_size, 0)
        start = (start // BLOCK_N) * BLOCK_N
        
    if CAUSAL:
        mid = (start_m // BLOCK_N) * BLOCK_N
    else:
        mid = S
        
    for start_n in tl.range(start, mid, BLOCK_N):
        k = desc_k.load([start_n, 0])
        v = desc_v.load([start_n, 0])
        qk = tl.dot(q, tl.permute(k, 1, 0))
        if D2 > 0:
            k2 = desc_k2.load([start_n, 0])
            qk = tl.dot(q2, tl.permute(k2, 1, 0), qk)
        if not WINDOW:
            qk *= sm_scale
        else:
            k_idx = start_n + tl.arange(0, BLOCK_N)
            mask = ((q_idx[:, None] - window_size) <= k_idx[None, :]) & (q_idx[:, None] >= k_idx[None, :])
            qk = tl.where(mask, qk * sm_scale, float('-inf'))
        # qk = tl.where(q_idx[:, None] >= k_idx[None, :], qk * sm_scale, float('-inf'))
        new_m_i = tl.maximum(tl.max(qk, -1), m_i)
        # alpha = tl.exp(m_i - new_m_i)
        # score = tl.exp(qk - new_m_i[:, None])
        alpha = tl.exp2(m_i - new_m_i)
        score = tl.exp2(qk - new_m_i[:, None])
        # l_i = l_i * alpha + tl.sum(score, -1)
        l_i = tl.fma(l_i, alpha, tl.sum(score, -1))
        acc = acc *  alpha[:, None] + tl.dot(score.to(v.dtype), v)
        m_i = new_m_i

    for start_n in tl.range(mid, start_m+BLOCK_M, BLOCK_N):
        k_idx = start_n + tl.arange(0, BLOCK_N)
        k = desc_k.load([start_n, 0])
        v = desc_v.load([start_n, 0])
        qk = tl.dot(q, tl.permute(k, 1, 0))
        if D2 > 0:
            k2 = desc_k2.load([start_n, 0])
            qk = tl.dot(q2, tl.permute(k2, 1, 0), qk)
        if not WINDOW:
            qk = tl.where(q_idx[:, None] >= k_idx[None, :], qk * sm_scale, float('-inf'))
        else:
            k_idx = start_n + tl.arange(0, BLOCK_N)
            mask = ((q_idx[:, None] - window_size) <= k_idx[None, :]) & (q_idx[:, None] >= k_idx[None, :])
            qk = tl.where(mask, qk * sm_scale, float('-inf'))
        new_m_i = tl.maximum(tl.max(qk, -1), m_i)
        # alpha = tl.exp(m_i - new_m_i)
        # score = tl.exp(qk - new_m_i[:, None])
        alpha = tl.exp2(m_i - new_m_i)
        score = tl.exp2(qk - new_m_i[:, None])
        # l_i = l_i * alpha + tl.sum(score, -1)
        l_i = tl.fma(l_i, alpha, tl.sum(score, -1))
        acc = acc *  alpha[:, None] + tl.dot(score.to(v.dtype), v)
        m_i = new_m_i

    acc /= l_i[:, None]
    desc_o.store([start_m, 0], acc.to(desc_o.dtype))
    m_i += tl.log2(l_i)
    tl.store(LSE + off_qh * T + bos + q_idx, m_i, q_idx<S)
    

# @triton.autotune(configs=[triton.Config({"BLOCK_M":BM}, num_stages=ns, num_warps=nw, pre_hook=_bwd_pre_host_descriptor_pre_hook)
#                           for BM in [32, 64, 128]
#                           for ns in [1,2,3, 4]
#                           for nw in [4, 8]], key=["VD"])
@triton.jit
def _bwd_preprocess_tma(O, 
                    DO, 
                    Delta,
                    o_stride_n, o_stride_h, o_stride_d,
                    delta_stride_h, delta_stride_n,
                    T:tl.constexpr, 
                    VD: tl.constexpr,
                    BLOCK_M: tl.constexpr=16
                    ):
    start_n = tl.program_id(0) * BLOCK_M
    off_h = tl.program_id(1)
    
    desc_o = tl.make_tensor_descriptor(O + off_h * o_stride_h, (T, VD), (o_stride_n, o_stride_d), (BLOCK_M, VD))
    desc_do = tl.make_tensor_descriptor(DO + off_h * o_stride_h, (T, VD), (o_stride_n, o_stride_d), (BLOCK_M, VD))

    
    
    o = desc_o.load([start_n, 0]).to(tl.float32)
    do = desc_o.load([start_n, 0]).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    tl.store(Delta + start_n + tl.arange(0, BLOCK_M) + off_h * T, delta, mask=(start_n + tl.arange(0, BLOCK_M)) < T)
    
@triton.jit
def _bwd_preprocess(O, 
                    DO, 
                    Delta,
                    o_stride_n, o_stride_h, o_stride_d,
                    T:tl.constexpr, 
                    VD: tl.constexpr,
                    BLOCK_M: tl.constexpr=16
                    ):
    off_n = tl.program_id(0).cast(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_h = tl.program_id(1).cast(tl.int64)

    O += off_h * o_stride_h
    DO += off_h * o_stride_h
    
    cols = tl.arange(0, VD)
    o = tl.load(O + off_n[:, None] * o_stride_n + cols[None, :], mask=off_n[:, None] < T, other=0.).to(tl.float32)
    do = tl.load(DO + off_n[:, None] * o_stride_n + cols[None, :], mask=off_n[:, None] < T, other=0.).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    tl.store(Delta + off_n + off_h * T , delta, mask=off_n < T)


# @triton.autotune(configs=[triton.Config({"BLOCK_M": BM, "BLOCK_N":BN}, num_stages=ns, num_warps=nw)
#                           for BM in [32, 64, 128]
#                           for BN in [32, 64, 128]
#                           for ns in [3, 4]
#                           for nw in [4, 8]], key=["D1", "D2", "VD"])
@triton.jit
def _dkdv_kernel(Q, 
            K,
            V,
            DK,
            DV,
            DO, 
            CU_SEQLENS, 
            Lse,
            Delta,
            q_stride_n, q_stride_h, q_stride_d,
            k_stride_n, k_stride_h, k_stride_d,
            v_stride_n, v_stride_h, v_stride_d,
            dk_stride_n, dk_stride_h, dk_stride_d,
            dv_stride_n, dv_stride_h, dv_stride_d,
            do_stride_n, do_stride_h, do_stride_d,
            sm_scale,
            T: tl.constexpr,
            G: tl.constexpr,
            D1: tl.constexpr,
            D2: tl.constexpr,
            VD: tl.constexpr,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
            ):
    start_n = tl.program_id(0) * BLOCK_N
    off_b = tl.program_id(1)
    off_qh = tl.program_id(2)
    off_kh = off_qh // G
    bos, eos = tl.load(CU_SEQLENS + off_b), tl.load(CU_SEQLENS + off_b + 1)
    S = eos - bos
    if start_n >= S:
        return
    
    
    desc_q = tl.make_tensor_descriptor(Q + bos * q_stride_n + off_qh * q_stride_h, (S, D1), (q_stride_n, q_stride_d), (BLOCK_M, D1))
    desc_k = tl.make_tensor_descriptor(K + bos * k_stride_n + off_kh * k_stride_h, (S, D1), (k_stride_n, k_stride_d), (BLOCK_N, D1))
    desc_v = tl.make_tensor_descriptor(V + bos * v_stride_n + off_kh * v_stride_h, (S, VD), (v_stride_n, v_stride_d), (BLOCK_N, VD))
    desc_dk = tl.make_tensor_descriptor(DK + bos * dk_stride_n + off_qh * dk_stride_h, (S, D1), (dk_stride_n, dk_stride_d), (BLOCK_N, D1))
    desc_dv = tl.make_tensor_descriptor(DV + bos * dv_stride_n + off_qh * dv_stride_h, (S, VD), (dv_stride_n, dv_stride_d), (BLOCK_N, VD))
    desc_do = tl.make_tensor_descriptor(DO + bos * do_stride_n + off_qh * do_stride_h, (S, VD), (do_stride_n, do_stride_d), (BLOCK_M, VD))
    if D2 > 0:
        desc_q2 = tl.make_tensor_descriptor(Q + bos * q_stride_n + off_qh * q_stride_h + D1, (S, D2), (q_stride_n, q_stride_d), (BLOCK_M, D2))
        desc_k2 = tl.make_tensor_descriptor(K + bos * k_stride_n + off_kh * k_stride_h + D1, (S, D2), (k_stride_n, k_stride_d), (BLOCK_N, D2))
        desc_dk2 = tl.make_tensor_descriptor(DK + bos * dk_stride_n + off_qh * dk_stride_h + D1, (S, D2), (dk_stride_n, dk_stride_d), (BLOCK_N, D2))
        
    ln2: tl.constexpr = 1.44269504
    sm_scale_ln2 = sm_scale * ln2
    
    k_idx = start_n + tl.arange(0, BLOCK_N)
    k = desc_k.load([start_n, 0])
    v = desc_v.load([start_n, 0])
    dv = tl.zeros((BLOCK_N, VD), dtype=tl.float32)
    dk = tl.zeros((BLOCK_N, D1), dtype=tl.float32)
    if D2 > 0:
        k2 = desc_k2.load([start_n, 0])
        dk2 = tl.zeros((BLOCK_N, D2), dtype=tl.float32)

    start = start_n
    mid = start + tl.cdiv(BLOCK_N, BLOCK_M) * BLOCK_M
    for start_m in range(start, mid, BLOCK_M):
        q_idx = start_m + tl.arange(0, BLOCK_M)
        q = desc_q.load([start_m, 0])
        do = desc_do.load([start_m, 0])
        lse = tl.load(Lse + (bos + q_idx) + off_qh * T, q_idx<S)
        delta = tl.load(Delta + (bos + q_idx) + off_qh * T, q_idx<S)
        qk = tl.dot(q, tl.permute(k, 1, 0))
        if D2 > 0:
            q2 = desc_q2.load([start_m, 0])
            qk = tl.dot(q2, tl.permute(k2, 1, 0), qk)
        mask = (q_idx[:, None] >= k_idx[None, :])
        qk = tl.where(mask, qk * sm_scale_ln2, float('-inf'))
        p = tl.exp2(qk - lse[:, None])
        dv = tl.dot(tl.permute(p, 1, 0).to(do.dtype), do, dv)
        dp = tl.dot(do, tl.permute(v, 1, 0))
        ds = p * (dp - delta[:, None])
        dk = tl.dot(tl.permute(ds, 1, 0).to(q.dtype), q, dk)
        if D2 > 0:
            dk2 = tl.dot(tl.permute(ds, 1, 0).to(q.dtype), q2, dk2)

    for start_m in range(mid, S, BLOCK_M):
        q_idx = start_m + tl.arange(0, BLOCK_M)
        q = desc_q.load([start_m, 0])
        do = desc_do.load([start_m, 0])
        lse = tl.load(Lse + (bos + q_idx) + off_qh * T, q_idx<S)
        delta = tl.load(Delta + (bos + q_idx) + off_qh * T, q_idx<S)
        qk = tl.dot(q, tl.permute(k, 1, 0))
        if D2 > 0:
            q2 = desc_q2.load([start_m, 0])
            qk = tl.dot(q2, tl.permute(k2, 1, 0), qk)
        p = tl.exp2(qk * sm_scale_ln2 - lse[:, None])
        dv = tl.dot(tl.permute(p, 1, 0).to(do.dtype), do, dv)
        dp = tl.dot(do, tl.permute(v, 1, 0))
        ds = p * (dp - delta[:, None])
        dk = tl.dot(tl.permute(ds, 1, 0).to(q.dtype), q, dk)
        if D2 > 0:
            dk2 = tl.dot(tl.permute(ds, 1, 0).to(q.dtype), q2, dk2)
            
    desc_dv.store([start_n, 0], dv.to(desc_dv.dtype))
    desc_dk.store([start_n, 0], (dk * sm_scale).to(desc_dk.dtype))
    if D2 > 0:
        desc_dk2.store([start_n, 0], (dk2 * sm_scale).to(desc_dk.dtype))
        
        
# @triton.autotune(configs=[triton.Config({"BLOCK_M": BM, "BLOCK_N":BN}, num_stages=ns, num_warps=nw)
#                           for BM in [32, 64, 128]
#                           for BN in [32, 64, 128]
#                           for ns in [3, 4]
#                           for nw in [4, 8]], key=["D1", "D2", "VD"])
@triton.jit
def _dkdv_kernel2(Q, 
            K,
            V,
            DK,
            DV,
            DO, 
            CU_SEQLENS, 
            Lse,
            Delta,
            q_stride_n, q_stride_h, q_stride_d,
            k_stride_n, k_stride_h, k_stride_d,
            v_stride_n, v_stride_h, v_stride_d,
            dk_stride_n, dk_stride_h, dk_stride_d,
            dv_stride_n, dv_stride_h, dv_stride_d,
            do_stride_n, do_stride_h, do_stride_d,
            sm_scale,
            CAUSAL: tl.constexpr,
            T: tl.constexpr,
            G: tl.constexpr,
            D1: tl.constexpr,
            D2: tl.constexpr,
            VD: tl.constexpr,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
            ):
    start_n = tl.program_id(0) * BLOCK_N
    off_b = tl.program_id(1)
    off_kh = tl.program_id(2)
    
    bos, eos = tl.load(CU_SEQLENS + off_b), tl.load(CU_SEQLENS + off_b + 1)
    S = eos - bos
    if start_n >= S:
        return
    
    ln2: tl.constexpr = 1.44269504
    sm_scale_ln2 = sm_scale * ln2
    
    desc_k = tl.make_tensor_descriptor(K + bos * k_stride_n + off_kh * k_stride_h, (S, D1), (k_stride_n, k_stride_d), (BLOCK_N, D1))
    desc_v = tl.make_tensor_descriptor(V + bos * v_stride_n + off_kh * v_stride_h, (S, VD), (v_stride_n, v_stride_d), (BLOCK_N, VD))
    desc_dk = tl.make_tensor_descriptor(DK + bos * dk_stride_n + off_kh * dk_stride_h, (S, D1), (dk_stride_n, dk_stride_d), (BLOCK_N, D1))
    desc_dv = tl.make_tensor_descriptor(DV + bos * dv_stride_n + off_kh * dv_stride_h, (S, VD), (dv_stride_n, dv_stride_d), (BLOCK_N, VD))
    if D2 > 0:
        desc_k2 = tl.make_tensor_descriptor(K + bos * k_stride_n + off_kh * k_stride_h + D1, (S, D2), (k_stride_n, k_stride_d), (BLOCK_N, D2))
        desc_dk2 = tl.make_tensor_descriptor(DK + bos * dk_stride_n + off_kh * dk_stride_h + D1, (S, D2), (dk_stride_n, dk_stride_d), (BLOCK_N, D2))
        
    k_idx = start_n + tl.arange(0, BLOCK_N)
    k = desc_k.load([start_n, 0])
    v = desc_v.load([start_n, 0])
    dv = tl.zeros((BLOCK_N, VD), dtype=tl.float32)
    dk = tl.zeros((BLOCK_N, D1), dtype=tl.float32)
    if D2 > 0:
        k2 = desc_k2.load([start_n, 0])
        dk2 = tl.zeros((BLOCK_N, D2), dtype=tl.float32)
        
    for off_qh in range(off_kh * G, off_kh * G + G):
    
        desc_q = tl.make_tensor_descriptor(Q + bos * q_stride_n + off_qh * q_stride_h, (S, D1), (q_stride_n, q_stride_d), (BLOCK_M, D1))
        desc_do = tl.make_tensor_descriptor(DO + bos * do_stride_n + off_qh * do_stride_h, (S, VD), (do_stride_n, do_stride_d), (BLOCK_M, VD))
        if D2 > 0:
            desc_q2 = tl.make_tensor_descriptor(Q + bos * q_stride_n + off_qh * q_stride_h + D1, (S, D2), (q_stride_n, q_stride_d), (BLOCK_M, D2))

        if CAUSAL:
            start = start_n
            mid = start + tl.cdiv(BLOCK_N, BLOCK_M) * BLOCK_M
        else:
            start = 0
            mid = 0
        for start_m in range(start, mid, BLOCK_M):
            q_idx = start_m + tl.arange(0, BLOCK_M)
            q = desc_q.load([start_m, 0])
            do = desc_do.load([start_m, 0])
            lse = tl.load(Lse + (bos + q_idx) + off_qh * T, q_idx<S)
            delta = tl.load(Delta + (bos + q_idx) + off_qh * T, q_idx<S)
            qk = tl.dot(q, tl.permute(k, 1, 0))
            if D2 > 0:
                q2 = desc_q2.load([start_m, 0])
                qk = tl.dot(q2, tl.permute(k2, 1, 0), qk)
            mask = (q_idx[:, None] >= k_idx[None, :])
            qk = tl.where(mask, qk * sm_scale_ln2, float('-inf'))
            p = tl.exp2(qk - lse[:, None])
            dv = tl.dot(tl.permute(p, 1, 0).to(do.dtype), do, dv)
            dp = tl.dot(do, tl.permute(v, 1, 0))
            ds = p * (dp - delta[:, None])
            dk = tl.dot(tl.permute(ds, 1, 0).to(q.dtype), q, dk)
            if D2 > 0:
                dk2 = tl.dot(tl.permute(ds, 1, 0).to(q.dtype), q2, dk2)

        for start_m in range(mid, S, BLOCK_M):
            q_idx = start_m + tl.arange(0, BLOCK_M)
            q = desc_q.load([start_m, 0])
            do = desc_do.load([start_m, 0])
            lse = tl.load(Lse + (bos + q_idx) + off_qh * T, q_idx<S)
            delta = tl.load(Delta + (bos + q_idx) + off_qh * T, q_idx<S)
            qk = tl.dot(q, tl.permute(k, 1, 0))
            if D2 > 0:
                q2 = desc_q2.load([start_m, 0])
                qk = tl.dot(q2, tl.permute(k2, 1, 0), qk)
            p = tl.exp2(qk * sm_scale_ln2 - lse[:, None])
            dv = tl.dot(tl.permute(p, 1, 0).to(do.dtype), do, dv)
            dp = tl.dot(do, tl.permute(v, 1, 0))
            ds = p * (dp - delta[:, None])
            dk = tl.dot(tl.permute(ds, 1, 0).to(q.dtype), q, dk)
            if D2 > 0:
                dk2 = tl.dot(tl.permute(ds, 1, 0).to(q.dtype), q2, dk2)
            
    desc_dv.store([start_n, 0], dv.to(desc_dv.dtype))
    desc_dk.store([start_n, 0], (dk * sm_scale).to(desc_dk.dtype))
    if D2 > 0:
        desc_dk2.store([start_n, 0], (dk2 * sm_scale).to(desc_dk.dtype))




# @triton.autotune(configs=[triton.Config({"BLOCK_M": BM, "BLOCK_N":BN}, num_stages=ns, num_warps=nw)
#                           for BM in [32, 64, 128]
#                           for BN in [32, 64, 128]
#                           for ns in [3, 4]
#                           for nw in [4, 8]], key=["D1", "D2", "VD"])
@triton.jit
def _dq_kernel(Q,
               K,
               V,
               DQ,
               DO, 
                CU_SEQLENS, 
                Lse,
                Delta,
                q_stride_n, q_stride_h, q_stride_d,
                k_stride_n, k_stride_h, k_stride_d,
                v_stride_n, v_stride_h, v_stride_d,
                dq_stride_n, dq_stride_h, dq_stride_d,
                do_stride_n, do_stride_h, do_stride_d,
                sm_scale,
                CAUSAL: tl.constexpr,
                T: tl.constexpr,
                G: tl.constexpr,
                D1: tl.constexpr,
                D2: tl.constexpr,
                VD: tl.constexpr,
                BLOCK_M: tl.constexpr,
                BLOCK_N: tl.constexpr,
            ):
    start_m = tl.program_id(0) * BLOCK_M
    off_b = tl.program_id(1)
    off_qh = tl.program_id(2)
    off_kh = off_qh // G
    bos, eos = tl.load(CU_SEQLENS + off_b), tl.load(CU_SEQLENS + off_b + 1)
    S = eos - bos
    if start_m >= S:
        return
    
    desc_q = tl.make_tensor_descriptor(Q + bos * q_stride_n + off_qh * q_stride_h, (S, D1), (q_stride_n, q_stride_d), (BLOCK_M, D1))
    desc_k = tl.make_tensor_descriptor(K + bos * k_stride_n + off_kh * k_stride_h, (S, D1), (k_stride_n, k_stride_d), (BLOCK_N, D1))
    desc_v = tl.make_tensor_descriptor(V + bos * v_stride_n + off_kh * v_stride_h, (S, VD), (v_stride_n, v_stride_d), (BLOCK_N, VD))
    desc_dq = tl.make_tensor_descriptor(DQ + bos * dq_stride_n + off_qh * dq_stride_h, (S, D1), (dq_stride_n, dq_stride_d), (BLOCK_M, D1))
    desc_do = tl.make_tensor_descriptor(DO + bos * do_stride_n + off_qh * do_stride_h, (S, VD), (do_stride_n, do_stride_d), (BLOCK_M, VD))
    if D2 > 0:
        desc_q2 = tl.make_tensor_descriptor(Q + bos * q_stride_n + off_qh * q_stride_h + D1, (S, D2), (q_stride_n, q_stride_d), (BLOCK_M, D2))
        desc_dq2 = tl.make_tensor_descriptor(DQ + bos * dq_stride_n + off_qh * dq_stride_h + D1, (S, D1), (dq_stride_n, dq_stride_d), (BLOCK_M, D2))
        desc_k2 = tl.make_tensor_descriptor(K + bos * k_stride_n + off_kh * k_stride_h + D1, (S, D2), (k_stride_n, k_stride_d), (BLOCK_N, D2))
        
    
    ln2: tl.constexpr = 1.44269504
    sm_scale_ln2 = sm_scale * ln2

    q_idx = start_m + tl.arange(0, BLOCK_M)
    q = desc_q.load([start_m, 0])
    do = desc_do.load([start_m, 0])
    lse = tl.load(Lse + (bos + q_idx) + off_qh * T, q_idx<S)
    delta = tl.load(Delta + (bos + q_idx) + off_qh * T, q_idx<S)
    dq = tl.zeros((BLOCK_M, D1), dtype=tl.float32)
    if D2 > 0:
        q2 = desc_q2.load([start_m, 0])
        dq2 = tl.zeros((BLOCK_M, D2), dtype=tl.float32)
    
    start = 0
    if CAUSAL:
        mid = (start_m // BLOCK_N) * BLOCK_N
    else:
        mid = S
    for start_n in tl.range(start, mid, BLOCK_N):
        k_idx = start_n + tl.arange(0, BLOCK_N)
        k = desc_k.load([start_n, 0])
        v = desc_v.load([start_n, 0])
        qk = tl.dot(q, tl.permute(k, 1, 0))
        if D2 > 0:
            k2 = desc_k2.load([start_n, 0])
            qk = tl.dot(q2, tl.permute(k2, 1, 0), qk)
        p = tl.exp2(qk * sm_scale_ln2 - lse[:, None])
        dp = tl.dot(do, tl.permute(v, 1, 0))
        ds = (p * (dp - delta[:, None])).to(k.dtype)
        dq = tl.dot(ds, k, dq)
        if D2 > 0:
            dq2 = tl.dot(ds, k2, dq2)

    for start_n in tl.range(mid, start_m+BLOCK_M, BLOCK_N):
        k_idx = start_n + tl.arange(0, BLOCK_N)
        k = desc_k.load([start_n, 0])
        v = desc_v.load([start_n, 0])
        qk = tl.dot(q, tl.permute(k, 1, 0))
        if D2 > 0:
            k2 = desc_k2.load([start_n, 0])
            qk = tl.dot(q2, tl.permute(k2, 1, 0), qk)
        qk = tl.where(q_idx[:, None] >= k_idx[None, :], qk * sm_scale_ln2, float('-inf'))
        p = tl.exp2(qk - lse[:, None])
        dp = tl.dot(do, tl.permute(v, 1, 0))
        ds = (p * (dp - delta[:, None])).to(k.dtype)
        dq = tl.dot(ds, k, dq)
        if D2 > 0:
            dq2 = tl.dot(ds, k2, dq2)

    desc_dq.store([start_m, 0], (dq * sm_scale).to(desc_dq.dtype))
    if D2 > 0:
        desc_dq2.store([start_m, 0], (dq2 * sm_scale).to(desc_dq.dtype))



@triton.autotune(configs=[triton.Config({"BLOCK_M": BM, "BLOCK_N":BN}, num_stages=ns, num_warps=nw)
                          for BM in [32, 64, 128]
                          for BN in [32, 64]
                          for ns in [2, 3, 4]
                          for nw in [4, 8]], key=["D1", "D2", "VD"])
@triton.jit
def _bwd_kernel(Q, 
            K,
            V,
            DQ,
            DK,
            DV,
            DO, 
            CU_SEQLENS, 
            Lse,
            Delta,
            q_stride_n, q_stride_h, q_stride_d,
            k_stride_n, k_stride_h, k_stride_d,
            v_stride_n, v_stride_h, v_stride_d,
            dq_stride_n, dq_stride_h, dq_stride_d,
            dk_stride_n, dk_stride_h, dk_stride_d,
            dv_stride_n, dv_stride_h, dv_stride_d,
            do_stride_n, do_stride_h, do_stride_d,
            sm_scale,
            T: tl.constexpr,
            G: tl.constexpr,
            D1: tl.constexpr,
            D2: tl.constexpr,
            VD: tl.constexpr,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
            ):
    start_n = tl.program_id(0) * BLOCK_N
    off_b = tl.program_id(1)
    off_qh = tl.program_id(2)
    off_kh = off_qh // G
    bos, eos = tl.load(CU_SEQLENS + off_b), tl.load(CU_SEQLENS + off_b + 1)
    S = eos - bos
    if start_n >= S:
        return
    
    
    desc_q = tl.make_tensor_descriptor(Q + bos * q_stride_n + off_qh * q_stride_h, (S, D1), (q_stride_n, q_stride_d), (BLOCK_M, D1))
    desc_k = tl.make_tensor_descriptor(K + bos * k_stride_n + off_kh * k_stride_h, (S, D1), (k_stride_n, k_stride_d), (BLOCK_N, D1))
    desc_v = tl.make_tensor_descriptor(V + bos * v_stride_n + off_kh * v_stride_h, (S, VD), (v_stride_n, v_stride_d), (BLOCK_N, VD))
    desc_dq = tl.make_tensor_descriptor(DQ + bos * dq_stride_n + off_qh * dq_stride_h, (S, D1), (dq_stride_n, dq_stride_d), (BLOCK_M, D1))
    desc_dk = tl.make_tensor_descriptor(DK + bos * dk_stride_n + off_qh * dk_stride_h, (S, D1), (dk_stride_n, dk_stride_d), (BLOCK_N, D1))
    desc_dv = tl.make_tensor_descriptor(DV + bos * dv_stride_n + off_qh * dv_stride_h, (S, VD), (dv_stride_n, dv_stride_d), (BLOCK_N, VD))
    desc_do = tl.make_tensor_descriptor(DO + bos * do_stride_n + off_qh * do_stride_h, (S, VD), (do_stride_n, do_stride_d), (BLOCK_M, VD))
    if D2 > 0:
        desc_q2 = tl.make_tensor_descriptor(Q + bos * q_stride_n + off_qh * q_stride_h + D1, (S, D2), (q_stride_n, q_stride_d), (BLOCK_M, D2))
        desc_k2 = tl.make_tensor_descriptor(K + bos * k_stride_n + off_kh * k_stride_h + D1, (S, D2), (k_stride_n, k_stride_d), (BLOCK_N, D2))
        desc_dq2 = tl.make_tensor_descriptor(DQ + bos * dq_stride_n + off_qh * dq_stride_h + D1, (S, D1), (dq_stride_n, dq_stride_d), (BLOCK_M, D2))
        desc_dk2 = tl.make_tensor_descriptor(DK + bos * dk_stride_n + off_qh * dk_stride_h + D1, (S, D2), (dk_stride_n, dk_stride_d), (BLOCK_N, D2))
        
    ln2: tl.constexpr = 1.44269504
    sm_scale_ln2 = sm_scale * ln2
    
    k_idx = start_n + tl.arange(0, BLOCK_N)
    k = desc_k.load([start_n, 0])
    v = desc_v.load([start_n, 0])
    dv = tl.zeros((BLOCK_N, VD), dtype=tl.float32)
    dk = tl.zeros((BLOCK_N, D1), dtype=tl.float32)
    if D2 > 0:
        k2 = desc_k2.load([start_n, 0])
        dk2 = tl.zeros((BLOCK_N, D2), dtype=tl.float32)

    start = start_n
    mid = start + tl.cdiv(BLOCK_N, BLOCK_M) * BLOCK_M
    for start_m in range(start, mid, BLOCK_M):
        q_idx = start_m + tl.arange(0, BLOCK_M)
        q = desc_q.load([start_m, 0])
        do = desc_do.load([start_m, 0])
        lse = tl.load(Lse + (bos + q_idx) + off_qh * T, q_idx<S)
        delta = tl.load(Delta + (bos + q_idx) + off_qh * T, q_idx<S)
        qk = tl.dot(q, tl.permute(k, 1, 0))
        if D2 > 0:
            q2 = desc_q2.load([start_m, 0])
            qk = tl.dot(q2, tl.permute(k2, 1, 0), qk)
        mask = (q_idx[:, None] >= k_idx[None, :])
        qk = tl.where(mask, qk * sm_scale_ln2, float('-inf'))
        p = tl.exp2(qk - lse[:, None])
        dv = tl.dot(tl.permute(p, 1, 0).to(do.dtype), do, dv)
        dp = tl.dot(do, tl.permute(v, 1, 0))
        ds = p * (dp - delta[:, None])
        ds = ds.to(k.dtype)
        dk = tl.dot(tl.permute(ds, 1, 0), q, dk)
        desc_dq.atomic_add([start_m, 0], tl.dot(ds, k) * sm_scale)
        if D2 > 0:
            dk2 = tl.dot(tl.permute(ds, 1, 0), q2, dk2)
            desc_dq2.atomic_add([start_m, 0], tl.dot(ds, k2) * sm_scale)

    for start_m in range(mid, S, BLOCK_M):
        q_idx = start_m + tl.arange(0, BLOCK_M)
        q = desc_q.load([start_m, 0])
        do = desc_do.load([start_m, 0])
        lse = tl.load(Lse + (bos + q_idx) + off_qh * T, q_idx<S)
        delta = tl.load(Delta + (bos + q_idx) + off_qh * T, q_idx<S)
        qk = tl.dot(q, tl.permute(k, 1, 0))
        if D2 > 0:
            q2 = desc_q2.load([start_m, 0])
            qk = tl.dot(q2, tl.permute(k2, 1, 0), qk)
        p = tl.exp2(qk * sm_scale_ln2 - lse[:, None])
        dv = tl.dot(tl.permute(p, 1, 0).to(do.dtype), do, dv)
        dp = tl.dot(do, tl.permute(v, 1, 0))
        ds = p * (dp - delta[:, None])
        ds = ds.to(k.dtype)
        dk = tl.dot(tl.permute(ds, 1, 0), q, dk)
        desc_dq.atomic_add([start_m, 0], tl.dot(ds, k) * sm_scale)
        if D2 > 0:
            dk2 = tl.dot(tl.permute(ds, 1, 0), q2, dk2)
            desc_dq2.atomic_add([start_m, 0], tl.dot(ds, k2) * sm_scale)
            
    desc_dv.store([start_n, 0], dv.to(desc_dv.dtype))
    desc_dk.store([start_n, 0], (dk * sm_scale).to(desc_dk.dtype))
    if D2 > 0:
        desc_dk2.store([start_n, 0], (dk2 * sm_scale).to(desc_dk.dtype))

class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, cu_seqlens, max_len, sm_scale, causal, window_size):
        T, QH, D = q.shape
        T2, KH, _D = k.shape
        T3, KH2, VD = v.shape
        D1, D2 = split_d(D)
        G = QH // KH
        B = len(cu_seqlens) - 1
        assert math.log2(VD).is_integer()
        assert T == T2 and T == T3 and KH == KH2 and D == _D
        assert QH % KH == 0
        window = window_size > -1
        if window:
            assert math.log2(window_size).is_integer()
        
        if sm_scale is None:
            sm_scale = D ** -0.5

        o = torch.zeros(T, QH, VD, dtype=q.dtype, device=q.device)
        lse = torch.zeros(QH, T, device=q.device, dtype=torch.float32)
        
        kwargs = {"BLOCK_M":128, "BLOCK_N": 64, "num_warps":4, "num_stages":2}
        if D>128:
            kwargs = {"BLOCK_M":128, "BLOCK_N": 64, "num_warps":8, "num_stages":3}
        grid = lambda meta: (triton.cdiv(max_len, meta['BLOCK_M']), B, QH)
        _attn_fwd[grid](
            q, 
            k,
            v, 
            o,
            cu_seqlens,
            lse,
            *q.stride(),
            *k.stride(),
            *v.stride(),
            *o.stride(),
            sm_scale,
            window,
            window_size,
            causal,
            T,
            KH,
            G,
            D1,
            D2,
            VD,
            **kwargs,
            )

        ctx.save_for_backward(q, k, v, o, lse, cu_seqlens)
        ctx.sm_scale = sm_scale
        ctx.infos = (B, T, QH, KH, G, D, D1, D2, VD, sm_scale, max_len)
        ctx.causal = causal
        ctx.window_size = window_size
        return o, lse

    @staticmethod
    def backward(ctx, do, dlse):
        q, k, v, o, lse, cu_seqlens = ctx.saved_tensors
        B, T, QH, KH, G, D, D1, D2, VD, sm_scale, max_len = ctx.infos
        assert do.is_contiguous()
        assert ctx.window_size == -1

        delta = torch.empty_like(lse)
        pre_kwargs = {"BLOCK_M": 16, "num_warps": 8, "num_stages": 4}
        pre_grid = lambda meta: (triton.cdiv(T, meta['BLOCK_M']), QH)               
        _bwd_preprocess[pre_grid](o, 
                                do, 
                                delta,
                                *o.stride(), 
                                T, 
                                VD,
                                **pre_kwargs
                                )
        
        
        # dkdv_kwargs = {"BLOCK_M":64, "BLOCK_N":64, "num_warps":4, "num_stages":3}
        # dkdv_grid = lambda meta: (triton.cdiv(max_len, meta["BLOCK_N"]), B, QH)
        
        # dk = torch.zeros(T, QH, D, device=q.device, dtype=torch.float32 if G>1 else k.dtype)
        # dv = torch.zeros(T, QH, VD, device=q.device, dtype=torch.float32 if G>1 else k.dtype)
        # _dkdv_kernel[dkdv_grid](q, 
        #                 k,
        #                 v,
        #                 dk,
        #                 dv,
        #                 do, 
        #                 cu_seqlens,
        #                 lse,
        #                 delta,
        #                 *q.stride(), 
        #                 *k.stride(), 
        #                 *v.stride(), 
        #                 *dk.stride(), 
        #                 *dv.stride(), 
        #                 *do.stride(),
        #                 sm_scale,
        #                 T,
        #                 G,
        #                 D1,
        #                 D2,
        #                 VD,
        #                 **dkdv_kwargs
        #                 )
        # if G > 1:
        #     dk = dk.view(T, KH, G, -1).sum(-2).to(k.dtype)
        #     dv = dv.view(T, KH, G, -1).sum(-2).to(v.dtype)
        
        dk = torch.empty(T, KH, D, device=q.device, dtype=k.dtype)
        dv = torch.empty(T, KH, VD, device=q.device, dtype=k.dtype)
        dkdv_kwargs = {"BLOCK_M":64, "BLOCK_N":64, "num_warps":4, "num_stages":3}
        if D > 128:
            dkdv_kwargs = {"BLOCK_M":64, "BLOCK_N":64, "num_warps":4, "num_stages":3}
        dkdv_grid = lambda meta: (triton.cdiv(max_len, meta["BLOCK_N"]), B, KH)
        _dkdv_kernel2[dkdv_grid](q, 
                        k,
                        v,
                        dk,
                        dv,
                        do, 
                        cu_seqlens,
                        lse,
                        delta,
                        *q.stride(), 
                        *k.stride(), 
                        *v.stride(), 
                        *dk.stride(), 
                        *dv.stride(), 
                        *do.stride(),
                        sm_scale,
                        ctx.causal,
                        T,
                        G,
                        D1,
                        D2,
                        VD,
                        **dkdv_kwargs
                        )
        
        dq = torch.empty(T, QH, D, device=q.device, dtype=q.dtype)
        dq_kwargs = {"BLOCK_M":128, "BLOCK_N":64, "num_warps":8, "num_stages":4}
        if D > 128:
            dq_kwargs = {"BLOCK_M":128, "BLOCK_N":64, "num_warps":8, "num_stages":3}
        dq_grid = lambda meta: (triton.cdiv(max_len, meta["BLOCK_M"]), B, QH)
        _dq_kernel[dq_grid](q, 
                            k,
                            v,
                            dq,
                            do,
                            cu_seqlens,
                            lse,
                            delta,
                            *q.stride(), 
                            *k.stride(), 
                            *v.stride(), 
                            *dq.stride(), 
                            *do.stride(),
                            sm_scale,
                            ctx.causal,
                            T,
                            G,
                            D1,
                            D2,
                            VD,
                            **dq_kwargs
                        )
        
        # dq = torch.zeros(T, QH, D, device=q.device, dtype=torch.float32)
        # dk = torch.zeros(T, QH, D, device=q.device, dtype=torch.float32 if G>1 else k.dtype)
        # dv = torch.zeros(T, QH, VD, device=q.device, dtype=torch.float32 if G>1 else k.dtype)
        # dkdv_kwargs = {"BLOCK_M":64, "BLOCK_N":64, "num_warps":4, "num_stages":3}
        # dkdv_grid = lambda meta: (triton.cdiv(max_len, meta["BLOCK_N"]), B, QH)
        # _bwd_kernel[dkdv_grid](q, 
        #                 k,
        #                 v,
        #                 dq,
        #                 dk,
        #                 dv,
        #                 do, 
        #                 cu_seqlens,
        #                 lse,
        #                 delta,
        #                 *q.stride(), 
        #                 *k.stride(), 
        #                 *v.stride(), 
        #                 *dq.stride(),
        #                 *dk.stride(), 
        #                 *dv.stride(), 
        #                 *do.stride(),
        #                 sm_scale,
        #                 T,
        #                 G,
        #                 D1,
        #                 D2,
        #                 VD,
        #                 # **dkdv_kwargs
        #                 )
        # if G > 1:
        #     dk = dk.view(T, KH, G, -1).sum(-2).to(k.dtype)
        #     dv = dv.view(T, KH, G, -1).sum(-2).to(v.dtype)
        # dq = dq.to(q.dtype)
        return dq, dk, dv, None, None, None, None, None

def triton_flash_attn_varlen_func(q, k, v, cu_seqlens, max_len, sm_scale=None, causal=True, window_size=-1) -> torch.Tensor:
    return _attention.apply(q, k, v, cu_seqlens, max_len, sm_scale, causal, window_size)