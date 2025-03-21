import torch
import triton
import triton.backends
import triton.language as tl
import math
import os
os.environ['TRITON_PRINT_AUTOTUNING'] = '1'
triton.__version__


# @triton.autotune([triton.Config({'BLOCK_SIZE_N': bsn, 'BLOCK_SIZE_M': bsm}, num_stages=ns, num_warps=nw)
#                  for bsm in [32, 64]
#                  for bsn in [32, 64]
#                  for ns in [2, 4]
#                  for nw in [4, 8]
#                  ], key=['N', 'KV_LORA_RANK', 'ROPE_HEAD_DIM'])
@triton.jit
def _mla_encode_kernel(Q, K, V, OUT, 
                    NUM_PADS, SCALE,
                    q_stride_b, q_stride_h, q_stride_n, q_stride_d,
                    k_stride_b, k_stride_m, k_stride_d,
                    v_stride_b, v_stride_m, v_stride_d,
                    out_stride_b, out_stride_h, out_stride_n, out_stride_d,
                    N, KV_LORA_RANK: tl.constexpr, ROPE_HEAD_DIM: tl.constexpr,
                    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_M: tl.constexpr
                    ):
    off_b = tl.cast(tl.program_id(0), tl.int64)
    off_h = tl.cast(tl.program_id(1), tl.int64)
    off_n = tl.cast(tl.program_id(2), tl.int64) * BLOCK_SIZE_N

    
    NUM_PADS += off_b
    num_pads = tl.cast(tl.load(NUM_PADS), tl.int64)

    if (off_n + BLOCK_SIZE_N) > num_pads:
        Q += off_b * q_stride_b + off_h * q_stride_h + off_n * q_stride_n
        K += off_b * k_stride_b + num_pads * k_stride_m
        V += off_b * v_stride_b
        OUT += off_b * out_stride_b + off_h * out_stride_h + off_n * out_stride_n

        nn = tl.arange(0, BLOCK_SIZE_N)
        mm = tl.arange(0, BLOCK_SIZE_M)
        
        dtype = Q.type.element_ty
    
        q_nope_ptrs = Q + nn[:, None] * q_stride_n + tl.arange(0, KV_LORA_RANK)[None, :]
        q_rope_ptrs = Q + nn[:, None] * q_stride_n + tl.arange(0, ROPE_HEAD_DIM)[None, :] + KV_LORA_RANK
        k_nope_ptrs = K + mm[None, :] * k_stride_m + tl.arange(0, KV_LORA_RANK)[:, None]
        k_rope_ptrs = K + mm[None, :] * k_stride_m + tl.arange(0, ROPE_HEAD_DIM)[:, None] + KV_LORA_RANK
        
        mask_n = (off_n + nn) < N
        q_nope = tl.load(q_nope_ptrs, mask=mask_n[:, None], other=0.) # q_nope是q的前kv_lora_rank维度
        q_rope = tl.load(q_rope_ptrs, mask=mask_n[:, None], other=0.)
        acc = tl.zeros((BLOCK_SIZE_N, KV_LORA_RANK), dtype=tl.float32)
        m_i = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32) - float('inf')
        l_i = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
        for start_m in range(num_pads, off_n + BLOCK_SIZE_N, BLOCK_SIZE_M): 
            mask_m = (start_m + mm) < N
            k_nope = tl.load(k_nope_ptrs, mask=mask_m[None, :], other=0.)
            k_rope = tl.load(k_rope_ptrs, mask=mask_m[None, :], other=0.) # k_nope是k的前kv_lora_rank维度

            attn_score= tl.dot(q_rope, k_rope)
            attn_score = tl.dot(q_nope, k_nope, acc=attn_score) * SCALE
            
            attn_mask = (off_n + nn)[:, None] >= (start_m + mm)[None, :]
            attn_score = tl.where(attn_mask & mask_m[None, :], attn_score, -60000)

            m_ij = tl.max(attn_score, axis=1)
            new_m_i = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - new_m_i)

            exp_attn_score = tl.exp(attn_score - new_m_i[:, None])

            l_i = l_i * alpha + tl.sum(exp_attn_score, axis=-1)
            acc = acc * alpha[:, None]
            # v就是compressed_kv，就是k的前kv_lora_rank维，也就是k_nope
            acc = tl.dot(exp_attn_score.to(dtype), tl.trans(k_nope, 1,0), acc=acc)

            m_i = new_m_i
            k_nope_ptrs += BLOCK_SIZE_M * k_stride_m
            k_rope_ptrs += BLOCK_SIZE_M * k_stride_m
        acc = acc / l_i[:, None]

        out_ptrs = OUT + nn[:, None] * out_stride_n + tl.arange(0, KV_LORA_RANK)[None, :]
        tl.store(out_ptrs, acc.to(dtype), mask=mask_n[:, None])
    else:
        pass


# @triton.autotune([triton.Config({'BLOCK_SIZE_M': bsn, 'BLOCK_SIZE_H': bsh}, num_stages=ns, num_warps=nw)
#                  for bsn in [32, 64, 128]
#                  for bsh in [32, 64, 128]
#                  for ns in [1, 2, 4]
#                  for nw in [4, 8]
#                  ], key=['M', 'KV_LORA_RANK', 'ROPE_HEAD_DIM'])
@triton.jit
def _mla_decode_kernel(Q, K, V, OUT, 
                    NUM_PADS, SCALE,
                    q_stride_b, q_stride_h, q_stride_n, q_stride_d,
                    k_stride_b, k_stride_m, k_stride_d,
                    v_stride_b, v_stride_m, v_stride_d,
                    out_stride_b, out_stride_h, out_stride_n, out_stride_d,
                    M, H, KV_LORA_RANK: tl.constexpr, ROPE_HEAD_DIM: tl.constexpr,
                    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_M: tl.constexpr
                    ):
    off_b = tl.cast(tl.program_id(0), tl.int64)
    off_h = tl.cast(tl.program_id(1), tl.int64) * BLOCK_SIZE_H

    
    NUM_PADS += off_b
    num_pads = tl.cast(tl.load(NUM_PADS), tl.int64)

    Q += off_b * q_stride_b + off_h * q_stride_h
    K += off_b * k_stride_b + num_pads * k_stride_m
    V += off_b * v_stride_b
    OUT += off_b * out_stride_b + off_h * out_stride_h

    hh = tl.arange(0, BLOCK_SIZE_H)
    mm = tl.arange(0, BLOCK_SIZE_M)
    
    dtype = Q.type.element_ty

    q_nope_ptrs = Q + hh[:, None] * q_stride_h + tl.arange(0, KV_LORA_RANK)[None, :]
    q_rope_ptrs = Q + hh[:, None] * q_stride_h + tl.arange(0, ROPE_HEAD_DIM)[None, :] + KV_LORA_RANK
    k_nope_ptrs = K + mm[None, :] * k_stride_m + tl.arange(0, KV_LORA_RANK)[:, None]
    k_rope_ptrs = K + mm[None, :] * k_stride_m + tl.arange(0, ROPE_HEAD_DIM)[:, None] + KV_LORA_RANK
    
    mask_h = (off_h + hh) < H
    q_nope = tl.load(q_nope_ptrs, mask=mask_h[:, None], other=0.)
    q_rope = tl.load(q_rope_ptrs, mask=mask_h[:, None], other=0.)
    acc = tl.zeros((BLOCK_SIZE_H, KV_LORA_RANK), dtype=tl.float32)
    m_i = tl.zeros((BLOCK_SIZE_H,), dtype=tl.float32) - float('inf')
    l_i = tl.zeros((BLOCK_SIZE_H,), dtype=tl.float32)
    for start_m in range(num_pads, M, BLOCK_SIZE_M): 
        mask_m = (start_m + mm) < M
        k_nope = tl.load(k_nope_ptrs, mask=mask_m[None, :], other=0.)
        k_rope = tl.load(k_rope_ptrs, mask=mask_m[None, :], other=0.)

        attn_score = tl.dot(q_rope, k_rope)
        attn_score = tl.dot(q_nope, k_nope, acc=attn_score) * SCALE
    
        attn_score = tl.where(mask_m[None, :], attn_score, -65000)

        m_ij = tl.max(attn_score, axis=1)
        new_m_i = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - new_m_i)

        exp_attn_score = tl.exp(attn_score - new_m_i[:, None])

        l_i = l_i * alpha + tl.sum(exp_attn_score, axis=-1)
        acc = acc * alpha[:, None]

        acc = tl.dot(exp_attn_score.to(dtype), tl.trans(k_nope, 1,0), acc=acc)

        m_i = new_m_i
        k_nope_ptrs += BLOCK_SIZE_M * k_stride_m
        k_rope_ptrs += BLOCK_SIZE_M * k_stride_m
    acc = acc / l_i[:, None]

    out_ptrs = OUT + hh[:, None] * out_stride_h + tl.arange(0, KV_LORA_RANK)[None, :]
    tl.store(out_ptrs, acc.to(dtype), mask=mask_h[:, None])

# @triton.autotune([triton.Config({'BLOCK_SIZE_M': bsm, 'BLOCK_SIZE_H': bsh, 'BLOCK_SIZE_K': bsk,}, num_stages=ns, num_warps=nw)
#                  for bsm in [32, 64, 128]
#                  for bsh in [32]
#                  for bsk in [32, 64, 128]
#                  for ns in [2, 4]
#                  for nw in [4, 8]
#                  ], key=['M', 'KV_LORA_RANK', 'ROPE_HEAD_DIM'])
@triton.jit
def _stage1_compute_qk(Q, K, QK, 
                    NUM_PADS, SCALE,
                    q_stride_b, q_stride_h, q_stride_n, q_stride_d,
                    k_stride_b, k_stride_m, k_stride_d,
                    M, H, KV_LORA_RANK: tl.constexpr, ROPE_HEAD_DIM: tl.constexpr,
                    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
                    ):
    off_b = tl.cast(tl.program_id(0), tl.int64)
    off_h = tl.cast(tl.program_id(1), tl.int64) * BLOCK_SIZE_H
    off_m = tl.cast(tl.program_id(2), tl.int64) * BLOCK_SIZE_M

    
    NUM_PADS += off_b
    num_pads = tl.cast(tl.load(NUM_PADS), tl.int64)
    if (off_m + BLOCK_SIZE_M) <= num_pads:
        return

    Q += off_b * q_stride_b + off_h * q_stride_h
    K += off_b * k_stride_b + off_m * k_stride_m
    QK += off_b * (M * H) + off_h * M + off_m
    

    hh = tl.arange(0, BLOCK_SIZE_H)
    mm = tl.arange(0, BLOCK_SIZE_M)
    kk = tl.arange(0, BLOCK_SIZE_K)
    
    dtype = Q.type.element_ty

    q_nope_ptrs = Q + hh[:, None] * q_stride_h + kk[None, :]
    q_rope_ptrs = Q + hh[:, None] * q_stride_h + tl.arange(0, ROPE_HEAD_DIM)[None, :] + KV_LORA_RANK
    k_nope_ptrs = K + mm[None, :] * k_stride_m + kk[:, None]
    k_rope_ptrs = K + mm[None, :] * k_stride_m + tl.arange(0, ROPE_HEAD_DIM)[:, None] + KV_LORA_RANK
    
    mask_h = (off_h + hh) < H
    mask_m = (off_m + mm) < M

    qk = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_M), dtype=tl.float32)

    q_rope = tl.load(q_rope_ptrs, mask=mask_h[:, None], other=0.)
    k_rope = tl.load(k_rope_ptrs, mask=mask_m[None, :], other=0.)
    qk = tl.dot(q_rope, k_rope, acc=qk)

    for _ in range(0, KV_LORA_RANK, BLOCK_SIZE_K): 
        k_nope = tl.load(k_nope_ptrs, mask=mask_m[None, :], other=0.)
        q_nope = tl.load(q_nope_ptrs, mask=mask_h[:, None], other=0.)
        qk = tl.dot(q_nope, k_nope, acc=qk)
        
        k_nope_ptrs += BLOCK_SIZE_K
        q_nope_ptrs += BLOCK_SIZE_K
    qk = qk * SCALE
    qk_ptrs = QK + hh[:, None] * M + mm[None, :]
    tl.store(qk_ptrs, qk, mask=mask_h[:, None] & mask_m[None, :])

# @triton.autotune([triton.Config({'BLOCK_SIZE_M': bsm, 'BLOCK_SIZE_H': bsh, 'BLOCK_SIZE_K': bsk,}, num_stages=ns, num_warps=nw)
#                  for bsm in [32, 64, 128]
#                  for bsh in [16]
#                  for bsk in [64, 128, 256]
#                  for ns in [2, 4]
#                  for nw in [4, 8]
#                  ], key=['M', 'KV_LORA_RANK', 'ROPE_HEAD_DIM'])
@triton.jit
def _stage2_compute_out(QK, V, OUT, 
                    NUM_PADS, 
                    v_stride_b, v_stride_m, v_stride_d,
                    out_stride_b, out_stride_h, out_stride_n, out_stride_d,
                    M, H, KV_LORA_RANK: tl.constexpr, ROPE_HEAD_DIM: tl.constexpr,
                    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
                    ):
    off_b = tl.cast(tl.program_id(0), tl.int64)
    off_h = tl.cast(tl.program_id(1), tl.int64) * BLOCK_SIZE_H
    off_k = tl.cast(tl.program_id(2), tl.int64) * BLOCK_SIZE_K

    
    NUM_PADS += off_b
    num_pads = tl.cast(tl.load(NUM_PADS), tl.int64)

    QK += off_b * (M * H) + off_h * M + num_pads
    V += off_b * v_stride_b + num_pads * v_stride_m + off_k
    OUT += off_b * out_stride_b + off_h * out_stride_h + off_k

    hh = tl.arange(0, BLOCK_SIZE_H)
    mm = tl.arange(0, BLOCK_SIZE_M)
    kk = tl.arange(0, BLOCK_SIZE_K)
    
    dtype = OUT.type.element_ty

    qk_ptrs = QK + hh[:, None] * M + mm[None, :]
    v_ptrs = V + mm[:, None] * v_stride_m + kk[None, :]
    mask_h = (off_h + hh) < H
    
    acc = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_K), dtype=tl.float32)
    m_i = tl.zeros((BLOCK_SIZE_H,), dtype=tl.float32) - float('inf')
    l_i = tl.zeros((BLOCK_SIZE_H,), dtype=tl.float32)
    for start_m in range(num_pads, M, BLOCK_SIZE_M): 
        mask_m = (start_m + mm) < M
        attn_score = tl.load(qk_ptrs, mask=mask_m[None, :] & mask_h[:, None], other=-65000)
        
        m_ij = tl.max(attn_score, axis=1)
        new_m_i = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - new_m_i)

        exp_attn_score = tl.exp(attn_score - new_m_i[:, None])

        l_i = l_i * alpha + tl.sum(exp_attn_score, axis=-1)
        acc = acc * alpha[:, None]

        v = tl.load(v_ptrs, mask=mask_m[:, None], other=0.) 
        # tl.static_print(exp_attn_score.shape, v.shape)
        acc = tl.dot(exp_attn_score.to(dtype), v, acc=acc)

        m_i = new_m_i
        qk_ptrs += BLOCK_SIZE_M
        v_ptrs += BLOCK_SIZE_M * v_stride_m
    acc = acc / l_i[:, None]

    out_ptrs = OUT + hh[:, None] * out_stride_h + kk[None, :]
    tl.store(out_ptrs, acc, mask=mask_h[:, None])

@torch.inference_mode()
def triton_mqa(q:torch.Tensor, k:torch.Tensor, v:torch.Tensor, scale=None, attention_mask:torch.Tensor=None, tow_stage_decode=None):
    """
    impl DeepSeek MLA by MQA with fast speed and low memory

    Args:
        q (Tensor): [bs, h, q_len, kv_lora_rank + rope_head_dim]
        k (Tensor): [bs, 1, kv_len, kv_lora_rank + rope_head_dim]
        v (Tensor): [bs, 1, kv_len, kv_lora_rank]
        scale (Float): softmax scaling factor
        attention (Tensor): [bs, kv_len], it's from the tokenizer(left_padding, 0 mean mask, 1 mean non-mask)
        tow_stage_decode (Bool): Default None, it will auto choose use which method to decode, if set true use 2-stage, if set false use 1-stage
    Return:
        out: [bs, h, q_len, kv_lora_rank]
    """
    B, H, N, D = q.shape
    M = k.shape[-2]
    # assert D == (256 + 32) or D == (512 + 64)
    if D == (256 + 32):
        KV_LORA_RANK = 256
        ROPE_HEAD_DIM = 32
    elif D == (512 + 64):
        KV_LORA_RANK = 512
        ROPE_HEAD_DIM = 64
    else:
        KV_LORA_RANK = 2**int(math.log2(D-1))
        ROPE_HEAD_DIM = D - KV_LORA_RANK
    assert math.log2(ROPE_HEAD_DIM).is_integer()
    assert v.size(-1) == KV_LORA_RANK
    assert N == 1 or M == N
    assert scale is not None, 'must provide the softmax scale value'
    # print(KV_LORA_RANK, ROPE_HEAD_DIM)
    if k.dim() == 4:
        k = k.squeeze(1)
    if v.dim() == 4:
        v = v.squeeze(1)

    if attention_mask is not None:
        num_pads = M - attention_mask.sum(-1)
    else:
        num_pads = torch.zeros((B,), dtype=torch.int32, device=q.device)
    
    out = torch.empty(B, H, N, KV_LORA_RANK, dtype=q.dtype, device=q.device)
    if N > 1:
        grids = lambda meta: (B, H, triton.cdiv(N, meta['BLOCK_SIZE_N']))
        kwargs = {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_M": 64, "num_warps": 8, "num_stages": 2}
        _mla_encode_kernel[grids](q, k, v, out, 
                                num_pads, scale,
                                *q.stride(),
                                *k.stride(),
                                *v.stride(),
                                *out.stride(),
                                N, KV_LORA_RANK, ROPE_HEAD_DIM,
                                **kwargs
                                )
        
    else:
        if tow_stage_decode or (tow_stage_decode is None and B <= 4):
            qk = torch.empty(B, H, M, dtype=torch.float32, device=q.device)
            grids = lambda meta: (B, triton.cdiv(H, meta['BLOCK_SIZE_H']), triton.cdiv(M, meta['BLOCK_SIZE_M']))
            kwargs = {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_H": 32, "BLOCK_SIZE_K": 64, "num_warps": 4, "num_stages": 4}
            _stage1_compute_qk[grids](q, k, qk, 
                                      num_pads, scale,
                                      *q.stride(),
                                      *k.stride(),
                                      M, H, KV_LORA_RANK, ROPE_HEAD_DIM,
                                      **kwargs
                                      )
            
            grids = lambda meta: (B, triton.cdiv(H, meta['BLOCK_SIZE_H']), triton.cdiv(KV_LORA_RANK, meta['BLOCK_SIZE_K']))
            kwargs = {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_H": 16, "BLOCK_SIZE_K": 128, "num_warps": 8, "num_stages": 4}
            _stage2_compute_out[grids](qk, v, out, 
                            num_pads,
                            *v.stride(),
                            *out.stride(),
                            M, H, KV_LORA_RANK, ROPE_HEAD_DIM,
                            **kwargs
                            )
        else:
            grids = lambda meta: (B, triton.cdiv(H, meta['BLOCK_SIZE_H']))
            kwargs = {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_H": 32, "num_warps": 8, "num_stages": 2}
            _mla_decode_kernel[grids](q, k, v, out, 
                                    num_pads, scale,
                                    *q.stride(),
                                    *k.stride(),
                                    *v.stride(),
                                    *out.stride(),
                                    M, H, KV_LORA_RANK, ROPE_HEAD_DIM,
                                    **kwargs
                                    )
    return out