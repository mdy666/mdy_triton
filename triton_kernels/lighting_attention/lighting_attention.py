import torch
import triton
import triton.language as tl
import os
from copy import deepcopy
import math

@triton.jit
def _lighting_attention_encode_kernel(QKV, KV, Y, SLOPE_RATE, NUM_PADDDINGS,
                                    qkv_sb, qkv_sn, qkv_sh, qkv_sd,
                                    kv_sb, kv_sh, kv_sd, kv_se,
                                    y_sb, y_sh, y_sn, y_sd,
                                    B, N, H, D:tl.constexpr, FP32:tl.constexpr,
                                    BLOCK_SIZE_N: tl.constexpr,
                                    ):
    off_b = tl.program_id(0)
    off_h = tl.program_id(1)

    QKV += off_b * qkv_sb + off_h * qkv_sh
    KV += off_b * kv_sb + off_h * kv_sh
    Y += off_b * y_sb + off_h * y_sh
    SLOPE_RATE += off_h
    NUM_PADDDINGS += off_b
    dd = tl.arange(0, D)
    nn = tl.arange(0, BLOCK_SIZE_N)

    num_paddings = tl.load(NUM_PADDDINGS)
    q_ptrs = QKV + (nn[:, None] + num_paddings) * qkv_sn + dd[None, :]
    k_ptrs = QKV + (nn[None, :] + num_paddings) * qkv_sn + dd[:, None] + D
    kv_ptrs = KV + dd[:, None] * kv_sd + dd[None, :]
    y_pts = Y + (nn[:, None] + num_paddings) * y_sn + dd[None, :]

    slope_rate = tl.load(SLOPE_RATE).to(tl.float32)

    array = (nn + 1)
    q_decay = tl.exp(-1. * slope_rate * array)[:, None]
    k_decay = tl.exp(-1. * slope_rate * (BLOCK_SIZE_N - array))[None, :]
    index = array[:, None] - array[None, :]
    s_index = slope_rate * index
    s_index = tl.where(index >= 0, -s_index, float('-inf'))
    diag_decay = tl.exp(s_index)

    if FP32:
        dtype = tl.float32
    else:
        dtype = tl.bfloat16
    kv = tl.zeros((D, D), dtype=tl.float32)
    
    for start_n in tl.range(num_paddings, N, BLOCK_SIZE_N):
        mask_nn = (nn + start_n) < N  
        m = tl.minimum(N-start_n, BLOCK_SIZE_N)
        if m < BLOCK_SIZE_N:
            k_decay = tl.exp(-1. * slope_rate * (m - array))[None, :]
            
        q = tl.load(q_ptrs, mask=mask_nn[:, None], other=0.).to(dtype)
        k = tl.load(k_ptrs, mask=mask_nn[None, :], other=0.).to(dtype)
        v = tl.load(q_ptrs + 2*D, mask=mask_nn[:, None], other=0.).to(dtype)

        qkv_none_diag = tl.dot((q * q_decay).to(dtype), kv.to(dtype))

        qk = tl.dot(q, k) * diag_decay
        qkv_diag = tl.dot(qk.to(dtype), v)
        y = qkv_diag + qkv_none_diag
        block_decay = tl.exp(-1. * slope_rate * m)
        kv = kv * block_decay + tl.dot((k * k_decay).to(v.dtype), v)
        # kv = tl.dot(tl.permute(k, (1,0)), k)
        
        tl.store(y_pts, y, mask=mask_nn[:, None])

        q_ptrs += BLOCK_SIZE_N * qkv_sn
        y_pts += BLOCK_SIZE_N * y_sn
        k_ptrs += BLOCK_SIZE_N * qkv_sn
    tl.store(kv_ptrs, kv)


@triton.jit
def _lighting_attention_decode_kernel(QKV, KV, Y, SLOPE_RATE,
                                    qkv_sb, qkv_sn, qkv_sh, qkv_sd,
                                    kv_sb, kv_sh, kv_sd, kv_se,
                                    y_sb, y_sh, y_sn, y_sd,
                                    B, N, H, D:tl.constexpr, FP32:tl.constexpr,
                                    ):
    off_b = tl.program_id(0)
    off_h = tl.program_id(1)

    QKV += off_b * qkv_sb + off_h * qkv_sh
    KV += off_b * kv_sb + off_h * kv_sh
    Y += off_b * y_sb + off_h * y_sh
    SLOPE_RATE += off_h
    dd = tl.arange(0, D)

    q_ptrs = QKV + dd
    kv_ptrs = KV + dd[:, None] * kv_sd + dd[None, :]
    y_ptrs = Y + dd

    slope_rate = tl.load(SLOPE_RATE).to(tl.float32)
    ratio = tl.exp(-1. * slope_rate)

    if FP32:
        dtype = tl.float32
    else:
        dtype = tl.bfloat16
    kv = tl.load(kv_ptrs).to(dtype)
    q = tl.load(q_ptrs).to(dtype)
    k = tl.load(q_ptrs+D) .to(dtype)
    v = tl.load(q_ptrs + 2*D).to(dtype)

    kv = ratio * kv + k[:, None] * v[None, :]
    y = tl.sum(q[:, None] * kv, axis=0)
    tl.store(kv_ptrs, kv)
    tl.store(y_ptrs, y)


def lighting_attention_encode(qkv, slope_rate, attention_mask=None, fp32=False):
    # b, n, h, d
    b, n, h, d3 = qkv.shape
    d = d3 // 3
    assert math.log2(d).is_integer(), 'd must be power of 2'
    slope_rate = slope_rate.squeeze()
    kv = torch.empty(b, h, d, d).to(torch.float32).to(qkv.device)
    y = torch.empty(b, h, n, d, device=qkv.device, dtype=qkv.dtype)
    if attention_mask is not None:
        assert attention_mask[-1, :].min().values != 0, 'please use left_padding'
        num_paddings = n - attention_mask.sum(-1)
    else:
        num_paddings = torch.full((b,), 0, device=qkv.device, dtype=torch.int32)
    
    grids = (b, h)

    _lighting_attention_encode_kernel[grids](qkv, kv, y, slope_rate, num_paddings,
                                            *qkv.stride(),
                                            *kv.stride(),
                                            *y.stride(),
                                            b, n, h, d, fp32,
                                            BLOCK_SIZE_N=32,
                                            num_warps=8, num_stages=4 if fp32 else 1,
                                            )
    return y, kv

def lighting_attention_decode(qkv, slope_rate, kv, fp32=False):
    # b, n, h, d
    b, n, h, d3 = qkv.shape
    assert n == 1, 'decoing phase need n=1'
    d = d3 // 3
    slope_rate = slope_rate.squeeze()
    y = torch.empty(b, h, n, d, device=qkv.device, dtype=qkv.dtype)

    grids = (b, h)

    _lighting_attention_decode_kernel[grids](qkv, kv, y, slope_rate,
                                            *qkv.stride(),
                                            *kv.stride(),
                                            *y.stride(),
                                            b, n, h, d, fp32,
                                            num_warps=8, num_stages=1,
                                            )
    return y, kv

def triton_lighting_attention(qkv, slope_rate, past_key_value=None, attention_mask=None, fp32=False):
    if past_key_value is None:
        y, kv = lighting_attention_encode(qkv, slope_rate, attention_mask, fp32)
    else:
        y, kv = lighting_attention_decode(qkv, slope_rate, past_key_value, fp32)
    return y, kv


BLOCK = 256
def torch_lighting_attention(qkv, slope_rate, past_key_value=None, attention_mask=None):
    n = qkv.size(1)
    q, k, v = torch.split(qkv, [qkv.size(-1)//3] * 3, dim=3)
    # [b, h, l, d]
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    if past_key_value is None:
        offset = q.shape[-2]
    else:
        offset = 1

    # for align with metaseq
    ratio = torch.exp(-slope_rate)

    # only use for the first time
    if past_key_value is None:
        slope_rate = slope_rate.to(torch.float32)
        if attention_mask is not None:
            v = v.masked_fill((1 - attention_mask).unsqueeze(1).unsqueeze(-1).to(torch.bool), 0)
        # print(v[0, 0, :32])
        NUM_BLOCK = (n + BLOCK - 1) // BLOCK
        b, h, n, d = q.shape
        e = v.shape[-1]
        # other
        array = torch.arange(BLOCK).to(q) + 1
        q_decay = torch.exp(-slope_rate * array.reshape(-1, 1)) # h, bn, 1 
        k_decay = torch.exp(-slope_rate * (BLOCK - array.reshape(-1, 1)))
        index = array[:, None] - array[None, :]
        s_index = slope_rate * index[
            None,
            None,
        ]
        s_index = torch.where(index >= 0, -s_index, float("-inf"))
        diag_decay = torch.exp(s_index)

        kv = torch.zeros(b, h, d, e).to(torch.float32).to(q.device)
        output = torch.empty((b, h, n, e), dtype=q.dtype, device=q.device)
        for i in range(NUM_BLOCK):
            si = i * BLOCK
            ei = min(si + BLOCK, n)
            m = ei - si
            qi = q[:, :, si:ei].contiguous()
            ki = k[:, :, si:ei].contiguous()
            vi = v[:, :, si:ei].contiguous()
            qkv_none_diag = torch.matmul(qi * q_decay[:, :m], kv).to(torch.float32)

            # diag
            qk = torch.matmul(qi, ki.transpose(-1, -2)).to(torch.float32) * diag_decay[:, :, :m, :m]
            qkv_diag = torch.matmul(qk, vi.to(torch.float32))
            block_decay = torch.exp(-slope_rate * m)
            output[:, :, si:ei] = qkv_none_diag + qkv_diag
            kv = block_decay * kv + torch.matmul((ki * k_decay[:, -m:]).transpose(-1, -2).to(vi.dtype), vi)
    else:
        kv = past_key_value
        output = []
        for i in range(n):
            kv = ratio * kv + torch.einsum(
                "... n d, ... n e -> ... d e",
                k[:, :, i:i + 1],
                v[:, :, i:i + 1],
            )
            qkv = torch.einsum("... n e, ... e d -> ... n d", q[:, :, i:i + 1], kv.to(q.dtype))
            output.append(qkv)
        output = torch.concat(output, dim=-2)
    return output, kv

