import torch
import triton
import triton.language as tl
from typing import Tuple
import deep_gemm


# @triton.autotune(
#         [triton.Config({'BLOCK_M': bsm}, num_stages=ns, num_warps=nw)
#                 for bsm in [16, 32, 64]
#                 for ns in [1, 2, 4]
#                 for nw in [4, 8]
#                 ], key=['M', 'N'])
@triton.jit
def _per_token_cast_to_fp8_kernel(X, Y, S, 
                           stride_xm, stride_xn,
                           stride_ym, stride_yn,
                           stride_sk, stride_sm,
                           M, N, K, MAX,
                           BLOCK_M: tl.constexpr=32, BLOCK_N: tl.constexpr=128,
                            ):
    off_m = tl.cast(tl.program_id(axis=0), tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    pid_k = tl.cast(tl.program_id(axis=1), tl.int64)
    off_n = pid_k * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = off_m < M

    x = tl.load(X + off_m[:, None] * stride_xm + off_n[None, :] * stride_xn, mask=mask[:, None], other=0.).to(tl.float32)
    x_max = tl.clamp(tl.max(tl.abs(x), -1), min=0, max=1e4)
    scale = x_max / MAX
    y = x / scale[:, None]
    tl.store(Y + off_m[:, None] * stride_ym + off_n[None, :] * stride_yn, y, mask=mask[:, None])
    tl.store(S + off_m * stride_sm + pid_k * stride_sk, scale, mask=mask)


def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    BLOCK_N = 128
    M, N = x.shape
    y = torch.empty(M, N, device=x.device, dtype=torch.float8_e4m3fn)
    K = N // BLOCK_N
    aligin_m = triton.cdiv(M, 8) * 8
    s = torch.empty(triton.cdiv(N, BLOCK_N), aligin_m, dtype=torch.float32, device=x.device)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), K)
    kwargs = {'BLOCK_M': 16, 'BLOCK_N': BLOCK_N, 'num_warps':4, 'num_stages':4}
    _per_token_cast_to_fp8_kernel[grid](x, y, s, 
                        *x.stride(),
                        *y.stride(),
                        *s.stride(),
                        M, N, K, torch.finfo(torch.float8_e4m3fn).max,
                        **kwargs,
                        )
    return y, s.transpose(0, 1)

# @triton.autotune(
#         [triton.Config({}, num_stages=ns, num_warps=nw)
#                 for ns in [1, 2, 3, 4, 5]
#                 for nw in [4, 8]
#                 ], key=['M', 'N']
# )
@triton.jit
def _per_block_cast_to_fp8_kernel(X, Y, S, 
                           stride_xm, stride_xn,
                           stride_ym, stride_yn,
                           stride_sm, stride_sn,
                           M, N, MAX,
                           BLOCK_M: tl.constexpr=128, BLOCK_N: tl.constexpr=128,
                            ):
    pid_m = tl.cast(tl.program_id(axis=0), tl.int64)
    pid_n = tl.cast(tl.program_id(axis=1), tl.int64)
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = pid_n * BLOCK_M + tl.arange(0, BLOCK_N)
    mask = (off_m[:, None] < M) & (off_n[None, :] < N)

    x = tl.load(X + off_m[:, None] * stride_xm + off_n[None, :] * stride_xn, mask=mask, other=0.).to(tl.float32)
    x_max = tl.clamp(tl.max(tl.abs(x)), min=0, max=1e4)
    scale = x_max / MAX
    y = x / scale
    tl.store(Y + off_m[:, None] * stride_ym + off_n[None, :] * stride_yn, y, mask=mask)
    tl.store(S + pid_m * stride_sm + pid_n * stride_sn, scale)

def per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    BLOCK_M = 128
    BLOCK_N = 128
    M, N = x.shape
    A, B = triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N)
    y = torch.empty(A * BLOCK_M, B * BLOCK_N, 
                    device=x.device, dtype=torch.float8_e4m3fn)
    s = torch.empty(A, B, dtype=torch.float32, device=x.device)
    grid = (A, B)
    kwargs = {'BLOCK_M': BLOCK_M, 'BLOCK_N': BLOCK_N, 'num_warps':8, 'num_stages':3}
    _per_block_cast_to_fp8_kernel[grid](x, y, s, 
                        *x.stride(),
                        *y.stride(),
                        *s.stride(),
                        M, N, torch.finfo(torch.float8_e4m3fn).max,
                        **kwargs,
                        )
    return y, s

def deep_matmul(a, b, out=None):
    # 注意，a是[m,k]，b是[n,k]，不要求连续
    assert a.dim() == 2 and a.dtype == torch.bfloat16
    assert b.dim() == 2 and b.dtype == torch.bfloat16
    m, k = a.shape
    n, k2 = b.shape
    assert k == k2 and k % 128 == 0
    a_fp8 = per_token_cast_to_fp8(a)
    y_fp8 = per_block_cast_to_fp8(b)
    if out is None:
        out = torch.empty(m, n, device=a.device, dtype=torch.bfloat16)
    deep_gemm.gemm_fp8_fp8_bf16_nt(a_fp8, y_fp8, out)
    return out

def deep_matmul(a_list, b_list, m_split, out=None):
    a = a_list[0]
    b = b_list[0]
    assert a.dim() == 2 and a.dtype == torch.bfloat16
    assert b.dim() == 2 and b.dtype == torch.bfloat16
    m, k = a.shape
    n, k2 = b.shape
    assert k == k2 and k % 128 == 0
    assert all([i % 4 == 0 for i in m_split])
    num_groups = len(m_split)

    a_fp8 = per_token_cast_to_fp8(torch.cat(a_list, axis=0))

    b_fp8 = (torch.empty(num_groups, n, k, dtype=torch.float8_e4m3fn, device=a.device), 
             torch.empty((num_groups, (n + 127) // 128, k // 128), device=a.device, dtype=torch.float))
    indices = []
    for i in range(len(b_list)):
        b_fp8[0][i], b_fp8[1][i] = per_block_cast_to_fp8(b_list[i])
        indices.extend([i] * m_split[i])
    indices = torch.tensor(indices, dtype=torch.int32, device=a.device)
    if out is None:
        out = torch.empty(sum(m_split), n, device=a.device, dtype=torch.bfloat16)
    deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(a_fp8, b_fp8, out)
    return out


 
class _DeepLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, weight, bias):
        ctx.inp_shape = inp.shape
        inp = inp.view(-1, inp.size(-1))
        ctx.save_for_backward(inp, weight, bias)
        out = deep_matmul(inp, weight)
        if bias is not None:
            out += bias
        return out.view(*ctx.inp_shape[:-1], -1)
    
    @staticmethod
    def backward(ctx, grad_outputs):
        inp, weight, bias = ctx.saved_tensors
        grad_outputs = grad_outputs.view(-1, grad_outputs.size(-1))
        dbias = None
        dinp, dweight, dbias = None, None, None
        
        dweight = deep_matmul(grad_outputs.T, inp.T)
        if inp.requires_grad:
            dinp = deep_matmul(grad_outputs, weight.T)
        
        if bias is not None:
            dbias = grad_outputs.sum(0)
        return dinp, dweight, dbias
    


class DeepLinear(torch.nn.Linear):
    def forward(self, input):
        return _DeepLinear.apply(input, self.weight, self.bias)
        

    