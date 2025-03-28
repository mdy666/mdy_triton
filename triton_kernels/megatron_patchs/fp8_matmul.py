import torch
import triton
import triton.language as tl
from typing import Tuple, List
import deep_gemm


# @triton.autotune(
#         [triton.Config({'BLOCK_M': bsm}, num_stages=ns, num_warps=nw)
#                 for bsm in [16, 32, 64]
#                 for ns in [1, 2, 4]
#                 for nw in [4, 8]
#                 ], key=['M', 'N'])
# @triton.jit
# def _per_token_cast_to_fp8_kernel(X, 
#                                   Y, 
#                                   S, 
#                                   YT,
#                                   ST,
#                                   stride_xm, 
#                                   stride_xn,
#                                   M: tl.constexpr, 
#                                   N: tl.constexpr, 
#                                   PM: tl.constexpr,
#                                   PN: tl.constexpr,
#                                   OUTPUT_T: tl.constexpr,
#                                   BLOCK_M: tl.constexpr=32, 
#                                   BLOCK_N: tl.constexpr=128,
#                                         ):
#     pid_m = tl.program_id(axis=0)
#     pid_n = tl.program_id(axis=1)
#     start_m = pid_m * BLOCK_M
#     start_n = pid_n * BLOCK_N

#     order = (stride_xm > stride_xn, stride_xm < stride_xn)
#     x_ptrs = tl.make_block_ptr(X, (M, N), (stride_xm, stride_xn), (start_m, start_n), (BLOCK_M, BLOCK_N), order)
#     y_ptrs = tl.make_block_ptr(Y, (PM, PN), (PN, 1), (start_m, start_n), (BLOCK_M, BLOCK_N), (1, 0))
#     x=tl.load(x_ptrs, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
#     x_max = tl.clamp(tl.max(tl.abs(x), -1), min=0, max=1e4) + 0.00000001
#     scale = x_max / 448
#     y = x / scale[:, None]
#     tl.store(y_ptrs, y.to(y_ptrs.type.element_ty))
#     tl.store(S + pid_m * BLOCK_M + tl.arange(0, BLOCK_M) + pid_n * PM, scale)

#     if OUTPUT_T:
#         x_max = tl.clamp(tl.max(tl.abs(x), 0), min=0, max=1e4) + 0.00000001
#         scale = x_max / 448
#         y = x / scale[None, :]
#         y_ptrs = tl.make_block_ptr(YT, (PM, PN), (1, PN), (start_m, start_n), (BLOCK_M, BLOCK_N), (0, 1))
#         tl.store(y_ptrs, y.to(y_ptrs.type.element_ty))
#         tl.store(ST + pid_n * BLOCK_N + tl.arange(0, BLOCK_N) + pid_m * PN, scale)


# @triton.autotune(
#         [triton.Config({'BLOCK_M': bsm}, num_stages=ns, num_warps=nw)
#                 for bsm in [16, 32, 64]
#                 for ns in [1, 2, 4]
#                 for nw in [4, 8]
#                 ], key=['M', 'N'])
@triton.jit
def _per_token_cast_to_fp8_kernel(X, 
                                  Y, 
                                  S, 
                                  YT,
                                  ST,
                                  stride_xm, 
                                  stride_xn,
                                  M: tl.constexpr, 
                                  N: tl.constexpr, 
                                  PM: tl.constexpr,
                                  PN: tl.constexpr,
                                  OUTPUT_T: tl.constexpr,
                                  BLOCK_M: tl.constexpr=32, 
                                  BLOCK_N: tl.constexpr=128,
                                        ):
    pid_m = tl.cast(tl.program_id(axis=0), tl.int64)
    pid_n = tl.cast(tl.program_id(axis=1), tl.int64)
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    x = tl.load(X + off_m[:, None] * stride_xm + off_n[None, :] * stride_xn, mask=(off_m[:, None] < M) & (off_n[None, :] < N), other=0.).to(tl.float32)
    x_max = tl.clamp(tl.max(tl.abs(x), -1), min=0, max=1e4) + 0.00000001
    scale = x_max / 448
    y = x / scale[:, None]
    tl.store(Y + off_m[:, None] * PN + off_n[None, :], y)
    tl.store(S + off_m + pid_n * PM, scale)

    if OUTPUT_T:
        x_max = tl.clamp(tl.max(tl.abs(x), 0), min=0, max=1e4) + 0.00000001
        scale = x_max / 448
        y = x / scale[None, :]
        tl.store(YT + off_m[:, None] + off_n[None, :] * PM, y)
        tl.store(ST + off_n + pid_m * PN, scale)


def per_token_cast_to_fp8(x: torch.Tensor, output_transpose=False) -> Tuple[torch.Tensor, torch.Tensor]:
    BLOCK_N = 128
    M, N = x.shape

    _B = triton.cdiv(N, 256)
    PN = _B * 256
    B = _B * 2
    if not output_transpose:
        A = triton.cdiv(M, 4)
        PM = A * 4
        y = torch.empty(PM, PN, device=x.device, dtype=torch.float8_e4m3fn)
        s = torch.empty(B, PM, dtype=torch.float32, device=x.device)
        y_t, s_t = None, None
        kwargs = {'BLOCK_M': 16, 'BLOCK_N': BLOCK_N, 'num_warps':4, 'num_stages':4}
    else:
        A = triton.cdiv(M, 128)
        PM = A * 128
        y = torch.empty(PM, PN, device=x.device, dtype=torch.float8_e4m3fn)
        s = torch.empty(B, PM, dtype=torch.float32, device=x.device)
        y_t = torch.empty(PN, PM, device=x.device, dtype=torch.float8_e4m3fn)
        s_t = torch.empty(A, PN, dtype=torch.float32, device=x.device)
        kwargs = {'BLOCK_M': 128, 'BLOCK_N': BLOCK_N, 'num_warps':4, 'num_stages':4}
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), B)
    _per_token_cast_to_fp8_kernel[grid](x, 
                                        y, 
                                        s, 
                                        y_t, 
                                        s_t,
                                        *x.stride(),
                                        M,
                                        N,
                                        PM,
                                        PN,
                                        output_transpose,
                                        **kwargs,
                                        )
    if not output_transpose:
        return y, s.transpose(0, 1)
    return y, s.transpose(0, 1), y_t, s_t.transpose(0, 1)


# @triton.autotune(
#         [triton.Config({}, num_stages=ns, num_warps=nw)
#                 for ns in [1, 2, 3, 4, 5]
#                 for nw in [4, 8]
#                 ], key=['M', 'N']
# )
# @triton.jit
# def _per_block_cast_to_fp8_kernel(X, 
#                          Y, 
#                          S, 
#                          YT, 
#                          ST,
#                          stride_xm,
#                          stride_xn,
#                          M:tl.constexpr, 
#                          N:tl.constexpr, 
#                          PM:tl.constexpr,
#                          PN:tl.constexpr,
#                          OUTPUT_T:tl.constexpr,
#                          BLOCK_M: tl.constexpr=128, 
#                          BLOCK_N: tl.constexpr=128,
#                             ):
#     pid_m = tl.program_id(axis=0)
#     pid_n = tl.program_id(axis=1)
#     start_m = pid_m * BLOCK_M
#     start_n = pid_n * BLOCK_N

#     # order = (stride_xm > stride_xn, stride_xm < stride_xn)
#     x_ptrs = tl.make_block_ptr(X, (M, N), (stride_xm, stride_xn), (pid_m * BLOCK_M, pid_n * BLOCK_N), (BLOCK_M, BLOCK_N), (0,1))
#     # x = tl.load(X + (start_m + tl.arange(0, BLOCK_M))[:, None] * stride_xm + (start_n + tl.arange(0, BLOCK_N))[None, :] * stride_xn)
#     y_ptrs = tl.make_block_ptr(Y, (PM, PN), (PN, 1), (pid_m * BLOCK_M, pid_n * BLOCK_N), (BLOCK_M, BLOCK_N), (1, 0))
#     x = tl.load(x_ptrs, boundary_check=(0, 1)).to(tl.float32)
#     x_max = tl.clamp(tl.max(tl.abs(x)), min=0, max=1e4) + 0.00000001
#     scale = x_max / 448
#     y = x / scale
#     tl.store(y_ptrs, y.to(y_ptrs.type.element_ty))
#     tl.store(S + pid_m * tl.num_programs(1) + pid_n, scale)
#     if OUTPUT_T:
#         yt_ptrs = tl.make_block_ptr(YT, (PM, PN), (1, PM), (pid_m * BLOCK_M, pid_n * BLOCK_N), (BLOCK_M, BLOCK_N), (0, 1))
#         tl.store(yt_ptrs, y.to(y_ptrs.type.element_ty))
#         tl.store(ST + pid_m + pid_n * tl.num_programs(0), scale)

# @triton.autotune(
#         [triton.Config({'BLOCK_M': bsm}, num_stages=ns, num_warps=nw)
#                 for bsm in [16, 32, 64]
#                 for ns in [1, 2, 4]
#                 for nw in [4, 8]
#                 ], key=['M', 'N'])
@triton.jit
def _group_per_token_cast_to_fp8_kernel(X, 
                                  Y, 
                                  S, 
                                  YT,
                                  ST,
                                  stride_xm, 
                                  stride_xn,
                                  M: tl.constexpr, 
                                  N: tl.constexpr, 
                                  TRUE_M: tl.constexpr,
                                  BLOCK_M: tl.constexpr=32, 
                                  BLOCK_N: tl.constexpr=128,
                                        ):
    pid_m = tl.cast(tl.program_id(axis=0), tl.int64)
    pid_n = tl.cast(tl.program_id(axis=1), tl.int64)
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    x = tl.load(X + off_m[:, None] * stride_xm + off_n[None, :] * stride_xn, mask=(off_m[:, None] < M) & (off_n[None, :] < N), other=0.).to(tl.float32)
    x_max = tl.clamp(tl.max(tl.abs(x), -1), min=0, max=1e4) + 0.00000001
    scale = x_max / 448
    y = x / scale[:, None]
    tl.store(Y + off_m[:, None] * N + off_n[None, :], y)
    tl.store(S + off_m + pid_n * TRUE_M, scale)

    x_max = tl.clamp(tl.max(tl.abs(x), 0), min=0, max=1e4) + 0.00000001
    scale = x_max / 448
    y = x / scale[None, :]
    tl.store(YT + off_m[:, None] + off_n[None, :] * M, y)
    tl.store(ST + off_n + pid_m * N, scale)

def group_per_token_cast_to_fp8(x_all: torch.Tensor, m_splits) -> Tuple[torch.Tensor, torch.Tensor]:
    BLOCK_N = 128
    BLOCK_M = 128
    y_all = torch.empty_like(x_all, dtype=torch.float8_e4m3fn)
    s_all = torch.empty(x_all.size(1) // BLOCK_N, x_all.shape[0], dtype=torch.float32, device=x_all.device)
    yt_all = []
    st_all = []
    for x,y,s in zip(x_all.split(m_splits, 0), y_all.split(m_splits, 0), s_all.split(m_splits, 1)):
        M, N = x.shape
        assert M % 256 == 0 and N % 256 == 0
        A = M // BLOCK_M
        y_t = torch.empty(N, M, device=x.device, dtype=torch.float8_e4m3fn)
        s_t = torch.empty(A, N, dtype=torch.float32, device=x.device)
        kwargs = {'BLOCK_M': BLOCK_M, 'BLOCK_N': BLOCK_N, 'num_warps':4, 'num_stages':4}
        
        grid = (M//BLOCK_M, N//BLOCK_N)
        _group_per_token_cast_to_fp8_kernel[grid](x, 
                                            y, 
                                            s, 
                                            y_t, 
                                            s_t,
                                            *x.stride(),
                                            M,
                                            N,
                                            x_all.size(0),
                                            **kwargs,
                                            )
        yt_all.append(y_t)
        st_all.append(s_t.transpose(0, 1))
    return y_all, s_all.transpose(0,1), yt_all, st_all


# @triton.autotune(
#         [triton.Config({}, num_stages=ns, num_warps=nw)
#                 for ns in [1, 2, 4]
#                 for nw in [1, 2, 4, 8, 16]
#                 ], key=['M', 'N']
# )
@triton.jit
def _per_block_cast_to_fp8_kernel(X, 
                         Y, 
                         S, 
                         YT, 
                         ST,
                         stride_xm,
                         stride_xn,
                         M:tl.constexpr, 
                         N:tl.constexpr, 
                         PM:tl.constexpr,
                         PN:tl.constexpr,
                         OUTPUT_N:tl.constexpr,
                         OUTPUT_T:tl.constexpr,
                         BLOCK_M: tl.constexpr=128, 
                         BLOCK_N: tl.constexpr=128,
                            ):

    pid_m = tl.cast(tl.program_id(axis=0), tl.int64)
    pid_n = tl.cast(tl.program_id(axis=1), tl.int64)
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    x = tl.load(X + off_m[:, None] * stride_xm + off_n[None, :] * stride_xn, mask=(off_m[:, None] < M) & (off_n[None, :] < N), other=0.).to(tl.float32)
    x_max = tl.clamp(tl.max(tl.abs(x)), min=0, max=1e4) + 0.00000001
    scale = x_max / 448
    y = x / scale

    if OUTPUT_N:
        tl.store(Y + off_m[:, None] * PN + off_n[None, :], y)
        tl.store(S + pid_m * tl.num_programs(1) + pid_n, scale)

    if OUTPUT_T:
        tl.store(YT + off_m[:, None] + off_n[None, :] * PM, y)
        tl.store(ST + pid_m + pid_n * tl.num_programs(0), scale)

# @triton.autotune(
#         [triton.Config({}, num_stages=ns, num_warps=nw)
#                 for ns in [1, 2, 4]
#                 for nw in [1, 2, 4, 8, 16]
#                 ], key=['PM', 'PN']
# )
@triton.jit
def _group_per_block_cast_to_fp8_kernel(P, 
                         Y, 
                         S, 
                         YT, 
                         ST,
                         stride_xm,
                         stride_xn,
                         NUM_GEMMS,
                         M:tl.constexpr, 
                         N:tl.constexpr, 
                         PM:tl.constexpr,
                         PN:tl.constexpr,
                         OUTPUT_T:tl.constexpr,
                         BLOCK_M: tl.constexpr=128, 
                         BLOCK_N: tl.constexpr=128,
                            ):
    x_id = tl.cast(tl.program_id(axis=0), tl.int64)
    pid_m = tl.cast(tl.program_id(axis=1), tl.int64)
    pid_n = tl.cast(tl.program_id(axis=2), tl.int64)
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    
    X = tl.load(P + x_id).to(tl.pointer_type(tl.bfloat16))
    x = tl.load(X + off_m[:, None] * stride_xm + off_n[None, :] * stride_xn, mask=(off_m[:, None] < M) & (off_n[None, :] < N), other=0.).to(tl.float32)
    x_max = tl.clamp(tl.max(tl.abs(x)), min=0, max=1e4) + 0.00000001
    scale = x_max / 448
    y = x / scale
    tl.store(Y + x_id * (PM * PN // NUM_GEMMS) + off_m[:, None] * PN + off_n[None, :], y)
    tl.store(S + x_id * tl.num_programs(1) * tl.num_programs(0) + pid_m * tl.num_programs(1) + pid_n, scale)

    if OUTPUT_T:
        tl.store(YT + x_id * (PM * PN // NUM_GEMMS) + off_m[:, None] + off_n[None, :] * (PM // NUM_GEMMS), y)
        tl.store(ST + x_id * tl.num_programs(1) * tl.num_programs(0) + pid_m + pid_n * tl.num_programs(0), scale)

def per_block_cast_to_fp8(x: torch.Tensor, output_normal=True, output_transpose=False) -> Tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(x, List):
        x = [x]
    num_gemms = len(x)

    device = x[0].device if isinstance(x, torch.Tensor) else x[0].device

    BLOCK_M = 128
    BLOCK_N = 128

    M, N = x[0].shape
    BA = 256 if output_transpose else 128
    BB = 256 if output_normal else 128
    _A, _B = triton.cdiv(M, BA), triton.cdiv(N, BB)
    PM, PN = BA * _A, BB * _B
    B = _B * BB // 128
    A = _A * BA // 128

    y, s = None, None
    if output_normal:
        y = torch.empty(num_gemms, PM, PN, 
                        device=device, dtype=torch.float8_e4m3fn)
        s = torch.empty(num_gemms, A, B, dtype=torch.float32, device=device)

    y_t, s_t = None, None
    if output_transpose:
        y_t = torch.empty(num_gemms, PN, PM, 
                    device=device, dtype=torch.float8_e4m3fn)
        s_t = torch.empty(num_gemms, B, A, dtype=torch.float32, device=device)

    grid = (A, B)
    kwargs = {'BLOCK_M': BLOCK_M, 'BLOCK_N': BLOCK_N, 'num_warps':16, 'num_stages':1}    
    for x_idx in range(num_gemms):
        _per_block_cast_to_fp8_kernel[grid](x[x_idx], 
                                y[x_idx] if y is not None else None, 
                                s[x_idx] if s is not None else None, 
                                y_t[x_idx] if y_t is not None else None, 
                                s_t[x_idx] if s_t is not None else None,
                                *x[x_idx].stride(),
                                M, 
                                N, 
                                PM, 
                                PN, 
                                output_normal,
                                output_transpose,
                                    **kwargs,
                                    )

    def post_process(t):
        if num_gemms == 1:
            return t.squeeze(0)
        return t

    out = ()
    if output_normal:
        out += (post_process(y), post_process(s))
    if output_transpose:
        out += (post_process(y_t), post_process(s_t))
    return out 

def fp8_matmul(qa, sa, qb, sb, out=None):
    if out is None:
        out = torch.empty(qa.size(0), qb.size(0), device=qa.device, dtype=torch.bfloat16)
    deep_gemm.gemm_fp8_fp8_bf16_nt((qa, sa), (qb, sb), out)
    return out

def fp8_grouped_matmul(qa, sa, qb, sb, m_splits, out=None):
    n = qa.size(0)
    indices = torch.empty(n, dtype=torch.int32, device="cuda")
    start = 0
    for i in range(len(m_splits)):
        size = m_splits[i]
        indices[start:start+size] = i
        start += size

    if out is None:
        out = torch.empty(n, qb.size(1), device=qa.device, dtype=torch.bfloat16)

    deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous((qa, sa), (qb, sb), out, indices)
    return out
