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
@triton.jit
def _quant_grad_output_kernel(X, 
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

    x_max = tl.clamp(tl.max(tl.abs(x), 0), min=0, max=1e4) + 0.00000001
    scale = x_max / 448
    y = x / scale[None, :]
    tl.store(YT + off_m[:, None] + off_n[None, :] * PM, y)
    tl.store(ST + off_n + pid_m * PN, scale)


def quant_grad_output(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    BLOCK_N = 128
    M, N = x.shape

    _B = triton.cdiv(N, 256)
    PN = _B * 256
    B = _B * 2

    _A = triton.cdiv(M, 256)
    PM = _A * 256
    A = _A * 2
    y = torch.empty(PM, PN, device=x.device, dtype=torch.float8_e4m3fn)
    s = torch.empty(B, PM, dtype=torch.float32, device=x.device)
    y_t = torch.empty(PN, PM, device=x.device, dtype=torch.float8_e4m3fn)
    s_t = torch.empty(A, PN, dtype=torch.float32, device=x.device)
    kwargs = {'BLOCK_M': 128, 'BLOCK_N': BLOCK_N, 'num_warps':4, 'num_stages':4}
    
    grid = (A, B)
    _quant_grad_output_kernel[grid](x, 
                                        y, 
                                        s, 
                                        y_t, 
                                        s_t,
                                        *x.stride(),
                                        M,
                                        N,
                                        PM,
                                        PN,
                                        **kwargs,
                                        )
    return y, s.transpose(0, 1), y_t, s_t.transpose(0, 1)



# @triton.autotune(
#         [triton.Config({'BLOCK_M': bsm}, num_stages=ns, num_warps=nw)
#                 for bsm in [128]
#                 for ns in [1, 2, 4]
#                 for nw in [1, 2, 4, 8]
#                 ], key=['M', 'N'])
@triton.jit
def _quant_input_kernel(X, 
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
                ONLY_FORWARD: tl.constexpr,
                BLOCK_M: tl.constexpr=128, 
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

    if not ONLY_FORWARD:
        x_max = tl.max(x_max)
        scale = x_max / 448
        y = x / scale
        tl.store(YT + off_m[:, None] + off_n[None, :] * PM, y)
        tl.store(ST + pid_n * tl.num_programs(0) + pid_m, scale)

def quant_input(x: torch.Tensor, only_forward=True) -> Tuple[torch.Tensor, torch.Tensor]:
    BLOCK_N = 128
    M, N = x.shape
    _B = triton.cdiv(N, 256)
    PN = _B * 256
    B = _B * 2

    if only_forward:
        tma_align_factor = 4
        A = triton.cdiv(M, tma_align_factor)
        PM = A * tma_align_factor
        y = torch.empty(PM, PN, device=x.device, dtype=torch.float8_e4m3fn)
        s = torch.empty(B, PM, dtype=torch.float32, device=x.device)
        y_t, s_t = None, None
        kwargs = {'BLOCK_M': 16, 'BLOCK_N': BLOCK_N, 'num_warps':4, 'num_stages':4}
    else:
        _A = triton.cdiv(M, 256)
        PM = _A * 256
        A = _A * 2
        y = torch.empty(PM, PN, device=x.device, dtype=torch.float8_e4m3fn)
        s = torch.empty(B, PM, dtype=torch.float32, device=x.device)
        y_t = torch.empty(PN, PM, device=x.device, dtype=torch.float8_e4m3fn)
        s_t = torch.empty(B, A, dtype=torch.float32, device=x.device)
        kwargs = {'BLOCK_M': 128, 'BLOCK_N': BLOCK_N, 'num_warps':8, 'num_stages':1}
    
    grid = lambda meta: (triton.cdiv(PM, meta['BLOCK_M']), B)
    _quant_input_kernel[grid](x, 
                        y, 
                        s, 
                        y_t, 
                        s_t,
                        *x.stride(),
                        M,
                        N,
                        PM,
                        PN,
                        only_forward,
                        **kwargs,
                        )
    return y, s.transpose(0, 1), y_t, s_t



# @triton.autotune(
#         [triton.Config({}, num_stages=ns, num_warps=nw)
#                 for ns in [1, 2, 4]
#                 for nw in [1, 2, 4, 8, 16]
#                 ], key=['M', 'N']
# )
@triton.jit
def _quant_weight_kernel(X, 
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
                         ONLY_FORWARD:tl.constexpr,
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

    tl.store(Y + off_m[:, None] * PN + off_n[None, :], y)
    tl.store(S + pid_m * tl.num_programs(1) + pid_n, scale)

    if not ONLY_FORWARD:
        tl.store(YT + off_m[:, None] + off_n[None, :] * PM, y)
        tl.store(ST + pid_m + pid_n * tl.num_programs(0), scale)


def quant_weight(x: torch.Tensor, only_forward=True) -> Tuple[torch.Tensor, torch.Tensor]:


    device = x.device
    BLOCK_M = 128
    BLOCK_N = 128

    M, N = x.shape

    BA, BB = 128 if only_forward else 256, 256
    _A, _B = triton.cdiv(M, BA), triton.cdiv(N, BB)
    PM, PN = BA * _A, BB * _B
    B = _B * BB // 128
    A = _A * BA // 128


    y = torch.empty(PM, PN, 
                    device=device, dtype=torch.float8_e4m3fn)
    s = torch.empty(A, B, dtype=torch.float32, device=device)
    y_t, s_t = None, None
    if not only_forward:
        y_t = torch.empty(PN, PM, 
                    device=device, dtype=torch.float8_e4m3fn)
        s_t = torch.empty(B, A, dtype=torch.float32, device=device)

    grid = (A, B)
    kwargs = {'BLOCK_M': BLOCK_M, 'BLOCK_N': BLOCK_N, 'num_warps':8, 'num_stages':1}
    _quant_weight_kernel[grid](x, 
                            y, 
                            s, 
                            y_t, 
                            s_t,
                            *x.stride(),
                            M, 
                            N, 
                            PM, 
                            PN, 
                            only_forward,
                                **kwargs,
                                )

    return y, s, y_t, s_t

# @triton.autotune(
#         [triton.Config({}, num_stages=ns, num_warps=nw)
#                 for ns in [1, 2, 3, 4]
#                 for nw in [1, 2, 4, 8]
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
    x_max = tl.clamp(tl.max(tl.abs(x)), min=0, max=1e4) + 0.000001
    scale = x_max / MAX
    y = x / scale
    tl.store(Y + off_m[:, None] * stride_ym + off_n[None, :] * stride_yn, y, mask=mask)
    tl.store(S + pid_m * stride_sm + pid_n * stride_sn, scale)



# @triton.autotune([triton.Config({"BLOCK_N":bn}, num_warps=nw, num_stages=ns)
#                   for bn in [2048, 4096, 8192]
#                   for nw in [1, 2, 4, 8, 16]
#                   for ns in [1, 2, 4]],
#                   key=["M", "N"])
@triton.jit
def _add_bias_kernel(X,
                     B,
                     N,
                     BLOCK_N:tl.constexpr=1024,
                     ):
    off_n = tl.cast(tl.program_id(0), tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)
    row_id = tl.cast(tl.program_id(1), tl.int64)
    mask = off_n < N,

    bias = tl.load(B + off_n, mask=mask, other=0.).to(tl.float32)
    x = tl.load(X + row_id * N + off_n, mask=mask, other=0.).to(tl.float32)
    tl.store(X + row_id * N + off_n, x + bias, mask=mask)

def add_bias(x, b):
    inp_shape = x.shape
    x = x.view(-1, inp_shape[-1])
    M, N = x.shape

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]), M)
    kwargs = {"BLOCK_N": min(triton.next_power_of_2(N), 2048), "num_warps":8, "num_stages":2}
    _add_bias_kernel[grid](x, b, N, **kwargs)
    return x.view(*inp_shape)

def fp8_matmul(qa, sa, qb, sb, out=None):
    if out is None:
        out = torch.empty(qa.size(0), qb.size(0), device=qa.device, dtype=torch.bfloat16)
    deep_gemm.gemm_fp8_fp8_bf16_nt((qa, sa), (qb, sb), out)
    return out

class _DeepLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, is_grad_enable):
        ctx.inp_shape = input.shape
        input = input.view(-1, input.size(-1))
        # print(is_grad_enable)
        q_inp, s_inp, q_inp_t, s_inp_t = quant_input(input, only_forward=(not is_grad_enable) and (not input.requires_grad))
        q_weight, s_weight, q_weight_t, s_weight_t = quant_weight(weight, only_forward=not is_grad_enable)
        if is_grad_enable:
            ctx.saved = (q_inp_t, s_inp_t, weight, q_weight_t, s_weight_t, bias)
            ctx.requires_dgrad = input.requires_grad
        out = fp8_matmul(q_inp, s_inp, q_weight, s_weight)
        out.requires_grad_(True)
        if bias is not None:
            add_bias(out, bias)
        return out.view(*ctx.inp_shape[:-1], -1)
    
    @staticmethod
    def backward(ctx, grad_output):
        q_inp_t, s_inp_t, weight, q_weight_t, s_weight_t, bias = ctx.saved
        
        grad_output = grad_output.view(-1, grad_output.size(-1))
        q_grad_output, s_grad_output, q_grad_output_t, s_grad_output_t = quant_grad_output(grad_output)
        dgrad, wgrad, grad_bias = None, None, None
        if ctx.requires_dgrad:
            dgrad = fp8_matmul(q_grad_output, s_grad_output, q_weight_t, s_weight_t)
        if weight.requires_grad:
            wgrad = fp8_matmul(q_grad_output_t, s_grad_output_t, q_inp_t, s_inp_t)
        if bias is not None:
            grad_bias = grad_output.sum(0)
        return dgrad.view(*ctx.inp_shape), wgrad, grad_bias, None
    


class DeepLinear(torch.nn.Linear):
    def forward(self, input):
        return _DeepLinear.apply(input, self.weight, self.bias, torch.is_grad_enabled())
        

    