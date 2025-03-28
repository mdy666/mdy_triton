import triton
import triton.language as tl
import torch

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

# @triton.autotune([triton.Config({"BLOCK_N":bn}, num_warps=nw, num_stages=ns)
#                   for bn in [2048, 4096, 8192]
#                   for nw in [1, 2, 4, 8, 16]
#                   for ns in [1, 2, 4]],
#                   key=["M", "N"])
@triton.jit
def _add_grad_kernel(X,
                     Y,
                     N,
                     BLOCK_N:tl.constexpr=1024,
                     ):
    off_n = tl.cast(tl.program_id(0), tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = off_n < N,

    x = tl.load(X + off_n, mask=mask, other=0.).to(tl.float32)
    y = tl.load(Y + off_n, mask=mask, other=0.).to(tl.float32)
    tl.store(X + off_n, x+y, mask=mask)


def add_grad(x:torch.Tensor, y):
    assert x.is_contiguous() and y.is_contiguous() and x.shape == y.shape
    N = x.numel()

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)
    kwargs = {"BLOCK_N": min(triton.next_power_of_2(N), 2048), "num_warps":8, "num_stages":2}
    _add_grad_kernel[grid](x, y, N, **kwargs)
    return x