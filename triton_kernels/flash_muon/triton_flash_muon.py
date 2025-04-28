import triton
import triton.language as tl
import torch

# @triton.autotune([triton.Config({"BLOCK_M":bm, "BLOCK_N":bn, "BLOCK_K":bk, "GROUP_SIZE_M":gm}, num_warps=nw, num_stages=4)
#                   for bm in [32, 64]
#                   for bn in [32, 64]
#                   for bk in [64]
#                   for nw in [1,2,4,8]
#                   for gm in [8]
#                   ], key=["N"])
@triton.jit
def matmul_kernel(X, 
                  Y, 
                  N: tl.constexpr,
                  BLOCK_M:tl.constexpr=64,
                  BLOCK_N:tl.constexpr=64,
                  BLOCK_K:tl.constexpr=64,
                  GROUP_SIZE_M:tl.constexpr=16,
                  ):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(N, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    if (pid_m * BLOCK_M + BLOCK_M) <= (pid_n * BLOCK_N):
        return

    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    a_ptrs = X + off_m[:, None] * N + tl.arange(0, BLOCK_K)[None, :]
    # b_ptrs = X + off_n[None, :] * N + tl.arange(0, BLOCK_K)[:, None]
    b_ptrs = X + off_n[:, None] * N + tl.arange(0, BLOCK_K)[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, tl.cdiv(N, BLOCK_K)):
        b = tl.load(b_ptrs)
        a = tl.load(a_ptrs)
        # acc = tl.dot(a, b, acc)
        acc = tl.dot(a, tl.permute(b, (1, 0)), acc)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K

    y_ptrs1 = Y + off_m[:, None] * N + off_n[None, :]
    tl.store(y_ptrs1, acc)

    y_ptrs2 = Y + off_m[None, :] + off_n[:, None] * N
    tl.store(y_ptrs2, tl.permute(acc, (1,0)))

    # y_ptrs2 = Y + off_m[:, None] + off_n[None, :] * N
    # tl.store(y_ptrs2, acc)

def matmul(x, out=None):
    if out is None:
        out = torch.zeros_like(x)
    M, N = x.shape
    assert N % 128 == 0 and M % 128 == 0
    kwargs = {"BLOCK_M":128 if N>=4096 else 64, "BLOCK_N":128 if N>=4096 else 64, "BLOCK_K":64, "num_warps":4, "num_stages":4, "GROUP_SIZE_M":8}
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']), )
    matmul_kernel[grid](x, 
                        out, 
                        N,
                        **kwargs
                        )
    return out

