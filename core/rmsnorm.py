import torch
import triton
import triton.language as tl

@triton.jit
def _rmsnorm_fwd(X, Y, W, RMS_STD, eps,
                 stride_n, stride_d,
                 BLOCK_N:tl.constexpr,
                 N: tl.constexpr):
    start_n = tl.program_id(0)
    offset = start_n * stride_n
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x_ptrs = X + offset + cols
    y_ptrs = Y + offset + cols
    w_ptrs = W + cols

    x = tl.load(x_ptrs, mask=mask, other=0.).to(tl.float32)
    w = tl.load(w_ptrs, mask=mask, other=0.)

    rms_std = tl.sqrt(tl.sum(x * x) / N + eps)
    x_hat = x / rms_std
    y = x_hat.to(w.dtype) * w

    tl.store(y_ptrs, y, mask=mask)
    tl.store(RMS_STD+start_n, rms_std)

@triton.jit
def _rmsnorm_bwd_dx_fused(DX, DY, DW, X, W, RMS_STD, Lock,
                 stride_n, stride_d,
                 BLOCK_N:tl.constexpr, GROUP_SIZE: tl.constexpr, 
                 N: tl.constexpr):
    start_n = tl.program_id(0)
    lock_id = start_n % GROUP_SIZE
    Lock += lock_id
    Count = Lock + GROUP_SIZE
    offset = start_n * stride_n
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x_ptrs = X + offset + cols
    w_ptrs = W + cols
    dx_ptrs = DX + offset + cols
    dy_ptrs = DY + offset + cols
    dw_ptrs = DW + lock_id * N + cols
    
    x = tl.load(x_ptrs, mask=mask, other=0.).to(tl.float32)
    w = tl.load(w_ptrs, mask=mask, other=0.)
    dy = tl.load(dy_ptrs, mask=mask, other=0.).to(tl.float32)
    rms_std = tl.load(RMS_STD+start_n)

    x_hat = x / rms_std
    wdy = w * dy
    dx = (wdy - (x_hat / N) * tl.sum(x_hat * wdy)) / rms_std
    tl.store(dx_ptrs, dx, mask=mask)

    partial_dw = x_hat * dy
    while tl.atomic_cas(Lock, 0, 1) == 1:
        pass
    count = tl.load(Count)
    # First store doesn't accumulate
    if count == 0:
        tl.atomic_xchg(Count, 1)
    else:
        partial_dw += tl.load(dw_ptrs, mask=mask, other=0.)
    tl.store(dw_ptrs, partial_dw, mask=mask)
    tl.atomic_xchg(Lock, 0)

@triton.jit
def _rmsnorm_bwd_dw(PART_DW, DW,
                 stride_m, stride_d,
                 BLOCK_NN:tl.constexpr, GROUP_SIZE: tl.constexpr, 
                 N: tl.constexpr):
    group_id = tl.program_id(0)
    offset_nn = group_id*BLOCK_NN

    partial_dw_ptrs = tl.make_block_ptr(
        base=PART_DW,
        shape=(GROUP_SIZE, N),
        strides=(stride_m, stride_d),
        offsets=(0, offset_nn),
        block_shape=(GROUP_SIZE, BLOCK_NN),
        order=(1,0),
    )
    
    partial_dw = tl.load(partial_dw_ptrs, boundary_check=(1,), padding_option='zero').to(tl.float32)
    dw = tl.sum(partial_dw, 0)
    tl.store(DW + offset_nn + tl.arange(0, BLOCK_NN), dw, mask=(offset_nn+tl.arange(0, BLOCK_NN)) < N)

class _TritronRMSNorm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, hidden_state, weight, eps):
        input_shape = hidden_state.shape
        output = torch.empty_like(hidden_state)
        hidden_state = hidden_state.reshape(-1, input_shape[-1])
        M,N = hidden_state.shape
        BLOCK_N = triton.next_power_of_2(N)
        rms_std = torch.empty(M, dtype=torch.float32, device=hidden_state.device)
        
        num_warps=8
        num_stages=1
        _rmsnorm_fwd[(M, )](hidden_state, output, weight, rms_std, eps,
                            *hidden_state.stride(),
                            BLOCK_N,
                            N,
                            num_warps=num_warps, num_stages=num_stages)
        ctx.save_for_backward(hidden_state, weight, rms_std)
        ctx.input_shape = input_shape
        ctx.BLOCK_N = BLOCK_N
        ctx.num_warps = num_warps
        ctx.num_stages = num_stages
        return output
    
    @staticmethod
    def backward(ctx, dy):
        # dy = dy.contiguous()
        hidden_state, weight, rms_std = ctx.saved_tensors
        hidden_state = hidden_state.reshape(-1, hidden_state.shape[-1])
        M,N = hidden_state.shape
        input_shape = ctx.input_shape
        N = input_shape[-1]
        GROUP_SIZE = 64
        if N <= 8192: GROUP_SIZE = 64
        if N <= 4096: GROUP_SIZE = 128
        if N <= 1024: GROUP_SIZE = 256
        # GROUP_SIZE = min(1024, triton.next_power_of_2(M))
        BLOCK_NN = min(128, triton.next_power_of_2(N))

        dw = torch.empty_like(weight)
        dx = torch.empty_like(dy)
        partial_dw = torch.zeros((GROUP_SIZE, input_shape[-1]), device=weight.device, dtype=weight.dtype)
        lock = torch.zeros(GROUP_SIZE*2, device=weight.device, dtype=torch.int32)
        
        _rmsnorm_bwd_dx_fused[(M,)](dx, dy, partial_dw, hidden_state, weight, rms_std, lock,
                 *hidden_state.stride(),
                 ctx.BLOCK_N, GROUP_SIZE, 
                 N,
                 num_warps=ctx.num_warps, num_stages=ctx.num_stages)
        # print(dy[0][0][:8])
        grid = lambda meta: (triton.cdiv(N, BLOCK_NN), )
        _rmsnorm_bwd_dw[grid](partial_dw, dw,
                 *partial_dw.stride(),
                 BLOCK_NN, GROUP_SIZE, 
                 N,
                 num_warps=ctx.num_warps, num_stages=ctx.num_stages)
        return dx, dw, None

triton_rmsnorm = _TritronRMSNorm.apply
class TritonRMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, hidden_state):
        return triton_rmsnorm(hidden_state, self.weight, self.eps)