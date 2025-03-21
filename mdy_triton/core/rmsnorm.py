import torch
import triton
import triton.language as tl

@triton.autotune([triton.Config({}, num_warps=nw)
                  for nw in [1, 2, 4, 8, 16]],
                  key=['BLOCK_N'])
@triton.jit
def _rmsnorm_fwd(X, Y, W, RMS_STD, eps,
                 row_stride,
                 N,
                 BLOCK_N:tl.constexpr,):
    row_id = tl.cast(tl.program_id(0), tl.int64)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    
    x = tl.load(X + row_id * row_stride + cols, mask=mask, other=0.).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.).to(tl.float32)

    rms_std = tl.sqrt(tl.sum(x * x) / N + eps)
    x_hat = x / rms_std
    y = x_hat * w

    tl.store(Y + row_id * row_stride + cols, y, mask=mask)
    tl.store(RMS_STD+row_id, rms_std)

@triton.autotune([triton.Config({}, num_warps=nw)
                  for nw in [1, 2, 4, 8, 16]],
                  key=['BLOCK_N'])
@triton.jit
def _rmsnorm_bwd_dx_fused(DX, DY, DW, X, W, RMS_STD,
                        row_stride,
                        M, N, BLOCK_N:tl.constexpr):
    start_id = tl.cast(tl.program_id(0), tl.int64)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    dw = tl.zeros([BLOCK_N,], dtype=tl.float32)
    for row_id in range(start_id, M, tl.num_programs(0)):
        x = tl.load(X + row_id * row_stride + cols, mask=mask, other=0.).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.)
        dy = tl.load(DY + row_id * row_stride + cols, mask=mask, other=0.).to(tl.float32)
        rms_std = tl.load(RMS_STD+row_id)

        x_hat = x / rms_std
        wdy = w * dy
        dx = (wdy - (x_hat / N) * tl.sum(x_hat * wdy)) / rms_std
        tl.store(DX + row_id * row_stride + cols, dx, mask=mask)

        dw += x_hat * dy
    tl.store(DW + start_id * row_stride + cols, dw, mask=mask)


class _TritronRMSNorm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, hidden_state, weight, eps):
        ctx.input_shape = hidden_state.shape
        output = torch.empty_like(hidden_state)
        hidden_state = hidden_state.reshape(-1, ctx.input_shape[-1])
        M,N = hidden_state.shape
        BLOCK_N = triton.next_power_of_2(N)
        rms_std = torch.empty(M, dtype=torch.float32, device=hidden_state.device)
        _rmsnorm_fwd[(M, )](hidden_state, output, weight, rms_std, eps,
                            hidden_state.stride(0),
                            N,
                            BLOCK_N,
                            )
        ctx.save_for_backward(hidden_state, weight, rms_std)
        return output
    
    @staticmethod
    def backward(ctx, dy):
        # dy = dy.contiguous()
        hidden_state, weight, rms_std = ctx.saved_tensors
        hidden_state = hidden_state.reshape(-1, hidden_state.shape[-1])
        M,N = hidden_state.shape
        BLOCK_N = triton.next_power_of_2(N)

        NUM_SMS = torch.cuda.get_device_properties('cuda').multi_processor_count
        dw = torch.empty(NUM_SMS, N, dtype=torch.float32, device=weight.device)
        dx = torch.empty_like(dy)
        
        _rmsnorm_bwd_dx_fused[(NUM_SMS,)](dx, dy, dw, hidden_state, weight, rms_std, 
                 hidden_state.stride(0),
                 M, N, BLOCK_N
                 )
        dw = dw.sum(0).to(weight.dtype)
        return dx.view(*ctx.input_shape), dw, None

triton_rmsnorm = _TritronRMSNorm.apply