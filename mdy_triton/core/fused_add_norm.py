import torch
import triton
import triton.language as tl

@triton.jit
def _add_rmsnorm_fwd(X, OLD_RESIDUAL, W, NEW_RESIDUAL, Y, RMS_STD, eps,
                 row_stride,
                 N,
                 BLOCK_N:tl.constexpr):
    row_id = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row_id * row_stride + cols, mask=mask, other=0.).to(tl.float32)
    old_residual = tl.load(OLD_RESIDUAL + row_id * row_stride + cols, mask=mask, other=0.).to(tl.float32)
    new_residual = x + old_residual
    tl.store(NEW_RESIDUAL + row_id * row_stride + cols, new_residual, mask=mask)
    
    w = tl.load(W+cols, mask=mask, other=0.).to(tl.float32)

    rms_std = tl.sqrt(tl.sum(new_residual * new_residual) / N + eps)
    new_residual_hat = new_residual / rms_std
    y = new_residual_hat * w

    tl.store(Y + row_id * row_stride + cols, y, mask=mask)
    tl.store(RMS_STD + row_id, rms_std)

@triton.jit
def _add_rmsnorm_bwd_dx_fused(DX, DW, DOLD_RES, DNEW_RES, DY, NEW_RES, W, RMS_STD,
                            row_stride,
                            M, N,
                            BLOCK_N:tl.constexpr
                            ):
    
    start_id = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    dw = tl.zeros([BLOCK_N,], dtype=tl.float32)
    for row_id in range(start_id, M, tl.num_programs(0)):

        new_res = tl.load(NEW_RES + row_id * row_stride + cols, mask=mask, other=0.).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.).to(tl.float32)
        dy = tl.load(DY + row_id * row_stride + cols, mask=mask, other=0.).to(tl.float32)
        rms_std = tl.load(RMS_STD + row_id)

        new_res_hat = new_res / rms_std
        wdy = w * dy
        dnew_res = (wdy - (new_res_hat / N) * tl.sum(new_res_hat * wdy)) / rms_std
        dnew_res += tl.load(DNEW_RES + row_id * row_stride + cols, mask=mask, other=0.).to(tl.float32)
        tl.store(DX + row_id * row_stride + cols, dnew_res, mask=mask)
        tl.store(DOLD_RES + row_id * row_stride + cols, dnew_res, mask=mask)

        dw += new_res_hat * dy
    tl.store(DW + start_id * row_stride + cols, dw, mask=mask)



class _TritronFusedAddRMSNorm(torch.autograd.Function):

    @staticmethod
    def forward(ctx, hidden_state, old_residual, weight, eps):
        ctx.input_shape = hidden_state.shape
        output = torch.empty_like(hidden_state)
        new_residual = torch.empty_like(hidden_state)
        hidden_state = hidden_state.reshape(-1, ctx.input_shape[-1])
        M,N = hidden_state.shape
        BLOCK_N = triton.next_power_of_2(N)
        rms_std = torch.empty(M, dtype=torch.float32, device=hidden_state.device)
        
        _add_rmsnorm_fwd[(M,)](hidden_state, old_residual, weight, new_residual, output, rms_std, eps,
                            hidden_state.stride(0),
                            N,
                            BLOCK_N,
                            )
        ctx.save_for_backward(hidden_state, new_residual, weight, rms_std)
        return output, new_residual
    
    @staticmethod
    def backward(ctx, dy, dnew_res, *args):
        hidden_state, new_residual, weight, rms_std = ctx.saved_tensors
        M,N = hidden_state.shape


        BLOCK_N = min(128, triton.next_power_of_2(N))

        NUM_SMS = torch.cuda.get_device_properties('cuda').multi_processor_count
        dw = torch.empty(NUM_SMS, N, dtype=torch.float32, device=weight.device)
        dx = torch.empty_like(dy)
        dold_residual = torch.empty_like(dy)

        _add_rmsnorm_bwd_dx_fused[(NUM_SMS,)](dx, dw, dold_residual, dnew_res, dy, new_residual, weight, rms_std,
                                hidden_state.stride(0), 
                                M, N, BLOCK_N)
        return dx, dold_residual, dw, None

triton_fused_add_norm = _TritronFusedAddRMSNorm.apply
    
triton_fused_add_norm = _TritronFusedAddRMSNorm.apply
