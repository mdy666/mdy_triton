import torch
import triton
import triton.language as tl


@triton.jit
def _fused_silu_fwd(
        UP, GATE, Y,
        M, N,
        stride_m, stride_n,  #
        BLOCK_SIZE_N: tl.constexpr, #
):
    pid = tl.program_id(axis=0)
    offset = pid * stride_m

    cols = tl.arange(0, BLOCK_SIZE_N)
    for start_n in range(0, N, BLOCK_SIZE_N):
        new_cols = cols + start_n
        mask = new_cols < N
        ptrs = offset + new_cols
        up = tl.load(UP+ptrs, mask=mask, other=0.)
        dtype = up.dtype
        up = up.to(tl.float32)
        gate = tl.load(GATE+ptrs, mask=mask, other=0.).to(tl.float32)
        act = gate * tl.sigmoid(gate)
        y = act * up
        tl.store(Y+ptrs, y.to(dtype), mask=mask)

@triton.jit
def _fused_silu_bwd_dupgate(UP, GATE, 
                               DY, DUP, DGATE,
                               stride_m, stride_n,
                               N, BLOCK_N: tl.constexpr,
                                ):
    pid = tl.program_id(0)
    offset = pid * stride_m

    cols = tl.arange(0, BLOCK_N)
    for start_n in range(0, N, BLOCK_N):
        new_cols = cols + start_n
        mask = new_cols < N
        ptrs = offset + new_cols
        
        dy = tl.load(DY+ptrs, mask=mask, other=0.)
        dtype = dy.dtype
        dy = dy.to(tl.float32)
        gate = tl.load(GATE+ptrs, mask=mask, other=0.).to(tl.float32)
        up = tl.load(UP+ptrs, mask=mask, other=0.).to(tl.float32)
        gate_sigmoid = tl.sigmoid(gate)
        act = gate_sigmoid * gate
        dup = act * dy
        dact = up * dy
        dgate = (gate_sigmoid + act * (1-gate_sigmoid)) * dact
        tl.store(DUP+ptrs, dup.to(dtype), mask=mask)
        tl.store(DGATE+ptrs, dgate.to(dtype), mask=mask)

class _FusedSiLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, up, gate):
        up = up.view(-1, up.shape[-1])
        M, N = up.shape
        y = torch.empty_like(gate)
        BLOCK_SIZE_N = triton.next_power_of_2(N)
        BLOCK_SIZE_N = min(4096, BLOCK_SIZE_N)
        num_warps = 4
        num_stages = 4
        _fused_silu_fwd[(M,)](
            up, gate, y, 
            M, N,  #
            *up.stride(),  #
            BLOCK_SIZE_N,
            num_warps=num_warps, num_stages=num_stages, 
        )
        ctx.infos = (M, N, BLOCK_SIZE_N, *up.stride())
        ctx.save_for_backward(up, gate)
        ctx.num_warps = num_warps
        ctx.num_stages = num_stages
        return y
    
    @staticmethod
    def backward(ctx, dy):
        M, N, BLOCK_SIZE_N, stride_m, stride_n = ctx.infos
        up, gate = ctx.saved_tensors

        dup = torch.empty_like(gate)
        dgate = torch.empty_like(gate)
        _fused_silu_bwd_dupgate[(M,)](up, gate,
                                   dy, dup, dgate,
                                   stride_m, stride_n,
                                   N, BLOCK_SIZE_N, 
                                   num_warps=ctx.num_warps, num_stages=ctx.num_stages)

        # up, gate = ctx.saved_tensors
        # up = up.view(*gate.shape)
        # # # print(dy.stride(), up.stride(), gate.stride())
        # # # print(up.shape, gate.shape)
        # act = torch.nn.functional.silu(gate)
        # dup = act * dy
        # dact = up * dy
        # gate_sigmoid = torch.nn.functional.sigmoid(gate)
        # dgate = (gate_sigmoid + act * (1-gate_sigmoid)) * dact
        return dup, dgate



def up_gate_silu(up, gate):
    return up * torch.nn.functional.silu(gate)

triton_fused_up_gate_silu = _FusedSiLU.apply