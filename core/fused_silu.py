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
    for start_n in tl.range(0, N, BLOCK_SIZE_N):
        cols += start_n
        mask = cols < N
        ptrs = offset + cols
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
                               N, BLOCK_N: tl.constexpr
                               ):
    pid = tl.program_id(0)
    offset = pid * stride_m

    cols = tl.arange(0, BLOCK_N)
    for start_n in tl.range(0, N, BLOCK_N):
        cols += start_n
        ptrs = offset + cols
        mask = cols < N
        
        dy = tl.load(DY+ptrs, mask=mask, other=0.)
        dtype = dy.dtype
        gate = tl.load(GATE+ptrs, mask=mask, other=0.).to(tl.float32)
        up = tl.load(UP+ptrs, mask=mask, other=0.).to(tl.float32)

        act = gate * tl.sigmoid(gate)
        dup = act * dy
        dact = up * dy
        gate_neg_exp = tl.exp(-gate)
        tmp = 1 + gate_neg_exp
        fenzi =  tmp + gate * gate_neg_exp
        fenmu = tmp * tmp
        dgate = (fenzi / fenmu) * dact
        tl.store(DUP+ptrs, dup.to(dtype), mask=mask)
        tl.store(DGATE+ptrs, dgate.to(dtype), mask=mask)

class _FusedSiLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, up, gate):
        up = up.view(-1, up.shape[-1])
        M, N = up.shape
        y = torch.empty_like(gate)
        BLOCK_SIZE_N = triton.next_power_of_2(N)
        BLOCK_SIZE_N = min(1024, BLOCK_SIZE_N)
        num_warps = 8
        num_stages = 1
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

        return dup, dgate


def up_gate_silu(up, gate):
    return up * torch.nn.functional.silu(gate)

fused_up_gate_silu = _FusedSiLU.apply


def up_gate_silu(up, gate):
    return up * torch.nn.functional.silu(gate)

triton_fused_up_gate_silu = _FusedSiLU.apply