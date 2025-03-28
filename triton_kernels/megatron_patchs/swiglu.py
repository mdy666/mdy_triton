import torch
import triton
import triton.language as tl


@triton.jit
def _fused_silu_fwd(X, Y,
                    N, 
                    stride_m, stride_n,  #
                    BLOCK_SIZE_N: tl.constexpr, ORDER:tl.constexpr,
                    ):

    pid = tl.program_id(axis=0)
    x_offset = pid * stride_m
    y_offset = x_offset // 2
    X += x_offset
    Y += y_offset
    if ORDER == 'up-gate':
        up_offset = 0
        gate_offset = N
    else:
        up_offset = N
        gate_offset = 0

    cols = tl.arange(0, BLOCK_SIZE_N)
    for start_n in tl.range(0, N, BLOCK_SIZE_N):
        new_cols = cols + start_n
        mask = new_cols < N
        up = tl.load(X+new_cols+up_offset, mask=mask, other=0.)
        dtype = up.dtype
        up = up.to(tl.float32)
        gate = tl.load(X+new_cols+gate_offset, mask=mask, other=0.).to(tl.float32)
        act = gate * tl.sigmoid(gate)
        y = act * up
        tl.store(Y+new_cols, y.to(dtype), mask=mask)


@triton.jit
def _fused_silu_bwd_dupgate(X, 
                            DY, DX,
                            N,
                            stride_m, stride_n,
                            BLOCK_SIZE_N: tl.constexpr,ORDER:tl.constexpr
                            ):
    pid = tl.program_id(0)
    x_offset = pid * stride_m
    y_offset = x_offset // 2
    X += x_offset
    DX += x_offset
    DY += y_offset
    if ORDER == 'up-gate':
        up_offset = 0
        gate_offset = N
    else:
        up_offset = N
        gate_offset = 0

    cols = tl.arange(0, BLOCK_SIZE_N)
    for start_n in range(0, N, BLOCK_SIZE_N):
        new_cols = cols + start_n
        mask = new_cols < N
        
        dy = tl.load(DY+new_cols, mask=mask, other=0.)
        dtype = dy.dtype
        gate = tl.load(X+new_cols+gate_offset, mask=mask, other=0.).to(tl.float32)
        up = tl.load(X+new_cols+up_offset, mask=mask, other=0.).to(tl.float32)
        gate_sigmoid = tl.sigmoid(gate)
        act = gate_sigmoid * gate
        dup = act * dy
        dact = up * dy
        dgate = (gate_sigmoid + act * (1-gate_sigmoid)) * dact
        tl.store(DX+new_cols+up_offset, dup.to(dtype), mask=mask)
        tl.store(DX+new_cols+gate_offset, dgate.to(dtype), mask=mask)

class _FusedSiLUNoSplit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, order='up-gate'):
        '''
        input:
            x     : torch.Tensor, [bs, L, 2*D], the output of self.fc1(x) in MLP, contain the up and gate
            order : str, the order of the x, must be gate-up or up-gate, default up-gate
        
        output:
            y    : torch.tensor, [bs, L, D], the result of up * silu(gate)

        example:
          original code:
            x = self.fc1(hidden_states)
            up, gate = x.chunk(2, -1)
            act = silu(gate)
            y = act * up

          new code:
            x = self.fc1(hidden_states)
            y = fused_up_gate_silu_no_split(x)
        '''
        assert order in ['up-gate', 'gate-up'], "please indicate the order of input, up-gate or gate-up"
        input_shape = x.shape
        x = x.view(-1, input_shape[-1])
        M, N2 = x.shape
        N = N2 // 2
        y = torch.empty(*input_shape[:-1], N, device=x.device, dtype=x.dtype)
        BLOCK_SIZE_N = triton.next_power_of_2(N)
        BLOCK_SIZE_N = min(4096, BLOCK_SIZE_N)
        num_warps = 8
        num_stages = 4
        _fused_silu_fwd[(M,)](
            x, y, 
            N,  #
            *x.stride(),  #
            BLOCK_SIZE_N, order,
            num_warps=num_warps, num_stages=num_stages, 
        )
        ctx.infos = (M, N, BLOCK_SIZE_N, *x.stride(), order)
        ctx.input_shape = input_shape
        ctx.save_for_backward(x)
        ctx.num_warps = num_warps
        ctx.num_stages = num_stages
        return y
    
    @staticmethod
    def backward(ctx, dy):
        M, N, BLOCK_SIZE_N, stride_m, stride_n, order = ctx.infos
        # print(stride_m, stride_n)
        x, = ctx.saved_tensors

        dx = torch.empty(ctx.input_shape, device=dy.device, dtype=dy.dtype)
        _fused_silu_bwd_dupgate[(M,)](x,
                                   dy, dx,
                                   N,
                                   stride_m, stride_n,
                                   BLOCK_SIZE_N, order,
                                   num_warps=ctx.num_warps, num_stages=ctx.num_stages)

        return dx.view(*ctx.input_shape), None
    

def swiglu_impl(x, bias, fp8_input_store=False):
    return _FusedSiLUNoSplit.apply(x, 'gate-up')

 



