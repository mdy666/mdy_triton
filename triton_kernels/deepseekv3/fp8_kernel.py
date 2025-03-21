import torch
import triton
import triton.language as tl
from typing import Tuple


# @triton.autotune(
#         [triton.Config({'BLOCK_M': bsm}, num_stages=ns, num_warps=nw)
#                 for bsm in [16, 32, 64]
#                 for ns in [1, 2, 3, 4, 5]
#                 for nw in [4, 8]
#                 ], key=['M', 'N', 'BLOCK_N']
# )
@triton.jit
def _act_quant_block_kernel(x_ptr, y_ptr, s_ptr, 
                           stride_xm, stride_xn,
                           stride_ym, stride_yn,
                           stride_sm, stride_sk,
                           M, N, K, MAX, TRANSPOSE:tl.constexpr,
                           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                            ):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N
    x_block_ptrs = tl.make_block_ptr(
        base=x_ptr,
        shape=(M, N),
        offsets=(off_m, off_n),
        strides=(stride_xm, stride_xn),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1,0)
    )
    y_block_ptrs = tl.make_block_ptr(
        base=y_ptr,
        shape=(M, N),
        offsets=(off_m, off_n),
        strides=(stride_ym, stride_yn),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1,0)
    )


    cols = off_m + tl.arange(0, BLOCK_M)
    mask = cols < M
    s_ptr += cols * stride_sm + pid_n
    if not TRANSPOSE:
        x = tl.load(x_block_ptrs, boundary_check=(0,), padding_option='zero').to(tl.float32)
    else:
        x = tl.load(x_block_ptrs, boundary_check=(1,), padding_option='zero').to(tl.float32)
    
    s = tl.max(tl.abs(x), 1) / MAX
    y = x / s[:, None]

    y = y.to(y_ptr.dtype.element_ty)
    if not TRANSPOSE:
        tl.store(y_block_ptrs, y, boundary_check=(0,))
    else:
        tl.store(y_block_ptrs, y, boundary_check=(1,))
    tl.store(s_ptr, s, mask=mask)

def act_quant_block(x: torch.Tensor, transpose=False, dtype=torch.float8_e5m2, BLOCK_N: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    
    # 理论上bs*seq_len 可以不是128的倍数，但一般训练中都是
    assert x.size(-1) % BLOCK_N == 0
    input_shape = x.shape
    x = x.view(-1, input_shape[-1])
    if transpose:
        x = x.T
        input_shape = (input_shape[-1], 1)
    M, N = x.shape
    y = torch.empty(M, N, device=x.device, dtype=dtype)

    K = N // BLOCK_N
    s = torch.empty(M, triton.cdiv(N, BLOCK_N), dtype=torch.float32, device=x.device)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), K)
    kwargs = {'BLOCK_M':32, 'num_warps':8, 'num_stages':5}
    _act_quant_block_kernel[grid](x, y, s, 
                        *x.stride(),
                        *y.stride(),
                        *s.stride(),
                        M, N, K, torch.finfo(dtype).max, transpose,
                        BLOCK_N=128,
                        **kwargs,
                        )
    return y.view(*input_shape[:-1], -1), s.view(*input_shape[:-1], -1)

@triton.jit
def _weight_quant_block_kernel(x_ptr, y_ptr, s_ptr, 
                           stride_xm, stride_xn,
                           stride_ym, stride_yn,
                           M, N, U, V, MAX,
                           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
                           ):
    pid = tl.program_id(axis=0)
    pid_m = pid // V
    pid_n = pid % V
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N
    x_block_ptrs = tl.make_block_ptr(
        base=x_ptr,
        shape=(M, N),
        offsets=(off_m, off_n),
        strides=(stride_xm, stride_xn),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1,0)
    )
    y_block_ptrs = tl.make_block_ptr(
        base=y_ptr,
        shape=(M, N),
        offsets=(off_m, off_n),
        strides=(stride_ym, stride_yn),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1,0)
    )
    s_ptr += pid
    x = tl.load(x_block_ptrs).to(tl.float32)
    s = tl.max(tl.abs(x)) / MAX
    y = x / s
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_block_ptrs, y)
    tl.store(s_ptr, s)

def weight_quant_block(x: torch.Tensor, BLOCK_M: int = 128, BLOCK_N: int = 128, dtype=torch.float8_e5m2) -> Tuple[torch.Tensor, torch.Tensor]:
    # assert x.is_contiguous()
    assert x.size(-1) % BLOCK_N == 0
    M, N = x.shape
    U = M // BLOCK_M
    V = N // BLOCK_N
    y = torch.empty(x.shape, device=x.device, dtype=dtype)
    s = torch.empty(U, V, dtype=torch.float32, device=x.device)
    grid = lambda meta: (U * V, )
    _weight_quant_block_kernel[grid](x, y, s, 
                        *x.stride(),
                        *y.stride(),
                        M, N, U,V, torch.finfo(dtype).max, 
                        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    return y, s

# def get_config(idx):
#     if idx == 0:
#         return [triton.Config({'BLOCK_SIZE_N': bsn, 'BLOCK_SIZE_M': bsm, 'BLOCK_SIZE_K': 128}, num_stages=ns, num_warps=nw)
#                 for bsn in [64, 128, 256]
#                 for bsm in [64, 128, 256]
#                 for ns in [3, 4, 5]
#                 for nw in [4, 8]
#                 ]
#     else:
#         return [triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 128}, num_stages=4, num_warps=8)]
    
# @triton.autotune(get_config(1), key=['M', 'N', 'K'])
@triton.jit
def _fp8_matmul_kernel(
        # Pointers to matrices
        x_ptr, w_ptr, y_ptr, scale_x_ptr, scale_w_ptr, bias_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_xm, stride_xk,  #
        stride_wn, stride_wk,  #
        stride_ym, stride_yn,
        stride_sx1, stride_sx2, 
        stride_sw1, stride_sw2, 
        HAVE_W: tl.constexpr, HAVE_BIAS: tl.constexpr, 
        # Meta-parameters
        BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        BLOCK_SIZE_M: tl.constexpr, 


):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    pid_m = tl.program_id(axis=0) 
    pid_n = tl.program_id(axis=1)


    # k = tl.cdiv(K, BLOCK_SIZE_K)
    off_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    off_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    row_mask = off_m < M
    col_mask = off_n < N

    x_block_ptrs = x_ptr + off_m[:, None]*stride_xm + tl.arange(0, BLOCK_SIZE_K)[None, :]*stride_xk
    w_block_ptrs = w_ptr + off_n[None, :]*stride_wn + tl.arange(0, BLOCK_SIZE_K)[:, None]*stride_wk
    scale_x_ptr += off_m * stride_sx1
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    if HAVE_W:
        # scale_w_ptr += (off_n // BLOCK_SIZE_K) * stride_sw1
        scale_w_ptr += pid_n * stride_sw1
        for start_k in tl.range(0, K, BLOCK_SIZE_K):
            x = tl.load(x_block_ptrs, mask=row_mask[:, None])
            w = tl.load(w_block_ptrs)
            scale_x = tl.load(scale_x_ptr, mask=row_mask, other=0.)
            scale_w = tl.load(scale_w_ptr)

            scale = scale_w * scale_x
            accumulator += tl.dot(x, w) * scale[:, None]
            # tl.dot(x, w, accumulator)，使用fp8时，累加的方式速度最快，但是没法进行量化，这是最需要解决的问题
            scale_x_ptr += stride_sx2
            scale_w_ptr += stride_sw2
            x_block_ptrs += BLOCK_SIZE_K * stride_xk
            w_block_ptrs += BLOCK_SIZE_K * stride_wk

    else:
        scale_w_ptr += off_n * stride_sw1
        kk = tl.arange(0, BLOCK_SIZE_K)
        for start_k in range(0, K, BLOCK_SIZE_K):
            mask = (start_k + kk) < K
            x = tl.load(x_block_ptrs, mask=mask[None, :])
            w = tl.load(w_block_ptrs, mask=mask[:, None]) 
            scale_x = tl.load(scale_x_ptr)
            scale_w = tl.load(scale_w_ptr)
            accumulator += tl.dot(x, w) * scale_x[:, None] * scale_w[None, :]
            x_block_ptrs += BLOCK_SIZE_K * stride_xk
            w_block_ptrs += BLOCK_SIZE_K * stride_wk
            scale_x_ptr += stride_sx2
            scale_w_ptr += stride_sw2
        
    if HAVE_BIAS:
        bias = tl.load(bias_ptr + off_n, mask=col_mask).to(tl.float32)
        accumulator += bias[None, :]
    c = accumulator.to(y_ptr.type.element_ty)

    y_ptr += (off_m * stride_ym)[:, None] + (off_n * stride_yn)[None, :]
    mask = row_mask[:, None] & col_mask[None, :]
    tl.store(y_ptr, c, mask=mask)




def fp8_matmul(x, w, scale_x, scale_w, bias=None, have_w=True):
    if have_w:
        assert x.size(-1) % 128 == 0, 'in forward x @ w.T'
    else:
        assert x.size(0) % 128 == 0, 'in backward dy.T @ x or dy @ w'
        assert bias is None

    input_shape = x.shape
    x = x.view(-1, input_shape[-1])
    scale_x = scale_x.view(-1, scale_x.size(-1))
    M, K = x.shape
    N, K2 = w.shape
    assert K2 == K
    # Allocates output.
    y = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']))
    # print(w.stride(), scale_w.stride())
    # print(x.device, y.device, w.device)
    have_bias = bias is not None
    kwargs = {'BLOCK_SIZE_K':128, 'BLOCK_SIZE_M':64, 'BLOCK_SIZE_N':256,'num_stages':4, 'num_warps':8}
    _fp8_matmul_kernel[grid](
        x, w, y, scale_x, scale_w, bias if have_bias else w,#
        M, N, K,  #
        x.stride(0), x.stride(1),  #
        w.stride(0), w.stride(1),  #
        y.stride(0), y.stride(1),  #
        scale_x.stride(0), scale_x.stride(1),
        scale_w.stride(0), scale_w.stride(1), 
        have_w, have_bias,
        **kwargs
    ) 
    return y.view(*input_shape[:-1],-1)

def fp8_matmul_bwd(xt, w, dy, dyt, scale_xt, scale_w, scale_dy, scale_dyt):
    dx = fp8_matmul(dy, w, scale_dy, scale_w, have_w=True)
    dw = fp8_matmul(dyt, xt, scale_dyt, scale_xt, have_w=False)
    return dx, dw

class _Fp8LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, bias, fp8_dtype):
        input_shape = x.shape
        x = x.view(-1, input_shape[-1])
        qx, sx = act_quant_block(x, dtype=fp8_dtype)
        qw, sw = weight_quant_block(w, dtype=fp8_dtype)
        y = fp8_matmul(qx, qw, sx, sw, bias, have_w=True)
        ctx.save_for_backward(x, w)
        ctx.bias = bias
        ctx.fp8_dtype = fp8_dtype
        ctx.input_shape = input_shape
        return y.view(*input_shape[:-1], -1)

    @staticmethod
    def backward(ctx, dy):
        dy = dy.view(-1, dy.size(-1))
        dbias = None
        if ctx.bias is not None:
            dbias = dy.sum(0)
        
        x,w = ctx.saved_tensors
        fp8_dtype = ctx.fp8_dtype

        qxt, sxt = act_quant_block(x, transpose=True, dtype=fp8_dtype)
        qdy, sdy = act_quant_block(dy, dtype=fp8_dtype)
        qdyt, sdyt = act_quant_block(dy, transpose=True, dtype=fp8_dtype)
        qw, sw = weight_quant_block(w.T, dtype=fp8_dtype)
        dx, dw = fp8_matmul_bwd(qxt, qw, qdy, qdyt, sxt, sw, sdy, sdyt)
        return dx.view(*ctx.input_shape), dw, dbias, None

fp8_linear_with_quant_ops = _Fp8LinearFunction.apply

class Fp8Linear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias = True, device=None, dtype=None, fp8_dtype=torch.float8_e5m2):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.fp8_dtype = fp8_dtype

    def forward(self, x):
        return fp8_linear_with_quant_ops(x, self.weight, self.bias, self.fp8_dtype)
    

# 按行处理
@triton.jit
def _fused_silu_fwdv2(X, Y,
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
def _fused_silu_bwd_dupgatev2(X, 
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


def silu_fwd(x, order='up-gate'):
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
    _fused_silu_fwdv2[(M,)](
        x, y, 
        N,  #
        *x.stride(),  #
        BLOCK_SIZE_N, order,
        num_warps=num_warps, num_stages=num_stages, 
        )
    return y

def silu_bwd(dy, x, order='up-gate'):
    input_shape = x.shape
    x = x.view(-1, input_shape[-1])
    M, N2 = x.shape
    N = N2 // 2
    BLOCK_SIZE_N = triton.next_power_of_2(N)
    BLOCK_SIZE_N = min(4096, BLOCK_SIZE_N)
    # print(stride_m, stride_n)
    dx = torch.empty(input_shape, device=dy.device, dtype=dy.dtype)
    # BLOCK_SIZE_N = min(8192, BLOCK_SIZE_N)
    _fused_silu_bwd_dupgatev2[(M,)](x,
                                dy, dx,
                                N,
                                *x.stride(),
                                BLOCK_SIZE_N, order,
                                num_warps=8, num_stages=4)

    return dx

class _SiluOps(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, order='up-gate'):
        ctx.x = x
        ctx.order = order
        return silu_fwd(x, order)

    @staticmethod
    def backward(ctx, dy):
        return silu_bwd(dy, ctx.x, ctx.order), None
triton_silu = _SiluOps.apply

class _Fp8MlpCkptOps(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, w_up_gate, w_down, fp8_dtype=torch.float8_e5m2, order='up-gate'):
        # quant for fwd and bwd
        input_shape = x.shape
        x = x.view(-1, input_shape[-1])
        qw_up_gate, sw_up_gate = weight_quant_block(w_up_gate, dtype=fp8_dtype)
        qw_down, sw_down = weight_quant_block(w_down, dtype=fp8_dtype)
        qx, sx = act_quant_block(x, dtype=fp8_dtype)

        
        # compute up_gate and act
        up_gate = fp8_matmul(qx, qw_up_gate, sx, sw_up_gate)
        act = silu_fwd(up_gate, order)

        # quant act
        qact, sact = act_quant_block(act, dtype=fp8_dtype)

        # compute output
        y = fp8_matmul(qact, qw_down, sact, sw_down)

        save4bwd = (x, w_up_gate, w_down)
        ctx.save_for_backward(*save4bwd)
        ctx.order = order
        ctx.fp8_dtype = fp8_dtype
        ctx.input_shape = input_shape
        return y.view(*input_shape)
    
    @staticmethod
    def backward(ctx, dy):
        dy = dy.view(-1, dy.size(-1))
        

        x, w_up_gate, w_down = ctx.saved_tensors
        qx, sx = act_quant_block(x, dtype=ctx.fp8_dtype)
        qxt, sxt = act_quant_block(x, transpose=True, dtype=ctx.fp8_dtype)
        qw_up_gate, sw_up_gate = weight_quant_block(w_up_gate, dtype=ctx.fp8_dtype)
        qw_downt, sw_downt = weight_quant_block(w_down.T, dtype=ctx.fp8_dtype)
        
        # recompute up_gate, act
        up_gate = fp8_matmul(qx, qw_up_gate, sx, sw_up_gate)
        act = silu_fwd(up_gate, ctx.order)
        qactt, sactt = act_quant_block(act, transpose=True, dtype=ctx.fp8_dtype)

        # quant dy
        qdy, sdy = act_quant_block(dy, dtype=ctx.fp8_dtype)
        qdyt, sdyt = act_quant_block(dy, transpose=True, dtype=ctx.fp8_dtype) 

        # compute dact, dw_down
        dact, dw_down = fp8_matmul_bwd(qactt, qw_downt, qdy, qdyt, sactt, sw_downt, sdy, sdyt)
        # compute dup_gate
        dup_gate = silu_bwd(dact, up_gate, ctx.order)

        # quant dup_gate
        qd_up_gate, sd_up_gate = act_quant_block(dup_gate, dtype=ctx.fp8_dtype)
        qd_up_gatet, sd_up_gatet = act_quant_block(dup_gate, transpose=True, dtype=ctx.fp8_dtype) 

        # compute dx, dwup_gate
        dx, dw_up_gate = fp8_matmul_bwd(qxt, qw_up_gate.T, qd_up_gate, qd_up_gatet, sxt, sw_up_gate.T, sd_up_gate, sd_up_gatet)
        return dx.view(*ctx.input_shape), dw_up_gate, dw_down, None, None

fp8_ckpt_mlp = _Fp8MlpCkptOps.apply

    
class Fp8MLP(torch.nn.Module):
    def __init__(self, hidden_size, intermediate_size, order='up-gate', fp8_dtype=torch.float8_e5m2, low_memory=False):
        super().__init__()
        self.up_gate_proj = Fp8Linear(hidden_size, intermediate_size*2, bias=False, fp8_dtype=fp8_dtype)
        self.down_proj = Fp8Linear(intermediate_size, hidden_size, bias=False, fp8_dtype=fp8_dtype)
        self.order = order
        self.fp8_dtype = fp8_dtype
        self.low_memory = low_memory
    
    def forward(self, hidden_states, low_memory=False):
        if low_memory or self.low_memory:
            return fp8_ckpt_mlp(hidden_states, self.up_gate_proj.weight, self.down_proj.weight, self.fp8_dtype, self.order)
        else:
            up_gate = self.up_gate_proj(hidden_states)
            act = triton_silu(up_gate, self.order)
            return self.down_proj(act)


  
# @triton.autotune([triton.Config({'BLOCK_SIZE_N': bsn, 'BLOCK_SIZE_M': bsm, 'BLOCK_SIZE_K': 128}, num_stages=ns, num_warps=nw)
#                 for bsn in [64, 128, 256]
#                 for bsm in [64, 128, 256]
#                 for ns in [3, 4, 5]
#                 for nw in [4, 8]
#                 ], key=['M', 'N', 'K'])
@triton.jit
def _fp8_matmul_wo_quant_kernel(
        # Pointers to matrices
        x_ptr, w_ptr, y_ptr, bias_ptr,
        # Matrix dimensions
        M, N, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_xm, stride_xk,  #
        stride_wn, stride_wk,  #
        stride_ym, stride_yn,
        HAVE_W: tl.constexpr, HAVE_BIAS: tl.constexpr, 
        # Meta-parameters
        BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        BLOCK_SIZE_M: tl.constexpr, 


):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    pid_m = tl.program_id(axis=0) 
    pid_n = tl.program_id(axis=1)


    # k = tl.cdiv(K, BLOCK_SIZE_K)
    off_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    off_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    row_mask = off_m < M
    col_mask = off_n < N

    x_block_ptrs = x_ptr + off_m[:, None]*stride_xm + tl.arange(0, BLOCK_SIZE_K)[None, :]*stride_xk
    w_block_ptrs = w_ptr + off_n[None, :]*stride_wn + tl.arange(0, BLOCK_SIZE_K)[:, None]*stride_wk
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    if HAVE_W:
        for start_k in tl.range(0, K, BLOCK_SIZE_K):
            x = tl.load(x_block_ptrs, mask=row_mask[:, None])
            w = tl.load(w_block_ptrs)
            accumulator = tl.dot(x, w, accumulator)
            x_block_ptrs += BLOCK_SIZE_K * stride_xk
            w_block_ptrs += BLOCK_SIZE_K * stride_wk

    else:
        kk = tl.arange(0, BLOCK_SIZE_K)
        for start_k in range(0, K, BLOCK_SIZE_K):
            mask = (start_k + kk) < K
            x = tl.load(x_block_ptrs, mask=mask[None, :])
            w = tl.load(w_block_ptrs, mask=mask[:, None]) 
            accumulator = tl.dot(x, w, accumulator)
            x_block_ptrs += BLOCK_SIZE_K * stride_xk
            w_block_ptrs += BLOCK_SIZE_K * stride_wk
        
    if HAVE_BIAS:
        bias = tl.load(bias_ptr + off_n, mask=col_mask).to(tl.float32)
        accumulator += bias[None, :]
    c = accumulator.to(y_ptr.type.element_ty)

    y_ptr += (off_m * stride_ym)[:, None] + (off_n * stride_yn)[None, :]
    mask = row_mask[:, None] & col_mask[None, :]
    tl.store(y_ptr, c, mask=mask)




def fp8_matmul_wo_quant(x, w, bias=None, have_w=True):
    if have_w:
        assert x.size(-1) % 128 == 0, 'in forward x @ w.T'
    else:
        assert x.size(0) % 128 == 0, 'in backward dy.T @ x or dy @ w'
        assert bias is None

    input_shape = x.shape
    x = x.view(-1, input_shape[-1])
    M, K = x.shape
    N, K2 = w.shape
    assert K2 == K
    # Allocates output.
    y = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']))
    have_bias = bias is not None
    kwargs = {'BLOCK_SIZE_K':128, 'BLOCK_SIZE_M':128, 'BLOCK_SIZE_N':256,'num_stages':4, 'num_warps':8}
    _fp8_matmul_wo_quant_kernel[grid](
        x, w, y, bias if have_bias else w,#
        M, N, K,  #
        x.stride(0), x.stride(1),  #
        w.stride(0), w.stride(1),  #
        y.stride(0), y.stride(1),  #
        have_w, have_bias,
        **kwargs
    ) 
    return y.view(*input_shape[:-1],-1)
        



