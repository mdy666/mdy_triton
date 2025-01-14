import torch
import triton
import triton.language as tl

# @triton.autotune([triton.Config({"BLOCK_SIZE_L":bsl, "BLOCK_SIZE_D":bsd}, num_stages=ns, num_warps=nw)
#                                  for bsl in [32, 64, 128]
#                                  for bsd in [32, 64, 128]
#                                  for ns in [1,2,3,4]
#                                  for nw in [4,8]], key=['L', 'D'])
@triton.jit
def _conv1d_fwd_kernel(X, W, Y, BIAS, HAVE_BIAS,
                       stride_xb, stride_xd, stride_xl,
                       stride_yb, stride_yd, stride_yl,
                       stride_wd, stride_wk,
                       B, D, L, K, ACT,
                       BLOCK_SIZE_L:tl.constexpr=64, BLOCK_SIZE_D:tl.constexpr=128,
                       num_stages: tl.constexpr=3,
                        ):
    off_b = tl.program_id(0)
    off_d = tl.program_id(1) * BLOCK_SIZE_D
    off_l = tl.program_id(2) * BLOCK_SIZE_L

    X += off_b * stride_xb 
    Y += off_b * stride_yb
    # kk = tl.arange(0, BLOCK_SIZE_K)
    dd = tl.arange(0, BLOCK_SIZE_D)
    ll = tl.arange(0, BLOCK_SIZE_L)
    rows = off_d + dd
    cols = off_l + ll - K
    x_ptrs = X + rows[:, None] * stride_xd + cols[None, :] * stride_xl 
    
    w_ptrs = W + rows * stride_wd
    row_mask = rows < D

    acc = tl.zeros((BLOCK_SIZE_D, BLOCK_SIZE_L), dtype=tl.float32)
    for _ in tl.range(K):
        w = tl.load(w_ptrs, mask=row_mask, other=0.).to(tl.float32)
        cols += 1
        x_ptrs += stride_xl
        col_mask = (cols >= 0) & (cols < L)
        x = tl.load(x_ptrs, mask=col_mask[None, :] & row_mask[:, None], other=0.).to(tl.float32)
        acc += x * w[:, None]
        w_ptrs += stride_wk 
    if HAVE_BIAS:
        bias = tl.load(BIAS+rows, mask=row_mask, other=0.).to(tl.float32)
        acc += bias[:, None]
    if ACT:
        acc *= tl.sigmoid(acc)
    y_ptrs = Y + rows[:, None] * stride_yd + (off_l + ll)[None, :] * stride_yl
    col_mask = off_l + ll < L
    tl.store(y_ptrs, acc, mask=col_mask[None, :] & row_mask[:, None])

@triton.jit
def _conv1d_bwd_dz_kernel(X, W, DY, DZ, BIAS, HAVE_BIAS,
                       stride_xb, stride_xd, stride_xl,
                       stride_yb, stride_yd, stride_yl,
                       stride_wd, stride_wk,
                       B, D, L, K, 
                       BLOCK_SIZE_L:tl.constexpr=64, BLOCK_SIZE_D:tl.constexpr=128,
                       num_stages: tl.constexpr=3,
                        ):
    off_b = tl.program_id(0)
    off_d = tl.program_id(1) * BLOCK_SIZE_D
    off_l = tl.program_id(2) * BLOCK_SIZE_L

    X += off_b * stride_xb 
    DY += off_b * stride_yb
    DZ += off_b * stride_yb

    # kk = tl.arange(0, BLOCK_SIZE_K)
    dd = tl.arange(0, BLOCK_SIZE_D)
    ll = tl.arange(0, BLOCK_SIZE_L)
    rows = off_d + dd
    cols = off_l + ll - K
    x_ptrs = X + rows[:, None] * stride_xd + cols[None, :] * stride_xl 
    
    w_ptrs = W + rows * stride_wd
    row_mask = rows < D

    acc = tl.zeros((BLOCK_SIZE_D, BLOCK_SIZE_L), dtype=tl.float32)
    for _ in tl.range(K):
        w = tl.load(w_ptrs, mask=row_mask, other=0.).to(tl.float32)
        cols += 1
        x_ptrs += stride_xl
        col_mask = (cols >= 0) & (cols < L)
        x = tl.load(x_ptrs, mask=col_mask[None, :] & row_mask[:, None], other=0.).to(tl.float32)
        acc += x * w[:, None]
        w_ptrs += stride_wk 
    if HAVE_BIAS:
        bias = tl.load(BIAS+rows, mask=row_mask, other=0.).to(tl.float32)
        acc += bias[:, None]

    sig_acc = tl.sigmoid(acc)
    dy_ptrs = DY + rows[:, None] * stride_yd + (off_l + ll)[None, :] * stride_yl
    dz_ptrs = DZ + rows[:, None] * stride_yd + (off_l + ll)[None, :] * stride_yl
    col_mask = off_l + ll<L
    dy = tl.load(dy_ptrs, mask=col_mask[None, :] & row_mask[:, None], other=0.)
    dz = (sig_acc + acc * sig_acc * (1 - sig_acc)) * dy
    tl.store(dz_ptrs, dz, mask=col_mask[None, :] & row_mask[:, None])

# @triton.autotune([triton.Config({"BLOCK_SIZE_D":bsd}, num_stages=ns, num_warps=nw)
#                                 #  for bsl in [32, 64, 128]
#                                  for bsd in [16, 32, 64, 128]
#                                  for ns in [1,2,3,4]
#                                  for nw in [4,8]], key=['L', 'D'])
@triton.jit
def _conv1d_bwd_dwdb_kernel(DY, DX, DW,
                        X, W, DB, HAVE_BIAS,
                       stride_dyb, stride_dyd, stride_dyl,
                       stride_dxb, stride_dxd, stride_dxl,
                       stride_xb, stride_xd, stride_xl,
                       stride_wd, stride_wk,
                       stride_dwb, stride_dwd, stride_dwk,
                       B, D, L, K, ACT,
                       BLOCK_SIZE_L:tl.constexpr, BLOCK_SIZE_D:tl.constexpr=16,
                       num_warps:tl.constexpr=4, num_stages: tl.constexpr=4,
                        ):
    off_b = tl.program_id(0)
    off_d = tl.program_id(1) * BLOCK_SIZE_D
    off_l = tl.program_id(2) * BLOCK_SIZE_L
    b = tl.cdiv(L, BLOCK_SIZE_L)

    X += off_b * stride_xb 
    DY += off_b * stride_dyb
    DX += off_b * stride_dxb
    DW += (off_b * b + tl.program_id(2)) * stride_dwb
    dd = tl.arange(0, BLOCK_SIZE_D)
    ll = tl.arange(0, BLOCK_SIZE_L)
    rows = off_d + dd
    cols = off_l + ll + K
    col_mask_x = off_l + ll < L
    x_ptrs = X + rows[:, None] * stride_xd + (off_l + ll)[None, :] * stride_xl 
    dx_ptrs = DX + rows[:, None] * stride_dxd + cols[None, :] * stride_dxl
    dy_ptrs = DY + rows[:, None] * stride_dyd + cols[None, :] * stride_dyl
    w_ptrs = W + rows * stride_wd
    dw_ptrs = DW + rows * stride_dwd
    row_mask = rows < D

    x = tl.load(x_ptrs, mask=col_mask_x[None, :] & row_mask[:, None], other=0.).to(tl.float32)
    acc_dx = tl.zeros((BLOCK_SIZE_D, BLOCK_SIZE_L), dtype=tl.float32)
    for idx in tl.range(K):
        w = tl.load(w_ptrs, mask=row_mask, other=0.).to(tl.float32)
        cols -= 1

        dy_ptrs -= stride_dyl
        col_mask = (cols >= 0) & (cols < L)
        
        dy = tl.load(dy_ptrs, mask=col_mask[None, :] & row_mask[:, None], other=0.).to(tl.float32)
        acc_dx += dy * w[:, None]
        dw = tl.sum(dy * x, 1)
        tl.store(dw_ptrs, dw, mask=row_mask)
        w_ptrs += stride_wk
        dw_ptrs += stride_dwk
        if idx == K-1 and HAVE_BIAS:
            DB += (off_b * b + tl.program_id(2)) * stride_dwb // K
            tl.store(DB+rows, tl.sum(dy, 1), mask=row_mask)

    dx_ptrs = DX + rows[:, None] * stride_dxd + cols[None, :] * stride_dxl
    tl.store(dx_ptrs, acc_dx, mask=col_mask_x[None, :] & row_mask[:, None])


def causal_conv1d_fwd(x, weight, bias=None, unuse_arg1=None, unuse_arg2=None, unuse_arg3=None, activation=False):
    # print(123)
    D, K = weight.shape
    B, D, L = x.shape
    y = torch.empty_like(x)
    HAVE_BIAS = bias is not None
    grid = lambda meta: (B, triton.cdiv(D, meta['BLOCK_SIZE_D']), triton.cdiv(L,meta['BLOCK_SIZE_L']))
    # print(activation is not None, activation)
    _conv1d_fwd_kernel[grid](x, weight, y, bias if HAVE_BIAS else weight, HAVE_BIAS,
                            *x.stride(),
                            *y.stride(),
                            *weight.stride(),
                            B, D, L, K, activation,
                            )
    return y
    
def causal_conv1d_bwd(x, weight, bias, dy, seq_idx=None, unuse_arg1=None, unuse_arg2=None, dx=None, unuse_arg3=None, activation=False):
    D, K = weight.shape
    B, D, L = x.shape
    HAVE_BIAS = bias is not None
    if activation:
        dz = torch.empty_like(x)
        grid = lambda meta: (B, triton.cdiv(D, meta['BLOCK_SIZE_D']), triton.cdiv(L,meta['BLOCK_SIZE_L']))
        # print(activation is not None, activation)
        _conv1d_bwd_dz_kernel[grid](x, weight, dy, dz, bias if HAVE_BIAS else weight, HAVE_BIAS,
                                *x.stride(),
                                *dz.stride(),
                                *weight.stride(),
                                B, D, L, K, 
                                )
    BLOCK_SIZE_L = 128
    # a = triton.cdiv(D, BLOCK_SIZE_D)
    b = triton.cdiv(L, BLOCK_SIZE_L)
    if dx is None:
        dx = torch.empty_like(dy)
    dw = torch.empty(B*b, D, K, dtype=x.dtype, device=x.device)
    db = None
    if HAVE_BIAS:
        db = torch.empty(B*b, D, dtype=x.dtype, device=x.device)
    grid = lambda meta: (B, triton.cdiv(D, meta['BLOCK_SIZE_D']), b)
    _conv1d_bwd_dwdb_kernel[grid](dz if activation else dy, dx, dw,
                            x, weight, db if HAVE_BIAS else dw, HAVE_BIAS,
                            *dy.stride(),
                            *dx.stride(),
                            *x.stride(),
                            *weight.stride(),
                            *dw.stride(),
                            B, D, L, K, activation,
                            BLOCK_SIZE_L, 
                            # BLOCK_SIZE_D,
                            # num_warps=4, num_stages=1
                            )
    dw = dw.sum(0)
    if HAVE_BIAS:
        db = db.sum(0)
    return dx, dw, db, None

class _TritonCausalConv1dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias=None, activation=None):
        assert activation in [None, 'silu', 'swish']
        ctx.activation = activation in ['silu', 'swish']
        y = causal_conv1d_fwd(x, weight, bias, None, None, None, ctx.activation)
        ctx.save_for_backward(x, weight)
        ctx.bias = bias
        return y
    
    @staticmethod
    def backward(ctx, dy):
        x, weight = ctx.saved_tensors
        dx, dw, db, *_ = causal_conv1d_bwd(x, weight, ctx.bias, dy, None, None, None, None, None, ctx.activation)
        return dx, dw, db, None
    
triton_causal_conv1d = _TritonCausalConv1dFunction.apply

def causal_conv1d_fn(
    x,
    weight,
    bias=None,
    seq_idx=None,# unuse
    initial_states=None, # unuse
    return_final_states=False,# unuse
    final_states_out=None,# unuse
    activation=None,
):
    """
    x: (batch, dim, seqlen)
    weight: (dim, width)
    bias: (dim,)
    seq_idx: (batch, seqlen)
    initial_states: (batch, dim, width - 1)
    final_states_out: (batch, dim, width - 1), to be written to
    activation: either None or "silu" or "swish"

    out: (batch, dim, seqlen)
    """
    return _TritonCausalConv1dFunction.apply(
        x,
        weight,
        bias,
        activation,
    )
