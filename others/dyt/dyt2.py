import torch
import triton
import triton.language as tl

@triton.jit
def tanh(x):
    a = tl.exp(x)
    b = tl.exp(-x)
    return (a - b) / (a + b)

# @triton.autotune([triton.Config({"BLOCK_N":bn, "BLOCK_M":bm}, num_stages=ns, num_warps=nw)
#                   for bn in [256, 512]
#                   for bm in [4, 8, 16]
#                   for ns in [1,2,4]
#                   for nw in [4, 8]
#                   ],
#                   key=['N'])
# @triton.jit
# def _dyt_fwd_kernel(X,
#                     Y,
#                     Alpha,
#                     Gemma,
#                     Beta,
#                     M,
#                     N:tl.constexpr,
#                     BLOCK_M:tl.constexpr=64,
#                     BLOCK_N:tl.constexpr=64
#                     ):
#     off_m = tl.cast(tl.program_id(1), tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
#     off_n = tl.cast(tl.program_id(0), tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)

#     alpha = tl.load(Alpha).to(tl.float32)
#     gemma = tl.load(Gemma + off_n, mask=off_n<N, other=0.).to(tl.float32)  
#     beta = tl.load(Beta + off_n, mask=off_n<N, other=0.).to(tl.float32) 
#     x = tl.load(X + off_m[:, None] * N + off_n[None, :], mask=(off_m[:, None] < M) & (off_n[None, :] < N), other=0.).to(tl.float32) 

#     tanh_x = tanh(alpha * x)
#     y = tanh_x * gemma[None, :] + beta[None, :]
#     tl.store(Y + off_m[:, None] * N + off_n[None, :], y, mask=(off_m[:, None] < M) & (off_n[None, :] < N))
    

# @triton.autotune([triton.Config({"BLOCK_N":bn}, num_stages=ns, num_warps=nw)
#                   for bn in [1024, 2048, 4096]
#                   for ns in [1, 4]
#                   for nw in [4, 8]
#                   ],
#                   key=['N'])
@triton.jit
def _dyt_fwd_kernel(X,
                    Y,
                    Alpha,
                    Gemma,
                    Beta,
                    HAVE_BETA:tl.constexpr,
                    M:tl.constexpr,
                    N:tl.constexpr,
                    BLOCK_N:tl.constexpr=1024,
                    ):
    offset = tl.cast(tl.program_id(0), tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)
    col = offset % N
    mask = offset < (M * N)
    alpha = tl.load(Alpha).to(tl.float32)
    gemma = tl.load(Gemma + col, mask=mask, other=0.).to(tl.float32)  
    if HAVE_BETA:
        beta = tl.load(Beta + col, mask=mask, other=0.).to(tl.float32)
    x = tl.load(X + offset, mask=mask, other=0.).to(tl.float32) 
    tanh_x = tanh(alpha * x)
    y = tanh_x * gemma
    if HAVE_BETA:
        y += beta
    tl.store(Y + offset, y, mask=mask)



# @triton.autotune([triton.Config({"BLOCK_N":bn}, num_stages=ns, num_warps=nw)
#                   for bn in [1024, 2048, 4096]
#                   for ns in [1,2,4]
#                   for nw in [4, 8, 16]
#                   ],
#                   key=['N'])
@triton.jit
def _dyt_bwd_kernel(DY,
                    DX,
                    DA,
                    DG,
                    DB,
                    X,
                    Alpha,
                    Gemma,
                    HAVE_BETA: tl.constexpr,
                    M,
                    N:tl.constexpr,
                    BLOCK_N:tl.constexpr=1024
                    ):
    col = tl.cast(tl.program_id(0), tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = col < N
    start_row_id = tl.cast(tl.program_id(1), tl.int64)

    alpha = tl.load(Alpha).to(tl.float32)
    da = 0.
    gemma = tl.load(Gemma + col, mask=mask, other=0.).to(tl.float32)  
    dg = tl.zeros((BLOCK_N,), dtype=tl.float32)
    if HAVE_BETA:
        db = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for row_id in range(start_row_id, M, tl.num_programs(1)):
        x = tl.load(X + row_id * N + col, mask=mask, other=0.).to(tl.float32) 
        dy = tl.load(DY + row_id * N + col, mask=mask, other=0.).to(tl.float32) 
        tanh_x = tanh(alpha * x)
        if HAVE_BETA:
            db += dy
        dg += dy * tanh_x
        tmp = (1 - tanh_x * tanh_x) * dy * gemma
        da += tl.sum(x * tmp, 0)
        dx = alpha * tmp
        tl.store(DX + row_id * N + col, dx, mask=mask)
    
    tl.store(DG + start_row_id * N + col, dg, mask=mask)
    if HAVE_BETA:
        tl.store(DB + start_row_id * N + col, db, mask=mask)
    tl.store(DA + start_row_id * tl.cdiv(N, 512) + tl.program_id(0), da)



class _DYT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha, gemma, beta):
        assert x.is_contiguous()
        ctx.HAVE_BETA = True if beta is not None else False
        ctx.input_shape = x.shape
        x = x.view(-1, ctx.input_shape[-1])
        M, N = x.shape
        
        y = torch.empty_like(x)

        
        kwargs = {"BLOCK_N": min(triton.next_power_of_2(N), 2048), "num_warps":4, "num_stages": 1}
        grid = lambda meta: (triton.cdiv(N * M, meta['BLOCK_N']), )
        _dyt_fwd_kernel[(grid)](x, 
                                y,
                                alpha, 
                                gemma, 
                                beta,
                                ctx.HAVE_BETA,
                                M,
                                N,
                                **kwargs,
                                )
        
        ctx.save_for_backward(x, alpha, gemma, beta)
        return y.view(ctx.input_shape)
    
    @staticmethod
    def backward(ctx, dy):
        assert dy.is_contiguous()
        x, alpha, gemma, beta = ctx.saved_tensors
        M, N = x.shape
        NUM_SMS = torch.cuda.get_device_properties('cuda').multi_processor_count
        da = torch.zeros(NUM_SMS, triton.cdiv(N, 512), dtype=torch.float32, device=x.device)
        dg = torch.empty(NUM_SMS, N, dtype=torch.float32, device=x.device)
        db = torch.empty(NUM_SMS, N, dtype=torch.float32, device=x.device) if ctx.HAVE_BETA else None
        dx = torch.empty_like(dy)

        kwargs = {"BLOCK_N": min(triton.next_power_of_2(N), 1024), "num_warps":8, "num_stages": 2}
        grid = lambda meta: (triton.cdiv(N, meta['BLOCK_N']), NUM_SMS)
        _dyt_bwd_kernel[grid](  dy, 
                                dx,
                                da, 
                                dg, 
                                db,
                                x,
                                alpha,
                                gemma,
                                ctx.HAVE_BETA,
                                M,
                                N,
                                **kwargs
                                )
        if ctx.HAVE_BETA:
            db = db.sum(0).to(x.dtype)
        dg = dg.sum(0).to(gemma.dtype)
        da = da.sum().to(x.dtype).unsqueeze(0)
        return dx, da, dg, db

@torch.compile    
def torch_dyt_with_beta(x, alpha, gemma, beta):
    return gemma * torch.tanh(x * alpha) + beta

@torch.compile    
def torch_dyt_without_beta(x, alpha, gemma):
    return gemma * torch.tanh(x * alpha)

class DYT(torch.nn.Module):
    def __init__(self, dim, beta=True, init_a=0.5):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.ones(1) * init_a)
        self.gemma = torch.nn.Parameter(torch.ones(dim))
        self.beta = None
        if beta:
            self.beta = torch.nn.Parameter(torch.zeros(dim))

    def forward(self, x, backend='triton'):
        if backend == "triton":
            return _DYT.apply(x, self.alpha, self.gemma, self.beta)
        else:
            if self.beta is None:
                return torch_dyt_without_beta(x, self.alpha, self.gemma)
            return torch_dyt_with_beta(x, self.alpha, self.gemma, self.beta)


        