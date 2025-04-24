import torch
import triton
import triton.language as tl
from triton.language.extra.libdevice import tanh

# @triton.jit
# def tanh(x):
#     a = tl.exp(x)
#     b = tl.exp(-x)
#     return (a - b) / (a + b)
    

# @triton.autotune([triton.Config({"BLOCK_N":bn}, num_stages=ns, num_warps=nw)
#                   for bn in [1024, 2048, 4096]
#                   for ns in [1,2,4]
#                   for nw in [4, 8, 16, 32]
#                   ],
#                   key=['N'])
@triton.jit
def _dyt_fwd_kernel(X,
                    Y,
                    Alpha,
                    Gemma,
                    Beta,
                    HAVE_BETA:tl.constexpr,
                    N:tl.constexpr,
                    BLOCK_N:tl.constexpr=1024
                    ):
    col = tl.cast(tl.program_id(0), tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = col < N
    row_id = tl.cast(tl.program_id(1), tl.int64)

    X += row_id * N
    Y += row_id * N
    alpha = tl.load(Alpha).to(tl.float32)
    
    gemma = tl.load(Gemma + col, mask=mask, other=0.).to(tl.float32)  
     
    x = tl.load(X + col, mask=mask, other=0.).to(tl.float32) 

    tanh_x = tanh(alpha * x)
    y = tanh_x * gemma
    if HAVE_BETA:
        beta = tl.load(Beta + col, mask=mask, other=0.).to(tl.float32)
        y += beta
    tl.store(Y + col, y, mask=mask)



# @triton.autotune([triton.Config({"BLOCK_N":bn}, num_stages=ns, num_warps=nw)
#                   for bn in [1024, 2048, 4096]
#                   for ns in [1,2,4]
#                   for nw in [1,2, 4, 8, 16]
#                   ],
#                   key=['N'])
@triton.jit
def _dyt_bwd_kernel(DY,
                    DX,
                    DA,
                    DG,
                    X,
                    Alpha,
                    Gemma,
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
    for row_id in range(start_row_id, M, tl.num_programs(1)):
        x = tl.load(X + row_id * N + col, mask=mask, other=0.).to(tl.float32) 
        dy = tl.load(DY + row_id * N + col, mask=mask, other=0.).to(tl.float32) 
        tanh_x = tanh(alpha * x)
        dg += dy * tanh_x
        tmp = (1 - tanh_x * tanh_x) * dy * gemma
        da += tl.sum(x * tmp, 0)
        dx = alpha * tmp
        tl.store(DX + row_id * N + col, dx, mask=mask)
    
    tl.store(DG + start_row_id * N + col, dg, mask=mask)
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

        if N >= 4096:
            kwargs = {"BLOCK_N": min(triton.next_power_of_2(N), 2048), "num_warps":4, "num_stages": 1}
        else:
            kwargs = {"BLOCK_N": min(triton.next_power_of_2(N), 1024), "num_warps":4, "num_stages": 1}
        grid = lambda meta: (triton.cdiv(N, meta['BLOCK_N']), M)
        _dyt_fwd_kernel[(grid)](x, 
                                y,
                                alpha, 
                                gemma, 
                                beta,
                                ctx.HAVE_BETA,
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
        dx = torch.empty_like(dy)

        kwargs = {"BLOCK_N": min(triton.next_power_of_2(N), 1024), "num_warps":4, "num_stages": 4}
        grid = lambda meta: (triton.cdiv(N, meta['BLOCK_N']), NUM_SMS)
        _dyt_bwd_kernel[grid](  dy, 
                                dx,
                                da, 
                                dg, 
                                x,
                                alpha,
                                gemma,
                                M,
                                N,
                                **kwargs
                                )
        db = None
        if ctx.HAVE_BETA:
            db = dy.sum(0)
        dg = dg.sum(0).to(gemma.dtype)
        da = da.sum().to(x.dtype).unsqueeze(0)
        return dx, da, dg, db

# @torch.compile    
def torch_dyt_with_beta(x, alpha, gemma, beta):
    return gemma * torch.tanh(x * alpha) + beta

# @torch.compile    
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


        