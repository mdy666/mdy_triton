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
    

# @triton.autotune([triton.Config({}, num_stages=ns, num_warps=nw)
#                   for ns in [1,2,4]
#                   for nw in [4, 8, 16, 32]
#                   ],
#                   key=['BLOCK_N'])
@triton.jit
def _dyt_fwd_kernel(X,
                    Y,
                    Alpha,
                    Gemma,
                    Beta,
                    N:tl.constexpr,
                    BLOCK_N:tl.constexpr=64
                    ):
    row_id = tl.cast(tl.program_id(0), tl.int64)
    col = tl.arange(0, BLOCK_N)
    mask = col < N

    alpha = tl.load(Alpha).to(tl.float32)
    gemma = tl.load(Gemma + col, mask=mask, other=0.).to(tl.float32)  
    beta = tl.load(Beta + col, mask=mask, other=0.).to(tl.float32) 
    x = tl.load(X + row_id * N + col, mask=mask, other=0.).to(tl.float32) 

    tanh_x = tanh(alpha * x)
    y = tanh_x * gemma + beta
    tl.store(Y + row_id * N + col, y, mask=mask)



# @triton.autotune([triton.Config({}, num_stages=ns, num_warps=nw)
#                   for ns in [1,2,4]
#                   for nw in [4, 8, 16, 32]
#                   ],
#                   key=['BLOCK_N'])
@triton.jit
def _dyt_bwd_kernel(DY,
                    DA,
                    DG,
                    DB,
                    X,
                    Alpha,
                    Gemma,
                    M,
                    N:tl.constexpr,
                    BLOCK_N:tl.constexpr
                    ):
    start_row_id = tl.cast(tl.program_id(0), tl.int64)
    col = tl.arange(0, BLOCK_N)
    mask = col < N
    alpha = tl.load(Alpha).to(tl.float32)
    gemma = tl.load(Gemma + col, mask=mask, other=0.).to(tl.float32)  
    # beta = tl.load(Beta + col, mask=mask, other=0.).to(tl.float32) 

    da = 0.
    dg = tl.zeros((BLOCK_N,), dtype=tl.float32)
    db = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for row_id in range(start_row_id, M, tl.num_programs(0), tl.num_programs(0)):
        x = tl.load(X + row_id * N + col, mask=mask, other=0.).to(tl.float32) 
        dy = tl.load(DY + row_id * N + col, mask=mask, other=0.).to(tl.float32)
        tanh_x = tanh(alpha * x)
        db += dy
        dg += dy * tanh_x
        tmp = (1 - tanh_x * tanh_x) * dy * gemma
        da += tl.sum(x * tmp, 0)
        dx = alpha * tmp
        tl.store(DY + row_id * N + col, dx, mask=mask)
    tl.store(DA + start_row_id, da)
    tl.store(DG + start_row_id * N + col, dg, mask=mask)
    tl.store(DB + start_row_id * N + col, db, mask=mask)
    # y = tanh(x)
    # tl.store(Y + off_m[:, None] * N + off_n[None, :], y, mask=(off_m[:, None] < M) & (off_n[None, :] < N))



class _DYT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha, gemma, beta):
        ctx.input_shape = x.shape
        x = x.view(-1, ctx.input_shape[-1])
        M, N = x.shape
        
        y = torch.empty_like(x)

        # if method == 1:
        #     kwargs = {'BLOCK_N':256, "BLOCK_M": 32, "num_warps":8, "num_stages": 1}
        #     grid = lambda meta: (triton.cdiv(N, meta['BLOCK_N']), triton.cdiv(M, meta['BLOCK_M']))
        #     _dyt_fwd_kernel[grid](x, 
        #                         y,
        #                         alpha, 
        #                         gemma, 
        #                         beta,
        #                         M, 
        #                         N,
        #                         # **kwargs
        #                         )
        # else:

        kwargs = {"num_warps":16, "num_stages": 1}
        BLOCK_N = triton.next_power_of_2(N)
        _dyt_fwd_kernel[(M,)](x, 
                                y,
                                alpha, 
                                gemma, 
                                beta,
                                N,
                                BLOCK_N,
                                **kwargs,
                                )
        
        ctx.save_for_backward(x, alpha, gemma, beta)
        return y.view(ctx.input_shape)
    
    @staticmethod
    def backward(ctx, dy):
        x, alpha, gemma, beta = ctx.saved_tensors
        M, N = x.shape
        NUM_SMS = torch.cuda.get_device_properties('cuda').multi_processor_count
        da = torch.empty(NUM_SMS, 1, dtype=torch.float32, device=x.device)
        dg = torch.empty(NUM_SMS, N, dtype=torch.float32, device=x.device)
        db = torch.empty(NUM_SMS, N, dtype=torch.float32, device=x.device)

        BLOCK_N = triton.next_power_of_2(N)

        kwargs = {"num_warps":32, "num_stages": 2}
        _dyt_bwd_kernel[(NUM_SMS,)](dy, 
                                    da, 
                                    dg, 
                                    db,
                                    x,
                                    alpha,
                                    gemma,
                                    M,
                                    N,
                                    BLOCK_N,
                                    **kwargs)
        return dy, da.sum(0).to(x.dtype), dg.sum(0).to(x.dtype), db.sum(0).to(x.dtype)
    

class DYT(torch.nn.Module):
    def __init__(self, dim, init_a=0.5):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.ones(1) * init_a)
        self.gemma = torch.nn.Parameter(torch.ones(dim))
        self.beta = torch.nn.Parameter(torch.zeros(dim))

    def forward(self, x, backend='triton'):
        if backend == "triton":
            return _DYT.apply(x, self.alpha, self.gemma, self.beta)
        else:
            return self.gemma * torch.tanh(x * self.alpha) + self.beta


        