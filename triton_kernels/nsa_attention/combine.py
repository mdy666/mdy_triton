import torch
import triton
import triton.language as tl
import math

# @triton.autotune([triton.Config({'BLOCK_N': bsn}, num_stages=ns, num_warps=nw)
#                  for bsn in [8, 16]
#                  for ns in [1, 2, 4]
#                  for nw in [1,2,4, 8, 16]
#                  ], key=['N', "D"])
@triton.jit
def _fused_sigmoid_combine_fwd_kernel(P,
                                      W,
                                      O,
                                      N:tl.constexpr,
                                      D:tl.constexpr,
                                      BLOCK_N:tl.constexpr=32,
                                      CHUNK_N:tl.constexpr=1024
                                      ):
    start_n = tl.cast(tl.program_id(0), tl.int64) * CHUNK_N + tl.program_id(1) * BLOCK_N
    if start_n >= N:
        return
    off_n = start_n + tl.arange(0, BLOCK_N)
    mask = off_n < N

    acc = tl.zeros((BLOCK_N, D), dtype=tl.float32)
    offset = off_n[:, None] * D + tl.arange(0, D)[None, :]
    for i in range(3):
        p = tl.load(P+i).to(tl.pointer_type(O.dtype.element_ty))
        o = tl.load(p + offset, mask=mask[:, None], other=0.).to(tl.float32)
        w = tl.load(W + off_n * 3 + i, mask=mask, other=0.).to(tl.float32)
        acc += o * tl.sigmoid(w)[:, None]
    tl.store(O + offset, acc, mask=mask[:, None])

# @triton.autotune([triton.Config({'BLOCK_N': bsn}, num_stages=ns, num_warps=nw)
#                  for bsn in [8, 16]
#                  for ns in [1, 2, 4]
#                  for nw in [1,2,4, 8, 16]
#                  ], key=['N', "D"])
@triton.jit
def _fused_sigmoid_combine_bwd_kernel(DP,
                                      DW,
                                      DO,
                                      P,
                                      W,
                                      N:tl.constexpr,
                                      D:tl.constexpr,
                                      BLOCK_N:tl.constexpr=32,
                                      CHUNK_N:tl.constexpr=1024
                                      ):
    start_n = tl.cast(tl.program_id(0), tl.int64) * CHUNK_N + tl.program_id(1) * BLOCK_N
    if start_n >= N:
        return
    off_n = start_n + tl.arange(0, BLOCK_N)
    mask = off_n < N

    offset = off_n[:, None] * D + tl.arange(0, D)[None, :]
    dcombine_o = tl.load(DO + offset, mask=mask[:, None], other=0.).to(tl.float32)
    for i in range(3):
        p = tl.load(P+i).to(tl.pointer_type(DO.dtype.element_ty))
        dp = tl.load(DP+i).to(tl.pointer_type(DO.dtype.element_ty))
        o = tl.load(p + offset, mask=mask[:, None], other=0.).to(tl.float32)
        w = tl.load(W + off_n * 3 + i, mask=mask, other=0.).to(tl.float32)
        sigmoid_w = tl.sigmoid(w)
        do = dcombine_o * sigmoid_w[:, None]
        dsigmoid_w = tl.sum(dcombine_o * o, -1)
        dw = sigmoid_w * (1 - sigmoid_w) * dsigmoid_w
        tl.store(dp + offset, do, mask=mask[:, None])
        tl.store(DW + off_n * 3 + i, dw, mask=mask)

    

class _FusedSigmoidCombine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, c, w):
        assert a.is_contiguous() and b.is_contiguous() and c.is_contiguous() and w.is_contiguous()
        B, S, H, D = a.shape
        assert w.size(-1) == 3
        assert math.log2(D).is_integer()
        o = torch.empty_like(a)
        N = B * S * H
        p = torch.tensor([a.data_ptr(), b.data_ptr(), c.data_ptr()], dtype=torch.int64, device=a.device)
        kwargs = {'BLOCK_N':16, "num_warps":8, "num_stages":2}
        grid = lambda meta: (triton.cdiv(N, meta['CHUNK_N']), triton.cdiv(meta['CHUNK_N'], meta['BLOCK_N']))
        _fused_sigmoid_combine_fwd_kernel[grid](p, 
                                                w, 
                                                o, 
                                                N, 
                                                D,
                                                **kwargs
                                                )
        ctx.save_for_backward(a, b, c, w)
        ctx.p = p
        ctx.N = N
        ctx.D = D

        return o 
    
    @staticmethod
    def backward(ctx, do):
        a, b, c, w = ctx.saved_tensors
        da = torch.empty_like(a)
        db = torch.empty_like(b)
        dc = torch.empty_like(c)
        dw = torch.empty_like(w)
        dp = torch.tensor([da.data_ptr(), db.data_ptr(), dc.data_ptr()], dtype=torch.int64, device=a.device)
        kwargs = {'BLOCK_N':8, "num_warps":4, "num_stages":4}
        grid = lambda meta: (triton.cdiv(ctx.N, meta['CHUNK_N']), triton.cdiv(meta['CHUNK_N'], meta['BLOCK_N']))
        _fused_sigmoid_combine_bwd_kernel[grid](dp, 
                                                dw,
                                                do,
                                                ctx.p,
                                                w, 
                                                ctx.N, 
                                                ctx.D,
                                                **kwargs
                                                )
        return da, db, dc, dw


def fused_sigmoid_combine(a, b, c, w):
    return _FusedSigmoidCombine.apply(a, b, c, w)