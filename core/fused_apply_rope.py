import torch
import triton
import triton.language as tl



def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def rotate_half_inv(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((x2, -x1), dim=-1)


@triton.jit
def _fused_apply_rope_fwd(X, X_R, X_EMBED, COS, SIN,
                        stride_xb, stride_xl, stride_xh, stride_xd,
                        stride_cb, stride_cl, stride_cd,
                        H, D:tl.constexpr, BLOCK_H:  tl.constexpr,
                        ):
    pid = tl.program_id(0)
    x_offset = pid * stride_xl
    cos_offset = pid * stride_cl

    X += x_offset
    X_R += x_offset
    X_EMBED += x_offset
    COS += cos_offset
    SIN += cos_offset
    cols = tl.arange(0, D)
    cos_ptrs = COS + cols
    sin_ptrs = SIN + cols

    x_ptrs = tl.make_block_ptr(
        base=X,
        shape=(H, D),
        offsets=(0, 0),
        strides=(stride_xh, stride_xd),
        block_shape=(BLOCK_H, D),
        order=(1,0)
    )
    xr_ptrs = tl.make_block_ptr(
        base=X_R,
        shape=(H, D),
        offsets=(0, 0),
        strides=(stride_xh, stride_xd),
        block_shape=(BLOCK_H, D),
        order=(1,0)
    )
    xembed_ptrs = tl.make_block_ptr(
        base=X_EMBED,
        shape=(H, D),
        offsets=(0, 0),
        strides=(stride_xh, stride_xd),
        block_shape=(BLOCK_H, D),
        order=(1,0)
    )
    cos = tl.load(cos_ptrs)
    sin = tl.load(sin_ptrs)
    x = tl.load(x_ptrs, boundary_check=(0,), padding_option='zero')
    xr = tl.load(xr_ptrs, boundary_check=(0,), padding_option='zero')
    x_embed = x * cos[None, :] + xr * sin[None, :]
    tl.store(xembed_ptrs, x_embed, boundary_check=(0,))

class _FusedApplyRope(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin):
        q = q.transpose(1,2)
        k = k.transpose(1,2)
        assert q.is_contiguous() and k.is_contiguous()
        qh = q.size(2)
        kh = k.size(2)
        x = torch.cat([q,k],axis=2)
        x_rotate_half = rotate_half(x)
        B, L, H, D = x.shape
        assert (D % 32 == 0) or (D % 64 == 0) or (D % 128 == 0)
        BLOCK_H = triton.next_power_of_2(H)
        x_embed = torch.empty_like(x)
        num_warps=8
        num_stages=1 
        M = B*L
        _fused_apply_rope_fwd[(M,)](x, x_rotate_half, x_embed, cos, sin,
                        *x.stride(),
                        *cos.stride(),
                        H,D, BLOCK_H,
                        num_warps=num_warps, num_stages=num_stages

        )
        q_embed, k_embed = x_embed.split([qh, kh], dim=2)
        # ctx.infos = (B, H, N, D)
        ctx.save_for_backward(cos, sin)
        # ctx.num_warps = num_warps
        # ctx.num_stages = num_stages
        return q_embed.transpose(1,2), k_embed.transpose(1,2)
    
    @staticmethod
    def backward(ctx, dq_embed, dk_embed):
        cos,sin = ctx.saved_tensors
        c = (cos + rotate_half_inv(sin)).unsqueeze(1)
        return dq_embed*c, dk_embed*c, None, None

fused_apply_rope = _FusedApplyRope.apply

