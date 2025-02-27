import torch
from torch.nn.attention import flex_attention
import math
from functools import partial


def torch_blcok_compress(x, weight, pe_embedding, stride):
    '''
    Args:
        x (Tensor): [bs, n, h, d]
        weight (Parameters): [kernel_size], 貌似可以加个维度h，作用看下面代码
        pe_embeding (Parameters): [kernel_size, d], 貌似也可以加个维度h，类似bert中的pe
        stride (int): 论文中的d
    Return:
        compress_x (Tensor): [bs, num_blocks, h, d]
    '''
    x = x.transpose(1,2)
    B, H, N, D = x.shape
    kernel_size = len(weight)  # 论文中的L，idx从0开始，最大idx+1就是num_blocks
    num_blocks = (N - kernel_size) // stride + 1

    # [bs, h, num_blocks, kernel_size, D]
    block_x = torch.cat(
        [
        torch.roll(x, shifts=-1 * idx * stride, dims=-2)[:, :, :num_blocks*stride]
        .reshape(B, H, num_blocks, stride, -1)[:, :, :, :min(stride, kernel_size-idx*stride)] 
        for idx in range(math.ceil(kernel_size/stride))
        ], 
        axis=-2
        )
    # print(block_x.shape)
    # 每个block加上embedding
    block_x = block_x + pe_embedding[None, None, None, :, :]
    # 压缩的话，直接乘上一个weight，然后求和，这是目前最简单的想法，之后可以用Linear之类的再对compress_x进行处理
    # 为什么说这样压缩合理呢，block_x明显是要比x更占内存的，如果有一个求和的过程
    # 使用triton时直接输入x，就直接输出compress_x了，一步完成
    # 而不是先对block_x乘上一个Linear做变换，然后再压缩，这样更消耗显存
    compress_x = (block_x * weight[None, None, None, :, None]).sum(-2)
    return compress_x.transpose(1, 2)

def compress_mask(score, batch, head, q_idx, k_idx, kernel_size, stride, value):
    # 具体怎么用请去torch官网搜索flex_attention
    # 这个输入的k_idx实际是压缩后的k的block_idx
    # 对于第一个block包含[0, 31]内容，那么只有q_idx>=31才能“看见”,0-30是看不见的
    # 找每个k_block的包含原始的k_idx的最大值，例如31,63，95...等
    true_k_idx = k_idx * stride + kernel_size - 1
    return score - (q_idx<true_k_idx) * value # “看不见”就让attn_score减去个inf

def flex_cmp_attn(q, k, v, kernel_size, stride, sm_scale=None):
    n = q.size(1)
    q = q.transpose(1,2)
    k = k.transpose(1,2)
    v = v.transpose(1,2)

    def compress_mask(score, batch, head, q_idx, k_idx, kernel_size, stride, value):
        # 具体怎么用请去torch官网搜索flex_attention
        # 这个输入的k_idx实际是压缩后的k的block_idx
        # 对于第一个block包含[0, 31]内容，那么只有q_idx>=31才能“看见”,0-30是看不见的
        # 找每个k_block的包含原始的k_idx的最大值，例如31,63，95...等
        true_k_idx = k_idx * stride + kernel_size - 1
        return score - (q_idx<true_k_idx) * value # “看不见”就让attn_score减去个inf
    
    score_mod = partial(compress_mask, 
                        kernel_size=kernel_size, 
                        stride=stride, 
                        value=torch.tensor(torch.finfo(q.dtype).max, device=q.device)
                        )
    o = flex_attention.flex_attention(q, k, v, score_mod=score_mod, enable_gqa=True, return_lse=False)
    q_idx = torch.arange(n, device=q.device, dtype=torch.int32)
    # q_idx在[0, 30]这个范围内，一个k block都是看不见的，给mask成0，
    # flex_attn的结果不让原地修改，没法用mask_fill_，只能这么mask
    o = o * (q_idx>=(kernel_size-1))[None, None, :, None]
    return o.transpose(1, 2)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def torch_cmp_attn(q, k, v, kernel_size, stride, sm_scale=None):
    q = q.transpose(1,2)
    k = k.transpose(1,2)
    v = v.transpose(1,2)
    b, qh, n, d = q.shape
    _, kh, m, vd = v.shape
    if sm_scale is None:
        sm_scale = d**-0.5
    k = repeat_kv(k, qh//kh)
    v = repeat_kv(v, qh//kh)

    s = (q @ k.transpose(-1, -2)) * sm_scale
    # mask和上面同理，找block_k_idx的包含的真实k_idx的最后一个值
    q_idx = torch.arange(n, device=q.device, dtype=torch.int32)
    block_idx = torch.arange(m, device=q.device, dtype=torch.int32)
    k_idx = block_idx * stride + kernel_size - 1
    mask = q_idx[:, None] < k_idx[None, :]
    s.masked_fill_(mask[None, None, :, :], torch.finfo(q.dtype).min)
    p = s.softmax(-1, dtype=torch.float32).to(s.dtype)
    o = p @ v
    # 同理，把前q_idx为[0, 30]的给mask掉
    o.masked_fill_((q_idx<(kernel_size-1))[None, None, :, None], 0)
    return o.transpose(1, 2)

@torch.inference_mode()
def torch_select_for_fwd(q, k, lse, kernel_size, stride, select_size, top_n, sm_scale=None):
    q = q.transpose(1,2)
    k = k.transpose(1,2)
    b, qh, n, d = q.shape
    _, kh, m, _ = k.shape
    if sm_scale is None:
        sm_scale = d**-0.5
    k = repeat_kv(k, qh//kh)
    q_idx = torch.arange(n, device=q.device, dtype=torch.int32)
    block_idx = torch.arange(m, device=q.device, dtype=torch.int32)
    k_idx = block_idx * stride + kernel_size - 1

    s = (q @ k.transpose(-1, -2))
    mask = q_idx[:, None] < k_idx[None, :]
    s.masked_fill_(mask[None, None, :, :], torch.finfo(q.dtype).min)
    p = torch.exp(s.float() * sm_scale - lse.unsqueeze(-1))
    # print(p.max())
    num_selcct_blocks = math.ceil(n / select_size)
    select_probs = torch.zeros(b, qh, n, num_selcct_blocks, device=q.device, dtype=torch.float32)
    for select_idx in range(num_selcct_blocks):
        acc_probs = select_probs[:, :, :, select_idx]
        select_start = select_idx * select_size
        select_end = min(select_start+select_size, n)
        compress_start_idx = max((select_start-kernel_size) // stride + 1, 0)
        compress_start = compress_start_idx * stride
        while compress_start < select_end and compress_start + kernel_size <= n:
            compress_end = compress_start + kernel_size
            area = min(compress_end, select_end) - max(compress_start, select_start)
            acc_probs += p[:, :, :, compress_start_idx] * area / stride
            compress_start_idx += 1
            compress_start += stride
    select_probs = select_probs.view(b, kh, -1, n, num_selcct_blocks).sum(2)
    select_probs[:, :, torch.arange(n), torch.arange(n)//select_size] = 9999
    values, indices = torch.topk(select_probs, k=top_n, dim=-1)
    for start in range(0, n, select_size):
        indices[:, :, start:start+select_size, (start+select_size)//select_size:] = num_selcct_blocks
    return select_probs, indices

def torch_select_attn(q, k, v, select_size, select_indices, sm_scale=None):
    q = q.transpose(1,2)
    k = k.transpose(1,2)
    v = v.transpose(1,2)
    b, qh, n, d = q.shape
    _, kh, m, vd = v.shape
    if sm_scale is None:
        sm_scale = d**-0.5
    k = repeat_kv(k, qh//kh)
    v = repeat_kv(v, qh//kh)

    s = (q @ k.transpose(-1, -2)) * sm_scale
    mask = None
    
    causal_mask = torch.ones(n, m, dtype=torch.int32, device=q.device).tril().bool()
    ignore_index = select_indices[0][0][0][-1]
    select_mask = torch.zeros(b, qh, n, ignore_index + 1, dtype=torch.int32, device=q.device)
    select_indices = repeat_kv(select_indices, qh//kh)
    select_mask.scatter_(-1, select_indices, 1)
    select_mask = select_mask.repeat_interleave(select_size, -1)[..., :m].contiguous().bool()
    mask = causal_mask[None, None, :, :] & select_mask
    # print(mask.shape, s.shape)
    s.masked_fill_(mask.logical_not(), float('-inf'))
    # lse = torch.logsumexp(s, -1)
    p = s.softmax(-1, dtype=torch.float32).to(s.dtype)
    o = p @ v
    return o.transpose(1, 2)

