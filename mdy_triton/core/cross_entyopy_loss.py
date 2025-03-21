import triton
import triton.language as tl
import torch
import torch.distributed as dist
import sys
# from megatron.core.parallel_state import (
#     get_tensor_model_parallel_group,
#     get_tensor_model_parallel_rank,
#     get_tensor_model_parallel_world_size,
# )

@triton.jit
def _cross_entropy_fwd_kernel(LOGITS, LABELS, LOSSES, LOGSUMEXP,
                             vocab_start_index, row_stride, 
                             M, N, SPLIT, BLOCK_SIZE: tl.constexpr, 
                             ):
    row_idx = tl.cast(tl.program_id(0), tl.int64)
    row_stride = row_stride.to(tl.int64)
    label_idx = tl.load(LABELS + row_idx).to(tl.int32)
    if (label_idx != -100):
        LOGITS += row_idx * row_stride
        base_cols = tl.arange(0, BLOCK_SIZE)
        m_i = -float("inf")
        l_i = 0.0
        for start_n in tl.range(0, N, BLOCK_SIZE):
            cols = start_n + base_cols
            mask = cols < N
            logits = tl.load(LOGITS+cols, mask=mask, other=-float('inf')).to(tl.float32)
            m_ij = tl.max(logits)
            new_m_i = tl.maximum(m_i, m_ij)
            l_i = l_i * tl.exp(m_i - new_m_i) + tl.sum(tl.exp(logits - new_m_i))
            m_i = new_m_i
        lse = tl.log(l_i) + m_i

        if (label_idx >= vocab_start_index) and (label_idx < (vocab_start_index + N)):
            x = -1.0 * tl.load(LOGITS+label_idx-vocab_start_index).to(tl.float32)
            if not SPLIT:
                loss = lse + x
                tl.store(LOSSES+row_idx, loss)
            else:
                tl.store(LOSSES+row_idx, x)
        tl.store(LOGSUMEXP+row_idx, lse)

@triton.jit
def _cross_entropy_bwd_kernel(DLOSSES, DLOGITS,
                            LOGITS, LABELS, LOGSUMEXP,
                             vocab_start_index, row_stride, dloss_row_stride,
                             M, N,  INPLACE,
                             BLOCK_SIZE: tl.constexpr,
                             ):
    row_idx = tl.cast(tl.program_id(0), tl.int64)
    LABELS += row_idx
    label_idx = tl.load(LABELS).to(tl.int32)
    row_stride = row_stride.to(tl.int64)
    if (label_idx != -100):
        # label_idx -= vocab_start_index
        LOGITS += row_idx * row_stride
        DLOGITS += row_idx * row_stride
        LOGSUMEXP += row_idx
        DLOSSES += row_idx * dloss_row_stride
        lse = tl.load(LOGSUMEXP)
        dloss = tl.load(DLOSSES).to(tl.float32)
        base_cols = tl.arange(0, BLOCK_SIZE)
        for start_n in tl.range(0, N, BLOCK_SIZE):
            cols = start_n + base_cols
            mask = cols < N
            logits = tl.load(LOGITS+cols, mask=mask, other=0.).to(tl.float32)
            probs = tl.exp(logits - lse)
            tmp = vocab_start_index + start_n
            if (label_idx >= tmp) and (label_idx < (tmp + BLOCK_SIZE)):
                probs = tl.where(cols+vocab_start_index != label_idx, probs, probs-1.)
            tl.store(DLOGITS+cols, probs * dloss, mask=mask)
    elif INPLACE:
        DLOGITS += row_idx * row_stride
        base_cols = tl.arange(0, BLOCK_SIZE)
        zeros = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for start_n in tl.range(0, N, BLOCK_SIZE):
            cols = start_n + base_cols
            mask = cols < N
            tl.store(DLOGITS+cols, zeros, mask=mask)

class _FastCrossEntropyLoss(torch.autograd.Function):

    @staticmethod
    def forward(ctx, logits, labels, inplace):
        assert logits.is_contiguous()
        assert labels.is_contiguous()
        ctx.input_shape = logits.shape
        tp_rank = 0
        tp_size = 1
        tp_group = None
        # tp_rank = get_tensor_model_parallel_rank()
        # tp_size = get_tensor_model_parallel_world_size()
        # tp_group = get_tensor_model_parallel_group()
        N = ctx.input_shape[-1]
        logits = logits.view(-1, N)
        M = logits.size(0)
        losses = torch.zeros(M, device=logits.device, dtype=torch.float32)
        split = tp_size > 1
        vocab_start_index = N * tp_rank
        logsumexp = torch.zeros(M, device=logits.device, dtype=torch.float32)
        # print(logsumexp.stride(), losses.stride())
        with torch.cuda.device(logits.device):
            _cross_entropy_fwd_kernel[(M,)](logits, labels, losses, logsumexp,
                                                vocab_start_index, logits.stride(0),
                                                M, N, split,
                                                BLOCK_SIZE=4096, num_warps=4, num_stages=3
                                                )
        if tp_size>1:
            lse_allgather = torch.empty(tp_size, M, dtype=logsumexp.dtype, device=logsumexp.device)
            torch.distributed.all_gather_into_tensor(lse_allgather, logsumexp, group=tp_group)
            torch.distributed.all_reduce(
                losses, op=torch.distributed.ReduceOp.SUM,
            )
            logsumexp = torch.logsumexp(lse_allgather, dim=0)
            losses += logsumexp
            losses.masked_fill_(labels.view(-1)==-100, 0.)
        ctx.save_for_backward(logits, labels, logsumexp)
        ctx.inplace = inplace
        ctx.tp_rank = tp_rank
        return losses.view(*ctx.input_shape[:-1])
    
    @staticmethod
    def backward(ctx, dlosses):
        # print(dlosses.shape, dlosses.stride())
        # assert dlosses.is_contiguous()
        logits, labels, logsumexp = ctx.saved_tensors
        dlogits = logits if ctx.inplace else torch.zeros_like(logits)
        N = logits.size(-1)
        logits = logits.view(-1, N)
        M = logits.size(0)
        vocab_start_index = N * ctx.tp_rank
        BLOCK_SIZE = min(triton.next_power_of_2(N), 32768)
        with torch.cuda.device(logits.device):
            _cross_entropy_bwd_kernel[(M,)](dlosses, dlogits, 
                                            logits, labels, logsumexp,
                                            vocab_start_index, logits.stride(0),
                                            dlosses.view(-1).stride(0),
                                            M, N, ctx.inplace, 
                                            BLOCK_SIZE=BLOCK_SIZE, num_warps=32, num_stages=1
                                            )
        return dlogits.view(*ctx.input_shape), None, None
    
def triton_entropy_loss(logits, labels, num_items_in_batch=None, inplace=False):
    loss =  _FastCrossEntropyLoss.apply(logits, labels, inplace)
    if num_items_in_batch is None:
        num_items_in_batch = torch.count_nonzero(labels != -100)
    return loss.sum() / num_items_in_batch