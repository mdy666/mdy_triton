import triton
import triton.language as tl
import torch
import torch.distributed as dist
from copy import deepcopy
import os
from functools import partial
import argparse
from tqdm import tqdm
import importlib
from mdy_triton.core import fast_cross_entropy_loss

import sys
# 添加Megatron仓库的路径
sys.path.append('/data/repo/Megatron-LM')
module = importlib.import_module('megatron.core.fusions.fused_cross_entropy')
module.get_tensor_model_parallel_group = lambda: None
module.get_tensor_model_parallel_rank = dist.get_rank
module.get_tensor_model_parallel_world_size = dist.get_world_size


command = '''
torchrun --nproc-per-node 8 test_ce.py \
    --func speed \
    --ce_type triton \
    --inplace \
    --sft

torchrun --nproc-per-node 8 test_ce.py \
    --func acc \
    --inplace \
    --sft

--func speed是测试速度，对应ce_type，megatron和triton可以多卡，剩下只能单卡。acc是对应精度测试
--ce_type triton or torch or megatron or unsloth。单卡就设置--nproc-per-node 1，多卡就设置大于1
--inplace grad是否是logit的视图，节省显存
--sft label中是否含有-100

'''
@triton.jit
def _cross_entropy_fwd_kernel(LOGITS, LABELS, LOSSES, LOGSUMEXP,
                             vocab_start_index, row_stride, 
                             M, N, SPLIT, BLOCK_SIZE: tl.constexpr, 
                             ):
    row_idx = tl.program_id(0)
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
                             vocab_start_index, row_stride, 
                             M, N,  INPLACE,
                             BLOCK_SIZE: tl.constexpr,
                             ):
    row_idx = tl.program_id(0)
    LABELS += row_idx
    label_idx = tl.load(LABELS).to(tl.int32)
    row_stride = row_stride.to(tl.int64)
    if (label_idx != -100):
        # label_idx -= vocab_start_index
        LOGITS += row_idx * row_stride
        DLOGITS += row_idx * row_stride
        LOGSUMEXP += row_idx
        DLOSSES += row_idx
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
        ctx.input_shape = logits.shape
        # tp_rank = 0
        # tp_size = 1
        tp_rank = dist.get_rank()
        tp_size = dist.get_world_size()
        tp_group = None
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
        logits, labels, logsumexp = ctx.saved_tensors
        dlogits = logits if ctx.inplace else torch.zeros_like(logits)
        N = logits.size(-1)
        logits = logits.view(-1, N)
        M = logits.size(0)
        vocab_start_index = N * ctx.tp_rank
        with torch.cuda.device(logits.device):
            _cross_entropy_bwd_kernel[(M,)](dlosses, dlogits, 
                                            logits, labels, logsumexp,
                                            vocab_start_index, logits.stride(0),
                                            M, N, ctx.inplace, 
                                            BLOCK_SIZE=32768, num_warps=32, num_stages=1
                                            )
        return dlogits.view(*ctx.input_shape), None, None
    
def triton_entropy_loss(logits, labels, inplace=False):
    return _FastCrossEntropyLoss.apply(logits, labels, inplace)

def test_speed(args):
    dtype = torch.bfloat16
    device = torch.device('cuda', args.rank)
    bs = 4
    seq_len = 1024
    vocab_size = 80000
    vocab_size = (vocab_size // args.world_size)*args.world_size
    part_vocab_size = vocab_size // args.world_size
    iters = 2000

    logits = torch.randn(bs, seq_len, part_vocab_size, dtype=dtype, device=device)
    logits.requires_grad_(True)
    if args.sft:
        labels = torch.randint(0, vocab_size*2-1, (bs, seq_len), device=device) - vocab_size
        labels.masked_fill_(labels<0, -100)
    else:
        labels = torch.randint(0, vocab_size-1, (bs, seq_len), device=device)
    dy = torch.randn(bs * seq_len, dtype=dtype, device=device)

    if args.ce_type == 'unsloth':
        loss_func = fast_cross_entropy_loss
    elif args.ce_type == 'megatron':
        loss_func = module.fused_vocab_parallel_cross_entropy
    elif args.ce_type == 'torch':
        loss_func = torch.nn.CrossEntropyLoss(reduce=False)
    elif args.ce_type == 'triton':
        loss_func = partial(triton_entropy_loss, inplace=args.inplace)
    else:
        assert False
    
    pbar = tqdm(total=iters) if rank == 0 else None
    for i in range(iters):
        if args.ce_type == 'torch':
            out = loss_func(logits.view(-1, logits.size(-1)).float(), labels.view(-1)).view(-1)
        else:
            out = loss_func(logits, labels).view(-1)
        if args.ce_type == 'unsloth':
            out.backward()
        else:
            out.backward(dy)
        if pbar is not None:
            pbar.update(1)
        
    memory_stats = torch.cuda.memory_stats()
    max_memory = memory_stats['allocated_bytes.all.peak']
    print(f"Peak GPU memory usage: {max_memory / 1024 ** 2:.2f} MB")

def test_acc(args):
    dtype = torch.bfloat16
    device = torch.device('cuda', args.rank)
    bs = 4
    seq_len = 512
    vocab_size = 1500
    vocab_size = (vocab_size // args.world_size)*args.world_size
    part_vocab_size = vocab_size // args.world_size

    logits1 = torch.randn(bs, seq_len, part_vocab_size, dtype=dtype, device=device)
    logits1.requires_grad_(True)
    logits2 = deepcopy(logits1)
    if args.sft:
        labels = torch.randint(0, vocab_size*2-1, (bs, seq_len), device=device) - vocab_size
        labels.masked_fill_(labels<0, -100)
    else:
        labels = torch.randint(0, vocab_size-1, (bs, seq_len), device=device)
    dy = torch.randn(bs * seq_len, dtype=dtype, device=device)
    dist.broadcast(dy, src=0)
    dist.broadcast(labels, src=0)

    # triton result
    y1 = triton_entropy_loss(logits1, labels, args.inplace).view(-1)
    y1.backward(dy)

    # megatron result
    y2:torch.Tensor = module.fused_vocab_parallel_cross_entropy(logits2, labels).view(-1)
    y2.masked_fill_(labels.view(-1) == -100, 0.)
    y2.backward(dy)

    # torch result
    # 需要将其它卡上的logit的聚集起来，然后计算
    # 作为gold进行对比
    tensor_list = [torch.empty_like(logits2) for _ in range(world_size)]
    dist.all_gather(tensor_list, logits2)
    gather_logits3 = torch.cat(tensor_list, axis=-1).to(torch.float32)
    gather_logits3.requires_grad_(True)
    y3 = torch.nn.CrossEntropyLoss(reduce=False)(gather_logits3.view(-1, vocab_size), labels.view(-1))
    y3.backward(dy)

    # 把triton结果的梯度聚集起来
    tensor_list = [torch.empty_like(logits1.grad) for _ in range(world_size)]
    dist.all_gather(tensor_list, logits1.grad)
    gather_logits1_grad = torch.cat(tensor_list, axis=-1).to(torch.float32)

    # 把megatron结果的梯度聚集起来
    tensor_list = [torch.empty_like(logits2.grad) for _ in range(world_size)]
    dist.all_gather(tensor_list, logits2.grad)
    gather_logits2_grad = torch.cat(tensor_list, axis=-1).to(torch.float32)

    atol = 1e-3
    rtol = 1e-3
    if args.rank == 0:
        print('='*80 + ' loss:')
        print(y1)
        print(y2)
        print(y3)
        print(torch.allclose(y1.to(torch.float32), y2.to(torch.float32), atol=atol, rtol=rtol))
        print(torch.allclose(y1.to(torch.float32), y3.to(torch.float32), atol=atol, rtol=rtol))
        print('\n\n')
        print('='*80 + ' grad:')
        atol = 1e-2
        rtol = 1e-2
        print(torch.allclose(gather_logits1_grad, gather_logits2_grad, atol=atol, rtol=rtol))
        print(torch.allclose(gather_logits1_grad, gather_logits3.grad, atol=atol, rtol=rtol))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ce_type', default='triton', type=str)
    parser.add_argument('--inplace', action='store_true')
    parser.add_argument('--func', default='acc', type=str)
    parser.add_argument('--sft', action='store_true')
    args = parser.parse_args()

    dist.init_process_group()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    args.rank = rank
    args.world_size = world_size

    if args.func == 'acc':
        test_acc(args)
    elif args.func == 'speed':
        test_speed(args)
    else:
        assert False
    dist.barrier()





