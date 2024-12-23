import time
import argparse
import sys
import os
import json
import random
import math
from functools import partial
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel, ShardingStrategy, MixedPrecision
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    # enable_wrap,
    # wrap,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
# from torch.cuda.amp import GradScaler, autocast
from train_model.utils import DistributedDS, print_rank0
from transformers import  AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer


def get_lr_lambda(max_steps, warm_up_steps, min_rate=0.05, lr_type='cosine'):
    assert lr_type in ['cosine', 'linear']
    if lr_type == 'cosine':
        def func(step, max_steps, warm_up_steps, min_rate):
            if step <=warm_up_steps:
                factor = step/warm_up_steps
            else:
                factor = min(math.cos((math.pi/2)*(step-warm_up_steps)/(max_steps-warm_up_steps)), min_rate)
            return factor
    elif lr_type == 'linear':
        def func(step, max_steps, warm_up_steps, min_rate):
            if step <=warm_up_steps:
                factor = step/warm_up_steps
            else:
                factor = min((max_steps-step)/(max_steps-warm_up_steps), min_rate)
            return factor
    
    return partial(func, max_steps=max_steps, warm_up_steps=warm_up_steps, min_rate=min_rate)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',type=str, default='qwen2-vl')
    parser.add_argument('--data_paths', nargs='+')
    parser.add_argument('--micro_batch_size', type=int, default=4)
    parser.add_argument('--global_batch_size', type=int, default=32)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--replace_kernel', action='store_true')
    parser.add_argument('--max_seq_len', type=int, default=1024)
    parser.add_argument('--max_steps', type=int, default=3000)
    parser.add_argument('--sync_dp_grad', action='store_true')
    #剩下自行添加
    return parser.parse_args()

if __name__ == '__main__':
    dist.init_process_group(backend='nccl')
    seed = 40
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print_rank0(rank, world_size)
    time.sleep(3)

    args = get_args()
    if args.replace_kernel:
        from replace_qwen2_kernel import trigger
        # from replace_llama_kernel import trigger
    print_rank0(args.data_paths)
    print_rank0(sys.path)
    print_rank0(args)
    time.sleep(3)

    assert args.global_batch_size % (args.micro_batch_size * world_size) == 0
    acc_grad_steps = args.global_batch_size // (args.micro_batch_size * world_size)

    device = torch.device('cuda', rank)
    dtype = torch.bfloat16

    print_rank0('====================loading tokenizer=======================') 
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    # base模型的bos和eos可能是预训练时候的，需要换，下面是qwen2 instruct的
    tokenizer.bos_token_id = 151644
    tokenizer.eos_token_id = 151645

    print_rank0('====================loading datasets=======================') 
    ds = DistributedDS(args.data_paths, tokenizer, max_seq_len=args.max_seq_len, padding_side='left', mask_labels=True, rank=rank, world_size=world_size)
    train_dl = DataLoader(ds, batch_size=args.global_batch_size // world_size, collate_fn=ds.process_batch,
                          shuffle=True, num_workers=32, prefetch_factor=2)
    dl_len = len(train_dl)
    def warp_dl(dataloader):
        while True:
            for inputs in dataloader:
                yield inputs


    print_rank0('====================loading model=======================')
    model = AutoModelForCausalLM.from_pretrained(args.model_path,  _attn_implementation='flash_attention_2', device_map=device)
    dist.barrier()
    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            Qwen2DecoderLayer,
        },
    )
    bfSixteen = MixedPrecision(
        param_dtype=dtype,
        # Gradient communication precision.
        reduce_dtype=dtype,
        # Buffer precision.
        buffer_dtype=dtype,
    )
    model = FullyShardedDataParallel(model, 
                                     auto_wrap_policy=auto_wrap_policy,
                                     device_id=device,
                                     sharding_strategy=ShardingStrategy.SHARD_GRAD_OP, 
                                     mixed_precision=bfSixteen)

    # for pretrain
    # if rank == 0:
    #     for name, module in model.named_children():
    #         # 如果是线性层或卷积层，重置参数并初始化
    #         if isinstance(module, torch.nn.Linear):
    #             module.reset_parameters()
    #             torch.nn.init.kaiming_normal_(module.weight)
    # dist.barrier()
    lr = 5e-5
    lr_type = 'cosine'
    max_steps = args.max_steps
    warm_up_steps = 300
    min_rate = 0.05
    lr_func = get_lr_lambda(max_steps=max_steps, warm_up_steps=warm_up_steps, min_rate=min_rate, lr_type=lr_type)
    optimizer = AdamW(model.parameters(), lr=lr)
    lr_scheduler = LambdaLR(optimizer, lr_func)

    print_rank0('====================start trainging=======================')
    step = 0
    # log_history = []
    args.sync_dp_grad
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    if rank == 0:
        pbar = tqdm(max_steps, desc='training')
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)
        fout = open(os.path.join(args.output_dir, 'log_history.json'), 'w')
    start_train_time = time.time()
    for batch in warp_dl(train_dl):
        start_step_time = time.time()
        batch = {key: batch[key].to(device) for key in batch}
        num_tokens_for_loss = (batch['labels'] != -100).sum().to(torch.bfloat16)
       
        if args.sync_dp_grad:
            dist.all_reduce(num_tokens_for_loss, op=dist.ReduceOp.SUM)
            num_tokens_for_loss /= world_size

        micro_batch_chunks = {key: batch[key].chunk(acc_grad_steps) for key in batch}
        
        loss_and_token = torch.tensor([0, num_tokens_for_loss.item()], device=device, dtype=torch.bfloat16)
        optimizer.zero_grad()
        for idx in range(acc_grad_steps):
            micro_batch = {key: micro_batch_chunks[key][idx] for key in micro_batch_chunks}
            shift_labels = micro_batch.pop('labels')[:, 1:].contiguous()
            out = model(**micro_batch)
            shift_logits = out.logits[:, :-1, :].contiguous()
            loss = loss_fn(shift_logits.view(-1, shift_logits.shape[-1]), 
                            shift_labels.view(-1)) / num_tokens_for_loss
            loss.backward()
            loss_and_token[0] += loss.detach()
        optimizer.step()
        lr_scheduler.step()
        dist.all_reduce(loss_and_token, op=dist.ReduceOp.SUM)
        step += 1
        logs = {'step': step, 
                'loss': round(loss_and_token[0].item() / world_size, 4), 
                'num_tokens': int(loss_and_token[1].item()), 
                'time':round(time.time() - start_step_time, 3),
                'lr': lr_scheduler.get_last_lr()[0], 
                'epoch': round(step / dl_len, 3)}
        print_rank0(logs)
        if rank == 0:
            fout.write(json.dumps(logs, ensure_ascii=False)+'\n')
            pbar.update(1)
        if step >= max_steps:
            break

    print_rank0('====================end trainging=======================')
    total_time = time.time() - start_train_time
    logs = f'iters: {max_steps}, total_time: {total_time:.2f}s, iters/s: {max_steps/total_time:.2f}'
    print_rank0(logs)
    if rank == 0:
        fout.write(json.dumps(logs, ensure_ascii=False)+'\n')
        fout.close()








            
        











