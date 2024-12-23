import torch
import torch.distributed as dist
from train_model.utils import DistributedDS, MyTrainer, print_rank0
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModelForCausalLM, AutoConfig
import time
import argparse
import sys
import os
import json
import random

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',type=str, default='qwen2-vl')
    parser.add_argument('--data_paths', nargs='+')
    parser.add_argument('--deepspeed', action='store_true')
    parser.add_argument('--fsdp', action='store_true')
    parser.add_argument('--micro_batch_size', type=int, default=4)
    parser.add_argument('--global_batch_size', type=int, default=32)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--replace_kernel', action='store_true')
    parser.add_argument('--max_seq_len', type=int, default=1024)
    parser.add_argument('--max_steps', type=int, default=3000)
    parser.add_argument('--pretrain', action='store_true')
    #剩下自行添加
    return parser.parse_args()



if __name__ == '__main__':
    dist.init_process_group(backend='nccl')
    seed = 42
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
        from replace_llama3_kernel import trigger
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

    print_rank0('====================loading datasets=======================') 
    # set mask_labels=False for pretrain
    ds = DistributedDS(args.data_paths, tokenizer, max_seq_len=args.max_seq_len, padding_side='left', mask_labels=not args.pretrain, rank=rank, world_size=world_size)
    
    print_rank0('====================loading model=======================')
    model = AutoModelForCausalLM.from_pretrained(args.model_path,  _attn_implementation='flash_attention_2', torch_dtype=dtype, device_map=device)
    # for pretrain
    if rank == 0 and args.pretrain:
        for name, module in model.named_children():
            # 如果是线性层或卷积层，重置参数并初始化
            if isinstance(module, torch.nn.Linear):
                module.reset_parameters()
                torch.nn.init.kaiming_normal_(module.weight)
    dist.barrier()

    print_rank0('====================start trainging=======================')
    train_args = TrainingArguments(output_dir=args.output_dir,
                            per_device_train_batch_size=args.micro_batch_size,
                            # per_device_eval_batch_size=args.batch_size_per_device * 2,
                            # evaluation_strategy='steps',
                            # eval_steps=100,
                            save_strategy='no',
                            # save_steps=1000,
                            # save_total_limit=10,
                            # num_train_epochs=2,
                            logging_strategy='steps',
                            logging_steps=1,
                            dataloader_num_workers=16,
                            dataloader_prefetch_factor=2,
                            learning_rate=5e-4,             
                            gradient_accumulation_steps=acc_grad_steps,
                            lr_scheduler_type='cosine',
                            max_steps=args.max_steps,
                            save_steps=args.max_steps,
                            warmup_steps=500,
                            disable_tqdm=False,
                            weight_decay=0.01,
                            seed=42,
                            logging_dir=args.output_dir,
                            # save_safetensors=False,
                            # save_only_model=True,
                            deepspeed='train_model/zero2.json' if args.deepspeed else None,
                            # fsdp='full_shard' if args.,
                            # fsdp_config={
                            #     "sharding_strategy": "FULL_SHARD",  # 全分片策略
                            #     "cpu_offload": False,               # 是否将参数卸载到 CPU
                            #     "min_num_params": 1e7,              # 最小参数数量以启用分片
                            #     "fp32_reduce_scatter": False,       # 是否在 reduce-scatter 中使用 fp32
                            #     "compute_dtype": torch.bfloat16,     # 计算数据类型
                            #     "buffer_dtype": torch.bfloat16,      # 缓冲区数据类型
                            #     "sharded_ddp": True,                # 是否启用分片数据并行
                            # },
                            bf16=True,
                            # fp16=True,
                            save_total_limit=1,
                            save_only_model=True,
                        )
    trainer = MyTrainer(model=model,
                      args=train_args,
                      data_collator=ds.process_batch,
                      train_dataset=ds
                      )
    dist.barrier()
    trainer.train()
    if rank == 0:
        with open(os.path.join(args.output_dir, 'log_history.json'), 'w') as f:
            for line in trainer.state.log_history:
                f.write(json.dumps(line, ensure_ascii=False)+'\n')




