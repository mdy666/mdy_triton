import torch
import torch.distributed as dist
from dataset_and_processor import CustomProcessor, get_batch, Qwen2VLDatasets
from transformers import TrainingArguments, Trainer, AutoConfig, Qwen2VLForConditionalGeneration

import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',type=str, default='qwen2-vl')
    parser.add_argument('--data_path', type=str, default='flickr8k/')
    parser.add_argument('--deepspeed_config', type=str, default='ds_config.json')
    parser.add_argument('--save_model_path', type=str, default='my_qwen2')
    #剩下自行添加
    return parser.parse_args()

def print_rank0(*args):
    if dist.get_rank() == 0:
        print(*args)

if __name__ == '__main__':
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    device = torch.device('cuda', rank)
    args = get_args()

    processor = CustomProcessor.from_pretrained(args.model_path)
    processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    processor.tokenizer.padding_side = 'right'
    
    train_ds = Qwen2VLDatasets(args.data_path, processor, max_length=512, padding='max_length', truncation=True)
    
    print_rank0('====================loading model=======================')
    config = AutoConfig.from_pretrained(args.model_path)
    model = Qwen2VLForConditionalGeneration(config)
    #我没下模型，下载好了的直接from_pretrained()
    model.to(torch.bfloat16).to(device)
    model.train()

    train_args = TrainingArguments(output_dir=args.save_model_path,
                            per_device_train_batch_size=2,
                            # per_device_eval_batch_size=args.batch_size_per_device * 2,
                            # evaluation_strategy='steps',
                            # eval_steps=100,
                            save_strategy='steps',
                            save_steps=1000,
                            save_total_limit=10,
                            num_train_epochs=2,
                            logging_strategy='steps',
                            logging_steps=10,
                            dataloader_num_workers=8,
                            learning_rate=1e-5,             
                            # gradient_accumulation_steps=1,
                            # lr_scheduler_type='linear',
                            warmup_steps=500,
                            disable_tqdm=False,
                            weight_decay=0.01,
                            save_safetensors=False,
                            save_only_model=True,
                            deepspeed=args.deepspeed_config,
                            bf16=True
                        )
    trainer = Trainer(model=model,
                      args=train_args,
                      data_collator=get_batch,
                      train_dataset=train_ds
                      )
    trainer.train()




