import numpy as np
from torch.utils.data import Dataset, DataLoader
import glob
from transformers import Trainer
import json
import torch
import os
from transformers.trainer_utils import seed_worker
from typing import Union, List

def print_rank0(*args):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(*args)
    else:
        print(*args)



def find_all_json_files(data_paths):
    all_json_files = []
    for path in data_paths:
        if path.endswith(".json") or path.endswith('.jsonl'):
            all_json_files.append(path)
            continue
        json_files = []
        for root, dirs, files in os.walk(path):
            for file in sorted(files):
                if file.endswith(".json") or file.endswith('.jsonl'):
                    json_files.append(os.path.join(root, file))
        all_json_files.extend(json_files)
    return all_json_files

class MyTrainer(Trainer):
    def get_train_dataloader(self):
        train_dataset = self.train_dataset
        data_collator = self.data_collator


        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        dataloader_params["sampler"] = self._get_train_sampler()
        dataloader_params["drop_last"] = self.args.dataloader_drop_last
        dataloader_params["worker_init_fn"] = seed_worker
        dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
        # accelerate会加载distributed sampler，不需要了
        return DataLoader(train_dataset, **dataloader_params)


# 每个卡加载不同的数据，不需要设置Distributed Sampler了，节省内存， 传入json文件或者，必须是messages数据 
# 目前数据处理方式只支持llama和qwen2
class DistributedDS(Dataset):
    def __init__(self, data_paths: Union[str, List[str]], tokenizer, max_seq_len=2048, padding_side='left', mask_labels=True, rank=0, world_size=1):
        super().__init__()
        '''
            args:
                data_paths: 可以是一个json文件，也可以是多个。也可以是一个目录，自动读取目录下所有json文件
            
            example:
                1. 'xxx.jsonl'
                2. ['xxx.json', 'json_dir/']

            data_format:
                必须是messages的json_line格式
                {"messages": [{"role": "user", "content": "user_content"}, 
                              {"role": "assistant", "content": "assistant_content"}, 
                                xxxx,
                                xxxx,
                                xxxx]}
        '''

        self.tokenizer = tokenizer
        if 'qwen' in tokenizer.name_or_path.lower():
            self.tokenizer_type = 'qwen'
            self.tokenizer.bos_token_id = 151644
            self.tokenizer.eos_token_id = 151645
        elif 'llama' in tokenizer.name_or_path.lower():
            self.tokenizer_type = 'llama'
            self.tokenizer.bos_token_id = 128006
            self.tokenizer.eos_token_id = 128009
        else:
            assert 'only support qwen or llama'
        
        self.max_seq_len = max_seq_len
        self.padding_side = padding_side
        self.mask_labels = mask_labels
        if isinstance(data_paths, str):
            data_paths = [data_paths]

        self.messages = []
        for file_path in find_all_json_files(data_paths):
            print_rank0(file_path)
            with open(file_path, 'r') as f:
                for line in f.readlines():
                    self.messages.append(json.loads(line))

        length = len(self.messages)
        if world_size > 1:
            num_samples_per_rank = length // world_size
            self.messages = self.messages[rank * num_samples_per_rank: (rank+1) * num_samples_per_rank]
        
    def __len__(self):
        return len(self.messages)
    
    def __getitem__(self, index):
        messages = self.messages[index]['messages']
        input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=True, return_tensors='pt')[0]
        attn_mask = np.ones_like(input_ids)
        position_ids = np.arange(len(input_ids))

        if self.mask_labels:
            labels = np.full_like(input_ids, -100)
            start_locs = np.where(input_ids==self.tokenizer.bos_token_id)[0]
            offset = 3
            if self.tokenizer_type == 'llama':
                start_locs = start_locs[1:]
                offset = 4
            end_locs = np.where(input_ids==self.tokenizer.eos_token_id)[0]
            for idx in range(2, len(start_locs), 2):
                start = start_locs[idx]
                end = end_locs[idx]
                labels[start+offset:end+1] = input_ids[start+offset:end+1]
        else:
            labels = input_ids

        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
            attn_mask = attn_mask[:self.max_seq_len]
            labels = labels[:self.max_seq_len]
            position_ids = position_ids[:self.max_seq_len]
        else:
            if self.padding_side == 'left':
                num_pad = self.max_seq_len - len(input_ids)
                input_ids = np.pad(input_ids, [num_pad, 0], constant_values=(self.tokenizer.eos_token_id, 0))
                labels = np.pad(labels, [num_pad, 0], constant_values=(-100, 0))
                attn_mask = np.pad(attn_mask, [num_pad, 0], constant_values=(0, 0))
                position_ids = np.pad(position_ids, [num_pad, 0], constant_values=(0, 0))
            else:
                num_pad = self.max_seq_len - len(input_ids)
                input_ids = np.pad(input_ids, [0, num_pad], constant_values=(0, self.tokenizer.eos_token_id))
                labels = np.pad(labels, [0, num_pad], constant_values=(0, -100))
                attn_mask = np.pad(attn_mask, [0, num_pad], constant_values=(0, 0))
                position_ids = np.pad(position_ids, [0, num_pad], constant_values=(0, 0))

        return {'input_ids': input_ids,
                'attention_mask': attn_mask,
                'labels': labels,
                'position_ids': position_ids}

    @classmethod
    def process_batch(cls, batch):
        input_ids = torch.from_numpy(np.stack([batch[i]['input_ids'] for i in range(len(batch))], axis=0)).long()
        attention_mask = torch.from_numpy(np.stack([batch[i]['attention_mask'] for i in range(len(batch))], axis=0)).long()
        labels = torch.from_numpy(np.stack([batch[i]['labels'] for i in range(len(batch))], axis=0)).long()
        position_ids = torch.from_numpy(np.stack([batch[i]['position_ids'] for i in range(len(batch))], axis=0)).long()
        return {'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'position_ids': position_ids}
