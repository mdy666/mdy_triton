import numpy as np
from torch.utils.data import Dataset, DataLoader
import glob
from transformers import Trainer
import json
import torch
from transformers import Trainer
from transformers.trainer_utils import seed_worker

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
        return DataLoader(train_dataset, **dataloader_params)
    
class DistributedDS(Dataset):
    def __init__(self, data_paths: str, tokenizer, max_seq_len=2048, rank=0, world_size=1):
        super().__init__()

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        if isinstance(data_paths, str):
            data_paths = [data_paths]
        self.messages = []
        for path in data_paths:
            for file_path in sorted(glob.glob(path)):
                print(file_path)
                if not file_path.endswith('.json') and not file_path.endswith('.jsonl'):
                    continue
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
        labels = np.full_like(input_ids, -100)
        attn_mask = np.ones_like(labels)

        start_locs = np.where(input_ids==self.tokenizer.bos_token_id)[0]
        end_locs = np.where(input_ids==self.tokenizer.eos_token_id)[0]
        for idx in range(2, len(start_locs), 2):
            start = start_locs[idx]
            end = end_locs[idx]
            labels[start+3:end+1] = input_ids[start+3:end+1]

        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[:self.max_seq_len]
            attn_mask = attn_mask[:self.max_seq_len]
            labels = labels[:self.max_seq_len]
        else:
            num_pad = self.max_seq_len - len(input_ids)
            input_ids = np.pad(input_ids, [0, num_pad], constant_values=(0, self.tokenizer.eos_token_id))
            labels = np.pad(labels, [0, num_pad], constant_values=(0, -100))
            attn_mask = np.pad(attn_mask, [0, num_pad], constant_values=(0, 0))

        return {'input_ids': input_ids,
                'attention_mask': attn_mask,
                'labels': labels}

    @classmethod
    def process_batch(cls, batch):
        input_ids = torch.from_numpy(np.stack([batch[i]['input_ids'] for i in range(len(batch))], axis=0)).long()
        attention_mask = torch.from_numpy(np.stack([batch[i]['attention_mask'] for i in range(len(batch))], axis=0)).long()
        labels= torch.from_numpy(np.stack([batch[i]['labels'] for i in range(len(batch))], axis=0)).long()
        return {'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels}
