import tqdm
import time
import json
import argparse

import torch
from transformers import AutoModelForCausalLM, Qwen2ForCausalLM
# from triton_grpo_loss.core import triton_grpo_loss
from triton_grpo_loss.decouple_logp_and_loss import triton_grpo_loss, fused_selective_log_softmax
from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--global_bs', type=int, default=1024)
    parser.add_argument('--mbs', type=int, default=16)
    parser.add_argument('--mmbs', type=int, default=4)
    parser.add_argument('--seq_len', type=int, default=2048)

    parser.add_argument('--model', type=str, default="/sharedata/mdy/models/DeepSeek-R1-Distill-Qwen-1.5B")

    parser.add_argument('--loss', type=str, choices=['my', 'liger'], default="my")

    args = parser.parse_args()
    assert args.global_bs % args.mbs == 0 and args.mbs % args.mmbs == 0

    return args

if __name__ == '__main__':
    args = get_args()
    device = 'cuda'

    # It should set B = args.mbs, then slice each part. But we just set B = args.mmbs
    B = args.mbs
    T = args.seq_len

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda")
    lin_weight = model.lm_head.weight
    input_ids = torch.randint(0, 1000-1, (B, T+100), dtype=torch.int64, device=device)
    completion_ids = input_ids[:, -T:].contiguous()
    advantages = torch.randn(B, device=device, dtype=torch.float32)
    ref_logp = torch.randn(B, T, device=device, dtype=torch.float32)
    old_logp = torch.randn(B, T, device=device, dtype=torch.float32)
    completion_mask = torch.ones_like(completion_ids, dtype=torch.int32)

    
    liger_loss_fn = LigerFusedLinearGRPOLoss(compiled=True, 
                                            chunk_size=1,
                                            temperature=0.9)

    pbar = tqdm.tqdm(total=args.global_bs)
    start_time = time.time()
    for mbs_idx in range(args.global_bs // args.mbs):
        total_token_this_mbs = completion_mask.sum()
        for start in range(0, args.mbs, args.mmbs):
            end = start + args.mmbs
            if args.loss == 'my':
                logits = model(input_ids[start:end], logits_to_keep=T+1).logits
                per_token_logps = fused_selective_log_softmax(logits, 
                                                            completion_ids[start:end], 
                                                            0.9, 
                                                            completion_mask[start:end]
                                                            )
                # trtton faster than compile code a little, you can change the loss when you need.
                per_token_loss, per_token_kl, is_clipped = triton_grpo_loss(per_token_logps, 
                                                                            advantages[start:end],
                                                                            old_logp[start:end] if old_logp is not None else None,
                                                                            ref_logp[start:end] if ref_logp is not None else None,
                                                                            )
                # per_token_loss = triton_grpo_loss(logits,
                #                 old_logp[start:end],
                #                 ref_logp[start:end],
                #                 completion_ids[start:end],
                #                 advantages[start:end],
                #                 completion_mask[start:end])[0]
                loss = (per_token_loss * completion_mask[start:end]).sum() / total_token_this_mbs
                loss.backward()
                logits.data = torch.Tensor()
                del logits

            else:
                _input = model.model(input_ids=input_ids[start:end],
                                    attention_mask=None, ).last_hidden_state[:, -(T+1):-1]
                # the loss is not right, it should use total_token_this_mbs to ruduce the loss
                loss, _ = liger_loss_fn(_input,
                                        lin_weight,
                                        completion_ids[start:end],
                                        completion_mask[start:end],
                                        advantages[start:end],
                                        ref_per_token_logps=ref_logp[start:end],
                                        old_per_token_logps=old_logp[start:end],
                                       )
                loss.backward()
            pbar.update(args.mmbs)
        # there is a optimizer
        # optimizer.step()

    memory_allocated = torch.cuda.memory_reserved()

    total_time = time.time() - start_time
    infos = {"global_bs": args.global_bs,
             "micro_bs": args.mbs,
             "micro_micro_bs": args.mmbs,
             "seq_len": args.seq_len,
             "loss": args.loss,
             "time(s)": total_time,
             "sample/s": round(args.global_bs / total_time, 2),
             "memory(G)": round(memory_allocated / 1024**3, 2)
             }
    
    print(infos)
    fin = open('./infos.jsonl', 'a+')
    fin.write(json.dumps(infos, ensure_ascii=False)+'\n')
    fin.close
    



