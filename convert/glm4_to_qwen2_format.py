import os
import math
import json
import time
import argparse

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config

# 只将模型转换为Qwen2，tokenizer不变，自动将glm4的tokenizer copy过去
command = '''
python convert/glm4_to_qwen2_format.py \
   --model_path /mnt/workspace/mdy/models/glm-4-9b-chat \
   --save_path /mnt/workspace/mdy/models/glm-4-qwen2-format \
   --test
'''

t1 = time.time()
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str)
parser.add_argument('--save_path', type=str)
parser.add_argument('--save_num_files', type=int, default=10)
parser.add_argument('--test', action='store_true')
args = parser.parse_args()

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)


def torch_copy(x):
    y = torch.empty(*x.shape, device=x.device, dtype=x.dtype)
    y.data.copy_(x)
    y = y.to(torch.bfloat16).cpu()
    return y

# 这个config只适用于glm4-9b
config = Qwen2Config()
config.intermediate_size = 13696
config.vocab_size = 151552
config.max_position_embeddings = 131072
config.num_hidden_layers = 40
config.num_key_value_heads = 2
config.rope_theta = 10000 * 500
config.torch_dtype = 'bfloat16'
config.transformers_version = '4.43.1'
config.rms_norm_eps = 1.5625e-07
config.eos_token_id = [151329, 151336, 151338]
config.architectures= ["Qwen2ForCausalLM"],

print('loading model ........')
model = AutoModelForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)
glm_state_dict = model.state_dict()

num_layers = config.num_hidden_layers
num_q_head = config.num_attention_heads
num_kv_head = config.num_key_value_heads
head_dim = config.hidden_size // num_q_head

file_id = 1
save_files = args.save_num_files
num_layers_per_file = math.ceil(num_layers/save_files)
n_bytes = 0
index_map = {}
qwen2_state_dict = {}

print('start transform ........')
for layer_idx in range(num_layers):
    if layer_idx == 0:
        qwen2_state_dict['model.embed_tokens.weight'] = torch_copy(glm_state_dict['transformer.embedding.word_embeddings.weight'])

    qkv = glm_state_dict[f'transformer.encoder.layers.{layer_idx}.self_attention.query_key_value.weight']
    qk, v = qkv.split([(num_kv_head+num_q_head)*head_dim, num_kv_head*head_dim], 0)
    qk = qk.reshape(num_kv_head+num_q_head, head_dim, -1)
    qk = torch.cat([qk[:, ::2, :], qk[:, 1::2]], axis=1).reshape((num_kv_head+num_q_head)*head_dim, -1)
    q, k = qk.split([num_q_head*head_dim, num_kv_head*head_dim], 0)
    qwen2_state_dict[f'model.layers.{layer_idx}.self_attn.q_proj.weight'] = torch_copy(q)
    qwen2_state_dict[f'model.layers.{layer_idx}.self_attn.k_proj.weight'] = torch_copy(k)
    qwen2_state_dict[f'model.layers.{layer_idx}.self_attn.v_proj.weight'] = torch_copy(v)
    qkv_bias = glm_state_dict[f'transformer.encoder.layers.{layer_idx}.self_attention.query_key_value.bias']
    # print(qkv_bias.shape)
    qk_bias, v_bias = qkv_bias.split([(num_kv_head+num_q_head)*head_dim, num_kv_head*head_dim], 0)
    # print(qk_bias.shape)
    qk_bias = qk_bias.reshape(num_kv_head+num_q_head, head_dim)
    qk_bias = torch.cat([qk_bias[:, ::2], qk_bias[:, 1::2]], axis=1).reshape((num_kv_head+num_q_head)*head_dim)
    q_bias, k_bias = qk_bias.split([num_q_head*head_dim, num_kv_head*head_dim], 0)
    qwen2_state_dict[f'model.layers.{layer_idx}.self_attn.q_proj.bias'] = torch_copy(q_bias)
    qwen2_state_dict[f'model.layers.{layer_idx}.self_attn.k_proj.bias'] = torch_copy(k_bias)
    qwen2_state_dict[f'model.layers.{layer_idx}.self_attn.v_proj.bias'] = torch_copy(v_bias)

    o = glm_state_dict[f'transformer.encoder.layers.{layer_idx}.self_attention.dense.weight']
    qwen2_state_dict[f'model.layers.{layer_idx}.self_attn.o_proj.weight'] = torch_copy(o)
    
    gate, up = glm_state_dict[f'transformer.encoder.layers.{layer_idx}.mlp.dense_h_to_4h.weight'].chunk(2, dim=0)
    qwen2_state_dict[f'model.layers.{layer_idx}.mlp.gate_proj.weight'] = torch_copy(gate)
    qwen2_state_dict[f'model.layers.{layer_idx}.mlp.up_proj.weight'] = torch_copy(up)
    down = glm_state_dict[f'transformer.encoder.layers.{layer_idx}.mlp.dense_4h_to_h.weight']
    qwen2_state_dict[f'model.layers.{layer_idx}.mlp.down_proj.weight'] = torch_copy(down)

    input_layernorm = glm_state_dict[f'transformer.encoder.layers.{layer_idx}.input_layernorm.weight']
    qwen2_state_dict[f'model.layers.{layer_idx}.input_layernorm.weight'] = torch_copy(input_layernorm)
    post_layernorm = glm_state_dict[f'transformer.encoder.layers.{layer_idx}.post_attention_layernorm.weight']
    qwen2_state_dict[f'model.layers.{layer_idx}.post_attention_layernorm.weight'] = torch_copy(post_layernorm)

    if layer_idx == num_layers-1:
        qwen2_state_dict['lm_head.weight'] = torch_copy(glm_state_dict['transformer.output_layer.weight'])
        qwen2_state_dict['model.norm.weight'] = torch_copy(glm_state_dict['transformer.encoder.final_layernorm.weight'])
        
    if ((layer_idx + 1) % num_layers_per_file == 0) or (layer_idx == num_layers - 1):
        chunk_filename = os.path.join(args.save_path, f"model-{str(file_id).zfill(5)}-of-{str(save_files).zfill(5)}.safetensors")
        print(chunk_filename)
        file_id += 1
        save_file(qwen2_state_dict, chunk_filename, metadata={'format':'pt'})
        index_map.update({key:chunk_filename.split('/')[-1] for key in qwen2_state_dict})
        n_bytes += sum([value.nbytes for key, value in qwen2_state_dict.items()])
        qwen2_state_dict = {}

os.system(f'cp {os.path.join(args.model_path, "token*")} {args.save_path}')
config.save_pretrained(args.save_path)

with open(os.path.join(args.save_path, 'model.safetensors.index.json'), 'w') as f:
    f.write(json.dumps({"metadata": {"total_size": n_bytes},
                        "weight_map": index_map}, ensure_ascii=False, indent=4))

print('transform and save done: {:.2f}s'.format(time.time()-t1))

if args.test:
    print('start test ........')
    model_path = args.save_path
    dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                _attn_implementation='flash_attention_2',
                                                 device_map='cuda', torch_dtype=dtype, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    text = [{'role':'user','content':'你是谁？介绍一下你自己'}]
    input_ids = tokenizer.apply_chat_template(text, add_generation_prompt=True, tokenize=True, return_tensors='pt').cuda()
    output = model.generate(input_ids, max_new_tokens=100)
    print('输入：')
    print(tokenizer.decode(input_ids[0]))
    print('输出：')
    print(tokenizer.decode(output[0][len(input_ids[0]):]))

'''
输入：
[gMASK] <sop> <|user|> 
你是谁？介绍一下你自己 <|assistant|>
输出：

您好！我是一个人工智能助手，名叫 ChatGLM。我是基于清华大学 KEG 实验室和智谱 AI 公司于 2024 年共同训练的语言模型 GLM-4 开发的。

我的主要功能是回答用户的问题、提供信息、辅助学习和工作等。 <|user|>
'''