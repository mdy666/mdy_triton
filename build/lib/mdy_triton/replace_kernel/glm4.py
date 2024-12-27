import importlib
from ..core.fused_add_norm import triton_fused_add_norm
from ..core.fused_silu import triton_fused_up_gate_silu_no_split
from ..core.rmsnorm import triton_rmsnorm
# rope实现太逆天了，没法加进去
# from ..core.fused_apply_rope import fused_apply_rope
from ..core.cross_entyopy_loss import fast_cross_entropy_loss
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, Union, Tuple, List
import torch
import sys

# 添加glm模型的存储位置
sys.path.append('/mnt/workspace/mdy/models')
module = importlib.import_module('glm-4-9b-chat.modeling_chatglm')

# 在主文件中也需要这样进行导入，如果有更优雅的办法请告诉我
# import sys
# sys.path.append('/mnt/workspace/mdy/models')
# module = importlib.import_module('glm-4-9b-chat.modeling_chatglm')
# model = module.ChatGLMForConditionalGeneration.from_pretrained(model_path, _attn_implementation='flash_attention_2',
#                                            device_map='cuda', torch_dtype=dtype, trust_remote_code=True)


def rmsnorm_forward(self, hidden_states):
    # print(hidden_state.device, self.weight.device, self.weight.dtype)
    return triton_rmsnorm(hidden_states, self.weight, self.eps)

def mlp_forward(self, hidden_states):      
    output = self.dense_4h_to_h(
        triton_fused_up_gate_silu_no_split(
            self.dense_h_to_4h(hidden_states), 'gate-up'
                                            )
                                )
    return output

def decoder_layer_forward(
            self, hidden_states, attention_mask, rotary_pos_emb, kv_cache=None, use_cache=True,
    ):
    # hidden_states: [s, b, h]
    # Layer norm at the beginning of the transformer layer.
    layernorm_output = self.input_layernorm(hidden_states)
    # Self attention.
    attention_output, kv_cache = self.self_attention(
        layernorm_output,
        attention_mask,
        rotary_pos_emb,
        kv_cache=kv_cache,
        use_cache=use_cache
    )

    # Residual connection.
    if self.apply_residual_connection_post_layernorm:
        residual = layernorm_output
    else:
        residual = hidden_states

    layernorm_input = torch.nn.functional.dropout(attention_output, p=self.hidden_dropout, training=self.training)

    layernorm_output, layernorm_input = triton_fused_add_norm(layernorm_input, 
                                                    residual, 
                                                    self.post_attention_layernorm.weight, 
                                                    self.post_attention_layernorm.eps)

    # MLP.
    mlp_output = self.mlp(layernorm_output)

    # Second residual connection.
    if self.apply_residual_connection_post_layernorm:
        residual = layernorm_output
    else:
        residual = layernorm_input

    output = torch.nn.functional.dropout(mlp_output, p=self.hidden_dropout, training=self.training)
    output = residual + output

    return output, kv_cache

# loss copy from unsloth/unsloth/models/llama.py
# loss copy from unsloth/unsloth/models/llama.py
# loss copy from unsloth/unsloth/models/llama.py
# loss copy from unsloth/unsloth/models/llama.py
# loss copy from unsloth/unsloth/models/llama.py
def causal_forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_last_logit: Optional[bool] = False,
        **loss_kwargs,
    ):
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    transformer_outputs = self.transformer(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    hidden_states = transformer_outputs[0]
    if return_last_logit:
        hidden_states = hidden_states[:, -1:]
    lm_logits = self.transformer.output_layer(hidden_states)

    loss = None
    if labels is not None:
        shift_logits = lm_logits
        if not hasattr(self, "extra_ignored_labels") or shift_logits.size(1) > self.max_seq_length:
            # Fixes https://github.com/unslothai/unsloth/issues/10
            self.max_seq_length = shift_logits.size(1)
            self.extra_ignored_labels = torch.full((self.max_seq_length, 1), -100, device = shift_logits.device)
        shift_labels = torch.hstack((labels[..., 1:], self.extra_ignored_labels[:labels.shape[0]]))
        loss = fast_cross_entropy_loss(shift_logits, shift_labels,
                                        n_items = loss_kwargs.get("num_items_in_batch", None) or loss_kwargs.get("n_items", None))

    if not return_dict:
        output = (lm_logits,) + transformer_outputs[1:]
        return ((loss,) + output) if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=lm_logits,
        past_key_values=transformer_outputs.past_key_values,
        hidden_states=transformer_outputs.hidden_states,
        attentions=transformer_outputs.attentions,
    )

module.RMSNorm.forward = rmsnorm_forward
module.MLP.forward = mlp_forward
module.GLMBlock.forward = decoder_layer_forward
module.ChatGLMForConditionalGeneration.forward = causal_forward
trigger = None