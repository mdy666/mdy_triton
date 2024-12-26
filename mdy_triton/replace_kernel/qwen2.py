import importlib
from ..core.fused_add_norm import triton_fused_add_norm
from ..core.fused_silu import triton_fused_up_gate_silu
from ..core.rmsnorm import triton_rmsnorm
from ..core.fused_apply_rope import fused_apply_rope
from ..core.cross_entyopy_loss import fast_cross_entropy_loss
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, Union, Tuple, List
import torch
from transformers import Qwen2ForCausalLM
module = importlib.import_module('transformers.models.qwen2.modeling_qwen2')

def rmsnorm_forward(self, hidden_state):
    # print(hidden_state.device, self.weight.device, self.weight.dtype)
    return triton_rmsnorm(hidden_state, self.weight, self.variance_epsilon)

def mlp_forward(self, hidden_state):
    return self.down_proj(triton_fused_up_gate_silu(self.up_proj(hidden_state),
                                                    self.gate_proj(hidden_state)))

def decoder_layer_forward(
        self,
        hidden_states,
        attention_mask = None,
        position_ids = None,
        past_key_value = None,
        output_attentions = False,
        use_cache = False,
        cache_position = None,
        position_embeddings = None,  # will become mandatory in v4.46
        **kwargs,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states, residual = triton_fused_add_norm(hidden_states, 
                                                        residual, 
                                                        self.post_attention_layernorm.weight, 
                                                        self.post_attention_layernorm.variance_epsilon)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

# loss copy from unsloth/unsloth/models/llama.py
# loss copy from unsloth/unsloth/models/llama.py
# loss copy from unsloth/unsloth/models/llama.py
# loss copy from unsloth/unsloth/models/llama.py
# loss copy from unsloth/unsloth/models/llama.py
def causal_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **loss_kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen2ForCausalLM

        >>> model = Qwen2ForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        loss = None
        if labels is not None:
            shift_logits = logits
            if not hasattr(self, "extra_ignored_labels") or shift_logits.size(1) > self.max_seq_length:
                # Fixes https://github.com/unslothai/unsloth/issues/10
                self.max_seq_length = shift_logits.size(1)
                self.extra_ignored_labels = torch.full((self.max_seq_length, 1), -100, device = logits.device)
            shift_labels = torch.hstack((labels[..., 1:], self.extra_ignored_labels[:labels.shape[0]]))
            loss = fast_cross_entropy_loss(shift_logits, shift_labels,
                                           n_items = loss_kwargs.get("num_items_in_batch", None) or loss_kwargs.get("n_items", None))
            
        # fix loss
        # loss = None
        # if labels is not None:
        #     logits = logits.to(torch.float32)
        #     shift_logits = logits[:, :-1].contiguous()
        #     vocab_size = shift_logits.size(-1)
        #     shift_labels = labels[:, 1:].contiguous()
        #     num_items_in_batch = loss_kwargs.get("num_items_in_batch", None)
        #     if num_items_in_batch is not None:
        #         loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, vocab_size), shift_labels.view(-1), reduction='sum') / num_items_in_batch
        #     else:
        #         loss = torch.nn.functional.cross_entropy(shift_logits.view(-1, vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

module.apply_rotary_pos_emb = fused_apply_rope
module.Qwen2RMSNorm.forward = rmsnorm_forward
module.Qwen2MLP.forward = mlp_forward
module.Qwen2DecoderLayer.forward = decoder_layer_forward
module.Qwen2ForCausalLM.forward = causal_forward
trigger = None