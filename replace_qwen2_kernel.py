import importlib
from core.fused_add_norm import triton_fused_add_norm
from core.fused_silu import triton_fused_up_gate_silu
from core.rmsnorm import triton_rmsnorm
module = importlib.import_module('transformers.models.qwen2.modeling_qwen2')

def rmsnorm_forward(self, hidden_state):
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

module.Qwen2RMSNorm.forward = rmsnorm_forward
module.Qwen2MLP.forward = mlp_forward
module.Qwen2DecoderLayer = decoder_layer_forward

trigger = None