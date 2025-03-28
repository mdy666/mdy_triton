import importlib
import transformer_engine as te

from .swiglu import swiglu_impl

from .bf16mv_adam import TritonAdamW
from .manger import CustomManger
from .dpsk_linear import Linear
from .dpsk_layernorm_linear import LayerNormLinear
from .dpsk_grouped_linear import GroupedLinear, Fp8Padding, Fp8Unpadding

# 先换TE的Linear，防止部分megatron先导入te的linear
# 训练moe模型时，建议不要使用FP8的GroupedLinear, 因为前期每个expert的token数目一直在变化，
# deep_gemm的kernel都是即时编译，当有大量cache时，速度才能恢复正常。
def patch(swiglu=True,
          cross_entropy_loss=True,
          bf16mv_adam=True,
          dpsk_fp8=True,
          zb2p=True,
          fa3=True,
          ):

    # print("start")
    if dpsk_fp8:
        CustomManger.fp8 = True
        module_te_linear = importlib.import_module('transformer_engine.pytorch')
        module_te_linear.Linear = Linear
        module_te_linear.LayerNormLinear = LayerNormLinear
        module_te_linear.GroupedLinear = GroupedLinear

        def set_is_first_microbatch(self):
            """Sets the is_first_microbatch flag if it exists and config.fp8==True.
            When this flag is set, TE modules will update their fp8 parameter cache.
            """
            if not hasattr(self, "modules_with_is_first_microbatch"):
                self.modules_with_is_first_microbatch = []
                for m in self.modules():
                    if hasattr(m, "is_first_microbatch"):
                        self.modules_with_is_first_microbatch.append(m)
            for m in self.modules_with_is_first_microbatch:
                m.is_first_microbatch = True

        base_module = importlib.import_module('megatron.core.transformer.module')
        base_module.MegatronModule.set_is_first_microbatch = set_is_first_microbatch

        def forward(
            self, permuted_local_hidden_states, tokens_per_expert
        ):
            """Forward of TEGroupedMLP

            Args:
                permuted_local_hidden_states (torch.Tensor): The permuted input hidden states of the
                local experts.
                tokens_per_expert (torch.Tensor): The number of tokens per expert.

            Return:
                output (torch.Tensor): The output of the local experts.
            """
            if not hasattr(self, "fp8_padding"):
                self.fp8_padding = Fp8Padding(self.num_local_experts)
                self.fp8_unpadding = Fp8Unpadding(self.num_local_experts)
                
            tokens_per_expert = tokens_per_expert.tolist()
            actual_tokens_per_expert = tokens_per_expert
            permuted_local_hidden_states, tokens_per_expert = self.fp8_padding(
                permuted_local_hidden_states, tokens_per_expert
            )

            intermediate_parallel, bias_parallel = self.linear_fc1(
                permuted_local_hidden_states, tokens_per_expert
            )

            intermediate_parallel = swiglu_impl(intermediate_parallel, bias_parallel)


            output, output_bias = self.linear_fc2(intermediate_parallel, tokens_per_expert)


            output = self.fp8_unpadding(output, actual_tokens_per_expert)

            return output, output_bias
        module_te_expert = importlib.import_module('megatron.core.transformer.moe.experts')
        module_te_expert.TEGroupedMLP.forward = forward

    if swiglu:
        moudel_swiglu = importlib.import_module('megatron.core.fusions.fused_bias_swiglu')
        moudel_swiglu.bias_swiglu_impl = swiglu_impl
        # print("swiglu")
    


    if bf16mv_adam:
        moudel_te_optimizer = importlib.import_module('transformer_engine.pytorch.optimizers')
        moudel_te_optimizer.FusedAdam = TritonAdamW
        # print("adam")



    if zb2p:
        from .zb2p import zb2p_v2
        # assert dpsk_fp8
        module_pp = importlib.import_module('megatron.core.pipeline_parallel.schedules')
        module_pp.forward_backward_pipelining_without_interleaving = zb2p_v2
        # print("zb2p")

    if fa3:
        from transformer_engine.pytorch import attention
        try:
            from flash_attn.flash_attn_interface import flash_attn_func, flash_attn_varlen_func
            from flash_attn.flash_attn_interface import (
                _flash_attn_varlen_forward as flash_attn_varlen_fwd,
            )
            from flash_attn.flash_attn_interface import (
                _flash_attn_varlen_backward as flash_attn_varlen_bwd,
            )
            from flash_attn_2_cuda import varlen_bwd as flash_attn_cuda_bwd

            attention.flash_attn_func = flash_attn_func
            attention.flash_attn_varlen_func = flash_attn_varlen_func
            attention.flash_attn_varlen_fwd = flash_attn_varlen_fwd
            attention.flash_attn_varlen_bwd = flash_attn_varlen_bwd
            attention.flash_attn_cuda_bwd = flash_attn_cuda_bwd
            attention._flash_attn_version = attention.PkgVersion("2.6.3")

            attention._flash_attn_is_installed = True
            attention._flash_attn_2_plus = True
            attention._flash_attn_2_1_plus = True
            attention._flash_attn_2_3_plus = True
            attention._flash_attn_2_4_plus = True
            attention._flash_attn_2_4_1_plus = True
            attention._flash_attn_2_5_7_plus = True
            attention._flash_attn_2_6_0_plus = True
        
        except:
            print("not install FA2")

        try:
            from flash_attn_interface import flash_attn_func as flash_attn_func_v3
            from flash_attn_interface import (
                flash_attn_varlen_func as flash_attn_varlen_func_v3,
            )
            from flash_attn_interface import (
                _flash_attn_forward as flash_attn_varlen_fwd_v3,
            )
            from flash_attn_interface import (
                _flash_attn_backward as flash_attn_varlen_bwd_v3,
            )

            
            attention._flash_attn_3_is_installed = True
            attention._use_flash_attn_3 = True
            attention.flash_attn_func_v3 = flash_attn_func_v3
            attention.flash_attn_varlen_func_v3 = flash_attn_varlen_func_v3
            attention.flash_attn_varlen_fwd_v3 = flash_attn_varlen_fwd_v3
            attention.flash_attn_varlen_bwd_v3 = flash_attn_varlen_bwd_v3
        except:
            print("not install FA3")
            pass

    if cross_entropy_loss:
        from .cross_entropy_losss import fast_cross_entropy_loss
        module_ce = importlib.import_module('megatron.core.tensor_parallel')
        module_ce.vocab_parallel_cross_entropy = fast_cross_entropy_loss
        module_fused_ce = importlib.import_module('megatron.core.fusions.fused_cross_entropy')
        module_fused_ce.fused_vocab_parallel_cross_entropy = fast_cross_entropy_loss
        # print("cross_entropy_loss")