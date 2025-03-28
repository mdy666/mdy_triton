# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""GroupedLinear API"""
from typing import Union, Optional, Callable, Tuple, List, Dict, Any

import torch

import transformer_engine_torch as tex

from transformer_engine.pytorch.module.base import (
    get_multi_stream_cublas_workspace,
    TransformerEngineBaseModule,
    _2X_ACC_FPROP,
    _2X_ACC_DGRAD,
    _2X_ACC_WGRAD,
)
from transformer_engine.pytorch.fp8 import get_fp8_te_dtype, FP8GlobalStateManager
from transformer_engine.pytorch.utils import (
    divide,
    cast_if_needed,
    assert_dim_for_fp8_exec,
    clear_tensor_data,
    init_method_constant,
    requires_grad,
)
from transformer_engine.pytorch.distributed import (
    set_tensor_model_parallel_attributes,
    get_distributed_world_size,
)
from transformer_engine.pytorch.cpp_extensions import (
    cast_to_fp8,
    fp8_cast_transpose_bgrad_fused,
    fp8_multi_cast_transpose_fused,
    fp8_grouped_gemm,
    grouped_gemm,
)
from transformer_engine.pytorch.constants import GemmParallelModes, dist_group_type
from transformer_engine.pytorch.jit import no_torch_dynamo
from transformer_engine.pytorch.graph import is_graph_capturing
from transformer_engine.pytorch.tensor import Float8Tensor, QuantizedTensor
from transformer_engine.pytorch.export import is_in_onnx_export_mode
from transformer_engine.pytorch.cpu_offload import is_cpu_offload_enabled

from .manger import CustomManger

__all__ = ["GroupedLinear"]


class _GroupedLinear(torch.autograd.Function):
    """GroupedLinear semi-top level module
    Calls custom cuda extensions.
    """

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        m_splits: List[int],
        use_bias: bool,
        is_first_microbatch: Union[bool, None],
        fp8: bool,
        fp8_calibration: bool,
        fp8_meta: Dict[str, Any],
        fuse_wgrad_accumulation: bool,
        cpu_offloading: bool,
        sequence_parallel: bool,
        activation_dtype: torch.dtype,
        fp8_meta_offsets: Dict[str, int],
        is_grad_enabled: bool,
        weights_fp8: List[Union[Float8Tensor, None]],
        *weights_and_biases: Union[Float8Tensor, torch.Tensor, None],
    ) -> torch.Tensor:
        # pylint: disable=missing-function-docstring
        num_gemms = len(m_splits)
        weights = weights_and_biases[:num_gemms]
        biases = weights_and_biases[num_gemms:]

        # Make sure input dimensions are compatible
        in_features = weights[0].shape[-1]
        assert inp.shape[-1] == in_features, "GEMM not possible"
        inputmats = torch.split(inp.view(-1, in_features), m_splits)
        if fp8:
            for i in range(num_gemms):
                assert_dim_for_fp8_exec(inputmats[i])
                assert_dim_for_fp8_exec(weights[i])

        # Cast input to expected dtype
        inputmats_no_fp8 = [cast_if_needed(mat, activation_dtype) for mat in inputmats]
        inputmats = []
        inputmats_t = []
        inputmat_scale_inv = None

        inputmats = inputmats_no_fp8

        if CustomManger.fp8:
            weights = [cast_if_needed(w, activation_dtype) for w in weights]
            biases = (
                [cast_if_needed(bias, activation_dtype) for bias in biases] if use_bias else biases
            )

            CustomManger.set_first_micro_batch(weights, is_first_microbatch)
            qx, sx = CustomManger.per_token_cast_to_fp8(inp.view(-1, in_features), output_transpose=False)
            qw, sw, _, _ = CustomManger.get_fp8_weight(weights)


            out = CustomManger.fp8_grouped_matmul(qx, sx, qw, sw, m_splits)

            if use_bias:
                start = 0
                for i in range(len(m_splits)):
                    size = m_splits[i]
                    CustomManger.add_bias(out[start:start+size], biases[i])
                    # out[start:start+size] += biases[i]
                    start += size


            CustomManger.clear_tensor_data(qx, sx)


        else:
            # Cast for native AMP
            weights = [cast_if_needed(w, activation_dtype) for w in weights]
            biases = (
                [cast_if_needed(bias, activation_dtype) for bias in biases] if use_bias else biases
            )

            if fp8_calibration:
                for i in range(num_gemms):
                    # amax of input
                    amin, amax = inputmats[i].aminmax()
                    fp8_meta["scaling_fwd"].amax_history[0][fp8_meta_offsets["input"] + i] = (
                        torch.max(-amin, amax).float()
                    )
                    # amax of weight
                    amin, amax = weights[i].aminmax()
                    fp8_meta["scaling_fwd"].amax_history[0][fp8_meta_offsets["weight"] + i] = (
                        torch.max(-amin, amax).float()
                    )

            out = torch.empty(
                [sum(m_splits), weights[0].size(0)],
                dtype=activation_dtype,
                device=inputmats[0].device,
            )

            _ = grouped_gemm(
                weights,
                inputmats,
                torch.split(out, m_splits),
                activation_dtype,
                get_multi_stream_cublas_workspace(),
                bias=biases,
                use_bias=use_bias,
            )

        if is_grad_enabled:
            saved_inputmats = [None] * num_gemms
            saved_inputmats_t = [None] * num_gemms
            if weights[0].requires_grad:
                if fp8 and not fp8_meta["recipe"].override_linear_precision.wgrad:
                    if not inputmats_t:
                        saved_inputmats = inputmats
                    else:
                        saved_inputmats_t = inputmats_t
                        if cpu_offloading:
                            for t in saved_inputmats_t:
                                t.activation_offloading = True
                else:
                    saved_inputmats = inputmats_no_fp8

                if cpu_offloading:
                    if fp8:
                        for w in weights_fp8:
                            if w is not None:
                                w.weight_offloading = True
                    for w in weights:
                        w.weight_offloading = True
                    for t in saved_inputmats:
                        if t is not None:
                            t.activation_offloading = True

            ctx.save_for_backward(
                inputmat_scale_inv,
                *saved_inputmats,
                *saved_inputmats_t,
                *weights,
                *weights_fp8,
                *[
                    w.main_grad if cpu_offloading and fuse_wgrad_accumulation else None
                    for w in weights
                ],
            )
            ctx.m_splits = m_splits
            ctx.num_gemms = num_gemms
            ctx.activation_dtype = activation_dtype
            ctx.fp8 = fp8
            ctx.fp8_meta = fp8_meta
            ctx.fuse_wgrad_accumulation = fuse_wgrad_accumulation
            ctx.cpu_offloading = cpu_offloading
            ctx.is_first_microbatch = is_first_microbatch
            ctx.use_bias = use_bias
            ctx.sequence_parallel = sequence_parallel
            ctx.inp_shape = inp.shape
            ctx.fp8_meta_offsets = fp8_meta_offsets
            ctx.requires_dgrad = inp.requires_grad
            ctx.reduce_and_update_bwd_fp8_tensors = False
            if ctx.fp8 and requires_grad(inp, weights[0], biases[0]):
                ctx.reduce_and_update_bwd_fp8_tensors = (
                    ctx.reduce_and_update_bwd_fp8_tensors
                    or FP8GlobalStateManager.is_first_fp8_module()
                )
        ctx.manger_fp8 = CustomManger.fp8
        # [*, in_features] -> [*, out_features] except first dimension changes for SP
        return out.view(-1, *inp.shape[1:-1], out.shape[-1])

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Union[torch.Tensor, None], ...]:
        # pylint: disable=missing-function-docstring
        with torch.cuda.nvtx.range("_GroupedLinear_backward"):
            (
                inputmat_scale_inv,
                *saved_tensors,
            ) = ctx.saved_tensors
            inputmats = saved_tensors[: ctx.num_gemms]
            inputmats_t = saved_tensors[ctx.num_gemms : 2 * ctx.num_gemms]
            weights = saved_tensors[2 * ctx.num_gemms : 3 * ctx.num_gemms]
            weights_fp8 = saved_tensors[3 * ctx.num_gemms : 4 * ctx.num_gemms]
            main_grads = saved_tensors[4 * ctx.num_gemms :]
            if ctx.cpu_offloading and ctx.fuse_wgrad_accumulation:
                for i in ctx.num_gemms:
                    w = torch.nn.Parameter(weights[i], weights[i].requires_grad)
                    w.main_grad = main_grads[i]
                    weights[i] = w

            # preprocess grad_output
            grad_output = grad_output.contiguous()
            grad_output_mats = torch.split(
                grad_output.view(-1, grad_output.shape[-1]), ctx.m_splits
            )
            grad_output_c = [None] * ctx.num_gemms
            grad_output_t = [None] * ctx.num_gemms
            grad_biases = [None] * ctx.num_gemms

            if ctx.is_first_microbatch is not None:
                accumulate_wgrad_into_param_main_grad = (
                    ctx.fuse_wgrad_accumulation and not ctx.is_first_microbatch
                )
            else:
                accumulate_wgrad_into_param_main_grad = ctx.fuse_wgrad_accumulation

            qdyt_all, sdyt_all = None, None
            if ctx.requires_dgrad:
                if ctx.manger_fp8:

                    qdy, sdy, qdyt_all, sdyt_all = CustomManger.group_per_token_cast_to_fp8(grad_output.view(-1, grad_output.shape[-1]), ctx.m_splits)
                    _, _, qwt, swt = CustomManger.get_fp8_weight(weights)
                    dgrad = CustomManger.fp8_grouped_matmul(qdy, sdy, qwt, swt, ctx.m_splits)
                    CustomManger.clear_tensor_data(qdy, sdy)

                else:
                    dgrad = torch.empty(
                        (sum(ctx.m_splits), weights[0].size(1)),
                        dtype=ctx.activation_dtype,
                        device=grad_output.device,
                    )
                    grouped_gemm(
                        weights,
                        grad_output_mats,
                        torch.split(dgrad, ctx.m_splits),
                        ctx.activation_dtype,
                        get_multi_stream_cublas_workspace(),
                        layout="NN",
                        grad=True,
                    )

            if weights[0].requires_grad:
                def _func(inputmats, grad_output_mats, qdyt_all, sdyt_all, weights):
                    if ctx.manger_fp8:
                        wgrad_list = [
                            torch.empty(w.size(), dtype=ctx.activation_dtype, device=w.device)
                            for w in weights
                        ]

                        for i in range(len(wgrad_list)):
                            weight = weights[i]
                            inp = inputmats[i]
                            if inp.size(0) == 0:
                                continue
                            qxt, sxt = CustomManger.per_block_cast_to_fp8(inp, output_normal=False, output_transpose=True)
                            qdyt, sdyt = qdyt_all[i], sdyt_all[i]

                            wgrad = CustomManger.fp8_matmul(qdyt, sdyt, qxt, sxt)
                            wgrad = CustomManger.fix_tensor_shape(wgrad, weight.data.size(0))

                            clear_tensor_data(qdyt, sdyt, qxt, sxt)

                            if hasattr(weight, "main_grad"):
                                CustomManger.add_grad(weight.main_grad, wgrad)
                                # weight.main_grad += wgrad
                            else:
                                if weight.grad is None:
                                    CustomManger.add_grad(weight.grad, wgrad)
                                    # weight.grad = wgrad
                                else:
                                    weight.grad += wgrad


                    else:
                        if ctx.fuse_wgrad_accumulation:
                            wgrad_list = [w.main_grad for w in weights]
                        else:
                            wgrad_list = [
                                torch.empty(w.size(), dtype=ctx.activation_dtype, device=w.device)
                                for w in weights
                            ]
                        # WGRAD
                        _, grad_biases, _ = grouped_gemm(
                            inputmats,
                            grad_output_mats,
                            wgrad_list,
                            ctx.activation_dtype,
                            get_multi_stream_cublas_workspace(),
                            layout="NT",
                            grad=True,
                            use_bias=ctx.use_bias,
                            accumulate=accumulate_wgrad_into_param_main_grad,
                        )

                        if not ctx.fuse_wgrad_accumulation:
                            for i in range(len(weights)):
                                weight = weights[i]
                                wgrad = wgrad_list[i]
                                if weight.grad is None:
                                    weight.grad = wgrad
                                else:
                                    weight.grad += wgrad

                    # Deallocate input tensor
                    clear_tensor_data(*inputmats)
                func = CustomManger.partial(_func, inputmats=inputmats, 
                                            grad_output_mats=None if ctx.manger_fp8 else grad_output_mats, 
                                            qdyt_all=qdyt_all,
                                            sdyt_all=sdyt_all,
                                            weights=weights)
                if CustomManger.store_wgrad_func:
                    CustomManger.wgrad_func.append(func)
                else:
                    func()


            wgrad_list = [None] * ctx.num_gemms
            if weights[0].requires_grad:
                def handle_custom_ddp_from_mcore(w, wgrad):
                    if w.requires_grad:
                        if ctx.fuse_wgrad_accumulation and hasattr(w, "grad_added_to_main_grad"):
                            w.grad_added_to_main_grad = True
                            if getattr(w, "zero_out_wgrad", False):
                                wgrad = torch.zeros(
                                    w.main_grad.shape,
                                    dtype=w.dtype,
                                    device=torch.cuda.current_device(),
                                    requires_grad=False,
                                )
                            else:
                                wgrad = torch.empty(
                                    w.main_grad.shape,
                                    dtype=w.dtype,
                                    device=torch.cuda.current_device(),
                                    requires_grad=False,
                                )
                        elif ctx.fuse_wgrad_accumulation:
                            wgrad = None
                    else:
                        wgrad = None
                    return wgrad
                
                wgrad_list = [
                    handle_custom_ddp_from_mcore(w, wgrad) for w, wgrad in zip(weights, wgrad_list)
                ]
            else:
                wgrad_list = [None] * ctx.num_gemms

            if not ctx.use_bias:
                grad_biases = [None] * ctx.num_gemms
            else:
                for i in range(len(grad_biases)):
                    if grad_biases[i] is None:
                        grad_biases[i] = grad_output_mats[i].sum(0)

        if ctx.reduce_and_update_bwd_fp8_tensors and not is_graph_capturing():
            FP8GlobalStateManager.reduce_and_update_fp8_tensors(forward=False)

        return (
            dgrad.view(ctx.inp_shape) if ctx.requires_dgrad else None,
            None,  # m_splits
            None,  # use_bias
            None,  # is_first_microbatch
            None,  # fp8
            None,  # fp8_calibration
            None,  # fp8_meta
            None,  # fuse_wgrad_accumulation
            None,  # cpu_offloading
            None,  # sequence_parallel
            None,  # activation_dtype
            None,  # fp8_meta_offsets
            None,  # is_grad_enabled
            None,  # weights_fp8
            *wgrad_list,
            *grad_biases,
        )


class GroupedLinear(TransformerEngineBaseModule):
    """Applies linear transformations to the incoming data list
       :math:`y_i = x_iA_i^T + b_i` in a grouped way.

    Parameters
    ----------
    num_gemms : int
                number of GEMMs to be performed simutaneously.
    in_features : int
                 size of each input sample.
    out_features : int
                  size of each output sample.
    bias : bool, default = `True`
          if set to `False`, the layer will not learn an additive bias.
    init_method : Callable, default = `None`
                 used for initializing weights in the following way: `init_method(weight)`.
                 When set to `None`, defaults to `torch.nn.init.normal_(mean=0.0, std=0.023)`.
    get_rng_state_tracker : Callable, default = `None`
                 used to get the random number generator state tracker for initializing weights.
    rng_tracker_name : str, default = `None`
                 the param passed to get_rng_state_tracker to get the specific rng tracker.
    device : Union[torch.device, str], default = "cuda"
          The device on which the parameters of the model will be allocated. It is the user's
          responsibility to ensure all parameters are moved to the GPU before running the
          forward pass.

    Optimization parameters
    -----------------------
    fuse_wgrad_accumulation : bool, default = 'False'
                             if set to `True`, enables fusing of creation and accumulation of
                             the weight gradient. When enabled, it is assumed that the weights
                             have an additional `main_grad` attribute (used instead of the
                             regular `grad`) which is a pre-allocated buffer of the correct
                             size to accumulate gradients in.
    return_bias : bool, default = `False`
                 when set to `True`, this module will not apply the additive bias itself, but
                 instead return the bias value during the forward pass together with the
                 output of the linear transformation :math:`y = xA^T`. This is useful when
                 the bias addition can be fused to subsequent operations.
    params_dtype : torch.dtype, default = `torch.get_default_dtype()`
                  it controls the type used to allocate the initial parameters. Useful when
                  the model is trained with lower precision and the original FP32 parameters
                  would not fit in GPU memory.

    """

    def __init__(
        self,
        num_gemms: int,
        in_features: int,
        out_features: int,
        sequence_parallel: bool = False,
        fuse_wgrad_accumulation: bool = False,
        tp_group: Optional[dist_group_type] = None,
        tp_size: int = 1,
        get_rng_state_tracker: Optional[Callable] = None,
        rng_tracker_name: Optional[str] = None,
        init_method: Optional[Callable] = None,
        bias: bool = True,
        return_bias: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        parallel_mode: Optional[str] = None,
        device: Union[torch.device, str] = "cuda",
        ub_overlap_rs: bool = False,
        ub_overlap_ag: bool = False,
        ub_name: Optional[str] = None,
    ) -> None:
        super().__init__()

        params_dtype = torch.get_default_dtype() if params_dtype is None else params_dtype
        self.num_gemms = num_gemms
        self.in_features = in_features
        self.out_features = out_features
        self.fuse_wgrad_accumulation = fuse_wgrad_accumulation
        self.use_bias = bias
        self.return_bias = return_bias
        self.apply_bias = bias and not return_bias
        self.ub_overlap_rs = ub_overlap_rs
        self.ub_overlap_ag = ub_overlap_ag
        self.ub_name = ub_name
        assert (
            not ub_overlap_rs and not ub_overlap_ag
        ), "GroupedLinear doesn't support Userbuffer overlap."
        self.get_rng_state_tracker = get_rng_state_tracker
        self.rng_tracker_name = rng_tracker_name

        self._offsets = {"input": 0, "weight": num_gemms, "output": 2 * num_gemms, "grad_output": 0}

        if tp_group is None:
            self.tp_size = tp_size
            if tp_size == 1:
                self.set_tensor_parallel_group(tp_group)
        else:
            self.tp_size = get_distributed_world_size(tp_group)
            self.set_tensor_parallel_group(tp_group)
        self.set_nccl_overlap_warning_if_tp()

        self.parallel_mode = parallel_mode
        assert (
            self.parallel_mode in GemmParallelModes
        ), f"parallel_mode {parallel_mode} not supported"

        if self.parallel_mode == "column":
            self.out_features = divide(self.out_features, self.tp_size)
        elif self.parallel_mode == "row":
            self.in_features = divide(self.in_features, self.tp_size)

        self.sequence_parallel = (self.tp_size > 1) and sequence_parallel

        for i in range(self.num_gemms):
            # Construct weight parameter
            self.register_parameter(
                f"weight{i}",
                torch.nn.Parameter(
                    torch.empty(
                        self.out_features,
                        self.in_features,
                        device=device,
                        dtype=params_dtype,
                    ),
                ),
                init_fn=init_method,
                get_rng_state_tracker=get_rng_state_tracker,
                fp8_meta_index=self._offsets["weight"] + i,
            )

            # Construct bias parameters if needed
            if self.use_bias:
                self.register_parameter(
                    f"bias{i}",
                    torch.nn.Parameter(
                        torch.empty(
                            self.out_features,
                            device=device,
                            dtype=params_dtype,
                        ),
                    ),
                    init_fn=init_method_constant(0.0),
                )
            else:
                bias = torch.Tensor().to(dtype=params_dtype, device=device)
                setattr(self, f"bias{i}", bias)

        if self.primary_weights_in_fp8:
            self.init_fp8_metadata(num_gemms=self.num_gemms)

        self.reset_parameters(defer_init=device == "meta")

        # For RPL, bias has to be added after TP collectives
        # So it cannot be fused with the GEMM
        if self.parallel_mode == "row" and self.apply_bias:
            self.gemm_bias_unfused_add = True
        else:
            self.gemm_bias_unfused_add = False

    def reset_parameters(self, defer_init=False):
        super().reset_parameters(defer_init=defer_init)

        if not defer_init:
            # Set parallelism attributes for linear weights
            for i in range(self.num_gemms):
                set_tensor_model_parallel_attributes(
                    tensor=getattr(self, f"weight{i}"),
                    is_parallel=True,
                    dim=1 if self.parallel_mode == "row" else 0,
                    stride=1,
                )

            # Set parallelism attributes for linear biases
            if self.use_bias:
                for i in range(self.num_gemms):
                    if self.parallel_mode == "row":
                        setattr(
                            getattr(self, f"bias{i}"),
                            "sequence_parallel",
                            self.sequence_parallel,
                        )
                    elif self.parallel_mode == "column":
                        set_tensor_model_parallel_attributes(getattr(self, f"bias{i}"), True, 0, 1)

    @no_torch_dynamo()
    def forward(
        self,
        inp: torch.Tensor,
        m_splits: List[int],
        is_first_microbatch: Optional[bool] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Apply the linear transformation to the input.

        Parameters
        ----------
        inp : torch.Tensor
             Input tensor.
        m_splits : List[int]
                 List of integers representing the split of the input tensor.
        is_first_microbatch : {True, False, None}, default = None
                             During training using either gradient accumulation or
                             pipeline parallelism a minibatch of data is further split
                             into microbatches. Between the microbatches of the same minibatch
                             the model weights are not updated. Setting this parameter indicates
                             whether the current microbatch is the first in a minibatch or not.
                             When set, this parameter enables additional optimizations:

                             * during FP8 training, it allows caching of the FP8 versions of
                               the weights
                             * it also allows skipping gradient accumulation during the
                               first microbatch (since it is the first gradient being
                               produced)
        """

        with self.prepare_forward(inp, is_first_microbatch, num_gemms=self.num_gemms) as inp:

            weight_tensors = [getattr(self, f"weight{i}") for i in range(self.num_gemms)]
            bias_tensors = [getattr(self, f"bias{i}") for i in range(self.num_gemms)]
            if not self.fp8:
                weight_tensors = [
                    w.dequantize() if isinstance(w, QuantizedTensor) else w for w in weight_tensors
                ]

            weight_tensors_fp8 = [None] * self.num_gemms


            if torch.is_grad_enabled():
                linear_fn = _GroupedLinear.apply
                args = []
            else:
                linear_fn = _GroupedLinear.forward
                args = [None]
            args += (
                inp,
                m_splits,
                self.apply_bias and not self.gemm_bias_unfused_add,
                is_first_microbatch,
                False, #self.fp8,
                self.fp8_calibration,
                self.fp8_meta,
                self.fuse_wgrad_accumulation,
                is_cpu_offload_enabled(),
                self.sequence_parallel,
                self.activation_dtype,
                self._offsets,
                torch.is_grad_enabled(),
                weight_tensors_fp8,
                *weight_tensors,
                *bias_tensors,
            )
            out = linear_fn(*args)

        if self.gemm_bias_unfused_add:
            out_shape = out.shape
            out = torch.cat(
                [
                    o + cast_if_needed(b, self.activation_dtype)
                    for o, b in zip(
                        torch.split(out.view(-1, self.out_features), m_splits), bias_tensors
                    )
                ]
            ).view(out_shape)

        if self.return_bias:
            return out, [cast_if_needed(b, self.activation_dtype) for b in bias_tensors]
        return out



from typing import Union, List

import torch

from transformer_engine.pytorch.cpp_extensions import (
    multi_padding_fused,
)
from transformer_engine.pytorch.jit import no_torch_dynamo

class _Fp8Unpadding(torch.autograd.Function):
    """functional FP8 unpadding"""

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        m_splits: List[int],
        padded_m_splits: List[int],
        is_grad_enabled: bool,
    ) -> torch.Tensor:
        # pylint: disable=missing-function-docstring
        inputmats = torch.split(inp.view(-1, inp.shape[-1]), padded_m_splits)
        out_ret = torch.cat(
            [grad_output_mat[: m_splits[i]] for i, grad_output_mat in enumerate(inputmats)], dim=0
        )

        if is_grad_enabled:
            ctx.m_splits = m_splits
            ctx.padded_m_splits = padded_m_splits
            ctx.requires_dgrad = inp.requires_grad

        return out_ret

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # pylint: disable=missing-function-docstring
        grad_input = None
        if ctx.requires_dgrad:
            grad_output = grad_output.contiguous()

            in_features = grad_output.shape[-1]

            # Allocate cast and transpose output tensor
            total_row = sum(ctx.padded_m_splits)
            grad_input = torch.empty(
                [total_row, in_features], dtype=grad_output.dtype, device=grad_output.device
            )
            # FP8 pad input for forward, FP8 input transpose for backward wgrad
            multi_padding_fused(
                grad_output.view(-1, in_features), ctx.m_splits, ctx.padded_m_splits, grad_input
            )

        return (grad_input, None, None, None)


class Fp8Unpadding(torch.nn.Module):
    """
    Apply the unpadding for Grouped GEMM input.

    Parameters
    ----------
    num_gemms: int
               number of GEMMs to be performed simutaneously.
    """

    def __init__(
        self,
        num_gemms,
    ) -> None:
        super().__init__()

        self.num_gemms = num_gemms

    @no_torch_dynamo()
    def forward(
        self,
        inp: torch.Tensor,
        m_splits: List[int],
    ) -> torch.Tensor:
        """
        Apply the unpadding to the input.

        Parameters
        ----------
        inp : torch.Tensor
                Input tensor.
        m_splits : List[int]
                    List of integers representing the split of the input tensor.
        """

        assert len(m_splits) == self.num_gemms, "Number of splits should match number of GEMMs."

        # FP8 padding calculate
        padded_m_splits = [(m + 255) // 256 * 256 for m in m_splits]

        if torch.is_grad_enabled():
            fn = _Fp8Unpadding.apply
            args = []
        else:
            fn = _Fp8Unpadding.forward
            args = [None]

        args += (
            inp,
            m_splits,
            padded_m_splits,
            torch.is_grad_enabled(),
        )
        out = fn(*args)

        return out
    



class _Fp8Padding(torch.autograd.Function):
    """functional FP8 padding"""

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        m_splits: List[int],
        padded_m_splits: List[int],
        is_grad_enabled: bool,
    ) -> torch.Tensor:
        # pylint: disable=missing-function-docstring
        # Make sure input dimensions are compatible
        in_features = inp.shape[-1]

        # Allocate cast and transpose output tensor
        total_row = sum(padded_m_splits)
        out = torch.empty([total_row, in_features], dtype=inp.dtype, device=inp.device)

        multi_padding_fused(inp.view(-1, in_features), m_splits, padded_m_splits, out)

        if is_grad_enabled:
            ctx.m_splits = m_splits
            ctx.padded_m_splits = padded_m_splits
            ctx.requires_dgrad = inp.requires_grad

        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # pylint: disable=missing-function-docstring

        grad_input = None
        if ctx.requires_dgrad:
            grad_output = grad_output.contiguous()

            grad_output_mats = torch.split(
                grad_output.view(-1, grad_output.shape[-1]), ctx.padded_m_splits
            )
            grad_input = torch.cat(
                [
                    grad_output_mat[: ctx.m_splits[i]]
                    for i, grad_output_mat in enumerate(grad_output_mats)
                ],
                dim=0,
            )

        return (grad_input, None, None, None)


class Fp8Padding(torch.nn.Module):
    """
    Apply the padding for Grouped GEMM input.

    Parameters
    ----------
    num_gemms: int
               number of GEMMs to be performed simutaneously.
    """

    def __init__(
        self,
        num_gemms,
    ) -> None:
        super().__init__()

        self.num_gemms = num_gemms

    @no_torch_dynamo()
    def forward(
        self,
        inp: torch.Tensor,
        m_splits: List[int],
    ) -> Union[torch.Tensor, List[int]]:
        """
        Apply the padding to the input.

        Parameters
        ----------
        inp : torch.Tensor
                Input tensor.
        m_splits : List[int]
                    List of integers representing the split of the input tensor.
        """

        assert len(m_splits) == self.num_gemms, "Number of splits should match number of GEMMs."

        # FP8 padding calculate
        padded_m_splits = [(m + 255) // 256 * 256 for m in m_splits]

        if torch.is_grad_enabled():
            fn = _Fp8Padding.apply
            args = []
        else:
            fn = _Fp8Padding.forward
            args = [None]

        args += (
            inp,
            m_splits,
            padded_m_splits,
            torch.is_grad_enabled(),
        )
        out = fn(*args)

        return out, padded_m_splits

