# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""LayerNormLinear API"""
import os
import warnings
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
from torch.nn import init

from transformer_engine.pytorch import cpp_extensions as tex

from transformer_engine.pytorch.module.base import (
    get_workspace,
    get_ub,
    TransformerEngineBaseModule,
    _2X_ACC_FPROP,
    _2X_ACC_DGRAD,
    _2X_ACC_WGRAD,
)
from transformer_engine.pytorch.fp8 import get_fp8_te_dtype, FP8GlobalStateManager
from transformer_engine.pytorch.utils import (
    divide,
    get_default_init_method,
    init_method_constant,
    cast_if_needed,
    assert_dim_for_fp8_exec,
    clear_tensor_data,
    requires_grad,
)
from transformer_engine.pytorch.distributed import (
    set_tensor_model_parallel_attributes,
    get_distributed_world_size,
    allreduce,
    reduce_scatter_along_first_dim,
    gather_along_first_dim,
    in_fp8_activation_recompute_phase,
    _fsdp_scatter_tensors,
    _fsdp_gather_tensors,
)
from transformer_engine.pytorch.constants import GemmParallelModes, dist_group_type, TE_DType
from transformer_engine.pytorch.jit import no_torch_dynamo
from transformer_engine.pytorch.graph import is_graph_capturing
from transformer_engine.pytorch.module._common import _apply_normalization, _noop_cat
from transformer_engine.pytorch.float8_tensor import Float8Tensor
from transformer_engine.pytorch.export import is_in_onnx_export_mode
from transformer_engine.pytorch.tensor import QuantizedTensor
from transformer_engine.pytorch.cpu_offload import is_cpu_offload_enabled

from .manger import CustomManger

__all__ = ["LayerNormLinear"]


class _LayerNormLinear(torch.autograd.Function):
    """LayerNormLinear semi-top level module
    Calls custom cuda extensions.
    """

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        ln_weight: torch.Tensor,
        ln_bias: Union[torch.Tensor, None],
        weight: torch.Tensor,
        weight_fp8: Optional[torch.Tensor],
        bias: torch.Tensor,
        use_bias: bool,
        eps: float,
        is_first_microbatch: Union[bool, None],
        fp8: bool,
        fp8_calibration: bool,
        fp8_meta: Dict[str, Any],
        fuse_wgrad_accumulation: bool,
        cpu_offloading: bool,
        tp_group: Union[dist_group_type, None],
        tp_size: int,
        sequence_parallel: bool,
        tensor_parallel: bool,
        activation_dtype: torch.dtype,
        parallel_mode: Union[str, None],
        return_layernorm_output: bool,
        return_layernorm_output_gathered: bool,
        is_grad_enabled: bool,
        fwd_ln_sm_margin: int,
        bwd_ln_sm_margin: int,
        zero_centered_gamma: bool,
        normalization: str,
        ub_bulk_wgrad: bool,
        ub_bulk_dgrad: bool,
        ub_overlap_rs_dgrad: bool,
        ub_overlap_ag: bool,
        ub_name: str,
        fp8_output: bool,
        fsdp_group: Union[dist_group_type, None],
    ) -> Union[Tuple[torch.Tensor, ...], torch.Tensor]:
        # pylint: disable=missing-function-docstring
        # Make sure input dimensions are compatible
        out_features, in_features = weight.shape
        inp_shape = inp.shape
        assert inp_shape[-1] == in_features, "GEMM not possible"
        inputmat = inp.view((-1, in_features))
        if fp8:
            assert_dim_for_fp8_exec(inputmat)
            assert_dim_for_fp8_exec(weight)

        # Cast for native AMP
        inputmat = cast_if_needed(inputmat, activation_dtype)
        ln_weight = cast_if_needed(ln_weight, activation_dtype)
        if ln_bias is not None:
            ln_bias = cast_if_needed(ln_bias, activation_dtype)

        if ub_overlap_ag:
            tp_world_size = get_distributed_world_size(tp_group)
            if tp_world_size == 1 or (not is_grad_enabled):
                ub_overlap_ag = False
        if ub_overlap_ag:
            dim_size = list(inputmat.size())
            dim_size[0] = dim_size[0] * tp_world_size
            ub_obj_lnout = get_ub(ub_name + "_fprop")
            if return_layernorm_output:
                # First prepare LN output in higher precision,
                # which will be later copied to a FP8 UB
                ln_out = torch.empty_like(inputmat, memory_format=torch.contiguous_format)
            else:
                ln_out = ub_obj_lnout.get_ubuf_output(0)
        else:
            ln_out_dtype = torch.uint8 if (fp8 and not return_layernorm_output) else inputmat.dtype
            ln_out = torch.empty_like(
                inputmat, dtype=ln_out_dtype, memory_format=torch.contiguous_format
            )

        # Objects for FP8 cast
        fp8_dtype_forward = get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=True)
        ln_out_scale_inv = None
        if fp8:
            ln_out_scale_inv = torch.empty([1], dtype=torch.float32, device=inputmat.device)

        # Launch normalization kernel
        ln_out, mu, rsigma = _apply_normalization(
            inputmat,
            ln_out,
            ln_weight,
            ln_bias,
            eps,
            fp8 and not return_layernorm_output,
            fp8_meta,
            normalization,
            fwd_ln_sm_margin,
            zero_centered_gamma,
            is_grad_enabled,
            fp8_scale_inv=ln_out_scale_inv,
        )

        # Column Parallel Linear
        ln_out_gathered = False
        ub_algo = None
        if ub_overlap_ag:
            ln_out_total = ub_obj_lnout.get_ubuf_output(1)
            if not return_layernorm_output:
                ln_out = torch.empty_like(ln_out)
            if ub_obj_lnout.is_atomic_gemm():
                ub_algo = tex.CommOverlapAlgo.ATOMIC_GEMM_AG_P2P
            else:
                ub_algo = tex.CommOverlapAlgo.SPLIT_PIPELINED_AG_P2P
        elif parallel_mode == "column" and sequence_parallel:
            ln_out_gathered = True
            ln_out_total, _ = gather_along_first_dim(ln_out, tp_group)
        else:
            ln_out_total = ln_out

        # If residual connection is after LN, we need `ln_out_return`
        # tensor in higher precision, this comes at the cost
        # of an extra fp8 cast.
        if return_layernorm_output:
            ln_out_return = ln_out_total if return_layernorm_output_gathered else ln_out

        if CustomManger.fp8:
            weight = cast_if_needed(weight, activation_dtype)
            bias = cast_if_needed(bias, activation_dtype) if use_bias else bias\
            
            CustomManger.set_first_micro_batch(weight, is_first_microbatch)
            qx, sx = CustomManger.per_token_cast_to_fp8(ln_out_total, output_transpose=False)
            qw, sw, _, _ = CustomManger.get_fp8_weight(weight)
            
            out = CustomManger.fp8_matmul(qx, sx, qw, sw)
            out = CustomManger.fix_tensor_shape(out, ln_out_total.size(0))

            if use_bias:
                # out += bias
                CustomManger.add_bias(out, bias)
            CustomManger.clear_tensor_data(qx, sx)

        else:
            # Cast for native AMP
            weight = cast_if_needed(weight, activation_dtype)
            bias = cast_if_needed(bias, activation_dtype) if use_bias else bias

            if fp8_calibration:
                # amax of input
                amin, amax = ln_out_total.aminmax()
                fp8_meta["scaling_fwd"].amax_history[0][tex.FP8FwdTensors.GEMM1_INPUT] = torch.max(
                    -amin, amax
                ).float()
                # amax of weight
                amin, amax = weight.aminmax()
                fp8_meta["scaling_fwd"].amax_history[0][tex.FP8FwdTensors.GEMM1_WEIGHT] = torch.max(
                    -amin, amax
                ).float()

            out, _, _ = tex.gemm(
                weight,
                ln_out_total,
                activation_dtype,
                get_workspace(),
                bias=bias,
                use_bias=use_bias,
                ub_algo=tex.CommOverlapAlgo.SPLIT_PIPELINED_AG_P2P if ub_overlap_ag else None,
                ub=ub_obj_lnout if ub_overlap_ag else None,
                extra_output_tensor=ln_out if ub_overlap_ag else None,
            )

        if is_grad_enabled:
            if cpu_offloading:
                if fp8 and weight_fp8 is not None:
                    weight_fp8.weight_offloading = True
                ln_weight.weight_offloading = True
                weight.weight_offloading = True

                inputmat.activation_offloading = True
                if normalization == "LayerNorm":
                    mu.activation_offloading = True
                rsigma.activation_offloading = True
                ln_out.activation_offloading = True

            # Scatter intermediate/activation tensors saved for the backward pass
            # NOTE: weight_fp8 = weight when ctx.fp8 == False and torch.disttributed.FSDP already
            #       shards/unshards the base weights so we don't do it ourselves
            ctx.fsdp_group = fsdp_group
            ctx.fsdp_shapes = _fsdp_scatter_tensors(
                fsdp_group,
                mu,
                rsigma,
                weight_fp8 if fp8 and not isinstance(weight, Float8Tensor) else None,
                ln_out if weight.requires_grad else None,
            )

            ctx.save_for_backward(
                inputmat,
                ln_weight,
                mu,
                rsigma,
                weight,
                weight_fp8,
                weight.main_grad if cpu_offloading and fuse_wgrad_accumulation else None,
                ln_out if weight.requires_grad else None,
                ln_out_scale_inv,
            )

            ctx.activation_dtype = activation_dtype
            ctx.fp8 = fp8
            ctx.fp8_meta = fp8_meta
            ctx.fuse_wgrad_accumulation = fuse_wgrad_accumulation
            ctx.cpu_offloading = cpu_offloading
            ctx.is_first_microbatch = is_first_microbatch
            ctx.use_bias = use_bias
            ctx.sequence_parallel = sequence_parallel
            ctx.tensor_parallel = tensor_parallel
            ctx.inp_shape = inp_shape
            ctx.parallel_mode = parallel_mode
            ctx.tp_group = tp_group
            ctx.tp_size = tp_size
            ctx.return_layernorm_output = return_layernorm_output
            ctx.return_layernorm_output_gathered = (
                return_layernorm_output_gathered and ln_out_gathered
            )
            ctx.bwd_ln_sm_margin = bwd_ln_sm_margin
            ctx.zero_centered_gamma = zero_centered_gamma
            ctx.ub_bulk_wgrad = ub_bulk_wgrad
            ctx.ub_bulk_dgrad = ub_bulk_dgrad
            ctx.ub_overlap_rs_dgrad = ub_overlap_rs_dgrad
            ctx.ub_name = ub_name
            ctx.requires_dgrad = inp.requires_grad
            ctx.normalization = normalization
            ctx.reduce_and_update_bwd_fp8_tensors = False
            if ctx.fp8 and requires_grad(inp, ln_weight, ln_bias, weight, bias):
                _first_fp8_module = FP8GlobalStateManager.IS_FIRST_FP8_MODULE
                ctx.reduce_and_update_bwd_fp8_tensors = FP8GlobalStateManager.is_first_fp8_module()
                if in_fp8_activation_recompute_phase():
                    FP8GlobalStateManager.IS_FIRST_FP8_MODULE = _first_fp8_module

        # Row Parallel Linear
        if parallel_mode == "row" and sequence_parallel:
            out, _ = reduce_scatter_along_first_dim(out, tp_group)
        elif parallel_mode == "row" and tensor_parallel:
            out, _ = allreduce(out, tp_group)

        # [*, in_features] -> [*, out_features] except first dimension changes for SP
        out = out.view(-1, *inp_shape[1:-1], out_features)

        ctx.manger_fp8 = CustomManger.fp8

        if return_layernorm_output:
            if return_layernorm_output_gathered:
                shape = list(inp.shape)
                shape[0] *= tp_size
                return out, ln_out_return.view(shape)
            return out, ln_out_return.view_as(inp)
        
        return out

    @staticmethod
    def backward(
        ctx, *grad_outputs: Tuple[torch.Tensor, ...]
    ) -> Tuple[Union[torch.Tensor, None], ...]:
        # pylint: disable=missing-function-docstring
        if isinstance(grad_outputs[0], Float8Tensor):
            ctx.fp8_meta["scaling_bwd"].scale_inv[tex.FP8BwdTensors.GRAD_OUTPUT1] = grad_outputs[
                0
            ]._scale_inv

        with torch.cuda.nvtx.range("_LayerNormLinear_backward"):
            (
                inputmat,
                ln_weight,
                mu,
                rsigma,
                weight,
                weight_fp8,
                main_grad,
                ln_out,
                ln_out_scale_inv,
            ) = ctx.saved_tensors

            # Gather intermediate/activation tensors if needed
            # NOTE: weight_fp8 = weight when ctx.fp8 == False and torch.disttributed.FSDP already
            #       shards/unshards the base weights so we don't do it ourselves
            _fsdp_gather_tensors(
                ctx.fsdp_group,
                ctx.fsdp_shapes,
                mu,
                rsigma,
                weight_fp8 if ctx.fp8 and not isinstance(weight, Float8Tensor) else None,
                ln_out,
            )

            if ctx.cpu_offloading and ctx.fuse_wgrad_accumulation:
                weight = torch.nn.Parameter(weight, weight.requires_grad)
                weight.main_grad = main_grad

            if ctx.ub_overlap_rs_dgrad:
                ctx.ub_bulk_dgrad = False
                ctx.ub_bulk_wgrad = False
                tp_world_size = get_distributed_world_size(ctx.tp_group)
                if tp_world_size == 1:
                    ctx.ub_overlap_rs_dgrad = False
            if ctx.ub_bulk_dgrad:
                tp_world_size = get_distributed_world_size(ctx.tp_group)
                if tp_world_size == 1 or not weight.requires_grad:
                    ctx.ub_bulk_dgrad = False
            if ctx.ub_bulk_dgrad:
                dim_size = list(ln_out.size())
                dim_size[0] = dim_size[0] * tp_world_size
                ub_obj_lnout = get_ub(ctx.ub_name + "_dgrad")
                ub_obj_lnout.copy_input_to_ubuf(ln_out, 1)
            (
                grad_output,
                grad_output_c,
                grad_output_t,
                grad_bias,
            ) = TransformerEngineBaseModule.grad_output_preprocess(
                ctx, grad_outputs[0], ctx.parallel_mode == "row"
            )

            if ctx.ub_bulk_wgrad:
                tp_world_size = get_distributed_world_size(ctx.tp_group)
                if tp_world_size == 1 or not weight.requires_grad:
                    ctx.ub_bulk_wgrad = False

            # Column Parallel Linear
            # Overlap input AG with dgrad
            if (
                weight.requires_grad
                and (not ctx.ub_bulk_dgrad)
                and ctx.parallel_mode == "column"
                and ctx.sequence_parallel
            ):
                ln_out_total, handle = gather_along_first_dim(ln_out, ctx.tp_group, async_op=True)
            else:
                ln_out_total = ln_out
                handle = None

            if ctx.is_first_microbatch is not None:
                accumulate_wgrad_into_param_main_grad = (
                    ctx.fuse_wgrad_accumulation and not ctx.is_first_microbatch
                )
            else:
                accumulate_wgrad_into_param_main_grad = ctx.fuse_wgrad_accumulation

            dgrad_size = list(grad_output.size())
            dgrad_size[1] = weight.size(1)
            if ctx.ub_bulk_wgrad:  # allocate dgrad output
                ub_obj_dgrad = get_ub(ctx.ub_name + "_wgrad")
                dgrad = ub_obj_dgrad.get_ubuf_output(1)  # AllGather output
            elif ctx.ub_overlap_rs_dgrad:
                ub_obj_dgrad = get_ub(ctx.ub_name + "_dgrad")
                dgrad = ub_obj_dgrad.get_ubuf_output(1)  # AllGather output
            else:
                dgrad = torch.empty(dgrad_size, dtype=ctx.activation_dtype, device=weight.device)

            rs_out = None
            if ctx.ub_bulk_dgrad:
                ub_algo = tex.CommOverlapAlgo.BULK_OVERLAP_AG
                ub_obj = ub_obj_lnout
            elif ctx.ub_overlap_rs_dgrad:
                dim_size = list(grad_output.size())
                dim_size[0] = dim_size[0] // tp_world_size
                dim_size[1] = weight.size(1)
                rs_out = torch.empty(
                    dim_size, dtype=ctx.activation_dtype, device=grad_output.device
                )
                if ub_obj_dgrad.is_p2p_overlap():
                    if ctx.fp8 and ub_obj_dgrad.is_atomic_gemm():
                        ub_algo = tex.CommOverlapAlgo.ATOMIC_GEMM_RS_P2P
                    else:
                        ub_algo = tex.CommOverlapAlgo.SPLIT_PIPELINED_RS_P2P
                else:
                    if ctx.fp8 and ub_obj_dgrad.is_atomic_gemm():
                        ub_algo = tex.CommOverlapAlgo.ATOMIC_GEMM_RS
                    else:
                        ub_algo = tex.CommOverlapAlgo.SPLIT_PIPELINED_RS
                ub_obj = ub_obj_dgrad
            else:
                ub_algo = None
                ub_obj = None

            qdyt, sdyt = None, None
            if ctx.manger_fp8:
                qdy, sdy, qdyt, sdyt = CustomManger.per_token_cast_to_fp8(grad_output, output_transpose=True)
                _, _, qwt, swt = CustomManger.get_fp8_weight(weight)
                dgrad = CustomManger.fp8_matmul(qdy, sdy, qwt, swt, dgrad)
                dgrad = CustomManger.fix_tensor_shape(dgrad, grad_output.size(0))
                CustomManger.clear_tensor_data(qdy, sdy)
                

            else:
                # DGRAD: Evaluated unconditionally to feed into Linear backward
                _, _, _ = tex.gemm(
                    weight,
                    grad_output,
                    ctx.activation_dtype,
                    get_workspace(),
                    out=dgrad,
                    layout="NN",
                    grad=True,
                    ub_algo=ub_algo,
                    ub=ub_obj,
                    extra_output_tensor=rs_out if ctx.ub_overlap_rs_dgrad else None,
                )
            if ctx.ub_bulk_dgrad:
                ln_out_total = ub_obj_lnout.get_ubuf_output(1)

            # Overlap dgrad-RS/AR with wgrad
            if ctx.parallel_mode == "column" and ctx.sequence_parallel:
                if not ctx.ub_bulk_dgrad and handle is not None:
                    handle.wait()
                if not ctx.ub_bulk_wgrad and not ctx.ub_overlap_rs_dgrad:
                    if ctx.return_layernorm_output and ctx.return_layernorm_output_gathered:
                        dgrad = dgrad + grad_outputs[1].view_as(dgrad)
                    dgrad, handle = reduce_scatter_along_first_dim(
                        dgrad, ctx.tp_group, async_op=True
                    )
            elif ctx.parallel_mode == "column" and ctx.tensor_parallel:
                dgrad, handle = allreduce(dgrad, ctx.tp_group, async_op=True)

            
            if weight.requires_grad:
                def _func(ln_out_total, grad_output, qdyt, sdyt, weight):
                    if ctx.manger_fp8 :
                        qxt, sxt = CustomManger.per_block_cast_to_fp8(ln_out_total, output_normal=False, output_transpose=True)
                        wgrad = CustomManger.fp8_matmul(qdyt, sdyt, qxt, sxt)
                        wgrad = CustomManger.fix_tensor_shape(wgrad, weight.data.size(0))
                        clear_tensor_data(qdyt, sdyt, qxt, sxt)

                        if hasattr(weight, "main_grad"):
                            CustomManger.add_grad(weight.main_grad, wgrad)
                            # weight.main_grad += wgrad
                        else:
                            if weight.grad is None:
                                # weight.grad = wgrad
                                CustomManger.add_grad(weight.grad, wgrad)
                            else:
                                weight.grad += wgrad

                    else:
                        # WGRAD
                        wgrad, grad_bias, _ = tex.gemm(
                            ln_out_total,
                            grad_output,
                            ctx.activation_dtype,
                            get_workspace(),
                            layout="NT",
                            grad=True,
                            use_bias=False,
                            accumulate=accumulate_wgrad_into_param_main_grad,
                            out=weight.main_grad if ctx.fuse_wgrad_accumulation else None,
                            ub_algo=tex.CommOverlapAlgo.BULK_OVERLAP_RS if ctx.ub_bulk_wgrad else None,
                            ub=ub_obj_dgrad if ctx.ub_bulk_wgrad else None,
                        )
                        

                        if not ctx.fuse_wgrad_accumulation:
                            if weight.grad is None:
                                weight.grad = wgrad
                            else:
                                weight.grad += wgrad
                    clear_tensor_data(ln_out_total)

                func = CustomManger.partial(_func, ln_out_total=ln_out_total, 
                                            grad_output=grad_output if not ctx.manger_fp8 else None,
                                            qdyt=qdyt, 
                                            sdyt=sdyt, 
                                            weight=weight)
                if CustomManger.store_wgrad_func:
                    CustomManger.wgrad_func.append(func)
                else:
                    func()

            wgrad = None
            if ctx.ub_bulk_wgrad:
                dgrad = ub_obj_dgrad.get_ubuf_output(0)  # Reduce-scatter output
            # Column Parallel Linear
            if (
                (not ctx.ub_bulk_wgrad)
                and ctx.parallel_mode == "column"
                and ctx.tensor_parallel
                and handle is not None
            ):
                handle.wait()

            # LayerNorm gradient
            if ctx.ub_overlap_rs_dgrad:
                dgrad = rs_out.view(inputmat.shape)
            else:
                dgrad = dgrad.view(inputmat.shape)

            # Residual gradient
            if ctx.return_layernorm_output and not ctx.return_layernorm_output_gathered:
                dgrad = dgrad + grad_outputs[1].view_as(dgrad)

            dgamma = None
            dbeta = None
            if ctx.normalization == "LayerNorm":
                dgrad, dgamma, dbeta = tex.layernorm_bwd(
                    dgrad,
                    inputmat,
                    mu,
                    rsigma,
                    ln_weight,
                    ctx.bwd_ln_sm_margin,
                    ctx.zero_centered_gamma,
                )
            elif ctx.normalization == "RMSNorm":
                dgrad, dgamma = tex.rmsnorm_bwd(
                    dgrad,
                    inputmat,
                    rsigma,
                    ln_weight,
                    ctx.bwd_ln_sm_margin,
                    ctx.zero_centered_gamma,
                )
                dbeta = None
            clear_tensor_data(mu)
            clear_tensor_data(rsigma)

            if not ctx.use_bias:
                grad_bias = None
            else:
                if grad_bias is None:
                    grad_bias = grad_output.sum(0)

        if weight.requires_grad:
            # Handle custom DDP from mcore.
            if ctx.fuse_wgrad_accumulation and hasattr(weight, "grad_added_to_main_grad"):
                weight.grad_added_to_main_grad = True
                if getattr(weight, "zero_out_wgrad", False):
                    wgrad = torch.zeros(
                        weight.main_grad.shape,
                        dtype=weight.dtype,
                        device=torch.cuda.current_device(),
                        requires_grad=False,
                    )
                else:
                    wgrad = torch.empty(
                        weight.main_grad.shape,
                        dtype=weight.dtype,
                        device=torch.cuda.current_device(),
                        requires_grad=False,
                    )
            elif ctx.fuse_wgrad_accumulation:
                wgrad = None
        else:
            wgrad = None

        if ctx.reduce_and_update_bwd_fp8_tensors and not is_graph_capturing():
            FP8GlobalStateManager.reduce_and_update_fp8_tensors(forward=False)

        # Scatter fp8 weight buffers
        if ctx.fp8 and not isinstance(weight, Float8Tensor):
            _fsdp_scatter_tensors(ctx.fsdp_group, weight_fp8)

        return (
            dgrad.view(ctx.inp_shape) if ctx.requires_dgrad else None,
            dgamma,
            dbeta,
            wgrad,
            None,  # weight_fp8
            grad_bias,
            None,  # use_bias
            None,  # eps
            None,  # is_first_microbatch
            None,  # fp8
            None,  # fp8_calibration
            None,  # fp8_meta
            None,  # fuse_wgrad_accumulation
            None,  # cpu_offloading
            None,  # tp_group
            None,  # tp_size
            None,  # sequence_parallel
            None,  # tensor_parallel
            None,  # activation_dtype
            None,  # parallel_mode
            None,  # return_layernorm_output
            None,  # return_layernorm_output_gathered
            None,  # is_grad_enabled
            None,  # fwd_ln_sm_margin
            None,  # bwd_ln_sm_margin
            None,  # zero_centered_gamma
            None,  # normalization
            None,  # ub_bulk_wgrad
            None,  # ub_bulk_dgrad
            None,  # ub_overlap_rs_dgrad
            None,  # ub_overlap_ag
            None,  # ub_name
            None,  # fp8_output
            None,  # fsdp_group
        )


class LayerNormLinear(TransformerEngineBaseModule):
    r"""
    Applies layer normalization followed by linear transformation to the incoming data.

    Parameters
    ----------
    in_features : int
                 size of each input sample.
    out_features : int
                  size of each output sample.
    eps : float, default = 1e-5
         a value added to the denominator of layer normalization for numerical stability.
    bias : bool, default = `True`
          if set to `False`, the layer will not learn an additive bias.
    normalization : { 'LayerNorm', 'RMSNorm' }, default = 'LayerNorm'
                   type of normalization applied.
    init_method : Callable, default = `None`
                 used for initializing weights in the following way: `init_method(weight)`.
                 When set to `None`, defaults to `torch.nn.init.normal_(mean=0.0, std=0.023)`.
    return_layernorm_output : bool, default = `False`
                             if set to `True`, output of layernorm is returned from the forward
                             together with the output of the linear transformation.
                             Example use case: residual connection for transformer module is
                             taken post layernorm.
    return_layernorm_output_gathered : bool, default = `False`
                             if set to `True`, output of layernorm is returned after the all
                             gather operation. Ignored if return_layernorm_output is False.
                             Example use case: with sequence parallel, input to residual connection
                             for transformer module (e.g. LoRA) will need to be gathered.
                             Returning layernorm output gathered will prevent a redundant gather.
    parameters_split : Optional[Union[Tuple[str, ...], Dict[str, int]]], default = None
                      Configuration for splitting the weight and bias tensors along dim 0 into
                      multiple PyTorch parameters. If a list or tuple of strings is provided,
                      they are used to make the names of equally-sized parameters. If a dict
                      (preferably an OrderedDict) is provided, the keys are used as names and
                      values as split sizes along dim 0. The resulting parameters will have
                      names that end in `_weight` or `_bias`, so trailing underscores are
                      stripped from any provided names.
    zero_centered_gamma : bool, default = 'False'
                         if set to 'True', gamma parameter in LayerNorm is initialized to 0 and
                         the LayerNorm formula changes to

                         .. math::
                            y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \varepsilon}} *
                            (1 + \gamma) + \beta
    device : Union[torch.device, str], default = "cuda"
          The device on which the parameters of the model will be allocated. It is the user's
          responsibility to ensure all parameters are moved to the GPU before running the
          forward pass.

    Parallelism parameters
    ----------------------
    sequence_parallel : bool, default = `False`
                       if set to `True`, uses sequence parallelism.
    tp_group : ProcessGroup, default = `None`
              tensor parallel process group.
    tp_size : int, default = 1
             used as TP (tensor parallel) world size when TP groups are not formed during
             initialization. In this case, users must call the
             `set_tensor_parallel_group(tp_group)` method on the initialized module before the
             forward pass to supply the tensor parallel group needed for tensor and sequence
             parallel collectives.
    parallel_mode : {None, 'column', 'row'}, default = `None`
                   used to decide whether this Linear layer is Column Parallel Linear or Row
                   Parallel Linear as described `here <https://arxiv.org/pdf/1909.08053.pdf>`_.
                   When set to `None`, no communication is performed.

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
        in_features: int,
        out_features: int,
        eps: float = 1e-5,
        sequence_parallel: bool = False,
        fuse_wgrad_accumulation: bool = False,
        tp_group: Optional[dist_group_type] = None,
        tp_size: int = 1,
        get_rng_state_tracker: Optional[Callable] = None,
        init_method: Optional[Callable] = None,
        bias: bool = True,
        normalization: str = "LayerNorm",
        return_bias: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        parallel_mode: Optional[str] = None,
        return_layernorm_output: bool = False,
        return_layernorm_output_gathered: bool = False,
        parameters_split: Optional[Union[Tuple[str, ...], Dict[str, int]]] = None,
        zero_centered_gamma: bool = False,
        device: Union[torch.device, str] = "cuda",
        ub_bulk_wgrad: bool = False,
        ub_bulk_dgrad: bool = False,
        ub_overlap_ag: bool = False,
        ub_overlap_rs_dgrad: bool = False,
        ub_name: Optional[str] = None,
    ) -> None:
        super().__init__()

        params_dtype = torch.get_default_dtype() if params_dtype is None else params_dtype
        self.in_features = in_features
        self.out_features = out_features
        self.fuse_wgrad_accumulation = fuse_wgrad_accumulation
        self.normalization = normalization
        assert normalization in ["LayerNorm", "RMSNorm"], "Unsupported normalization type!"
        self.use_bias = bias
        self.return_bias = return_bias
        self.apply_bias = self.use_bias and not return_bias
        self.return_layernorm_output = return_layernorm_output
        self.return_layernorm_output_gathered = return_layernorm_output_gathered
        self.zero_centered_gamma = zero_centered_gamma
        self.ub_bulk_wgrad = ub_bulk_wgrad
        self.ub_bulk_dgrad = ub_bulk_dgrad
        self.ub_overlap_ag = ub_overlap_ag
        self.ub_overlap_rs_dgrad = ub_overlap_rs_dgrad
        if any([ub_bulk_wgrad, ub_bulk_dgrad, ub_overlap_ag, ub_overlap_rs_dgrad]):
            assert ub_name is not None, "Userbuffer name [string] is not set."
        self.ub_name = ub_name

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

        if init_method is None:
            init_method = get_default_init_method()

        self.sequence_parallel = (self.tp_size > 1) and sequence_parallel

        self.eps = eps
        layer_norm_weight = torch.nn.Parameter(
            torch.empty(in_features, device=device, dtype=params_dtype)
        )
        self.register_parameter(
            "layer_norm_weight",
            layer_norm_weight,
            init_fn=init_method_constant(float(not self.zero_centered_gamma)),
        )
        if self.normalization != "RMSNorm":
            layer_norm_bias = torch.nn.Parameter(
                torch.empty(in_features, device=device, dtype=params_dtype)
            )
            self.register_parameter(
                "layer_norm_bias", layer_norm_bias, init_fn=init_method_constant(0.0)
            )
        else:
            self.layer_norm_bias = None

        # Initialize params in FP8
        with_fp8_params = FP8GlobalStateManager.with_fp8_parameters()

        # Contiguous buffers for params
        weight_tensor = torch.empty(
            self.out_features,
            self.in_features,
            device=device,
            dtype=params_dtype,
        )
        bias_tensor = None
        if self.use_bias:
            bias_tensor = torch.empty(
                self.out_features,
                device=device,
                dtype=params_dtype,
            )

        # Configure parameter splits
        self.weight_names = []
        self.bias_names = []
        self.parameter_split_sizes = []
        if parameters_split is None:
            # Split into a single parameter by default
            self.weight_names = ["weight"]
            self.bias_names = ["bias"]
            self.parameter_split_sizes = [out_features]
        elif not parameters_split:
            raise ValueError("Cannot split weight buffer into 0 parameters")
        elif isinstance(parameters_split, dict):
            # Split parameters with provided sizes
            for name, split_size in parameters_split.items():
                self.weight_names.append(f"{name.rstrip('_')}_weight")
                self.bias_names.append(f"{name.rstrip('_')}_bias")
                self.parameter_split_sizes.append(split_size)
        elif all(isinstance(name, str) for name in parameters_split):
            # Split parameters evenly
            split_size = out_features // len(parameters_split)
            for name in parameters_split:
                self.weight_names.append(f"{name.rstrip('_')}_weight")
                self.bias_names.append(f"{name.rstrip('_')}_bias")
                self.parameter_split_sizes.append(split_size)
        else:
            raise TypeError("Invalid configuration for parameters split")

        # Make sure parameter splits are valid
        if sum(self.parameter_split_sizes) != out_features:
            raise ValueError(
                f"Trying to split weight buffer ({out_features=}) "
                f"with split sizes {self.parameter_split_sizes}"
            )

        # Adjust parameter splits for tensor-parallel distribution
        if self.parallel_mode == "column":
            for i, size in enumerate(self.parameter_split_sizes):
                if size % self.tp_size != 0:
                    raise RuntimeError(
                        f"Attempting to distribute a parameter with out_features={size} "
                        f"between {self.tp_size} tensor-parallel processes"
                    )
                self.parameter_split_sizes[i] = size // self.tp_size

        # Construct weight parameters
        # Note: Register weights together so that they are adjacent to
        # each other in LayerNormLinear.parameters(). This makes it
        # more likely that they will stay contiguous if the weights
        # are manipulated externally, e.g. by FSDP.
        offset = 0
        for i, split_size in enumerate(self.parameter_split_sizes):
            split_start = offset
            offset += split_size
            split_end = offset

            # Check if parameters are subviews of buffers
            is_subview = (split_start, split_end) != (0, self.out_features)
            if is_subview and with_fp8_params:
                raise RuntimeError("Splitting Float8Tensor into multiple params is not supported")

            # Construct weight parameter
            self.register_parameter(
                self.weight_names[i],
                torch.nn.Parameter(weight_tensor[split_start:split_end]),
                init_fn=init_method,
                get_rng_state_tracker=get_rng_state_tracker,
                fp8_meta_index=tex.FP8FwdTensors.GEMM1_WEIGHT,
            )

        # Construct bias parameters if needed
        if self.use_bias:
            offset = 0
            for i, split_size in enumerate(self.parameter_split_sizes):
                split_start = offset
                offset += split_size
                split_end = offset
                self.register_parameter(
                    self.bias_names[i],
                    torch.nn.Parameter(bias_tensor[split_start:split_end]),
                    init_fn=init_method_constant(0.0),
                )
        else:
            for name in self.bias_names:
                bias = torch.Tensor().to(dtype=params_dtype, device=device)
                setattr(self, name, bias)

        if with_fp8_params:
            self.init_fp8_metadata()

        self.reset_parameters(defer_init=device == "meta")

        # For RPL, bias has to be added after TP collectives
        # So it cannot be fused with the GEMM
        if self.parallel_mode == "row" and self.apply_bias:
            self.gemm_bias_unfused_add = True
        else:
            self.gemm_bias_unfused_add = False

        # These many SMs are subtracted from the total SM count when calling forward
        # and backward LayerNorm C APIs. These envvars can be used to prevent the LN
        # kernels from using all SMs in the device. This is useful for cases such as
        # communication overlap with LN.
        self.fwd_ln_sm_margin = int(os.getenv("NVTE_FWD_LAYERNORM_SM_MARGIN", "0"))
        self.bwd_ln_sm_margin = int(os.getenv("NVTE_BWD_LAYERNORM_SM_MARGIN", "0"))
        self.inf_ln_sm_margin = int(os.getenv("NVTE_INF_LAYERNORM_SM_MARGIN", "0"))

    def reset_layer_norm_parameters(self) -> None:
        """Init LN params"""
        warnings.warn(
            "This method will be deprecated in an upcoming release. "
            "Update your code to use LayerNormLinear.reset_parameters() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not self.zero_centered_gamma:
            init.ones_(self.layer_norm_weight)
        else:
            init.zeros_(self.layer_norm_weight)
        if self.layer_norm_bias is not None:
            init.zeros_(self.layer_norm_bias)

    def reset_parameters(self, defer_init=False):
        super().reset_parameters(defer_init=defer_init)

        if not defer_init:
            # Set parallelism attributes for layer norm parameters
            setattr(self.layer_norm_weight, "sequence_parallel", self.sequence_parallel)
            if self.normalization != "RMSNorm":
                setattr(self.layer_norm_bias, "sequence_parallel", self.sequence_parallel)

            # Set parallelism attributes for linear weights
            for weight in self.weight_names:
                set_tensor_model_parallel_attributes(
                    tensor=getattr(self, weight),
                    is_parallel=True,
                    dim=1 if self.parallel_mode == "row" else 0,
                    stride=1,
                )

            # Set parallelism attributes for linear biases
            if self.use_bias:
                for bias in self.bias_names:
                    if self.parallel_mode == "row":
                        setattr(getattr(self, bias), "sequence_parallel", self.sequence_parallel)
                    elif self.parallel_mode == "column":
                        set_tensor_model_parallel_attributes(getattr(self, bias), True, 0, 1)

    @no_torch_dynamo()
    def forward(
        self,
        inp: torch.Tensor,
        is_first_microbatch: Optional[bool] = None,
        fp8_output: Optional[bool] = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Apply layer normalization to the input followed by a linear transformation.

        Parameters
        ----------
        inp : torch.Tensor
             Input tensor.
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

        with self.prepare_forward(inp, is_first_microbatch) as inp:

            # Get concatenated weight and bias tensors
            unfused_weights = [getattr(self, name) for name in self.weight_names]
            weight_tensor = _noop_cat(unfused_weights)
            if self.use_bias:
                bias_tensor = _noop_cat([getattr(self, name) for name in self.bias_names])
            else:
                bias_tensor = getattr(self, self.bias_names[0])  # Unused

            # Initialize FP8 weights if needed
            weight_fp8 = None

            if torch.is_grad_enabled():
                fwd_fn = _LayerNormLinear.apply
                args = []
            else:
                fwd_fn = _LayerNormLinear.forward
                args = [None]

            fp8 = CustomManger.fp8
            args += (
                inp,
                self.layer_norm_weight,
                self.layer_norm_bias,
                weight_tensor,
                weight_fp8,
                bias_tensor,
                self.apply_bias and not self.gemm_bias_unfused_add,
                self.eps,
                is_first_microbatch,
                False, # self.fp8,
                self.fp8_calibration,
                self.fp8_meta,
                self.fuse_wgrad_accumulation,
                is_cpu_offload_enabled(),
                self.tp_group,
                self.tp_size,
                self.sequence_parallel,
                self.tp_size > 1,
                self.activation_dtype,
                self.parallel_mode,
                self.return_layernorm_output,
                self.return_layernorm_output_gathered,
                torch.is_grad_enabled(),
                self.fwd_ln_sm_margin if torch.is_grad_enabled() else self.inf_ln_sm_margin,
                self.bwd_ln_sm_margin,
                self.zero_centered_gamma,
                self.normalization,
                self.ub_bulk_wgrad if not fp8 else False,
                self.ub_bulk_dgrad if not fp8 else False,
                self.ub_overlap_rs_dgrad if not fp8 else False,
                self.ub_overlap_ag if not fp8 else False,
                self.ub_name if not fp8 else None,
                fp8_output,
                self.fsdp_group,
            )
            out = fwd_fn(*args)

        if self.return_layernorm_output:
            out, ln_out = out

        if self.gemm_bias_unfused_add:
            out = out + cast_if_needed(bias_tensor, self.activation_dtype)

        if self.return_bias:
            if self.return_layernorm_output:
                return out, cast_if_needed(bias_tensor, self.activation_dtype), ln_out
            return out, cast_if_needed(bias_tensor, self.activation_dtype)
        if self.return_layernorm_output:
            return out, ln_out
        return out
