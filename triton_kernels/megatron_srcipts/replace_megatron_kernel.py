
from copy import deepcopy
from itertools import chain
from typing import cast, Tuple, Union

import torch
from torch import Tensor
from torch.optim import Optimizer
import triton
import triton.language as tl

from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)

import importlib


@triton.jit
def _cross_entropy_fwd_kernel(LOGITS, LABELS, LOSSES, LOGSUMEXP,
                             vocab_start_index, row_stride, 
                             M, N, SPLIT, BLOCK_SIZE: tl.constexpr, 
                             ):
    row_idx = tl.program_id(0)
    row_stride = row_stride.to(tl.int64)
    label_idx = tl.load(LABELS + row_idx).to(tl.int32)
    if (label_idx != -100):
        LOGITS += row_idx * row_stride
        base_cols = tl.arange(0, BLOCK_SIZE)
        m_i = -float("inf")
        l_i = 0.0
        for start_n in tl.range(0, N, BLOCK_SIZE):
            cols = start_n + base_cols
            mask = cols < N
            logits = tl.load(LOGITS+cols, mask=mask, other=-float('inf')).to(tl.float32)
            m_ij = tl.max(logits)
            new_m_i = tl.maximum(m_i, m_ij)
            l_i = l_i * tl.exp(m_i - new_m_i) + tl.sum(tl.exp(logits - new_m_i))
            m_i = new_m_i
        lse = tl.log(l_i) + m_i

        if (label_idx >= vocab_start_index) and (label_idx < (vocab_start_index + N)):
            x = -1.0 * tl.load(LOGITS+label_idx-vocab_start_index).to(tl.float32)
            if not SPLIT:
                loss = lse + x
                tl.store(LOSSES+row_idx, loss)
            else:
                tl.store(LOSSES+row_idx, x)
        tl.store(LOGSUMEXP+row_idx, lse)

@triton.jit
def _cross_entropy_bwd_kernel(DLOSSES, DLOGITS,
                            LOGITS, LABELS, LOGSUMEXP,
                             vocab_start_index, row_stride, 
                             M, N,  INPLACE,
                             BLOCK_SIZE: tl.constexpr,
                             ):
    row_idx = tl.program_id(0)
    LABELS += row_idx
    label_idx = tl.load(LABELS).to(tl.int32)
    row_stride = row_stride.to(tl.int64)
    if (label_idx != -100):
        # label_idx -= vocab_start_index
        LOGITS += row_idx * row_stride
        DLOGITS += row_idx * row_stride
        LOGSUMEXP += row_idx
        DLOSSES += row_idx
        lse = tl.load(LOGSUMEXP)
        dloss = tl.load(DLOSSES).to(tl.float32)
        base_cols = tl.arange(0, BLOCK_SIZE)
        for start_n in tl.range(0, N, BLOCK_SIZE):
            cols = start_n + base_cols
            mask = cols < N
            logits = tl.load(LOGITS+cols, mask=mask, other=0.).to(tl.float32)
            probs = tl.exp(logits - lse)
            tmp = vocab_start_index + start_n
            if (label_idx >= tmp) and (label_idx < (tmp + BLOCK_SIZE)):
                probs = tl.where(cols+vocab_start_index != label_idx, probs, probs-1.)
            tl.store(DLOGITS+cols, probs * dloss, mask=mask)
    elif INPLACE:
        DLOGITS += row_idx * row_stride
        base_cols = tl.arange(0, BLOCK_SIZE)
        zeros = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for start_n in tl.range(0, N, BLOCK_SIZE):
            cols = start_n + base_cols
            mask = cols < N
            tl.store(DLOGITS+cols, zeros, mask=mask)

class _FastCrossEntropyLoss(torch.autograd.Function):

    @staticmethod
    def forward(ctx, logits, labels, inplace):
        ctx.input_shape = logits.shape
        # tp_rank = 0
        # tp_size = 1
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        tp_group = get_tensor_model_parallel_group()
        N = ctx.input_shape[-1]
        logits = logits.view(-1, N)
        M = logits.size(0)
        losses = torch.zeros(M, device=logits.device, dtype=torch.float32)
        split = tp_size > 1
        vocab_start_index = N * tp_rank
        logsumexp = torch.zeros(M, device=logits.device, dtype=torch.float32)
        # print(logsumexp.stride(), losses.stride())
        with torch.cuda.device(logits.device):
            _cross_entropy_fwd_kernel[(M,)](logits, labels, losses, logsumexp,
                                                vocab_start_index, logits.stride(0),
                                                M, N, split,
                                                BLOCK_SIZE=4096, num_warps=4, num_stages=3
                                                )
        if tp_size>1:
            lse_allgather = torch.empty(tp_size, M, dtype=logsumexp.dtype, device=logsumexp.device)
            torch.distributed.all_gather_into_tensor(lse_allgather, logsumexp, group=tp_group)
            torch.distributed.all_reduce(
                losses, op=torch.distributed.ReduceOp.SUM,
            )
            logsumexp = torch.logsumexp(lse_allgather, dim=0)
            losses += logsumexp
            losses.masked_fill_(labels.view(-1)==-100, 0.)
        ctx.save_for_backward(logits, labels, logsumexp)
        ctx.inplace = inplace
        ctx.tp_rank = tp_rank
        return losses.view(*ctx.input_shape[:-1])
    
    @staticmethod
    def backward(ctx, dlosses):
        logits, labels, logsumexp = ctx.saved_tensors
        dlogits = logits if ctx.inplace else torch.zeros_like(logits)
        N = logits.size(-1)
        logits = logits.view(-1, N)
        M = logits.size(0)
        vocab_start_index = N * ctx.tp_rank
        BLOCK_SIZE = min(triton.next_power_of_2(N), 32768)
        with torch.cuda.device(logits.device):
            _cross_entropy_bwd_kernel[(M,)](dlosses, dlogits, 
                                            logits, labels, logsumexp,
                                            vocab_start_index, logits.stride(0),
                                            M, N, ctx.inplace, 
                                            BLOCK_SIZE=BLOCK_SIZE, num_warps=32, num_stages=1
                                            )
        return dlogits.view(*ctx.input_shape), None, None

   
@triton.jit
def _group_tensor_apply_adam(
                            P,
                            G,
                            M,
                            V,
                            NUMELS,
                            lr, beta1, beta2, weight_decay, eps,
                            beta1t, beta2t, bf16_p:tl.constexpr, bf16_mv:tl.constexpr, bf16_g:tl.constexpr,
                            N, BLOCK_SIZE:tl.constexpr,
                            NUM_SM:tl.constexpr=512):
    tile_idx = tl.program_id(0)

    p_dtype = tl.bfloat16 if bf16_p else tl.float32
    g_dtype = tl.bfloat16 if bf16_g else tl.float32   
    mv_dtype = tl.bfloat16 if bf16_mv else tl.float32
    last_problem_end = 0
    for n in range(N):
        numels = tl.load(NUMELS+n)
        num_tiles = tl.cdiv(numels, BLOCK_SIZE)
        if (tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles):  
            p_ptr = tl.load(P+n).to(tl.pointer_type(p_dtype))
            g_ptr = tl.load(G+n).to(tl.pointer_type(g_dtype))
            m_ptr = tl.load(M+n).to(tl.pointer_type(mv_dtype))
            v_ptr = tl.load(V+n).to(tl.pointer_type(mv_dtype))
            cols = tl.arange(0,BLOCK_SIZE)
            while (tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles):

                offset = (tile_idx - last_problem_end)*BLOCK_SIZE + cols
                mask = offset < numels

                p = tl.load(p_ptr+offset, mask=mask).to(tl.float32)
                g = tl.load(g_ptr+offset, mask=mask).to(tl.float32)
                m = tl.load(m_ptr+offset, mask=mask).to(tl.float32)
                v = tl.load(v_ptr+offset, mask=mask).to(tl.float32)

                p *= (1-lr*weight_decay)
                m = m*beta1 + (1-beta1)*g
                v = v*beta2 + (1-beta2)*g*g
                tl.store(m_ptr+offset, m, mask=mask)
                tl.store(v_ptr+offset, v, mask=mask)

                m = m / (1 - beta1t)
                v = v / (1 - beta2t)

                p = p - lr * m / (tl.sqrt(v) + eps)
                tl.store(p_ptr+offset, p, mask=mask)

                tile_idx += NUM_SM

        last_problem_end = last_problem_end + num_tiles

@triton.jit
def _group_tensor_apply_adam_with_master_p(MASTER_P,
                            P,
                            G,
                            M,
                            V,
                            NUMELS,
                            lr, beta1, beta2, weight_decay, eps,
                            beta1t, beta2t, bf16_p:tl.constexpr, bf16_mv:tl.constexpr, bf16_g:tl.constexpr,
                            N, BLOCK_SIZE:tl.constexpr,
                            NUM_SM:tl.constexpr=512):
    tile_idx = tl.program_id(0)
    p_dtype = tl.bfloat16 if bf16_p else tl.float32
    g_dtype = tl.bfloat16 if bf16_g else tl.float32   
    mv_dtype = tl.bfloat16 if bf16_mv else tl.float32

    last_problem_end = 0
    for n in range(N):
        numels = tl.load(NUMELS+n)
        num_tiles = tl.cdiv(numels, BLOCK_SIZE)
        if (tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles):            
            p_ptr = tl.load(P+n).to(tl.pointer_type(p_dtype))
            master_p_ptr = tl.load(MASTER_P+n).to(tl.pointer_type(tl.float32))
            g_ptr = tl.load(G+n).to(tl.pointer_type(g_dtype))
            m_ptr = tl.load(M+n).to(tl.pointer_type(mv_dtype))
            v_ptr = tl.load(V+n).to(tl.pointer_type(mv_dtype))
            cols = tl.arange(0,BLOCK_SIZE)
            while (tile_idx >= last_problem_end and tile_idx < last_problem_end + num_tiles):

                offset = (tile_idx - last_problem_end)*BLOCK_SIZE + cols
                mask = offset < numels

                p = tl.load(master_p_ptr+offset, mask=mask).to(tl.float32)
                g = tl.load(g_ptr+offset, mask=mask).to(tl.float32)
                m = tl.load(m_ptr+offset, mask=mask).to(tl.float32)
                v = tl.load(v_ptr+offset, mask=mask).to(tl.float32)

                p *= (1-lr*weight_decay)
                m = m*beta1 + (1-beta1)*g
                v = v*beta2 + (1-beta2)*g*g
                tl.store(m_ptr+offset, m, mask=mask)
                tl.store(v_ptr+offset, v, mask=mask)

                m = m / (1 - beta1t)
                v = v / (1 - beta2t)

                p = p - lr * m / (tl.sqrt(v) + eps)
                tl.store(p_ptr+offset, p, mask=mask)
                tl.store(master_p_ptr+offset, p, mask=mask)

                tile_idx += NUM_SM

        last_problem_end = last_problem_end + num_tiles

@torch.no_grad
def triton_adam_group(master_params,
                params, 
                grads, 
                exp_avgs, 
                exp_avg_sqs, 
                numels, 
                lr, 
                beta1, 
                beta2, 
                weight_decay, 
                eps, 
                beta1t,
                beta2t,
                master_weight,
                bf16_p,
                bf16_mv,
                bf16_g,
                ):
    
    N = len(params)
    BLOCK_SIZE = 2048
    grid = lambda meta: (meta['NUM_SM'], )
    if not master_weight:
        with torch.cuda.device(params.device):
            _group_tensor_apply_adam[grid](params,
                                        grads,
                                        exp_avgs,
                                        exp_avg_sqs,
                                        numels,
                                        lr, beta1, beta2, weight_decay, eps,
                                        beta1t, beta2t, bf16_p, bf16_mv, bf16_g,
                                        N, BLOCK_SIZE, num_warps=4, num_stages=1)
    else:
        with torch.cuda.device(params.device):
            _group_tensor_apply_adam_with_master_p[grid](master_params, params,
                                        grads,
                                        exp_avgs,
                                        exp_avg_sqs,
                                        numels,
                                        lr, beta1, beta2, weight_decay, eps,
                                        beta1t, beta2t, bf16_p, bf16_mv, bf16_g,
                                        N, BLOCK_SIZE, num_warps=4, num_stages=1)


class TritonAdamW(Optimizer):
    def __init__(
        self,
        params,
        lr: Union[float, Tensor] = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        master_weights=False,
        use_decoupled_grad=False,
        exp_avg_dtype=torch.float32,
        exp_avg_sq_dtype=torch.float32,
        master_weight_dtype=torch.float32,
        **kwargs,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        assert exp_avg_dtype == exp_avg_sq_dtype

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

        self.bf16_mv = exp_avg_sq_dtype != torch.float32
        self.master_weights = master_weights
        self.use_decoupled_grad = use_decoupled_grad

        self.name_to_dtype_map = {
            "exp_avg": torch.bfloat16 if exp_avg_dtype!=torch.float32 else exp_avg_dtype,
            "exp_avg_sq": torch.bfloat16 if exp_avg_sq_dtype!=torch.float32 else exp_avg_sq_dtype,
            "master_param": torch.float32,
        }

        self.p_ptrs_groups = []
        self.m_ptrs_groups = []
        self.v_ptrs_groups = []
        self.numels_groups = []
        self.master_p_ptrs_groups = []

    #te
    def set_scaled_state(self, param, state_name, unscaled_state):
        """Set the optimizer state.

        If the dtype of the corresponding optimizer state is not FP32,
        it will do scaling automatically.

        Arguments:
            param (torch.nn.Parameter): One of parameters in this optimizer.
            state_name (string): Name of optimizer states, can be one of 'exp_avg', 'exp_avg_sq',
                and 'master_param`.
            unscaled_state (torch.Tensor): The original high-precision(FP32) state.
        """
        state = self.state[param]
        if state_name not in state:
            state[state_name] = torch.empty_like(param, dtype=self.name_to_dtype_map[state_name])

        state[state_name].copy_(unscaled_state)

    #te
    def get_unscaled_state(self, param, state_name):
        dtype = self.state[param][state_name].dtype 
        assert dtype == self.name_to_dtype_map[state_name], dtype
        return self.state[param][state_name].float()
    
    def custom_init(self):
        for group in self.param_groups:
            p_ptrs = []
            m_ptrs = []
            v_ptrs = []
            numels = []
            master_p_ptrs = []
            # steps = []
            for p in group['params']:
                assert p.is_contiguous()
                if not self.use_decoupled_grad and not p.requires_grad:
                    continue
                state = self.state[p]
                # Exponential moving average of gradient  values
                if state.get("exp_avg", None) is None:
                    state["exp_avg"] = torch.zeros_like(p, dtype=self.name_to_dtype_map['exp_avg'])
                assert state["exp_avg"].dtype == self.name_to_dtype_map['exp_avg'], state["exp_avg"].dtype
                assert state["exp_avg"].is_contiguous()

                # Exponential moving average of squared gradient values
                if state.get("exp_avg_sq", None) is None:
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=self.name_to_dtype_map['exp_avg_sq'])
                assert state["exp_avg_sq"].dtype == self.name_to_dtype_map['exp_avg_sq'], state["exp_avg_sq"].dtype
                assert state["exp_avg_sq"].is_contiguous()
                
                if self.master_weights:
                    if state.get('master_param', None) is None:
                        state["master_param"] = torch.empty_like(p, dtype=self.name_to_dtype_map['master_param'])
                        state["master_param"].copy_(p.clone().detach().float())
                    assert state["master_param"].dtype == self.name_to_dtype_map['master_param'], state["master_param"].dtype
                    assert state["master_param"].is_contiguous()



                # state['step'] = torch.tensor(0.0, dtype=torch.int32)
                p_ptrs.append(p.data_ptr())
                m_ptrs.append(state["exp_avg"].data_ptr())
                v_ptrs.append(state["exp_avg_sq"].data_ptr())
                numels.append(p.numel())
                if self.master_weights:
                    master_p_ptrs.append(state["master_param"].data_ptr())
            if group.get('step', None) is None:
                group['step'] = 0
            
            if p_ptrs:
                self.p_ptrs_groups.append(torch.tensor(p_ptrs, dtype=torch.int64).to(p.device))
                self.m_ptrs_groups.append(torch.tensor(m_ptrs, dtype=torch.int64).to(p.device))
                self.v_ptrs_groups.append(torch.tensor(v_ptrs, dtype=torch.int64).to(p.device))
                self.numels_groups.append(torch.tensor(numels, dtype=torch.int32).to(p.device))
                if self.master_weights:
                    self.master_p_ptrs_groups.append(torch.tensor(master_p_ptrs, dtype=torch.int64).to(p.device))
                else:
                    self.master_p_ptrs_groups.append([])
        torch.cuda.empty_cache() 
        self.bf16_p = p.dtype == torch.bfloat16
        master_weight_dtype = state['master_param'].dtype if self.master_weights else None
        print(f"finish_custom_init, p_dtype: {p.dtype}, master_p_dtype: {master_weight_dtype}")
        
    def step(self, *args):
        
        if len(self.p_ptrs_groups) == 0:
            self.custom_init()
        self._cuda_graph_capture_health_check()

        for idx, group in enumerate(self.param_groups):
            group['step'] += 1
            t = group['step']
            if idx>= len(self.p_ptrs_groups):
                continue
            beta1, beta2 = cast(Tuple[float, float], group["betas"])
            beta1t = beta1 ** t
            beta2t = beta2 ** t

            master_p_ptrs = self.master_p_ptrs_groups[idx]
            p_ptrs = self.p_ptrs_groups[idx]
            m_ptrs = self.m_ptrs_groups[idx]
            v_ptrs = self.v_ptrs_groups[idx]
            numels = self.numels_groups[idx]

            if hasattr(group['params'][0], 'decoupled_grad'):
                g_ptrs = torch.tensor([p.decoupled_grad.data_ptr() for p in group['params'] if (p.requires_grad or self.use_decoupled_grad)], dtype=torch.int64).to(p_ptrs.device)
                bf16_grad = group['params'][0].decoupled_grad.dtype == torch.bfloat16
            else:
                g_ptrs = torch.tensor([p.grad.data_ptr() for p in group['params'] if p.requires_grad], dtype=torch.int64).to(p_ptrs.device)
                bf16_grad = group['params'][0].grad.dtype == torch.bfloat16
            triton_adam_group(master_p_ptrs,
                p_ptrs, 
                g_ptrs, 
                m_ptrs, 
                v_ptrs, 
                numels, 
                lr=group["lr"], 
                beta1=beta1, 
                beta2=beta2, 
                weight_decay=group['weight_decay'], 
                eps=group['eps'], 
                beta1t=beta1t,
                beta2t=beta2t,
                master_weight=self.master_weights,
                bf16_p=self.bf16_p,
                bf16_mv=self.bf16_mv,
                bf16_g=bf16_grad,
                )
            
    def zero_grad(self):
        # pylint: disable=missing-function-docstring
        if not self.use_decoupled_grad:
            super().zero_grad()
            return

        for group in self.param_groups:
            for p in group["params"]:
                if self.use_decoupled_grad:
                    p.decoupled_grad = None


    def state_dict(self):
        """Override the state_dict() of pytorch. Before returning the state_dict, cast all
        non-fp32 states to fp32.
        """
        state_dict = super().state_dict()

        groups = self.param_groups
        saved_groups = deepcopy(state_dict["param_groups"])
        id_map = dict(
            zip(
                chain.from_iterable(g["params"] for g in saved_groups),
                chain.from_iterable(g["params"] for g in groups),
            )
        )
        for k, v in state_dict["state"].items():
            if k in id_map:
                param = id_map[k]
                new_v = {}
                for name in v:
                    new_v[name] = self.get_unscaled_state(param, name)
                state_dict["state"][k] = new_v

        return state_dict

    def load_state_dict(self, state_dict):
        """Override the load_state_dict() of pytorch. Since pytorch's load_state_dict forces the
        state to be the same dtype as param, We need to manully set the state again.
        """
        super().load_state_dict(state_dict)

        groups = self.param_groups
        saved_groups = deepcopy(state_dict["param_groups"])
        id_map = dict(
            zip(
                chain.from_iterable(g["params"] for g in saved_groups),
                chain.from_iterable(g["params"] for g in groups),
            )
        )
        for k, v in state_dict["state"].items():
            if k in id_map:
                param = id_map[k]
                self.state[param] = {}
                for name in v:
                    self.set_scaled_state(param, name, v[name].float())

@triton.jit
def _fused_silu_fwdv2(X, Y,
                    N, 
                    stride_m, stride_n,  #
                    BLOCK_SIZE_N: tl.constexpr, ORDER:tl.constexpr,
                    ):

    pid = tl.program_id(axis=0)
    x_offset = pid * stride_m
    y_offset = x_offset // 2
    X += x_offset
    Y += y_offset
    if ORDER == 'up-gate':
        up_offset = 0
        gate_offset = N
    else:
        up_offset = N
        gate_offset = 0

    cols = tl.arange(0, BLOCK_SIZE_N)
    for start_n in tl.range(0, N, BLOCK_SIZE_N):
        new_cols = cols + start_n
        mask = new_cols < N
        up = tl.load(X+new_cols+up_offset, mask=mask, other=0.)
        dtype = up.dtype
        up = up.to(tl.float32)
        gate = tl.load(X+new_cols+gate_offset, mask=mask, other=0.).to(tl.float32)
        act = gate * tl.sigmoid(gate)
        y = act * up
        tl.store(Y+new_cols, y.to(dtype), mask=mask)


@triton.jit
def _fused_silu_bwd_dupgatev2(X, 
                            DY, DX,
                            N,
                            stride_m, stride_n,
                            BLOCK_SIZE_N: tl.constexpr,ORDER:tl.constexpr
                            ):
    pid = tl.program_id(0)
    x_offset = pid * stride_m
    y_offset = x_offset // 2
    X += x_offset
    DX += x_offset
    DY += y_offset
    if ORDER == 'up-gate':
        up_offset = 0
        gate_offset = N
    else:
        up_offset = N
        gate_offset = 0

    cols = tl.arange(0, BLOCK_SIZE_N)
    for start_n in range(0, N, BLOCK_SIZE_N):
        new_cols = cols + start_n
        mask = new_cols < N
        
        dy = tl.load(DY+new_cols, mask=mask, other=0.)
        dtype = dy.dtype
        gate = tl.load(X+new_cols+gate_offset, mask=mask, other=0.).to(tl.float32)
        up = tl.load(X+new_cols+up_offset, mask=mask, other=0.).to(tl.float32)
        gate_sigmoid = tl.sigmoid(gate)
        act = gate_sigmoid * gate
        dup = act * dy
        dact = up * dy
        dgate = (gate_sigmoid + act * (1-gate_sigmoid)) * dact
        tl.store(DX+new_cols+up_offset, dup.to(dtype), mask=mask)
        tl.store(DX+new_cols+gate_offset, dgate.to(dtype), mask=mask)

class _FusedSiLUNoSplit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, order='up-gate'):
        '''
        input:
            x     : torch.Tensor, [bs, L, 2*D], the output of self.fc1(x) in MLP, contain the up and gate
            order : str, the order of the x, must be gate-up or up-gate, default up-gate
        
        output:
            y    : torch.tensor, [bs, L, D], the result of up * silu(gate)

        example:
          original code:
            x = self.fc1(hidden_states)
            up, gate = x.chunk(2, -1)
            act = silu(gate)
            y = act * up

          new code:
            x = self.fc1(hidden_states)
            y = fused_up_gate_silu_no_split(x)
        '''
        assert order in ['up-gate', 'gate-up'], "please indicate the order of input, up-gate or gate-up"
        input_shape = x.shape
        x = x.view(-1, input_shape[-1])
        M, N2 = x.shape
        N = N2 // 2
        y = torch.empty(*input_shape[:-1], N, device=x.device, dtype=x.dtype)
        BLOCK_SIZE_N = triton.next_power_of_2(N)
        BLOCK_SIZE_N = min(4096, BLOCK_SIZE_N)
        num_warps = 8
        num_stages = 4
        _fused_silu_fwdv2[(M,)](
            x, y, 
            N,  #
            *x.stride(),  #
            BLOCK_SIZE_N, order,
            num_warps=num_warps, num_stages=num_stages, 
        )
        ctx.infos = (M, N, BLOCK_SIZE_N, *x.stride(), order)
        ctx.input_shape = input_shape
        ctx.save_for_backward(x)
        ctx.num_warps = num_warps
        ctx.num_stages = num_stages
        return y
    
    @staticmethod
    def backward(ctx, dy):
        M, N, BLOCK_SIZE_N, stride_m, stride_n, order = ctx.infos
        # print(stride_m, stride_n)
        x, = ctx.saved_tensors

        dx = torch.empty(ctx.input_shape, device=dy.device, dtype=dy.dtype)
        # BLOCK_SIZE_N = min(8192, BLOCK_SIZE_N)
        _fused_silu_bwd_dupgatev2[(M,)](x,
                                   dy, dx,
                                   N,
                                   stride_m, stride_n,
                                   BLOCK_SIZE_N, order,
                                   num_warps=ctx.num_warps, num_stages=ctx.num_stages)

        return dx, None
    

import time 
def swiglu_impl(x, bias, fp8_input_store=False):
    print(1)
    return _FusedSiLUNoSplit.apply(x, 'gate-up')

def cross_entropy_loss(logits, labels):
    print(2)
    time.sleep(5)
    return _FastCrossEntropyLoss.apply(logits, labels, True)

moudel_swiglu = importlib.import_module('megatron.core.fusions.fused_bias_swiglu')
moudel_swiglu.bias_swiglu_impl = swiglu_impl

module_ce = importlib.import_module('megatron.core.tensor_parallel')
module_ce.vocab_parallel_cross_entropy = cross_entropy_loss
module_fused_ce = importlib.import_module('megatron.core.fusions.fused_cross_entropy')
module_fused_ce.fused_vocab_parallel_cross_entropy = cross_entropy_loss

moudel_te_optimizer = importlib.import_module('transformer_engine.pytorch.optimizers')
moudel_te_optimizer.FusedAdam = TritonAdamW

trigger = None
print('kenrel replace done')
 



