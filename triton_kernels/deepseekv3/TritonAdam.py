from copy import deepcopy
from itertools import chain
from typing import cast, Tuple, Union

import torch
from torch import Tensor
from torch.optim import Optimizer
import triton
import triton.language as tl



# don‘t use it in training, the tuning will change the params, please do it in test
# @triton.autotune(
#     configs = [
#         triton.Config(
#             {
#                 'BLOCK_SIZE': 2048,
#                 'NUM_SM': num_sm,
#             },
#             num_stages=1,
#             num_warps=4,
#         )
#         for num_sm in [96, 128, 160, 256, 512]
#     ],
#     key=['N'],
# )    
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

# master_weight只可以是fp32
# 1，2阶动量不是fp32的话，自动转换为bf16
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
                    # print('init exp_avg')
                    state["exp_avg"] = torch.zeros_like(p, dtype=self.name_to_dtype_map['exp_avg'])
                assert state["exp_avg"].dtype == self.name_to_dtype_map['exp_avg'], state["exp_avg"].dtype
                assert state["exp_avg"].is_contiguous()

                # Exponential moving average of squared gradient values
                if state.get("exp_avg_sq", None) is None:
                    # print('init exp_avg_sq')
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=self.name_to_dtype_map['exp_avg_sq'])
                assert state["exp_avg_sq"].dtype == self.name_to_dtype_map['exp_avg_sq'], state["exp_avg_sq"].dtype
                assert state["exp_avg_sq"].is_contiguous()
                
                if self.master_weights:
                    if state.get('master_param', None) is None:
                        # print('init master_weight')
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

    # te        
    def zero_grad(self):
        # pylint: disable=missing-function-docstring
        if not self.use_decoupled_grad:
            super().zero_grad()
            return

        for group in self.param_groups:
            for p in group["params"]:
                if self.use_decoupled_grad:
                    p.decoupled_grad = None

    # te 
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
    
    # te 
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



# update weight one by one, it is not efficient
@triton.jit
def _single_tensor_apply_adam(P, G, M, V,
                            lr, beta1, beta2, weight_decay, eps,
                            beta1t, beta2t,
                            N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    offset = pid * BLOCK_SIZE + cols
    mask = offset < N

    p = tl.load(P+offset, mask=mask).to(tl.float32)
    g = tl.load(G+offset, mask=mask).to(tl.float32)
    m = tl.load(M+offset, mask=mask).to(tl.float32)
    v = tl.load(V+offset, mask=mask).to(tl.float32)
    beta1t = tl.load(beta1t).to(tl.float32)
    beta2t = tl.load(beta2t).to(tl.float32)

    p *= (1-lr*weight_decay)
    m = m*beta1 + (1-beta1)*g
    v = v*beta2 + (1-beta2)*g*g
    tl.store(M+offset, m, mask=mask)
    tl.store(V+offset, v, mask=mask)

    m = m / (1 - beta1t)
    v = v / (1 - beta2t)

    p = p - lr * m / (tl.sqrt(v) + eps)
    tl.store(P+offset, p, mask=mask)


@torch.no_grad
def triton_adam_no_group(params, 
                grads, 
                exp_avgs, 
                exp_avg_sqs, 
                state_steps, 
                lr, 
                beta1, 
                beta2, 
                weight_decay, 
                eps, 
                ):
    for idx in range(len(params)):
        t = state_steps[idx]
        t += 1
        N = params[idx].numel()
        if N <= 8192:
            BLOCK_SIZE = 128
        else:
            BLOCK_SIZE = 512
        # print(beta1**t)
        _single_tensor_apply_adam[(triton.cdiv(N, BLOCK_SIZE), )](params[idx], 
                                    grads[idx],
                                    exp_avgs[idx],
                                    exp_avg_sqs[idx],
                                    lr, beta1, beta2, weight_decay, eps,
                                    beta1**t, beta2**t, 
                                    N, BLOCK_SIZE)



