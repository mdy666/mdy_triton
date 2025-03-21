import torch
import math
from functools import partial
from typing import Tuple

try:
    # fa3
    from flash_attn_interface import flash_attn_func 
except:
    # fa2
    from flash_attn import flash_attn_func

from compress_attn import CompressAttn
from select_attn_v3 import select_attn, select_for_fwd_bwd
from combine import fused_sigmoid_combine

class NsaAttention(torch.nn.Module):
    """
    native sparse attention.

    Args:
        qk_head_dim (int): head dim of q and k head

        v_head_dim (int): head dim of v head

        kernel_size (int): how many kv will be compressed and become a compressed kv block, the "l" in the paper

        stride (int): like conv stride, compress the next block will move how many kv, the "d" in the paper

        select_size (int): select block size, the "l'" in the paper

        top_n (int): q will chosses how many select blocks.

        window_size (int): sliding window size for window attention
    """
    def __init__(self, qk_head_dim, v_head_dim, kernel_size=32, stride=16, select_size=64, top_n=16, window_size=512):
        super().__init__()
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.select_size = select_size
        self.top_n = top_n
        self.window_size = window_size
        self.sm_scale = qk_head_dim ** -0.5
        assert math.log2(self.stride).is_integer()
        assert kernel_size % stride == 0 and select_size % kernel_size == 0

        self.compress_attn = CompressAttn(qk_head_dim, v_head_dim, kernel_size, stride)
        self.select_for_fwd_bwd = partial(select_for_fwd_bwd, 
                                      kernel_size=self.kernel_size, 
                                      stride=self.stride, 
                                      select_size=self.select_size, 
                                      top_n=self.top_n, 
                                      sm_scale=self.sm_scale)
        
        self.select_attn = partial(select_attn, 
                                   select_size=self.select_size, 
                                   sm_scale=self.sm_scale)
        
        self.window_attn = partial(flash_attn_func, 
                                   softmax_scale=self.sm_scale, 
                                   causal=True, 
                                   window_size=(self.window_size, -1) )

        self.attn_gate = torch.nn.Linear(qk_head_dim, 3)

    
    def forward(self, q, k, v, inplace=True):
        """
        Forward pass for the NSA Attention module.

        Args:
            q (torch.Tensor): [b, seq_len, num_q_head, qk_head_dim]
            k (torch.Tensor): [b, seq_len, num_kv_head, qk_head_dim]
            v (torch.Tensor): [b, seq_len, num_kv_head, v_head_dim]
            inplace (bool): in the backward the bwd_ind will be update in-place, set False for benchmark
        Returns:
            o (torch.Tensor): [b, seq_len, num_q_head, v_head_dim]
        """
        # inplace用于测试，bwd_ind会被原地修改，改为不原地修改
        cmp_o, lse, cmp_k = self.compress_attn(q, k, v) # 17ms
        _, fwd_ind, bwd_ind = self.select_for_fwd_bwd(q, cmp_k, lse) # 14ms
        select_o = self.select_attn(q, k, v, fwd_ind=fwd_ind, bwd_ind=bwd_ind, inplace=inplace) # 16ms
        window_o = self.window_attn(q, k, v) # 2.7ms
        if isinstance(window_o, Tuple):
            window_o = window_o[0]
        weight = self.attn_gate(q) # 1ms 
        combine_o = fused_sigmoid_combine(cmp_o, select_o, window_o, weight) # 1.2ms
        return combine_o
