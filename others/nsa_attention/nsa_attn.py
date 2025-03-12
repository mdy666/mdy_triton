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
from fla.ops.nsa import parallel_nsa


    

class NsaAttention(torch.nn.Module):
    def __init__(self, qk_head_dim, v_head_dim, kernel_size, stride, select_size, top_n, window_size):
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
        
        # self.select_attn = partial(parallel_nsa, 
        #                            block_size=self.select_size, 
        #                            scale=self.sm_scale)
        self.window_attn = partial(flash_attn_func, 
                                   softmax_scale=self.sm_scale, 
                                   causal=True, 
                                   window_size=(self.window_size, -1) )

        self.attn_weight = torch.nn.Sequential(torch.nn.Linear(qk_head_dim, 3),
                                               torch.nn.Sigmoid())

    
    def forward(self, q, k, v, **kwargs):
        cmp_o, lse, cmp_k = self.compress_attn(q, k, v) # 17ms
        
        _, fwd_ind, bwd_ind = self.select_for_fwd_bwd(q, cmp_k, lse) # 14ms
        # return
        select_o = self.select_attn(q, k, v, fwd_ind=fwd_ind, bwd_ind=bwd_ind,**kwargs) # 31ms
        # select_o = self.select_attn(q, k, v, fwd_ind.transpose(1,2).contiguous())
        window_o = self.window_attn(q, k, v) # 2.7ms
        # return
        if isinstance(window_o, Tuple):
            window_o = window_o[0]
        weight = self.attn_weight(q) # 1ms
        # return 
        combine_o = cmp_o * weight[..., 0].unsqueeze(-1) \
                    + select_o * weight[..., 1].unsqueeze(-1) \
                    + window_o * weight[..., 2].unsqueeze(-1) # 6ms
        return combine_o

