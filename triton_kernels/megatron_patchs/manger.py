import torch
from typing import Tuple, Optional
import re
import torch.distributed as dist
from .fp8_matmul import per_block_cast_to_fp8, per_token_cast_to_fp8, fp8_matmul, fp8_grouped_matmul, group_per_token_cast_to_fp8
from functools import partial
from .add_kernel import add_bias, add_grad


class CustomManger:
    store_wgrad_func = False
    wgrad_func = []

    fp8 = True
    fp8_weight_map = {}

    send_ops = []
    recv_ops = []

    partial = partial
    add_bias = add_bias
    add_grad = add_grad

    group_per_token_cast_to_fp8 = group_per_token_cast_to_fp8

    @classmethod
    def weight_backward(cls):
        if not cls.store_wgrad_func:
            return
        func = cls.wgrad_func.pop(0)
        func()

    @classmethod
    def per_block_cast_to_fp8(cls, x, output_normal=True, output_transpose=False):
        return per_block_cast_to_fp8(x, output_normal, output_transpose)
    
    @classmethod
    def per_token_cast_to_fp8(cls, x, output_transpose=False):
        return per_token_cast_to_fp8(x, output_transpose)
    
    @classmethod
    def fp8_matmul(cls, qa, sa, qb, sb, out=None):
        return fp8_matmul(qa, sa, qb, sb, out)

    @classmethod
    def fp8_grouped_matmul(cls, qa, sa, qb, sb, m_splits, out=None):
        return fp8_grouped_matmul(qa, sa, qb, sb, m_splits, out)
    
    @classmethod
    def fix_tensor_shape(cls, tensor, n):
        tensor = tensor[:n]
        if not tensor.is_contiguous():
            new_tensor = tensor.contiguous()
            cls.clear_tensor_data(tensor)
            return new_tensor
        return tensor
    
    # @classmethod
    # def partial(cls):
    #     return partial
    
    @classmethod
    def get_fp8_weight(cls, p):
        p_key = p if isinstance(p, torch.Tensor) else p[0]
        if cls.fp8_weight_map.get(p_key, None) is not None:
            if cls.fp8_weight_map[p_key]['first'] or cls.fp8_weight_map[p_key]['first'] is None:
                cls.clear_tensor_data(*cls.fp8_weight_map[p_key]['data'])
                cls.fp8_weight_map[p_key]['data'] = per_block_cast_to_fp8(p, output_transpose=True)
                cls.fp8_weight_map[p_key]['first'] = False
        else:
            cls.fp8_weight_map[p_key] = {}
            cls.fp8_weight_map[p_key]['data'] = per_block_cast_to_fp8(p, output_transpose=True)
            cls.fp8_weight_map[p_key]['first'] = False

        return cls.fp8_weight_map[p_key]['data']


    @classmethod
    def set_first_micro_batch(cls, p, flag):
        p_key = p if isinstance(p, torch.Tensor) else p[0]
        if cls.fp8_weight_map.get(p_key , None) is not None:
            cls.fp8_weight_map[p_key]["first"] = flag
        else:
            cls.fp8_weight_map[p_key] = {}
            cls.fp8_weight_map[p_key]['first'] = flag
            cls.fp8_weight_map[p_key]['data'] = [None]

    @classmethod
    def wait_send(cls):
        if len(cls.send_ops) == 0:
            return
        reqs = dist.batch_isend_irecv(cls.send_ops)
        for req in reqs:
            req.wait()
        torch.cuda.synchronize()
        cls.send_ops = []

    @classmethod
    def wait_recv(cls):
        if len(cls.recv_ops) == 0:
            return
        reqs = dist.batch_isend_irecv(cls.recv_ops)
        for req in reqs:
            req.wait()
        torch.cuda.synchronize()
        cls.recv_ops = []

    @classmethod
    def wait(cls):
        # tmp = cls.recv_ops + cls.send_ops
        if len(cls.recv_ops + cls.send_ops) == 0:
            return
        reqs = dist.batch_isend_irecv(cls.send_ops + cls.recv_ops)
        for req in reqs:
            req.wait()
        torch.cuda.synchronize()
        cls.recv_ops = []
        cls.send_ops = []

    @classmethod
    def clear_tensor_data(cls, *tensors: Tuple[Optional[torch.Tensor], ...]) -> None:
        """
        Trick to deallocate tensor memory when delete operation does not
        release the tensor due to PyTorch override.

        Must be used carefully.
        """

        for t in tensors:
            if t is not None:
                t.data = torch.Tensor()
                del t

    @classmethod
    def clear(cls):
        for i in [cls.wgrad_func, cls.fp8_weight_map, cls.send_ops, cls.recv_ops]:
            i.clear()