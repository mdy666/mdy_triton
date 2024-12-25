import torch
import triton
import triton.language as tl

class VALUES_INDICES:
    def __init__(self, values, indices):
        self.values = values
        self.indices = indices

@triton.jit
def _max_short(INPUT, VALUES, INDICES,
          stride0, stride1,
          M, N: tl.constexpr, 
          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    
    start_m = tl.program_id(0)
    input_offset = start_m * BLOCK_M * stride0
    input_ptrs = INPUT + input_offset +  tl.arange(0, BLOCK_M)[:, None] * stride0 + tl.arange(0, BLOCK_N)[None, :]
    mask_row = (start_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]) < M
    mask_col = tl.arange(0, BLOCK_N) < N
    mask = mask_row & mask_col
    inp = tl.load(input_ptrs, mask=mask, other=float('-inf'))
    max_num, index = tl.max(inp, -1, return_indices=True)
    output_ptrs = VALUES + start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    tl.store(output_ptrs, max_num, mask=(start_m * BLOCK_M + tl.arange(0, BLOCK_M)) < M)
    indices_ptrs = INDICES + start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    tl.store(indices_ptrs, index, mask=(start_m * BLOCK_M + tl.arange(0, BLOCK_M)) < M)

@triton.jit
def _max_long(INPUT, VALUES, INDICES,
          stride0, stride1,
          M, N: tl.constexpr, 
          BLOCK_N: tl.constexpr):
    start_m = tl.program_id(0)
    base_ptrs = INPUT + start_m * stride0

    INPUT_ptrs = base_ptrs + tl.arange(0, BLOCK_N)
    mask = tl.arange(0, BLOCK_N) < N
    inp = tl.load(INPUT_ptrs, mask=mask, other=float('-inf'))
    max_num, indices = tl.max(inp, 0, return_indices=True)  

    for start_n in range(BLOCK_N, N, BLOCK_N):
        INPUT_ptrs = base_ptrs + start_n  + tl.arange(0, BLOCK_N)
        mask = (start_n + tl.arange(0, BLOCK_N)) < N
        inp = tl.load(INPUT_ptrs, mask=mask,  other=float('-inf'))
        new_max_num, new_indices = tl.max(inp, 0, return_indices=True)  
        if new_max_num > max_num:
            max_num = new_max_num
            indices = start_n + new_indices

    tl.store(VALUES + start_m, max_num)
    tl.store(INDICES + start_m, indices)

def triton_max(tensor, axis=-1):
    tensor = torch.movedim(tensor, axis, -1)
    tensor_shape = tensor.shape
    tensor = tensor.reshape(-1, tensor_shape[-1])
    B,D = tensor.shape
    values = torch.empty(B, device=tensor.device, dtype=tensor.dtype)
    indices = torch.empty(B, device=tensor.device, dtype=torch.int64)
    if D <=256:
        tmp = triton.next_power_of_2(B)
        BLOCK_M= min(256, tmp)
        BLOCK_N=triton.next_power_of_2(D)
        grid = lambda meta: (triton.cdiv(B, meta['BLOCK_M']),)
        _max_short[grid](tensor, values, indices,
                tensor.stride(0),tensor.stride(1),
                    B,D,
                    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                    num_stages=4, num_warps=8
        )
    else:
        BLOCK_N = min(triton.next_power_of_2(D), 64*1024)
        _max_long[(B,)](tensor, values, indices, 
                tensor.stride(0),tensor.stride(1),
                    B,D,
                    BLOCK_N=BLOCK_N,
                    num_stages=4, num_warps=8
        )
    return VALUES_INDICES(values.reshape(*tensor_shape[:-1]),indices.reshape(*tensor_shape[:-1]))

@triton.jit
def _min_short(INPUT, VALUES, INDICES,
          stride0, stride1,
          M, N: tl.constexpr, 
          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    
    start_m = tl.program_id(0)
    input_offset = start_m * BLOCK_M * stride0
    input_ptrs = INPUT + input_offset +  tl.arange(0, BLOCK_M)[:, None] * stride0 + tl.arange(0, BLOCK_N)[None, :]
    mask_row = (start_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]) < M
    mask_col = tl.arange(0, BLOCK_N) < N
    mask = mask_row & mask_col
    inp = tl.load(input_ptrs, mask=mask, other=float('inf'))
    min_num, index = tl.min(inp, -1, return_indices=True)
    output_ptrs = VALUES + start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    tl.store(output_ptrs, min_num, mask=(start_m * BLOCK_M + tl.arange(0, BLOCK_M)) < M)
    indices_ptrs = INDICES + start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    tl.store(indices_ptrs, index, mask=(start_m * BLOCK_M + tl.arange(0, BLOCK_M)) < M)

@triton.jit
def _min_long(INPUT, VALUES, INDICES,
          stride0, stride1,
          M, N: tl.constexpr, 
          BLOCK_N: tl.constexpr):
    start_m = tl.program_id(0)
    base_ptrs = INPUT + start_m * stride0


    INPUT_ptrs = base_ptrs + tl.arange(0, BLOCK_N)
    mask = tl.arange(0, BLOCK_N) < N
    inp = tl.load(INPUT_ptrs, mask=mask, other=float('inf'))
    min_num, indices = tl.min(inp, 0, return_indices=True)  

    for start_n in range(BLOCK_N, N, BLOCK_N):
        INPUT_ptrs = base_ptrs + start_n  + tl.arange(0, BLOCK_N)
        mask = (start_n + tl.arange(0, BLOCK_N)) < N
        inp = tl.load(INPUT_ptrs, mask=mask, other=float('inf'))
        new_min_num, new_indices = tl.min(inp, 0, return_indices=True)  
        if new_min_num < min_num:
            min_num = new_min_num
            indices = start_n + new_indices

    tl.store(VALUES + start_m, min_num)
    tl.store(INDICES + start_m, indices)

def triton_min(tensor, axis=-1):
    tensor = torch.movedim(tensor, axis, -1)
    tensor_shape = tensor.shape
    tensor = tensor.reshape(-1, tensor_shape[-1])
    B,D = tensor.shape
    values = torch.empty(B, device=tensor.device, dtype=tensor.dtype)
    indices = torch.empty(B, device=tensor.device, dtype=torch.int64)
    if D <=256:
        tmp = triton.next_power_of_2(B)
        BLOCK_M= min(256, tmp)
        BLOCK_N=triton.next_power_of_2(D)
        grid = lambda meta: (triton.cdiv(B, meta['BLOCK_M']),)
        _min_short[grid](tensor, values, indices,
                tensor.stride(0),tensor.stride(1),
                    B,D,
                    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                    num_stages=4, num_warps=8
        )
    else:
        BLOCK_N = min(triton.next_power_of_2(D), 64*1024)
        _min_long[(B,)](tensor, values, indices, 
                tensor.stride(0),tensor.stride(1),
                    B,D,
                    BLOCK_N=BLOCK_N,
                    num_stages=4, num_warps=8
        )
    return VALUES_INDICES(values.reshape(*tensor_shape[:-1]),indices.reshape(*tensor_shape[:-1]))