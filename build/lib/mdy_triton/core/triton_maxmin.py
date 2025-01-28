import torch
import triton
import triton.language as tl

class VALUES_INDICES:
    def __init__(self, values, indices, func_type):
        self.values = values
        self.indices = indices
        self.func_type = func_type

    def __str__(self):
        return f"func_type={self.func_type}\nvalues={self.values}\nindices={self.indices})"
    
    def __repr__(self):
        return f"func_type={self.func_type}\nvalues={self.values}\nindices={self.indices})"

@triton.jit
def _max_short(INPUT, VALUES, INDICES,
          stride0, stride1,
          M, N: tl.constexpr, 
          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    
    start_m = tl.program_id(0)
    row_offset = start_m * BLOCK_M
    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    input_ptrs = INPUT + row_offset * stride0 +  rows[:, None] * stride0 + cols[None, :] * stride1
    mask_row = (row_offset + rows) < M
    mask_col = cols < N
    mask = mask_row[:, None] & mask_col[None, :]
    inp = tl.load(input_ptrs, mask=mask, other=float('-inf'))
    max_num, index = tl.max(inp, -1, return_indices=True)
    output_ptrs = VALUES + row_offset + rows
    tl.store(output_ptrs, max_num, mask=mask_row)
    indices_ptrs = INDICES + row_offset + rows
    tl.store(indices_ptrs, index, mask=mask_row)

@triton.jit
def _max_long(INPUT, VALUES, INDICES,
          stride0, stride1,
          M, N: tl.constexpr, 
          BLOCK_N: tl.constexpr):
    start_m = tl.program_id(0)
    base_ptrs = INPUT + start_m * stride0

    cols = tl.arange(0, BLOCK_N)
    INPUT_ptrs = base_ptrs + cols * stride1
    mask = tl.arange(0, BLOCK_N) < N
    inp = tl.load(INPUT_ptrs, mask=mask)
    max_num, indices = tl.max(inp, 0, return_indices=True)  

    for start_n in range(BLOCK_N, N, BLOCK_N):
        cols += start_n 
        INPUT_ptrs = base_ptrs + cols * stride1
        mask = cols < N
        inp = tl.load(INPUT_ptrs, mask=mask)
        new_max_num, new_indices = tl.max(inp, 0, return_indices=True)  
        if new_max_num > max_num:
            max_num = new_max_num
            indices = start_n + new_indices
        

    tl.store(VALUES + start_m, max_num)
    tl.store(INDICES + start_m, indices)

def triton_max(tensor, axis=-1, keepdim=False):
    tensor = torch.movedim(tensor, axis, -1)
    tensor_shape = tensor.shape
    tensor = tensor.reshape(-1, tensor_shape[-1])
    # print(tensor.stride())
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
        BLOCK_N = min(triton.next_power_of_2(D), 2048)
        _max_long[(B,)](tensor, values, indices, 
                tensor.stride(0),tensor.stride(1),
                    B,D,
                    BLOCK_N=BLOCK_N,
                    num_stages=4, num_warps=8
        )
    values = values.reshape(*tensor_shape[:-1])
    indices = indices.reshape(*tensor_shape[:-1])
    if keepdim:
        values.unsqueeze_(axis)
    return VALUES_INDICES(values=values, indices=indices, func_type='triton_max')

@triton.jit
def _min_short(INPUT, VALUES, INDICES,
          stride0, stride1,
          M, N: tl.constexpr, 
          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    
    start_m = tl.program_id(0)
    row_offset = start_m * BLOCK_M
    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    input_ptrs = INPUT + row_offset * stride0 +  rows[:, None] * stride0 + cols[None, :] * stride1
    mask_row = (row_offset + rows) < M
    mask_col = cols < N
    mask = mask_row[:, None] & mask_col[None, :]
    inp = tl.load(input_ptrs, mask=mask, other=float('inf'))
    max_num, index = tl.min(inp, -1, return_indices=True)
    output_ptrs = VALUES + row_offset + rows
    tl.store(output_ptrs, max_num, mask=mask_row)
    indices_ptrs = INDICES + row_offset + rows
    tl.store(indices_ptrs, index, mask=mask_row)

@triton.jit
def _min_long(INPUT, VALUES, INDICES,
          stride0, stride1,
          M, N: tl.constexpr, 
          BLOCK_N: tl.constexpr):
    start_m = tl.program_id(0)
    base_ptrs = INPUT + start_m * stride0


    cols = tl.arange(0, BLOCK_N)
    INPUT_ptrs = base_ptrs + cols * stride1
    mask = tl.arange(0, BLOCK_N) < N
    inp = tl.load(INPUT_ptrs, mask=mask)
    min_num, indices = tl.min(inp, 0, return_indices=True)  

    for start_n in range(BLOCK_N, N, BLOCK_N):
        cols += start_n 
        INPUT_ptrs = base_ptrs + cols * stride1
        mask = cols < N
        inp = tl.load(INPUT_ptrs, mask=mask)
        new_min_num, new_indices = tl.min(inp, 0, return_indices=True)  
        if new_min_num < min_num:
            min_num = new_min_num
            indices = start_n + new_indices
        

    tl.store(VALUES + start_m, min_num)
    tl.store(INDICES + start_m, indices)

def triton_min(tensor, axis=-1, keepdim=False):
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
        BLOCK_N = min(triton.next_power_of_2(D), 2048)
        _min_long[(B,)](tensor, values, indices, 
                tensor.stride(0),tensor.stride(1),
                    B,D,
                    BLOCK_N=BLOCK_N,
                    num_stages=4, num_warps=8
        )
    values = values.reshape(*tensor_shape[:-1])
    indices = indices.reshape(*tensor_shape[:-1])
    if keepdim:
        values.unsqueeze_(axis)
    return VALUES_INDICES(values=values, indices=indices, func_type='triton_min')