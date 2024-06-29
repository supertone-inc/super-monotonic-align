import torch

import triton
import triton.language as tl

@triton.jit
def maximum_path(
    path, value, t_x, t_y,
    t_x_stride, t_y_stride,
    path_stride_b, path_stride_x, path_stride_y,
    value_stride_b, value_stride_x, value_stride_y,
    BLOCK_SIZE_X: tl.constexpr
    ):
    batch = tl.program_id(axis=0)
    path += batch * path_stride_b
    t_x += batch * t_x_stride
    x_length = tl.load(t_x)
    index = tl.load(t_x)-1
    t_y += batch * t_y_stride   
    y_length = tl.load(t_y)
    value += batch * value_stride_b
    offs_prev = tl.arange(0, BLOCK_SIZE_X)
    for j in range(2, y_length+1, 1):
        x_start = 0 #tl.maximum(0, x_length + j - y_length-1) # for backup
        x_end = x_length #tl.minimum(x_length, j) # for backup
        v_cur= tl.load(value + (offs_prev+1)* value_stride_x + (j-1)*value_stride_y, mask=(offs_prev < x_end) & (offs_prev >= x_start), other=-1e9)
        v_prev =tl.load(value + offs_prev* value_stride_x + (j-1)*value_stride_y, mask=(offs_prev < x_end) & (offs_prev >= x_start), other=-1e9)
        # compare v_cur and v_prev, and update v with larger value
        v = tl.maximum(v_cur, v_prev) + tl.load(value + (offs_prev+1)* value_stride_x + j*value_stride_y, mask=(offs_prev < x_end) & (offs_prev >= x_start))
        tl.store(value + (offs_prev+1)* value_stride_x + j*value_stride_y, v, mask=(offs_prev < x_end) & (offs_prev >= x_start))


    for j in range(y_length-1,-1,-1):
        tl.store(path + (index)*path_stride_x + (j)*path_stride_y, 1)
        if (index > 0):
            if (index>1):
                v_left = tl.load(value+ (index+1) * value_stride_x+ j*value_stride_y) # remember that value is padded
                v_leftdown =  tl.load(value+(index) * value_stride_x + j*value_stride_y) 
                if (v_left <= v_leftdown):
                    index += - 1

            #elif  #((index== j-skipped) or 
            elif (tl.load(value+ (index+1) * value_stride_x+ j*value_stride_y)<=tl.load(value+(index) * value_stride_x + j*value_stride_y)):
            # don't know, why but <= (not <) has identical output with original Cython implementation.
                index += - 1
            
            
                        
@torch.no_grad()
def maximum_path_triton(path, value, t_x, t_y, max_neg_val=-1e9):
    B,T,S = path.shape
    value = torch.nn.functional.pad(value, (1, 0, 1, 0, 0, 0), value=max_neg_val) #[B, T+1, S+1]
    value[:, 2:, 1] = max_neg_val
    value[:, 0, 0] = 0
    BLOCK_SIZE_X = triton.next_power_of_2(T+1)
    num_warps = 1 # Need to be 1 to prevent wrong output by slicing the operation
    with torch.cuda.device(value.device.index):
        maximum_path[(B, )](
            path, value, t_x, t_y, 
            t_x.stride(0), t_y.stride(0),
            path.stride(0), path.stride(1), path.stride(2),
            value.stride(0), value.stride(1), value.stride(2),
            num_warps = num_warps,
            BLOCK_SIZE_X = BLOCK_SIZE_X)
    return path

