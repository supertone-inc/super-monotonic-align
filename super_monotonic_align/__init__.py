import torch
from super_monotonic_align.core import maximum_path_triton

@torch.no_grad()
def maximum_path(value, mask, dtype=torch.float32):
    """ Triton optimized version.
    value: [b, t_x, t_y]
    mask: [b, t_x, t_y]
    skip_mask: [b, t_x]
    """
    # check value is contiguous
    value = value.contiguous()
    # Use masked_fill_ to avoid new tensor creation
    value = value.masked_fill_(mask.logical_not(), 0)
    path = torch.zeros_like(value, dtype=dtype)
    t_x_max = mask.sum(1)[:, 0].to(torch.int32) 
    t_y_max = mask.sum(2)[:, 0].to(torch.int32)
    path = maximum_path_triton(path, value, t_x_max, t_y_max)
    return path