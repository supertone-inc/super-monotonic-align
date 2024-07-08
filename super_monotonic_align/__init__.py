import numpy as np
import torch
import triton
from super_monotonic_align.core import maximum_path_triton

@torch.no_grad()
def maximum_path(value, mask, dtype=torch.float32):
    """ Triton optimized version.
    value: [b, t_x, t_y]
    mask: [b, t_x, t_y]
    skip_mask: [b, t_x]
    """
    # Use masked_fill_ to avoid new tensor creation
    value = value.masked_fill_(mask.logical_not(), 0)
    path = torch.zeros_like(value, dtype=dtype)
    t_x_max = mask.sum(1)[:, 0].to(torch.int32) 
    t_y_max = mask.sum(2)[:, 0].to(torch.int32)
    path = maximum_path_triton(path, value, t_x_max, t_y_max)
    return path
    
def identical_test(B,T,S):
    from cython_monotonic_align import maximum_path as maximum_path_cython

    value = torch.randn((B, T, S), dtype=torch.float32, device='cuda')
    attn_mask = torch.ones((B, T, S), dtype=torch.int32, device='cuda')
    path_c = maximum_path_cython(value, attn_mask)
    path_tri = maximum_path(value, attn_mask)
    # not 100% equal due to precision issue
    assert torch.allclose(path_c, path_tri, atol=1e-2, rtol=0), f"Failed on shape=({B,T,S}), {path_c} {path_tri} diff:{(path_c-path_tri).abs().sum()}"

# benchmark
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['T'],
        x_vals=[128 * i for i in range(1, 17)],
        line_arg='provider',
        line_vals=['triton', 'cython'],
        line_names=['Triton', 'Cython'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='ms',
        plot_name='MAS in ms',
        y_log=True,
        args={'B': 16},
    ))
def bench_mas(B, T, provider, device='cuda'):
    from cython_monotonic_align import maximum_path as maximum_path_cython
    # create data
    quantiles = [0.5, 0.2, 0.8]

    S = 4*T
    value = torch.randn((B, T, S), dtype=torch.float32, device=device)
    attn_mask = torch.ones((B, T, S), dtype=torch.int32, device=device)
 
    # utility functions
    if provider == 'triton':

        def y_fwd():
            return maximum_path(value, attn_mask)  # noqa: F811, E704

    if provider == 'cython':

        def y_fwd():
            return maximum_path_cython(value, attn_mask)  # noqa: F811, E704
    ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles, rep=500)

    return (ms), (max_ms), (min_ms)

if __name__ == "__main__":
    for (b,t,s) in [(32, 16, 16), (32, 128, 512), (32, 256, 1024), (32, 511, 2048)]:
        identical_test(b,t,s)
    bench_mas.run(save_path='.', print_data=True)
