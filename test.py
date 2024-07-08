import torch
import triton
from super_monotonic_align import maximum_path as maximum_path_trion
from cython_monotonic_align import maximum_path as maximum_path_cython
from jit_monotonic_align import maximum_path1 as maximum_path_jit_v1
from jit_monotonic_align import maximum_path2 as maximum_path_jit_v2


def identical_test(B,T,S):
    value = torch.randn((B, T, S), dtype=torch.float32, device='cuda') * 0.01
    attn_mask = torch.ones((B, T, S), dtype=torch.int32, device='cuda')
    path_c = maximum_path_cython(value, attn_mask)
    path_jit1 = maximum_path_jit_v1(value, attn_mask)
    path_jit2 = maximum_path_jit_v2(value, attn_mask)
    path_tri = maximum_path_trion(value, attn_mask)

    # not 100% equal due to precision issue
    assert torch.allclose(path_c, path_tri, atol=1e-2, rtol=0), f"Failed on shape=({B,T,S})\n{path_c}\n{path_tri}\ndiff:{(path_c-path_tri).abs().sum()}"
    assert torch.allclose(path_c, path_jit1, atol=1e-2, rtol=0), f"Failed on shape=({B,T,S})\n{path_c}\n{path_jit1}\ndiff:{(path_c-path_jit1).abs().sum()}"
    assert torch.allclose(path_c, path_jit2, atol=1e-2, rtol=0), f"Failed on shape=({B,T,S})\n{path_c}\n{path_jit2}\ndiff:{(path_c-path_jit2).abs().sum()}"

# benchmark
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['T'],
        x_vals=[128 * i for i in range(1, 17)],
        line_arg='provider',
        line_vals=['triton', 'cython', 'jit_v1', 'jit_v2'],
        line_names=['Triton', 'Cython', 'JIT_v1', 'JIT_v2'],
        styles=[('blue', '-'), ('green', '-'), ('red', '-'), ('orange', '-')],
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
            return maximum_path_trion(value, attn_mask)  # noqa: F811, E704

    if provider == 'cython':

        def y_fwd():
            return maximum_path_cython(value, attn_mask)  # noqa: F811, E704
    
    if provider == 'jit_v1':
            
        def y_fwd():
            return maximum_path_jit_v1(value, attn_mask)
        
    if provider == 'jit_v2':

        def y_fwd():
            return maximum_path_jit_v2(value, attn_mask)
        
    ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles, rep=500)

    return (ms), (max_ms), (min_ms)

if __name__ == "__main__":
    for (b,t,s) in [(32, 16, 16), (32, 128, 512), (32, 256, 1024), (32, 511, 2048)]:
        identical_test(b,t,s)
        print(f"Passed on shape=({b},{t},{s})")
    bench_mas.run(save_path='.', print_data=True)
