# Super-Monotonic-Alignment-Search

This repo contains [Triton-Lang](https://github.com/triton-lang/triton) and PyTorch implementation of the monotonic alignment search (MAS), originally from [Glow-TTS](https://arxiv.org/abs/2005.11129).
MAS is an effective algorithm for estimating the alignment between paired speech and text in a self-supervised manner.


The authors of Glow-TTS noted:
> "The time complexity of the algorithm is O(T_{text} × T_{mel}). Even though the algorithm is difficult to parallelize, it runs efficiently on CPU without the need for GPU executions. In our experiments, it spends less than 20 ms on each iteration, which amounts to less than 2% of the total training time. Furthermore, we do not need MAS during inference, as the duration predictor is used to estimate the alignment."

However, we found two things.
1. MAS can be parallelized in the text-length dimension.
2. CPU execution consumes an inordinate amount of time for large inputs due to the need to copy large tensors between CPU and GPU.

Therefore, we implemented a Triton kernel to accelerate MAS on GPU without inter-device copy.

# Requirments
1. PyTorch (tested with version `torch==2.3.0+cu121`)
2. Triton-Lang (tested with version `triton==2.3.0`)
3. Cython (optional for bench, tested with version `Cython== 0.29.36`)

Please ensure you have these packages installed to run the code in this repository, as version checks are not enforced.

# How to use
1. Install super-monotonic-align
```
cd super_monotonic_align; pip install -e ./
```
or
```
pip install git+https://github.com/supertone-inc/super-monotonic-align.git
```
2. Import `super_monotonic_align` and use it!
```python
from super_monotonic_align import maximum_path
...
# You need to know value's value is modified by triton kernel.
# If you want to keep value without changing, you need to clone it before maximum_path.
value = torch.randn((B, T, S), dtype=torch.float32, device='cuda')
attn_mask = torch.ones((B, T, S), dtype=torch.int32, device='cuda')
# path: [B,T,S]
path = maximum_path(value, attn_mask)
```

# Benchmark
```
MAS in ms:
         T      Triton       Cython
0    128.0    0.498688     8.835520
1    256.0    1.615872    42.645790
2    384.0    3.428352   138.791458
3    512.0    5.874688   305.674438
4    640.0    9.065472   448.026978
5    768.0   12.222464   499.826355
6    896.0   15.326208   613.331116
7   1024.0   19.742207  1348.791382
8   1152.0   33.110016  1425.197388
9   1280.0   39.672832  1879.560547
10  1408.0   47.270401  2160.256836
11  1536.0   59.043327  2851.383789
12  1664.0   72.378372  2873.137939
13  1792.0   82.171906  3634.574219
14  1920.0   98.664452  3959.088379
15  2048.0  107.194878  7263.244629
```

The Triton MAS implementation is at least 17 times faster and up to 67 times faster than the Cython implementation.
| ms in linear scale | ms in log scale |
|----------|----------|
| ![Image 1](./assets/MAS.png) | ![Image 2](./assets/MAS_log.png) |

## How to run benchmark
```bash
cd cython_monotonic_align; python setup.py build_ext --inplace
cd ../super_monotonic_align; pip install -e ./
python __init__.py
```

# References
This implementation uses code from following repositories:
- [Official Glow-TTS Implementation](https://github.com/jaywalnut310/glow-tts)
- [Triton-Lang](https://github.com/triton-lang/triton)


# Authors
- Junhyeok Lee ([jlee843@jhu.edu](mailto:jlee843@jhu.edu))
- Hyoungju Kim([hyeongju@supertone.ai](mailto:hyeongju@supertone.ai))


Feel free to create an issue if you encounter any problems or have any questions.

Additionally, [Supertone](https://supertone.ai) is hiring TTS researchers. 
If you are interested, please check out our career opportunities!
