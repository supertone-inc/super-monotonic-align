# Super-Monotonic-Alignment-Search

This repo contains [Triton-Lang](https://github.com/triton-lang/triton) and PyTorch implementation of the monotonic alignment search (MAS), originally from [Glow-TTS](https://arxiv.org/abs/2005.11129).
MAS is an effective algorithm for estimating the alignment between paired speech and text in a self-supervised manner.


The authors of Glow-TTS noted:
> "The time complexity of the algorithm is O(T_{text} × T_{mel}). Even though the algorithm is difficult to parallelize, it runs efficiently on CPU without the need for GPU executions. In our experiments, it spends less than 20 ms on each iteration, which amounts to less than 2% of the total training time. Furthermore, we do not need MAS during inference, as the duration predictor is used to estimate the alignment."

However, we found two things.
1. MAS can be parallelized in the text-length dimension.
2. CPU execution consumes an inordinate amount of time for large inputs due to the need to copy large tensors between CPU and GPU.

Therefore, we implemented a Triton kernel to accelerate MAS without inter-device copy.

# Requirments
1. PyTorch (tested with version `torch==2.3.0+cu121`)
2. Triton-Lang (tested with version `triton==2.3.0`)

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
value = torch.randn((B, T, S), dtype=torch.float32, device='cuda')
attn_mask = torch.ones((B, T, S), dtype=torch.int32, device='cuda')
# path: [B,T,S], you can specify dtype of path
path = maximum_path(value, attn_mask, dtype=dtype)
```

# Benchmark
```
MAS in ms:
         T      Triton       Cython
0    128.0    0.413696     8.216721
1    256.0    1.649664    37.776081
2    384.0    3.086336    93.494621
3    512.0    4.545536   218.105865
4    640.0    6.973440   311.186676
5    768.0    9.482240   473.866119
6    896.0   12.257792   617.156738
7   1024.0   27.352560   909.698120
8   1152.0   34.011139  1013.993713
9   1280.0   40.676353  1279.761597
10  1408.0   47.934464  1516.422241
11  1536.0   58.279938  1960.699097
12  1664.0   68.456451  2163.526855
13  1792.0   80.745468  2560.196045
14  1920.0   94.303230  2795.177002
15  2048.0  272.310272  5583.550781
```
![](./assets/MAS.png)


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
