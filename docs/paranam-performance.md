# HyperA2 training throughput investigation

Measured 2026-11-01 on NVIDIA GeForce RTX 3070 Laptop GPU, 8 GiB,
PyTorch 2.10.0+cu128, Windows. The reported user case was an RTX 4060 with
identical captures and input signals; these measurements use a different GPU.

## Controlled comparison

The HyperWaveNet reference configuration was read from
[Phillip Self's capture exporter](https://github.com/phillipmself/neural-amp-modeler-parametric/blob/main/nam/capture/export.py).
Its audio template and HyperA2 Full both have 23 layers and receptive field 6347.
HyperA2 additionally trains an independent 3-channel Lite branch.

Each measured step included the actual parametric training loss (including MRSTFT),
forward/backward, and an AdamW update. Two knob parameters, uniform capture-grouped
batches, 8192 target samples, float32, cudnn benchmark enabled, compilation disabled
for BOTH architectures. Five warmup steps, then 20 synchronized measured steps.
Input tensors were already on the GPU. This excludes file I/O, host transfers,
Lightning overhead, validation, checkpoint/export costs and compilation warmup.

| Architecture | Batch | ms/step | Windows/s | Peak allocated VRAM (GiB) |
| --- | ---: | ---: | ---: | ---: |
| HyperWaveNet | 32 | 77.6 | 412.3 | 0.982 |
| HyperWaveNet | 4 | 36.2 | 110.6 | 0.139 |
| HyperA2 Full+Lite | 4 | 74.6 | 53.7 | 0.189 |
| HyperA2 Full+Lite | 16 | 80.3 | 199.2 | 0.720 |
| HyperA2 Full+Lite | 32 | 118.8 | 269.4 | 1.374 |

At batch 32, the extra branch costs approximately 1.53x; changing HyperA2 from
batch 4 to 32 improves throughput approximately 5.02x. Comparing the reference
batch 32 against ParaNAM's former default 4 produced a 7.68x difference here.
This can explain a large slowdown without an architecture defect; it does not
prove the exact cause or predict timings for the user's RTX 4060.

Larger batches also mean fewer optimizer updates per epoch and can change
convergence. The new default matches the reference recipe; existing projects
are not silently altered. A resumed job can change physical batch size only
while keeping model and capture identity fixed, and restarts at a safe epoch
boundary. Reduce batch size if memory is insufficient.

The reference configuration enables torch.compile, whereas the self-contained
app keeps it disabled to avoid requiring a local compiler/toolkit. The benchmark
does not attribute any benefit to compilation, since both sides ran eager.

Reproduce from a developer environment with a CUDA PyTorch installation:

```sh
python packaging/benchmark_training.py --output benchmark.json
```

Use `--gpu-runtime` to select an already installed ParaNAM GPU engine.
