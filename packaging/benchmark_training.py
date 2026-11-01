"""Developer-only CUDA throughput comparison; does not play or record audio."""

import argparse
import copy
import gc
import json
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-runtime", action="store_true")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.gpu_runtime:
        from paranam.gpu_runtime import activate

        activate()
    import torch
    from nam.capture.project import CaptureProject, KnobModel
    from nam.capture.export import build_hyper_model_config, build_hyper_a2_model_config
    from nam.train.parametric import _ParametricLightningModule

    if not torch.cuda.is_available():
        raise RuntimeError(
            "A CUDA GPU engine is required; --gpu-runtime selects the installed ParaNAM engine"
        )
    torch.set_num_threads(8)
    torch.manual_seed(938)
    torch.backends.cudnn.benchmark = True
    project = CaptureProject(
        knobs=[
            KnobModel(name="Gain", min=0, max=10),
            KnobModel(name="Tone", min=0, max=10),
        ],
        sample_rate=48000,
    )
    results = []
    for architecture, build, sizes in (
        ("HyperWaveNet", build_hyper_model_config, (32, 4)),
        ("HyperA2", build_hyper_a2_model_config, (4, 16, 32)),
    ):
        for batch_size in sizes:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            model = _ParametricLightningModule.init_from_config(
                copy.deepcopy(build(project))
            ).cuda()
            model.net.set_uniform_batch_params(True)
            model.on_train_epoch_start()
            model.train()
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=0.002, weight_decay=3.17e-7
            )
            x = 0.1 * torch.randn(
                batch_size, model.net.receptive_field + 8191, device="cuda"
            )
            p = torch.tensor([[3.0, 7.0]], device="cuda").expand(batch_size, -1)
            y = 0.1 * torch.randn(batch_size, 8192, device="cuda")

            def step():
                optimizer.zero_grad(set_to_none=True)
                loss = model.training_step((x, p, y), 0)
                loss.backward()
                optimizer.step()

            for _ in range(5):
                step()
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(args.steps):
                step()
            torch.cuda.synchronize()
            duration = (time.perf_counter() - start) / args.steps
            result = dict(
                architecture=architecture,
                batch_size=batch_size,
                seconds_per_step=duration,
                windows_per_second=batch_size / duration,
                peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024**3,
            )
            print(json.dumps(result), flush=True)
            results.append(result)
            del model, optimizer, x, p, y
            gc.collect()
    report = dict(
        gpu=torch.cuda.get_device_name(), torch=torch.__version__, results=results
    )
    if args.output:
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    main()
