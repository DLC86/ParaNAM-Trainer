"""Configure reproducibility before loading the optional CUDA runtime."""

import json
import os
from pathlib import Path


def prepare_worker_environment(job):
    # Keep this module torch-free: the GPU probe can create a cuBLAS handle.
    spec = json.loads((Path(job) / "spec.json").read_text(encoding="utf-8"))
    if spec.get("learning", {}).get("trainer", {}).get("deterministic") is True:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def apply_backend_flags(spec):
    import torch

    config = spec["learning"]["trainer"]
    deterministic = config.get("deterministic") is True
    benchmark = bool(config.get("benchmark", True)) and not deterministic
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark
    print(
        f"Training backend: cuDNN benchmark={benchmark}; deterministic algorithms={deterministic}",
        flush=True,
    )
