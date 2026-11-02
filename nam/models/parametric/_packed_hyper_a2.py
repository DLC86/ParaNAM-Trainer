"""Differentiable Full/Lite weight packing for HyperA2 training.

Only the execution layout changes. The cache owns no trained parameters and is
not registered in the model's state dict, optimizer, or exported PNAM weights.
"""

import torch
from torch.nn import functional as F
from torch.func import functional_call

from ..wavenet import PackedWaveNet
from ..wavenet._packed_conv import PackedConv1dBase


class PackedHyperA2Execution:
    def __init__(self, model):
        # Constructing the execution shell must not change training RNG state.
        with torch.random.fork_rng(devices=[]):
            self.shell = PackedWaveNet.init_from_config(
                dict(
                    submodels=[
                        dict(
                            name=size,
                            config=model.branch(size)._template.export_config(),
                        )
                        for size in model.sizes
                    ]
                )
            )
        self.shell.requires_grad_(False)
        self.plan = []
        for name, module in self.shell.named_modules():
            if isinstance(module, PackedConv1dBase):
                self.plan.append((name, module.shared_input, module.bias is not None))

    def __call__(self, model, x, params, pad_start):
        p = model._encode_params(params)
        if p.ndim == 2:
            if p.shape[0] != x.shape[0]:
                raise ValueError("Audio and parameter batch sizes differ")
            for branch in model._branches.values():
                branch._check_uniform_batch_params(p)
            p = p[0]
        branches = [
            model.branch(size)._conditioned_weight_dict(p) for size in model.sizes
        ]
        weights = {}
        for name, shared_input, has_bias in self.plan:
            key = name + ".weight"
            source_key = key.removeprefix("_net.")
            lite, full = (branch[source_key] for branch in branches)
            if shared_input:
                weights[key] = torch.cat((lite, full), dim=0)
            else:
                weights[key] = torch.cat(
                    (
                        F.pad(lite, (0, 0, 0, full.shape[1])),
                        F.pad(full, (0, 0, lite.shape[1], 0)),
                    ),
                    dim=0,
                )
            if has_bias:
                key = name + ".bias"
                weights[key] = torch.cat(
                    [branch[key.removeprefix("_net.")] for branch in branches]
                )
        shell_parameter = next(self.shell.parameters())
        if shell_parameter.device != x.device or shell_parameter.dtype != x.dtype:
            self.shell.to(device=x.device, dtype=x.dtype)
        if self.shell.training != model.training:
            self.shell.train(model.training)
        return functional_call(
            self.shell, weights, (x,), dict(pad_start=pad_start), strict=False
        )
