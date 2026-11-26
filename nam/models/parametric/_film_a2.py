
"""Jointly trained A2 widths with a shared control-to-affine conditioner."""

from copy import deepcopy
from itertools import groupby

import torch
from torch import nn
from torch.func import functional_call

from .._from_nam import convert_nam_wavenet_config
from ..wavenet import PackedWaveNet, WaveNet
from ..wavenet._packed_conv import PackedConv1dBase
from ..wavenet._wavenet import WaveNet as WaveNetCore
from ._hyper_a2 import SlimmableA2, _CHANNELS, _KERNELS, _template_config
from ._spec import ParamSpec


class FiLMA2(SlimmableA2):
    """The Lite convolutions, residuals and head are prefixes of Full."""

    pad_start_default = True

    def __init__(self, *, param_specs, conditioning="neural", head_scale=0.01,
                 size="full", sample_rate=None):
        super().__init__(param_specs=param_specs, sample_rate=sample_rate)
        if conditioning not in ("neural", "linear"):
            raise ValueError(f"Unknown FiLMA2 conditioning: {conditioning!r}")
        self._conditioning = conditioning
        self.set_size(size)
        self._template = self._make_core(8, head_scale)
        # The Lite template supplies structure only. Every parameter is replaced by
        # a differentiable view of Full for functional_call.
        self._lite = (self._make_core(3, head_scale),)
        self._layers = tuple(
            f"_layer_arrays.0._layers.{i}" for i in range(len(_KERNELS))
        )
        self._hidden_sizes = (16,) if conditioning == "neural" else ()
        width = 2 * len(_KERNELS) * _CHANNELS["full"]
        readout = nn.Linear(self._hidden_sizes[-1] if self._hidden_sizes else
                            self.encoded_param_dim, width)
        nn.init.zeros_(readout.weight)
        nn.init.zeros_(readout.bias)
        self._conditioner = (
            nn.Sequential(nn.Linear(self.encoded_param_dim, 16), nn.ReLU(), readout)
            if conditioning == "neural" else nn.Sequential(readout)
        )
        self._uniform_batch_params = False
        self._uniform_batch_params_checked = False

        # Cache module/attribute lookup rather than Tensor views: .to(), state
        # loading and deepcopy may replace Parameters.
        base = dict(self._template.named_parameters())
        self._parameter_plan = tuple(
            (name, self._template.get_submodule(name.rsplit(".", 1)[0]),
             name.rsplit(".", 1)[1])
            for name in base
        )
        # Batch identically shaped layer weights. Consecutive kernel runs avoid
        # building a device index tensor or issuing one multiply per layer.
        plan = []
        for suffix in ("_conv.weight", "_input_mixer.weight", "_conv.bias"):
            keys = tuple(f"{prefix}.{suffix}" for prefix in self._layers)
            start = 0
            for _, group in groupby(keys, key=lambda key: tuple(base[key].shape)):
                group = tuple(group)
                stop = start + len(group)
                plan.append((start, stop, group, suffix == "_conv.bias"))
                start = stop
        self._fold_plan = tuple(plan)
        self._packed_execution_cache = None

    @staticmethod
    def _make_core(channels, head_scale):
        return WaveNetCore.init_from_config(
            convert_nam_wavenet_config(_template_config(channels, head_scale))
        )

    @classmethod
    def parse_config(cls, config):
        config = super().parse_config(config)
        config["param_specs"] = tuple(ParamSpec.from_dict(p) for p in config.pop("params"))
        submodels = config.pop("submodels", None)
        if submodels is not None:
            expected = cls(**config)._export_submodels()
            # Per-width loudness can be added by calibration tools.
            layout = [{k: v for k, v in entry.items() if k != "metadata"}
                      for entry in submodels]
            if layout != expected:
                raise ValueError("FiLMA2 submodels do not match the container configuration")
        return config

    @property
    def receptive_field(self):
        return self._template.receptive_field

    def set_uniform_batch_params(self, uniform):
        self._uniform_batch_params = bool(uniform)
        self._uniform_batch_params_checked = False

    def _check_uniform_batch_params(self, encoded):
        if not self._uniform_batch_params_checked:
            if not torch.equal(encoded, encoded[:1].expand_as(encoded)):
                raise RuntimeError("FiLMA2 uniform batch promise was violated")
            self._uniform_batch_params_checked = True

    @staticmethod
    def _lite_prefix(value):
        value = value[:_CHANNELS["lite"]]
        return value[:, :_CHANNELS["lite"]] if value.ndim == 3 else value

    def _full_parameters(self):
        return {name: getattr(module, attr) for name, module, attr in self._parameter_plan}

    def _base_parameters(self, size):
        weights = self._full_parameters()
        if size == "lite":
            return {name: self._lite_prefix(value) for name, value in weights.items()}
        return weights

    def _fold_full(self, coeff):
        weights = self._full_parameters()
        gamma, beta = 1.0 + coeff[..., 0], coeff[..., 1]
        for start, stop, keys, bias in self._fold_plan:
            stacked = torch.stack([weights[key] for key in keys])
            g = gamma[start:stop]
            folded = (stacked * g + beta[start:stop] if bias
                      else stacked * g[..., None, None])
            weights.update(zip(keys, folded.unbind(0)))
        return weights

    def _fold(self, size, coeff):
        weights = self._fold_full(coeff)
        if size == "lite":
            return {name: self._lite_prefix(value) for name, value in weights.items()}
        return weights

    def _prepare_packed(self, x):
        cache = self._packed_execution_cache
        if cache is None:
            # Do not register a second set of trainable parameters or change RNG.
            cache = (_PackedFiLMA2Execution(self),)
            self._packed_execution_cache = cache
        execution = cache[0]
        execution.prepare(self, x)
        return execution

    @property
    def supports_compiled_step(self):
        return True

    def _compilable_step(self, x, p):
        # _prepare_packed is called outside the compiled region.
        return self._packed_execution_cache[0](self, x, p)

    def _coefficients(self, encoded):
        return self._conditioner(encoded).reshape(len(_KERNELS), 8, 2)

    def _evaluate(self, x, encoded, sizes):
        if encoded.ndim == 2 and x.shape[0] != encoded.shape[0]:
            raise ValueError("FiLMA2 requires one setting per audio row")
        if encoded.ndim == 1 or self._uniform_batch_params:
            if encoded.ndim == 2:
                self._check_uniform_batch_params(encoded)
                encoded = encoded[0]
            full = self._fold_full(self._coefficients(encoded))
            return torch.stack([
                functional_call(
                    self._core(size),
                    full if size == "full" else {
                        name: self._lite_prefix(value) for name, value in full.items()
                    },
                    (x[:, None],),
                )[:, 0]
                for size in sizes
            ], dim=1)
        unique, inverse = torch.unique(encoded, dim=0, return_inverse=True)
        outputs = None
        for index, setting in enumerate(unique):
            mask = inverse == index
            result = self._evaluate(x[mask].contiguous(), setting, sizes)
            if outputs is None:
                outputs = result.new_empty((x.shape[0], *result.shape[1:]))
            outputs[mask] = result
        if outputs is None:
            raise ValueError("FiLMA2 requires a nonempty audio batch")
        return outputs

    def _run_conditioned(self, x, p):
        return self._evaluate(x, p, (self.size,))[:, 0]

    def forward_sizes(self, x, params, pad_start=False):
        if pad_start:
            x = nn.functional.pad(x, (self.receptive_field - 1, 0))
        encoded = self._encode_params(params)
        if encoded.ndim == 2 and x.shape[0] != encoded.shape[0]:
            raise ValueError("FiLMA2 requires one setting per audio row")
        if self._uniform_batch_params:
            if encoded.ndim == 2:
                self._check_uniform_batch_params(encoded)
                encoded = encoded[0]
            self._prepare_packed(x)
            return self._run_step(x, encoded)
        return self._evaluate(x, encoded, self.sizes)

    def _core(self, size):
        return self._template if size == "full" else self._lite[0]

    def _get_export_architecture(self):
        return "ParametricSlimmableContainer"

    def _branch_config(self, size):
        config = self._core(size).export_config(sample_rate=self.sample_rate)
        config.update(
            film={"conditioning": self._conditioning},
            params=[spec.to_dict() for spec in self.param_specs],
        )
        return config

    def _state(self, size):
        channels = _CHANNELS[size]
        readout = self._conditioner[-1]
        width = 2 * len(_KERNELS) * channels
        return [
            *[p.flatten() for p in list(self._conditioner.parameters())[:-2]],
            readout.weight.reshape(len(_KERNELS), 8, 2, readout.in_features)
            [:, :channels].flatten(),
            readout.bias.reshape(len(_KERNELS), 8, 2)[:, :channels].flatten(),
            # Reserved zero anchor to use the same conditioner serializer as HyperWaveNet.
            readout.bias.new_zeros(width),
        ]

    def _export_submodels(self):
        offset = 0
        entries = []
        for size in self.sizes:
            base_count = len(self._core(size).export_weights())
            count = base_count + sum(p.numel() for p in self._state(size))
            entries.append({
                "name": size,
                "max_value": 0.5 if size == "lite" else 1.0,
                "architecture": "FiLMA2",
                "config": self._branch_config(size),
                "weight_offset": offset,
                "weight_count": count,
                "base_weight_count": base_count,
            })
            offset += count
        return entries

    def _export_inner_config(self):
        return {
            "size": self.size,
            "head_scale": self._template._head_scale,
            "conditioning": self._conditioning,
            "submodels": self._export_submodels(),
        }

    def _base_flat(self, size):
        weights = self._base_parameters(size)
        # Stock A2 uses module registration order for its convolution parameters.
        return torch.cat([
            *[p.flatten() for p in weights.values()],
            self.nominal_params.new_tensor([self._template._head_scale]),
        ])

    def _export_weights(self):
        return torch.cat([
            tensor for size in self.sizes
            for tensor in (self._base_flat(size), *self._state(size))
        ]).detach().cpu().numpy()

    @torch.no_grad()
    def import_weights(self, weights, i=0):
        weights = torch.as_tensor(weights, device=self.nominal_params.device,
                                  dtype=self.nominal_params.dtype)
        lite, full = self._export_submodels()
        count = lite["weight_count"] + full["weight_count"]
        if weights.ndim != 1 or i < 0 or len(weights) - i != count:
            raise ValueError(f"Expected {count} FiLMA2 weights after offset {i}")
        values = weights[i:]
        if not torch.isfinite(values).all():
            raise ValueError("FiLMA2 weights must be finite")
        start = lite["weight_count"]
        end = start + full["base_weight_count"]
        scale = values[end - 1]
        self._template.import_weights(values[start:end].cpu())
        self._lite[0]._head_scale = self._template._head_scale
        for param in self._conditioner.parameters():
            param.copy_(values[end:end + param.numel()].reshape(param.shape))
            end += param.numel()
        expected = torch.cat([self._base_flat("lite"), *self._state("lite")])
        # The export hook scales float32 tail values and JSON config floats separately.
        expected[lite["base_weight_count"] - 1] = scale
        if not torch.equal(values[:start], expected) or torch.count_nonzero(values[end:]):
            raise ValueError("FiLMA2 serialized shared weights are inconsistent")
        return i + count

    @torch.no_grad()
    def bake_size(self, size, params, *, output_scale=None):
        from ...data import Dataset
        from ._export import _normalize_output_scale, _resolve_raw_params

        params = _resolve_raw_params(self, params)
        core = deepcopy(self._core(size)).to(self.nominal_params)
        core._head_scale = self._template._head_scale
        conditioned = self._fold(size, self._coefficients(self._encode_params(params)))
        for name, parameter in core.named_parameters():
            parameter.copy_(conditioned[name])
        baked = WaveNet(wavenet=core, sample_rate=self.sample_rate).eval()
        scale = _normalize_output_scale(output_scale)
        if scale is not None:
            baked.export_model_dict_post_hooks.append(Dataset._ScaleOutputHook(scale=1 / scale))
        return baked

    def bake(self, params, *, output_scale=None):
        from ._hyper_a2 import _pack_baked_sizes

        return _pack_baked_sizes(self, [self.bake_size(size, params) for size in self.sizes],
                                 output_scale)


class _PackedFiLMA2Execution:
    """One packed A2 forward with differentiable shared Full/Lite weights."""

    def __init__(self, model):
        with torch.random.fork_rng(devices=[]):
            self.shell = PackedWaveNet.init_from_config({
                "submodels": [
                    {"name": size, "config": model._core(size).export_config()}
                    for size in model.sizes
                ]
            })
        self.shell.requires_grad_(False)
        base = model._full_parameters()
        groups = {}
        for name, module in self.shell.named_modules():
            if not isinstance(module, PackedConv1dBase):
                continue
            for suffix in ("weight", "bias"):
                if suffix == "bias" and module.bias is None:
                    continue
                key = name + "." + suffix
                source = key.removeprefix("_net.")
                value = base[source]
                lite = model._lite_prefix(value)
                group = (suffix, module.shared_input, tuple(value.shape), tuple(lite.shape))
                groups.setdefault(group, []).append((key, source))
        self.plan = tuple((suffix, shared, tuple(entries))
                          for (suffix, shared, _, _), entries in groups.items())

    def prepare(self, model, x):
        parameter = next(self.shell.parameters())
        if parameter.device != x.device or parameter.dtype != x.dtype:
            self.shell.to(device=x.device, dtype=x.dtype)
        if self.shell.training != model.training:
            self.shell.train(model.training)
        self.shell._net._head_scale = model._template._head_scale

    def __call__(self, model, x, encoded):
        full = model._fold_full(model._coefficients(encoded))
        weights = {}
        for suffix, shared, entries in self.plan:
            # Stack across layers of the same shape, then pack in a handful of
            # GPU operations rather than padding/concatenating every layer.
            wide = torch.stack([full[source] for _, source in entries])
            narrow = wide[:, :_CHANNELS["lite"]]
            if suffix == "weight":
                narrow = narrow[:, :, :_CHANNELS["lite"]]
                if not shared:
                    narrow, wide = (
                        nn.functional.pad(narrow, (0, 0, 0, wide.shape[2])),
                        nn.functional.pad(wide, (0, 0, narrow.shape[2], 0)),
                    )
            packed = torch.cat((narrow, wide), dim=1)
            weights.update(zip((key for key, _ in entries), packed.unbind(0)))
        return functional_call(self.shell, weights, (x,),
                               dict(pad_start=False), strict=False)
