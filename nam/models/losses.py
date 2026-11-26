# File: losses.py
# Created Date: Sunday January 22nd 2023
# Author: Steven Atkinson (steven@atkinson.mn)

"""
Loss functions
"""

from math import isfinite as _isfinite
from typing import Optional as _Optional

import torch as _torch

from .._dependencies.auraloss.freq import (
    MultiResolutionSTFTLoss as _MultiResolutionSTFTLoss,
)

DEFAULT_SEGMENT_ESR_MIN_RMS_DBFS = -58.0
SEGMENT_ESR_RMS_RANGE_DBFS = (-120.0, -1.0)


def waveform_loss_settings(config):
    """Resolve the parametric waveform objective; old configs use MSE."""
    loss = config.get("waveform_loss", "mse")
    if loss not in ("mse", "segment_esr", "dual_file_esr"):
        raise ValueError("Waveform loss must be mse, segment_esr or dual_file_esr")
    threshold = config.get("segment_esr_min_rms_dbfs", DEFAULT_SEGMENT_ESR_MIN_RMS_DBFS)
    low, high = SEGMENT_ESR_RMS_RANGE_DBFS
    if loss == "segment_esr" and (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not _isfinite(threshold)
        or not low <= threshold <= high
    ):
        raise ValueError(f"Segment ESR minimum target RMS must be between {low:g} and {high:g} dBFS")
    if loss != "segment_esr":
        threshold = DEFAULT_SEGMENT_ESR_MIN_RMS_DBFS
    return loss, float(threshold)


def apply_pre_emphasis_filter(x: _torch.Tensor, coef: float) -> _torch.Tensor:
    """
    Apply first-order pre-emphsis filter

    :param x: (*, L)
    :param coef: The coefficient

    :return: (*, L-1)
    """
    return x[..., 1:] - coef * x[..., :-1]


def dual_file_esr_settings(config):
    pre = float(config.get("dual_pre_emphasis", 0.85))
    raw = float(config.get("dual_raw_weight", 1.0))
    floor = float(config.get("dual_floor_db", 30.0))
    if not (0 <= pre <= 1 and _isfinite(raw) and raw >= 0 and 0 < floor < 200):
        raise ValueError("Invalid dual file ESR parameters")
    return pre, raw, floor


def dual_file_energies(signals, synthetic_indices, pre=0.85, floor_db=30.0):
    """V4 paired raw/pre-emphasized energy floor; synthetic mutes are excluded."""
    pairs = []
    for i, y in enumerate(signals):
        if i in synthetic_indices:
            pairs.append(None)
            continue
        y = y.detach().double()
        raw = y.square().mean().item()
        emphasized = apply_pre_emphasis_filter(y, pre).square().mean().item()
        if not (_isfinite(raw) and raw > 0 and _isfinite(emphasized) and emphasized > 0):
            raise ValueError("Dual file ESR requires positive real capture energy; mark silent captures as Mute")
        pairs.append((raw, emphasized))
    raw_values = sorted(pair[0] for pair in pairs if pair is not None)
    if not raw_values:
        raise ValueError("Dual file ESR needs a real training capture")
    floor = raw_values[(len(raw_values) - 1) // 2] * 10.0 ** (-floor_db / 10.0)
    return tuple(None if pair is None else tuple(
        e * max(1.0, floor / pair[0]) for e in pair
    ) for pair in pairs)


def dual_file_esr(preds, targets, energies, pre=0.85, raw_weight=1.0):
    """Capture-grouped equivalent of Faverooo's V4 waveform objective."""
    raw, emphasized = energies
    error = preds.float() - targets.float()
    return (apply_pre_emphasis_filter(error, pre).square().mean() / emphasized
            + raw_weight * error.square().mean() / raw)


def esr(preds: _torch.Tensor, targets: _torch.Tensor) -> _torch.Tensor:
    """
    ESR of (a batch of) predictions & targets

    :param preds: (N,) or (B,N)
    :param targets: Same as preds
    :return: ()
    """
    if preds.ndim == 1 and targets.ndim == 1:
        preds, targets = preds[None], targets[None]
    if preds.ndim != 2:
        raise ValueError(
            f"Expect 2D predictions (batch_size, num_samples). Got {preds.shape}"
        )
    if targets.ndim != 2:
        raise ValueError(
            f"Expect 2D targets (batch_size, num_samples). Got {targets.shape}"
        )
    return _torch.mean(
        _torch.mean(_torch.square(preds - targets), dim=1)
        / _torch.mean(_torch.square(targets), dim=1)
    )


def segment_esr(preds: _torch.Tensor, targets: _torch.Tensor, min_power: float) -> _torch.Tensor:
    """Mean of per-window error/target energies above a target RMS threshold.

    `min_power` is the squared minimum target RMS in full-scale units.
    A batch without eligible windows has a differentiable zero waveform loss.
    """
    if preds.shape != targets.shape or preds.ndim not in (1, 2):
        raise ValueError("Segment ESR requires matching (samples,) or (batch, samples) tensors")
    if preds.ndim == 1:
        preds, targets = preds[None], targets[None]
    # Accumulate in at least FP32, even under mixed precision.
    dtype = _torch.promote_types(_torch.float32, _torch.promote_types(preds.dtype, targets.dtype))
    preds, targets = preds.to(dtype), targets.to(dtype)
    energy = targets.square().sum(dim=-1)
    eligible = energy >= min_power * targets.shape[-1]
    error = _torch.where(eligible[:, None], preds - targets, 0.0)
    denominator = _torch.where(eligible, energy + 1e-12, 1.0)
    return (error.square().sum(dim=-1) / denominator).sum() / eligible.sum().clamp_min(1)


def multi_resolution_stft_loss(
    preds: _torch.Tensor,
    targets: _torch.Tensor,
    loss_func: _Optional[_MultiResolutionSTFTLoss] = None,
    device: _Optional[_torch.device] = None,
) -> _torch.Tensor:
    """
    Experimental Multi Resolution Short Time Fourier Transform Loss using auraloss implementation.
    B: Batch size
    L: Sequence length

    :param preds: (B,L)
    :param targets: (B,L)
    :param loss_func: A pre-initialized instance of the loss function module. Providing
        this saves time.
    :param device: If provided, send the preds and targets to the provided device.
    :return: ()
    """

    def ensure_shape(z: _torch.Tensor) -> _torch.Tensor:
        """
        Required for auraloss v0.4

        :param z: (L,) or (B,L)
        :return: (B,C,L)
        """
        if z.ndim == 1:
            return z[None, None, :]
        elif z.ndim == 2:
            return z[:, None, :]
        else:
            assert z.ndim == 3, f"Expected 1D or 2D tensor. Got {z.shape}"
            return z

    loss_func = _MultiResolutionSTFTLoss() if loss_func is None else loss_func
    if device is not None:
        preds, targets = [z.to(device) for z in (preds, targets)]
    preds, targets = [ensure_shape(z) for z in (preds, targets)]
    return loss_func(preds, targets)


def mse(preds: _torch.Tensor, targets: _torch.Tensor) -> _torch.Tensor:
    """
    MSE loss
    """
    return _torch.nn.functional.mse_loss(preds, targets)


def mse_fft(preds: _torch.Tensor, targets: _torch.Tensor) -> _torch.Tensor:
    """
    Fourier loss

    :param preds: (N,) or (B,N)
    :param targets: Same as preds
    :return: ()
    """
    fp = _torch.fft.fft(preds)
    ft = _torch.fft.fft(targets)
    e = fp - ft
    return _torch.mean(_torch.square(e.abs()))
