#!/usr/bin/env python3
"""Upgrade the existing single-network sample-accurate delta convolutions to GEMM.

Applied AFTER apply-direct-kernel-smoothing.py on the pinned core. No new
hypernetwork instance, no crossfade and no worker thread. The base kernels,
100 ms update cadence and 100 ms linear per-sample alpha remain unchanged.
"""
from pathlib import Path
import sys

core = Path(sys.argv[1] if len(sys.argv) > 1 else "external/nam_core_parametric") / "NAM"
marker = "GEMM delta kernel smoothing installed"
if marker in (core / "conv1d.cpp").read_text():
    if marker not in (core / "dsp.cpp").read_text():
        raise RuntimeError("Partially installed GEMM delta smoothing")
    print("GEMM delta smoothing already installed", core)
    sys.exit(0)


def replace(path, before, after):
    target = core / path
    old = target.read_text()
    n = old.count(before)
    if n != 1:
        raise RuntimeError(f"{path}: expected one anchor, got {n}: {before[:90]!r}")
    target.write_text(old.replace(before, after))


replace("dsp.h",
        "  Eigen::MatrixXf _output;\n  bool _do_bias;",
        """  Eigen::MatrixXf _output;
  Eigen::MatrixXf _ramp_output; // Allocated on Reset, only used during weight ramps.
  bool _do_bias;""")

replace("dsp.cpp",
        """void nam::Conv1x1::SetMaxBufferSize(const int maxBufferSize)
{
  _output.resize(get_out_channels(), maxBufferSize);
}""",
        """void nam::Conv1x1::SetMaxBufferSize(const int maxBufferSize)
{
  _output.resize(get_out_channels(), maxBufferSize);
  _ramp_output.resize(get_out_channels(), maxBufferSize);
}""")

p = core / "dsp.cpp"
content = p.read_text()
start = content.index("  // Existing fast/fused base convolution stays intact. Add only the delta.")
end = content.index("\n  }\n", start) + len("\n  }\n")
content = content[:start] + """  // GEMM delta kernel smoothing installed: y = W*x + alpha[f] * (dW*x).
  // Allocate scratch at Reset, not in Process. No extra work when idle.
  const float* alpha = kernel_weight_smoothing::block_alphas;
  if (alpha)
  {
    if (_ramp_weight_changed)
    {
      if (_is_depthwise)
      {
        for (int f = 0; f < num_frames; ++f)
          for (int o = 0; o < get_out_channels(); ++o)
            _output(o, f) += alpha[f] * _ramp_depthwise_weight(o) * input(o, f);
      }
      else
      {
        _ramp_output.leftCols(num_frames).noalias() =
            _ramp_weight * input.leftCols(num_frames);
        for (int f = 0; f < num_frames; ++f)
          _output.col(f).noalias() += alpha[f] * _ramp_output.col(f);
      }
    }
    if (_ramp_bias_changed)
      for (int f = 0; f < num_frames; ++f)
        _output.col(f).noalias() += alpha[f] * _ramp_bias;
  }
""" + content[end:]
p.write_text(content)

replace("conv1d.h",
        "  Eigen::MatrixXf _output; // Pre-allocated output buffer (out_channels x maxBufferSize)",
        """  Eigen::MatrixXf _output; // Pre-allocated output buffer (out_channels x maxBufferSize)
  Eigen::MatrixXf _ramp_output; // Preallocated block GEMM scratch for delta kernels""")

replace("conv1d.cpp",
        """  _output.resize(out_channels, maxBufferSize);
  _output.setZero();""",
        """  _output.resize(out_channels, maxBufferSize);
  _output.setZero();
  _ramp_output.resize(out_channels, maxBufferSize);""")

p = core / "conv1d.cpp"
content = p.read_text()
start = content.index("  // Evaluate a second *convolution delta* only for conditioned kernels;")
end = content.index("  if (alpha && _ramp_bias_changed)", start)
content = content[:start] + """  // GEMM delta kernel smoothing installed. Linear alpha is still per-sample;
  // multiply delta matrices by contiguous input frames, then scale each column
  // by its individual alpha. Unchanged kernels skip every extra multiplication.
  const float* alpha = kernel_weight_smoothing::block_alphas;
  if (alpha && _ramp_weight_changed)
  {
    const int taps = static_cast<int>(get_kernel_size());
    if (_is_depthwise)
    {
      const int out_ch = static_cast<int>(get_out_channels());
      for (int k = 0; k < taps; ++k)
      {
        const long lookback = this->_dilation * (taps - 1 - k);
        const auto history = _input_buffer.Read(num_frames, lookback);
        for (int f = 0; f < num_frames; ++f)
          for (int o = 0; o < out_ch; ++o)
            _output(o, f) += alpha[f] * _ramp_depthwise_weight[k](o) * history(o, f);
      }
    }
    else
    {
      _ramp_output.leftCols(num_frames).setZero();
      for (int k = 0; k < taps; ++k)
      {
        const long lookback = this->_dilation * (taps - 1 - k);
        const auto history = _input_buffer.Read(num_frames, lookback);
        _ramp_output.leftCols(num_frames).noalias() += _ramp_weight[k] * history;
      }
      for (int f = 0; f < num_frames; ++f)
        _output.col(f).noalias() += alpha[f] * _ramp_output.col(f);
    }
  }
""" + content[end:]
p.write_text(content)

print("GEMM delta-kernel smoothing installed (single DSP, no extra thread)", core)
