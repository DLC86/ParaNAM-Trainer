# Credits and acknowledgements

ParaNAM Trainer is a derivative of existing open-source NAM projects. The credits
below identify the origin of the work; they do not replace the original licenses
or imply endorsement by the original authors.

## Original NAM and parametric work

| Author / project | Contribution used by this application |
| --- | --- |
| **[Phillip Self](https://github.com/phillipmself)** — [neural-amp-modeler-parametric](https://github.com/phillipmself/neural-amp-modeler-parametric) | Original parametric trainer, hypernetwork/conditioning implementation, parametric model families, capture/planning and active-learning foundations. ParaNAM was created from DLC86's `Knobs+Switch` derivative of this project. |
| **Phillip Self** — [NamParametricPlugin](https://github.com/phillipmself/NamParametricPlugin) and [NeuralAmpModelerCoreParametric](https://github.com/phillipmself/NeuralAmpModelerCoreParametric) | Original plugin and parametric inference implementation underlying the native playback runtime. |
| **[Steven Atkinson](https://github.com/sdatkinson)** and NAM contributors — [Neural Amp Modeler](https://github.com/sdatkinson/neural-amp-modeler), [NeuralAmpModelerCore](https://github.com/sdatkinson/NeuralAmpModelerCore), [AudioDSPTools](https://github.com/sdatkinson/AudioDSPTools) | Original NAM training/model infrastructure, static NAM/A2 implementation, inference core and resampling infrastructure. Original copyright/license notices remain in [LICENSE](LICENSE) and the pinned dependencies. |
| **[TONE3000](https://github.com/tone-3000)** and NAM/A2 contributors | Published A2 training references and bundled standard capture sweep; web-player IRs and DI preview recordings, as itemized below. |
| **[DLC86](https://github.com/DLC86)** | ParaNAM desktop wizard, packaging, HyperA2 extensions, knobs/switches workflow, calibration, training queue and Compare integration. The native runtime is built from the [DLC86 plugin fork](https://github.com/DLC86/NamParametricPlugin). |

The exact playback-runtime revision is pinned by the Git submodule
`external/playback-runtime`. Its licenses and third-party notices are bundled
with the application. Upstream commit history retains additional contributor
attribution; this summary does not replace it.

## Training loss and numerical libraries

**Christian J. Steinmetz and Joshua D. Reiss** created
[auraloss](https://github.com/csteinmetz1/auraloss), used for audio-focused loss
functions including MRSTFT. The vendored copy in `nam/_dependencies/auraloss`
includes adaptations from the
[Atkinson Advanced Modeling fork](https://github.com/Atkinson-Advanced-Modeling/auraloss).
Its complete Apache-2.0 license is retained in
[nam/_dependencies/auraloss/LICENSE](nam/_dependencies/auraloss/LICENSE).

Reference: Steinmetz, Christian J. and Reiss, Joshua D. (2020),
*auraloss: Audio focused loss functions in PyTorch*, DMRN+15.

The native engine also uses **Eigen's authors and contributors** (MPL-2.0 subset)
and **Niels Lohmann and contributors**' nlohmann/json (MIT). The pinned source
trees retain their notices; the installer includes their license texts.

AudioDSPTools' resampling code also derives from the **iPlug 2 developers**,
**Oli Larkin**, **John Schwartz / Cockos**, **Cockos WDL contributors**, and
**Paul Walker (baconpaul) / Surge Synth Team**'s Lanczos resampler. Their original
notices, including the Lanczos file's specific MIT/BSD-context permission, are
reproduced in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Runtime and tools

Thanks to the authors and maintainers of these projects and their dependencies:

| Project | Role / upstream |
| --- | --- |
| Python / Python Software Foundation | Bundled interpreter — [python.org](https://www.python.org/) |
| PySide6 / Qt Project and The Qt Company | Desktop interface — [Qt for Python](https://doc.qt.io/qtforpython/) |
| PyTorch and its contributors | Training and tensor operations — [pytorch.org](https://pytorch.org/) |
| Lightning AI and contributors | PyTorch Lightning, Lightning Fabric and TorchMetrics — [Lightning](https://github.com/Lightning-AI/pytorch-lightning), [TorchMetrics](https://github.com/Lightning-AI/torchmetrics) |
| NumPy, SciPy, librosa and Matplotlib contributors | Numerical, signal-processing, audio-loading and plotting support — [NumPy](https://numpy.org/), [SciPy](https://scipy.org/), [librosa](https://librosa.org/), [Matplotlib](https://matplotlib.org/) |
| Matthias Geier and sounddevice contributors; PortAudio authors | Audio-device access and streaming — [sounddevice](https://github.com/spatialaudio/python-sounddevice), [PortAudio](https://github.com/PortAudio/portaudio) |
| SoundFile and libsndfile contributors; Warren Weckesser and wavio contributors | WAV/audio file I/O — [SoundFile](https://github.com/bastibe/python-soundfile), [libsndfile](https://github.com/libsndfile/libsndfile), [wavio](https://github.com/WarrenWeckesser/wavio) |
| Pydantic, filelock, psutil, tqdm and TensorBoard contributors | Validation, process/file handling and training support — [Pydantic](https://github.com/pydantic/pydantic), [filelock](https://github.com/tox-dev/filelock), [psutil](https://github.com/giampaolo/psutil), [tqdm](https://github.com/tqdm/tqdm), [TensorBoard](https://github.com/tensorflow/tensorboard) |
| PyInstaller, Inno Setup, CMake and platform packaging-tool contributors | Application and installer build tools — [PyInstaller](https://pyinstaller.org/), [Inno Setup](https://jrsoftware.org/isinfo.php), [CMake](https://cmake.org/) |
| NVIDIA and PyTorch contributors | Optional GPU runtime, redistributed through the official wheels with their own notices and terms. |

Each component retains its own license; ParaNAM's MIT license does not relicense
its dependencies, fonts, graphics or recordings.

## Recordings, fonts and branding

- **TONE3000** supplies `T3K-sweep-v3.wav`. The fixed `validation.wav` is distributed
  through **Phillip Self's parametric trainer**. Exact download sources and hashes
  are recorded in [paranam/assets/manifest.json](paranam/assets/manifest.json).
- **TONE3000's web-player contributors** supply the six IRs and eight DI preview
  recordings. Their original MIT license, exact upstream revision, URLs and hashes
  are retained in [the IR collection](paranam/assets/irs/manifest.json) and
  [the preview collection](paranam/assets/previews/manifest.json).
- **The Tone Scientist** supplies the TTS logo. The logo and NAM On Steroids styling/calibration references
  come from [NAM On Steroids / NAM Oversampler](https://github.com/DLC86/NAM-Oversampler).
  Branding retains its owner's rights.
- **TTS input v11** is a user-provided capture signal, included at the repository
  owner's request. Its exact bytes and checksum are recorded in the capture asset
  manifest; the recording retains its owner's rights.
- **The Inter Project Authors** — [Inter](https://github.com/rsms/inter),
  [SIL Open Font License 1.1](paranam/assets/Inter-OFL.txt).
- **The Michroma Project Authors** — [Michroma](https://github.com/googlefonts/Michroma-font),
  [SIL Open Font License 1.1](paranam/assets/Michroma-OFL.txt).

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for distribution notices and
the original source/license locations.
