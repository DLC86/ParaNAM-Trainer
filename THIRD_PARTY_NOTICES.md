# Third-party software and capture signals

ParaNAM Trainer includes the MIT-licensed Neural Amp Modeler code by Steven
Atkinson and the parametric capture/model code by Phillip Self and contributors.
The original MIT license remains in LICENSE. The source baseline is DLC86's
neural-amp-modeler-parametric branch Knobs+Switch, commit cb442001e78655b1cd3470031b66a8d7bf212424.

The installer includes Python (PSF license), PySide6/Qt (LGPLv3/GPLv3 or commercial
terms), PyTorch (BSD-style license), PyTorch Lightning (Apache 2.0), NumPy and SciPy
(BSD), PortAudio (MIT-style), sounddevice (MIT), and their dependencies. Their
license notices and shared libraries are distributed with the application bundle.
The one-folder distribution leaves Qt shared libraries replaceable.

Capture assets are the unmodified TONE3000 sweep T3K-sweep-v3.wav and the
validation.wav asset published with the upstream parametric trainer's
training-inputs-v1 release. Source URLs, sample counts and SHA-256 hashes are
recorded in paranam/assets/manifest.json. They are bundled for offline capture,
credited to their respective authors, and are not synthetic substitutes.

Optional NVIDIA acceleration downloads the redistributable runtime libraries supplied
by official PyTorch/NVIDIA wheels and retains their complete license notices. No CUDA toolkit or local
compiler is used. An NVIDIA driver compatible with that wheel is provided by the
operating system/user's normal graphics driver installation, not this application.


The Inter typeface is bundled under the SIL Open Font License 1.1; its full license
is in paranam/assets/Inter-OFL.txt.

The Michroma title typeface is bundled under the SIL Open Font
License 1.1; its full license is in paranam/assets/Michroma-OFL.txt.

The TTS logo is included from NAM On Steroids and belongs to The Tone Scientist.

TTS input v11.wav is bundled at the repository owner's request. Its exact original
bytes and SHA-256 are recorded in paranam/assets/manifest.json.

The bundled native playback library uses the NAM Parametric Plugin DSP engine
at commit 03b423f336dbfb4f86308e65445f468e2e00ad46, its pinned parametric NAM Core,
AudioDSPTools and Eigen dependencies. Their license files are included in the
application's licenses/playback-* folders. The library does not link JUCE; the
desktop interface and audio streaming use Qt and PortAudio as before.


## Tone3000 preview impulse responses

The six WAV files in `paranam/assets/irs/` (Ampeg, Celestion, Eminence,
Mesa, Plate and Spring) are reproduced unchanged from the official TONE3000
web player, revision `ea8dc285462d6cedd982691b9f090cd03dac04a2`:
https://github.com/tone-3000/neural-amp-modeler-wasm/tree/ea8dc285462d6cedd982691b9f090cd03dac04a2/ui/public/irs

Copyright (c) 2026 TONE3000. MIT license; the complete upstream UI license
is included as `paranam/assets/irs/LICENSE.txt`. `manifest.json` records
source URLs and SHA-256 digests. IR normalization follows the upstream
`ui/src/audio/irGain.ts` left-channel, 64-frequency mean-power method.

## TONE3000 preview inputs

The eight bundled audition WAVs in `paranam/assets/previews` come from the
TONE3000 web preview player, under its MIT license. Original file names, pinned
source URLs and SHA-256 hashes are retained in that directory's `manifest.json`;
the complete upstream license is included as `LICENSE.txt` and bundled with the app.
