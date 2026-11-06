# Third-party software and capture signals

## Original authors and project lineage

**Phillip Self** is the author of the original
[parametric trainer](https://github.com/phillipmself/neural-amp-modeler-parametric),
[NAM Parametric Plugin](https://github.com/phillipmself/NamParametricPlugin) and
[parametric inference fork](https://github.com/phillipmself/NeuralAmpModelerCoreParametric)
on which this application and its playback integration build. **Steven Atkinson**
and the NAM contributors created the underlying Neural Amp Modeler, NAM Core and
AudioDSPTools. DLC86 maintains this derivative application.

The original MIT license in [LICENSE](LICENSE) is preserved verbatim. Phillip
Self's original plugin MIT license is preserved in
`external/playback-runtime/LICENSE` and shipped under `licenses/playback-plugin`.
See [CREDITS.md](CREDITS.md) for component-specific contributions and sources.

## Training and runtime foundations

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


## auraloss

Christian J. Steinmetz and Joshua D. Reiss created the audio-focused loss functions
used by the vendored auraloss implementation, including MRSTFT. Sources:
https://github.com/csteinmetz1/auraloss and the adapted
https://github.com/Atkinson-Advanced-Modeling/auraloss.
The original Apache-2.0 license is retained at `nam/_dependencies/auraloss/LICENSE`
and is copied into the installer under `licenses/auraloss`.

Reference: Steinmetz, Christian J. and Reiss, Joshua D. (2020),
*auraloss: Audio focused loss functions in PyTorch*, DMRN+15.

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

## Resampling component license notices

The native playback runtime uses AudioDSPTools resampling components derived from
iPlug 2, Cockos WDL and Paul Walker / Surge Synth Team. These notices are retained
from the pinned `external/playback-runtime/external/AudioDSPTools` source tree.

### iPlug 2 / Lanczos resampler

````text
// File: LanczosResampler.h
// Created Date: Saturday December 16th 2023
// Author: Steven Atkinson (steven@atkinson.mn)


// This file originally came from the iPlug2 library and has been subsequently modified;
// the following license is copied as required from
// https://github.com/iPlug2/iPlug2/blob/40ebb560eba68f096221e99ef0ae826611fc2bda/LICENSE.txt
// -------------------------------------------------------------------------------------

/*
iPlug 2 C++ Plug-in Framework.

Copyright (C) the iPlug 2 Developers. Portions copyright other contributors, see each source file for more information.

Based on WDL-OL/iPlug by Oli Larkin (2011-2018), and the original iPlug v1 (2008) by John Schwartz / Cockos

LICENSE:

This software is provided 'as-is', without any express or implied warranty.  In no event will the authors be held liable
for any damages arising from the use of this software.

Permission is granted to anyone to use this software for any purpose, including commercial applications, and to alter it
and redistribute it freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If
you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not
required.
1. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original
software.
1. This notice may not be removed or altered from any source distribution.

iPlug 2 includes the following 3rd party libraries (see each license info):

* Cockos WDL https://www.cockos.com/wdl
* NanoVG https://github.com/memononen/nanovg
* NanoSVG https://github.com/memononen/nanosvg
* MetalNanoVG https://github.com/ollix/MetalNanoVG
* RTAudio https://www.music.mcgill.ca/~gary/rtaudio
* RTMidi https://www.music.mcgill.ca/~gary/rtmidi
*/
// -------------------------------------------------------------------------------------

/*
 This code is derived from
 https://github.com/surge-synthesizer/sst-basic-blocks/blob/main/include/sst/basic-blocks/dsp/LanczosResampler.h

 The following license info is copied from the above file:

 * sst-basic-blocks - an open source library of core audio utilities
 * built by Surge Synth Team.
 *
 * Provides a collection of tools useful on the audio thread for blocks,
 * modulation, etc... or useful for adapting code to multiple environments.
 *
 * Copyright 2023, various authors, as described in the GitHub
 * transaction log. Parts of this code are derived from similar
 * functions original in Surge or ShortCircuit.
 *
 * sst-basic-blocks is released under the GNU General Public Licence v3
 * or later (GPL-3.0-or-later). The license is found in the "LICENSE"
 * file in the root of this repository, or at
 * https://www.gnu.org/licenses/gpl-3.0.en.html
 *
 * All source in sst-basic-blocks available at
 * https://github.com/surge-synthesizer/sst-basic-blocks

 * A special note on licensing: This file (and only this file)
 * has Paul Walker (baconpaul) as the sole author to date.
 *
 * In order to make this handy small function based on public
 * information available to a set of open source projects
 * adapting hardware to software, but which are licensed under
 * MIT or BSD or similar licenses, this file and only this file
 * can be used in an MIT/BSD context as well as a GPL3 context, by
 * copying it and modifying it as you see fit.
 *
 * If you do that, you will need to replace the `sum_ps_to_float`
 * call below with either an hadd if you are SSE3 or higher or
 * an appropriate reduction operator from your toolkit.
 *
 * But basically: Need to resample 48k to variable rate with
 * a small window and want to use this? Go for it!
 *
 * For avoidance of doubt, this license exception only
 * applies to this file.
 */
````

### Cockos WDL — pointer list

````text
/*
    WDL - ptrlist.h
    Copyright (C) 2005 and later, Cockos Incorporated

    This software is provided 'as-is', without any express or implied
    warranty.  In no event will the authors be held liable for any damages
    arising from the use of this software.

    Permission is granted to anyone to use this software for any purpose,
    including commercial applications, and to alter it and redistribute it
    freely, subject to the following restrictions:

    1. The origin of this software must not be misrepresented; you must not
       claim that you wrote the original software. If you use this software
       in a product, an acknowledgment in the product documentation would be
       appreciated but is not required.
    2. Altered source versions must be plainly marked as such, and must not be
       misrepresented as being the original software.
    3. This notice may not be removed or altered from any source distribution.

*/
````

### Cockos WDL — heap buffer

````text
/*
    WDL - heapbuf.h
    Copyright (C) 2005 and later Cockos Incorporated

    This software is provided 'as-is', without any express or implied
    warranty.  In no event will the authors be held liable for any damages
    arising from the use of this software.

    Permission is granted to anyone to use this software for any purpose,
    including commercial applications, and to alter it and redistribute it
    freely, subject to the following restrictions:

    1. The origin of this software must not be misrepresented; you must not
       claim that you wrote the original software. If you use this software
       in a product, an acknowledgment in the product documentation would be
       appreciated but is not required.
    2. Altered source versions must be plainly marked as such, and must not be
       misrepresented as being the original software.
    3. This notice may not be removed or altered from any source distribution.

*/
````
