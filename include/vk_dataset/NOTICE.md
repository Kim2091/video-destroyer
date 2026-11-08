# Third-party notices for vk_dataset

## video-destroyer

The HDR tonemap and the post-processing (tiling, blank, motion and
completeness filters) are ported from video-destroyer by Kim2091
(https://github.com/Kim2091/video-destroyer), which is licensed under the GNU
General Public License, version 3. Video Destroyer is licensed under the same
licence and has the same author, so the ported code is distributed under GPL-3.0.

## Lens Blur kernels (Real-ESRGAN, paper only)

`nodes/optics.py` builds the gaussian, generalized and plateau kernels from the
formulas in Wang et al., "Real-ESRGAN: Training Real-World Blind
Super-Resolution with Pure Synthetic Data" (ICCVW 2021), section 3.1. The code
was written from the paper; no Real-ESRGAN or BasicSR code is included.

## VapourSynth plugins, installed from PyPI, not copied

The Bad Telecine Fix node calls bwdif (vapoursynth-bwdif, LGPL-3.0-or-later),
the Player Deblock node calls Deblock (vapoursynth-deblock, GPL-2.0-or-later)
and the Player DNR node calls FluxSmoothST from zsmooth (vapoursynth-zsmooth, MIT). The
Lens Blur and grain nodes use OpenCV (opencv-python-headless, Apache-2.0).

## ffmpeg

The tonemap curves follow the formulas of ffmpeg's tonemap filter; no ffmpeg
source code is included. The tone curves are the
published ones by John Hable, Reinhard et al. and the Mobius operator, and
the bt2390 curve follows ITU-R BT.2408.

## VSR-Dataset-Creator (Apache License 2.0)

Pair mode (`nodes/pair_align.py`, `nodes/pair_select.py`, `nodes/pair_dedup.py`,
the 'detail' tiles of Cut into Tiles in `nodes/post.py`, and the examples
`presets/06-two-releases-to-aligned-pairs.json` and
`presets/14-real-pairs-extra-damage.json`) follows the "Extract Sequences"
workflow of VSR-Dataset-Creator: its output layout, defaults, sequence
selection, crop selection, motion weighting, alignment score and alignment
gates. The code was rewritten for VapourSynth; `laplacian_complexity_map`,
`select_tiles`, `motion_weights` and `alignment_score` in `pair_select.py` are
modified ports of its `core/tiling.py`, `core/part1.py` and `core/spatial.py`,
which `post.py` calls without copying, and the ConvNeXt-S network in
`pair_dedup.py` is a modified port of its `archs/convnext.py`.
Used with the author's permission, under the Apache License 2.0
(http://www.apache.org/licenses/LICENSE-2.0).

## vs_align (MIT), installed from PyPI, not copied

Copyright (c) 2024, pifroggi. Pair mode calls `vs_align.temporal` and
`vs_align.spatial`. vs_align carries its own notices for the parts below.

- XFeat, by Guilherme Potje, Felipe Cadar, Andre Araujo, Renato Martins and
  Erickson R. Nascimento: Apache License 2.0.
- RIFE (Practical-RIFE / ECCV2022-RIFE, `flownet_v4.14.pkl`), copyright (c)
  Megvii Inc.: MIT License.
- TOPIQ / IQA-PyTorch, by Chaofeng Chen: CC BY-NC-SA 4.0. Pair mode uses it
  only when Align Time is set to precision 3, and says so in that setting.

## PepeDP (MIT)

Copyright (c) 2024 umzi. Crop selection (best paired tile) and perceptual
deduplication follow PepeDP, through VSR-Dataset-Creator.

## ConvNeXt (MIT)

Copyright (c) Meta Platforms, Inc. and affiliates. The ConvNeXt-S feature
extractor in `nodes/pair_dedup.py` is based on Meta's ConvNeXt. Its weights
(`convnextS_1kpretrained_official_style.pth`) are not part of Video Destroyer; the
dedup node reads them from the path it is given.

MIT License text (applies to vs_align, RIFE, PepeDP and ConvNeXt, each with
its own copyright line above):

> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

## HR Process, Denoise Dataset and sequence renaming

These parts follow **VSR-Dataset-Creator**, licensed under the Apache License 2.0
(https://www.apache.org/licenses/LICENSE-2.0), used with the author's permission.
Ported and changed from it: the Match Gamma estimator (`match_gamma.py`, itself a
port of chaiNNer's Match Gamma), the static-run detector and sigma-clipped stack
(`nodes/static.py`), the dataset folder rules (`dataset_folders.py`) and the
sequence renaming (`rename.py`). The files were rewritten for VapourSynth and
changed from the originals, as Apache-2.0 section 4(b) asks to be stated.

## ntsc-simulator (Apache-2.0), as a git submodule

The analogue video nodes (`nodes/ntsc_chain.py`: Composite Encode, VHS Tape,
Playback Deck, Composite Decode) run ntsc-simulator by Kim2091, the author
of Video Destroyer, and Joey Ballentine (https://github.com/Kim2091/ntsc-simulator),
through the VapourSynth plugin in `native/vs-ntsc`. The simulator is a git
submodule at `native/ntsc-simulator`, used unmodified; the plugin is built
into `plugins/vsntsc.dll`. ntsc-simulator is licensed under the Apache
License 2.0; its licence and NOTICE ship next to the plugin as
`plugins/ntsc-simulator-LICENSE.txt` and `plugins/ntsc-simulator-NOTICE.txt`.
Its dependencies are permissively licensed (MIT, Apache-2.0 and similar);
their list and licence texts ship next to the plugin as
`plugins/ntsc-simulator-THIRD_PARTY_LICENSES.md`, and the plugin's own two
crates (vapoursynth-sys, MIT OR Apache-2.0; libloading, ISC) as
`plugins/vsntsc-THIRD_PARTY_NOTICES.md`. vapoursynth-sys is generated from
VapourSynth's API header, which is LGPL-2.1-or-later.

## Filter templates (`filters/`)

The `.vkfilter` templates come from Vapourkit (GPL-3.0, the app this one is
built on) unchanged. Many call the libraries below; none of their code is
copied into the templates.

## Libraries used, not copied

- **vs-jetpack** (vsrgtools, vsdehalo, vstools): MIT, Copyright (c) 2020 - 2025
  Jaded-Encoding-Thaumaturgy. Fast Line Darken and Fine Dehalo call it. The
  upstream algorithms descend from havsfunc (Unlicense).
- **vs-mlrt**: GPL-3.0. The vs-mlrt plugin wheels come from PyPI. The app ships
  its own copy of `vsmlrt.py` (`include/vsmlrt.py`), from AmusementClub/vs-mlrt
  at commit 1404bba, modified so that the plugins' `Version()` and
  `DeviceProperties()` strings read the same on vs-mlrt 16 and 17; the file
  header states the change.
- **onnxruntime**: MIT, used by AniCC.
- **Filter template libraries**, installed by setup, not shipped: vsjetpack's
  extras and the plugin wheels they pull, pifroggi's vs_temporalfix and
  vs_colorfix (Apache-2.0), vs_grain and vs_tiletools (MIT), vs_undistort and
  vs_deepdeinterlace (no licence file; pip fetches them from PyPI and GitHub on
  the user's machine), and Selur's VapoursynthScriptsInHybrid (no licence file,
  mixed third-party content; downloaded at install, pinned to a commit).
  PLUGINS.md at the repository root lists each with its licence.

## Models

No model weights are bundled or redistributed; nodes take a path to an ONNX
file, and `scripts/export_dataset_models.py` converts checkpoints the user has.

- GaterV3 is based on Umzi's PepeDP (https://github.com/umzi2/PepeDP), MIT
  license as declared on its repository.
- SCUNet (https://github.com/cszn/SCUNet) is Apache License 2.0. The Prism
  de-H264 models by Bubblemint864 are SCUNet variants whose licence is not
  stated; they are not shipped, and anyone redistributing them or their ONNX
  exports must check the terms with the author first.
- The exporter's network definitions (GaterV3, BGCC v4, Prism SCUNet and
  their helpers) are copied unchanged from VSR-Dataset-Creator into
  `scripts/dataset_archs/`, under the Apache License 2.0. They are used only
  by the exporter and are not shipped with the app.
