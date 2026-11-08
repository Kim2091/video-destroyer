# What setup installs

Everything is installed into the app's private Python on the user's machine. Only `include/plugins/plugins.7z`
(four DLLs) and `include/vsmlrt.py` ship inside the app; everything else is downloaded at install time.
"Templates" means the Filter Template node's catalog (`include/vk_dataset/filters/`).

## Core (first-run setup)

| Name | What it is for here | Licence | Source |
|---|---|---|---|
| Python 3.13 (embedded) | Runs the engine | PSF | python.org |
| VapourSynth R79 | The frame server every node uses | LGPL-2.1 | PyPI `vapoursynth==79` |
| BestSource | Frame-accurate video loading | MIT | PyPI |
| FFmpeg (Windows) | Codec round trips, ffmpeg Filter node | GPL-3.0 (full build) | gyan.dev download |

## Plugin phase

| Name | What it is for here | Licence | Source |
|---|---|---|---|
| PyTorch, torchvision | vs_align, near-duplicate removal, vs_deepdeinterlace, vs_undistort (PyTorch path) | BSD-3-Clause | PyTorch index (CUDA on NVIDIA, CPU elsewhere) |
| vsjetpack 2.2.0 `[full,nvidia]` / `[full,amd]` / `[full,cl,vulkan]` | Fine Dehalo, Line Darken; templates for denoise (BM3D, DFTTest, MVTools), deband, dehalo, AA, QTGMC, masks, descale | MIT (wrappers); the ~35 plugin wheels it pulls are MIT, LGPL, GPL-2/3 | PyPI + JET index |
| vs_temporalfix | Removes temporal flicker (templates) | Apache-2.0 | PyPI |
| vs_undistort | Heat haze / wobble removal (templates) | **none** | PyPI (NVIDIA with TensorRT deps; elsewhere `--no-deps`) |
| vs_grain | Film grain generation (CUDA) and overlay (FGrain template) | MIT | PyPI |
| vs_tiletools | Tiling, pad, crop, auto trim (templates) | MIT | PyPI (`>=1.0.0.dev0`) |
| vs_colorfix (`[tensorrt]` on NVIDIA) | Average / wavelet colour fix against a reference (templates) | Apache-2.0 | PyPI |
| vapoursynth-mvtools | Motion vectors, degrain, stabilize | GPL-2.0 | PyPI |
| vapoursynth-mvutensils 9 | MVTools helpers vsdenoise calls (MC_Degrain, QTGMC, TemporalFix Classic) | GPL-2.0 | PyPI; pinned, 10 dropped the arguments vsjetpack 2.2.0 passes |
| vapoursynth-cas | Contrast adaptive sharpening | MIT | PyPI |
| vapoursynth-adaptivegrain | Luma-adaptive grain mask | MIT | PyPI |
| vapoursynth-wnnm | Slow high-quality denoise | GPL-3.0+ (PyPI; upstream repo MIT) | PyPI |
| vapoursynth-nlm-cuda | GPU NLMeans (NLM Denoise CUDA template) | GPL-3.0+ | PyPI, NVIDIA only |
| vapoursynth-scxvid | Scene change detection | GPL-2.0 | PyPI |
| vapoursynth-dctfilter | DCT-domain filtering (Deblock QED) | MIT | PyPI |
| vapoursynth-timecube | 3D LUTs (Apply LUT template) | WTFPL | PyPI |
| vs_deepdeinterlace | AI deinterlacers DDD, DeF, DfConvEkSA (Deep Deinterlace, IVTC AI templates) | **none** | GitHub `pifroggi/vs_deepdeinterlace` @ `0516244`, `--no-deps` |
| positional-encodings, einops, timm | vs_deepdeinterlace's models | MIT / MIT / Apache-2.0 | PyPI |
| vs_align | Align HR to LR (Position), Sync (Timing) | MIT | PyPI, `--no-deps` |
| opencv-python-headless | Image writer, vs_align | Apache-2.0 (OpenCV), MIT (wrapper) | PyPI |
| onnxruntime | AniCC node, engine builder | MIT | PyPI |
| vs-mlrt backends (trt 16.1, ort 16.1, ncnn 15.16, migx 17.0) | Model nodes | GPL-3.0 | PyPI; MIGraphX on Windows from the vs-mlrt release |
| tensorrt-cu13, nvidia-modelopt, polygraphy, onnx-graphsurgeon, onnxscript, onnxslim, lief | TensorRT engine builder | NVIDIA SLA / Apache-2.0 / Apache-2.0 / Apache-2.0 / MIT / MIT / Apache-2.0 | PyPI, NVIDIA only |
| `plugins.7z`: TCanny, MiscFilters, EEDI2CUDA, cv_inpaint | API4 DLLs with no wheel: Edge Mask template and Hybrid scripts (tcanny, misc.SCDetect, eedi2cuda); inpainting | GPL-3.0+ / LGPL-2.1 / GPL-2.0+ / Apache-2.0 | Vapourkit's bundle, Windows only; licences inside the archive |
| Hybrid scripts (VapoursynthScriptsInHybrid @ `d430e19`, 70 modules) | SMDegrain, QTGMC (old), SpotLess, sRestore, dehalo, dering and ~45 other templates | **none** (mixed third-party files inside) | GitHub zip, downloaded into `vs-scripts` |
| `vsmlrt.py` | Model nodes | GPL-3.0 | shipped (`include/vsmlrt.py`) |

## Flagged

- **No licence:** vs_undistort, vs_deepdeinterlace, the Hybrid scripts. None is shipped with the app; setup downloads
  them on the user's machine. Its weights (deepdeinterlace's DDD/DeF/DfConvEkSA/RIFE files) have unknown terms.
- **GPL-2.0 (only vs. or-later not checked):** vapoursynth-mvtools, vapoursynth-mvutensils, vapoursynth-scxvid, and the GPL-2.0 wheels
  vsjetpack pulls (deblock, sangnom, znedi3, eedi3, dmetrics, dvdsrc2). EEDI2CUDA's headers say GPL-2.0 or later.
- **GPU-only:** vapoursynth-nlm-cuda, the `[nvidia]` extra (CUDA builds of BM3D, DFTTest, vszip), vs_grain's grain
  generator, vs_colorfix `[tensorrt]`, TensorRT and the engine builder, EEDI2CUDA. vs_undistort and
  vs_deepdeinterlace run on CPU off NVIDIA, slowly.
- **NVIDIA licence:** tensorrt-cu13 (NVIDIA Software License Agreement), installed from PyPI, never shipped.
- **vsjetpack is pinned to 2.2.0:** later releases require vs-mlrt ort >= 16.2 (2.2.1) or vs-mlrt 17 (2.2.5), which
  conflict with the app's vs-mlrt pins.
- **Templates:** 138 of Vapourkit's 161. Dropped 23: upscaling-UI or script-chain ones (Resize presets, Color
  Grade, Create/Load LUT, Crop, Crop with Preview, Move Original Clip Reference, Wavelet Color Fix from Step,
  Side by Side, Stack H/V, Interleave/Blend/Splice Clips) and ones needing something not installed (DLSS Neural
  Uplift, DPIR, RIFE, Grain Synth, Guided Color Fix, Motion Mask, Retinex, Retinex Edge Mask). On the 12-frame
  test clip 125 run; the other 13 need a setting first (a LUT or image path, frame numbers, descale size, field
  order) or a Tile/Pad step before them.
- **Not restored:** vsview (viewer), vs_grainsynth and vs_guidedcolorfix (weights of unknown or NC licence),
  Vapourkit's `basic_resize.py` / `upscale-script.py`, and 39 of `plugins.7z`'s 43 DLLs (wheels replace them, or
  they are API3-only: EEDI2, Retinex, motionmask, TEdgeMask, mvsf, neo_f3kdb, plus FFTW).
