# Video Destroyer

![License](https://img.shields.io/badge/license-GPL%203.0-green)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey)

**Video Destroyer** builds training datasets for AI video and image models. A dataset job is a node graph: connect sources, degradations, sequence pickers and filters to a writer, run it, and get matching HR/LR folders. It runs on VapourSynth, on Windows and Linux.

Version 2 replaces two older tools, the first Video Destroyer (on the [`v1` branch](https://github.com/Kim2091/video-destroyer/tree/v1)) and VSR-Dataset-Creator, and ships presets that reproduce their defaults:

- **One video into HR/LR pairs**: split by scene, duration or frame count, then degrade the LR side the way real releases are made: lens blur, grain and noise, SD resampling, mastering halos, telecine and bad deinterlacing, banding, chroma damage, DVD/broadcast/web compression, player smoothing, or any ffmpeg filter. HDR sources can be tonemapped first.
- **Two releases into aligned pairs**: the LR release is matched to the HR one in time and space with vs_align, crops are picked for detail and motion, bad matches are rejected and near-duplicates removed.
- **HR processing** of an existing dataset: line darkening, dehaloing, Match Gamma, tone adjustment, and ONNX models for denoising, de-H264 and colour correction. Denoise datasets from static shots, and renumbering after culling.
- **Graphs, not just chains**: Random Branch picks one whole damage chain per chunk (only that one runs), Blend mixes two, and masks (brightness, edges, flat areas, motion) with Masked Merge put damage only where it really happens. Several writers can share a source and its chunks, and a writer can write several LR variants per HR. Examples 10 to 14 show each.

Random settings are reproducible: any setting can be a fixed value, a range or a list, each chunk draws its own values from the job's seed, and `manifest.json` records what every sequence got. Settings can also be driven by wires from value nodes (Random Number, Random Pick, Math, If Equals), so two nodes share one draw or one setting is worked out from another; example 17 shows both. Jobs save as `.vkdataset` files.

## Install

### Windows

1. Run the setup installer, or extract the portable archive into a folder you own (not under Program Files).
2. Start Video Destroyer.

### Linux (x86_64 AppImage)

1. Install Python 3.12, 3.13 or 3.14 with `venv`, `ffmpeg` and `ffprobe` on `PATH`, and a working Vulkan driver, through your distribution's package manager.
2. `chmod +x VideoDestroyer-*.AppImage`, then run it. If FUSE is unavailable, run it with `APPIMAGE_EXTRACT_AND_RUN=1`.

## First run

On first launch, click **Start Setup**. It installs a private Python with VapourSynth, FFmpeg (Windows), and the plugin packages: vsjetpack with your GPU's extras, the bwdif, Deblock and zsmooth plugins the degradation nodes use, vs-mlrt for your GPU, vs_align, PyTorch, OpenCV, ONNX Runtime, the VapourSynth plugins and scripts the Filter Template node's templates use (pifroggi's filters, MVTools, the Hybrid scripts, and more; [PLUGINS.md](PLUGINS.md) lists them with their licences). Nothing is installed globally. Setup needs an internet connection. An NVIDIA install downloads about 6.5 GB and takes about 10 GB of disk, because of PyTorch's CUDA build and TensorRT; on other GPUs it is about 3 GB.

The data lives in a `data` folder beside the app on Windows, and under `~/.config/Video Destroyer/data/` on Linux. **Plugins** in the left rail reinstalls or removes the plugin packages; **Settings** has the Default AI backend, the ROCm choice for MIGraphX on Linux, the appearance, and the folders.

Model nodes run ONNX files through vs-mlrt with TensorRT (NVIDIA), DirectML (Windows), NCNN Vulkan or MIGraphX (AMD). A node set to "auto" runs on the Default AI backend in Settings, whose Auto picks the best one this machine has; a backend chosen on the node wins. The run log and the preview say which backend each model node uses, and the run log follows a TensorRT engine build as it goes. No model weights are bundled; `scripts/export_dataset_models.py` converts the checkpoints you have to ONNX.

## Building a job

With no job open, the workspace lists the example workflows; example 1 is the smallest useful job and the place to start. A job reads left to right through eight stages, from Load to Save, and every node card shows its stage, what it does, its key settings and any note. The palette on the left groups the nodes by stage and searches their names and descriptions; clicking one adds it after the selected node and joins the two when there is only one way to. The shaded region on the canvas holds the nodes that run once per chunk, where random settings are drawn again for every chunk. **Arrange** lays a job out left to right, the HR path above the LR path, and a hint appears when an edge runs back through the stages.

The analogue video nodes (Composite Encode, VHS Tape, Playback Deck, Composite Decode) run a VapourSynth plugin built from `native/vs-ntsc`. After cloning, run `git submodule update --init` and `npm run build:native` (needs Rust's cargo); the packaging scripts build it first.

## Credits

- [Vapourkit](https://github.com/Kim2091/vapourkit), the application this one is built on
- Video Destroyer v1 ([`v1` branch](https://github.com/Kim2091/video-destroyer/tree/v1)), for the degradations and the default preset
- [ntsc-simulator](https://github.com/Kim2091/ntsc-simulator) by Kim2091 and Joey Ballentine (Apache License 2.0), for the analogue video nodes (a git submodule in `native/`)
- VSR-Dataset-Creator by Bendel, for pair mode, HR processing and renaming (Apache License 2.0, used with permission)
- [VapourSynth](https://www.vapoursynth.com) and [vs-mlrt](https://github.com/AmusementClub/vs-mlrt)
- [vs_align](https://github.com/pifroggi/vs_align) by pifroggi
- [vs-jetpack](https://github.com/Jaded-Encoding-Thaumaturgy/vs-jetpack)

The third-party notices for the dataset engine, including the licences of the ported code and the models it can load, are in [include/vk_dataset/NOTICE.md](include/vk_dataset/NOTICE.md).

## License

GPL 3.0, see [LICENSE](LICENSE). As an additional term under section 7(b), copies and modified versions must keep [NOTICE](NOTICE), with its author attribution and links, intact.

If Video Destroyer helps you, you can support it on [Ko-fi](https://ko-fi.com/kim20913944).
