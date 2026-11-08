# INCLUDE
This folder contains files the app ships and copies into its data folder.

- `vk_dataset` - the dataset engine the Dataset workspace runs, with its presets, filter templates (`filters/`) and third-party notices
- `align_videos.py` - temporal alignment the dataset engine imports
- `vk_runtime.py` - the helpers the model nodes use to pick a vs-mlrt backend
- `vsmlrt.py` - vs-mlrt's Python wrapper, which the model nodes run models through (GPL-3.0; its header names the upstream commit and the change made to it)
- `build_trt_engine.py` - the TensorRT engine builder the trtexec shim calls
- `plugins` - `plugins.7z`, Windows VapourSynth plugins without a wheel (TCanny, MiscFilters, EEDI2CUDA, cv_inpaint), with their licences
