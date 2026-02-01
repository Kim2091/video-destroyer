# Video Destroyer
[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/J3J3BCC3L)

A tool designed to create datasets from a single high quality input video for AI training purposes. It processes videos by splitting them into chunks, applying random codecs and quality settings (more degredations to come), and finally extracting frame sequences.

## 🎯 Features

This toolkit consists of two main components:
1. Video Processing (`main.py`)
   - Splits videos into chunks
   - Applies random codecs with varying quality settings (more degredations to come)
   - Creates paired HR/LR video chunks

2. Frame Extraction (`frame_extractor.py`)
   - Extracts frame sequences from video chunks
   - Supports scene-based or time-based extraction
   - Creates paired HR/LR frame sequences
   
- Note: You MUST have ffmpeg in your PATH to use this program

## 🚀 Quick Start

1. Run this command to clone the repository
```bash
git clone https://github.com/Kim2091/video-destroyer/
```

2. Install Python dependencies (and ensure ffmpeg is in your PATH):
```bash
pip install -r requirements.txt
```

3. **Edit ```config.yaml``` and configure it as you wish**
    - Set `input` to a video file or folder of videos (auto-detects which mode to use)
    - Customize video degradations
    - NOTE: Only resizing and compression are enabled by default

4. Run the video processor:
```bash
python main.py --config config.yaml
```

5. Extract frame sequences:
```bash
python frame_extractor.py
```

## 💡 Advanced Usage Guide

### Video Processing (main.py)
The first step creates paired high-quality and degraded video chunks:
- Splits input video into chunks using scene detection or fixed duration
- Processes each chunk with random codecs and quality settings
- Creates HR (original) and LR (degraded) pairs

The tool automatically detects whether you're processing a single video or a folder of videos.

**Single Video Mode:**
Edit `config.yaml` and set the `input` to a video file:
```yaml
input: "C:/path/to/video.mp4"
```

**Batch Processing Mode:**
Edit `config.yaml` and set the `input` to a folder path:
```yaml
input: "C:/path/to/video/folder"
```

Then run:
```bash
python main.py --config config.yaml
```

When processing a folder, each video will be processed independently and output to its own subfolder:
- `chunks_directory/video1_name/HR/`
- `chunks_directory/video1_name/LR/`
- `chunks_directory/video2_name/HR/`
- `chunks_directory/video2_name/LR/`
- etc.

The batch processor will:
- Automatically detect all videos in the folder (supports .mp4, .mov, .mkv, .avi, .webm, .flv)
- Process each video independently
- Continue processing even if one video fails
- Provide a summary report at the end

### Frame Extraction (frame_extractor.py)
The second step extracts frame sequences from the video chunks.

To use, you can either edit the configuration in config.yaml for frame extraction, or use the arguments below.

Arguments:
```bash
-c, --chunks_dir     Directory containing HR and LR chunks
-o, --output_dir     Directory to save extracted frames
-s, --sequence_length Number of frames in each sequence
-d, --use_scene_detection Use scene detection for frame selection
-m, --max_sequences  Maximum sequences per chunk pair
-t, --time_gap       Time gap between sequences in seconds
```

Example usage:
```bash
# Extract using time gaps
python frame_extractor.py -c chunks -o frames -s 5 -t 3.0

# Extract using scene detection
python frame_extractor.py -c chunks -o frames -s 5 -d
```

## 📝 Tips & Additional Info

- The video processor creates two directories for each chunk:
  - `chunks/HR/`: Original quality chunks
  - `chunks/LR/`: Degraded video chunks

- Frame sequences are saved as:
  - `frames/HR/show{N}_Frame{M}.png`: High-quality frames
  - `frames/LR/show{N}_Frame{M}.png`: Degraded frames

- Scene detection is recommended for videos with distinct scene changes
- Time-based extraction is better for continuous footage

- When using scene detection, the time gap parameter is ignored
- Default sequence length is 5 frames
- Default time gap is 3 seconds when not using scene detection

## 🎓 Training Dataset Creation

This tool is particularly useful for creating training datasets for video enhancement AI models:
1. Use high-quality source videos as input
2. Process them to create controlled degradations
3. Extract matching frame sequences
4. Use the resulting HR/LR pairs for training

Train one yourself with [traiNNer-redux](https://github.com/the-database/traiNNer-redux), using the TSCUNet architecture!
