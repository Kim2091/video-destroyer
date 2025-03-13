import os
import shutil
import subprocess
import json
import ffmpeg
from typing import Dict, Any, Tuple, List, Optional
import logging
import tempfile
import platform
from utils.scene_detector import SceneDetector

logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Handles video processing operations using FFmpeg.
    """
    
    def __init__(self, config: Dict[str, Any], codec_handler):
        """
        Initialize the video processor.
        
        Args:
            config: Configuration dictionary
            codec_handler: CodecHandler instance for codec selection
        """
        self.input_path = config['input_video']
        self.output_directory = config['output_directory']
        self.chunks_directory = os.path.join(config.get('chunks_directory', 'chunks'))
        self.hr_directory = os.path.join(self.chunks_directory, 'HR')
        self.lr_directory = os.path.join(self.chunks_directory, 'LR')
        self.chunk_strategy = config.get('chunk_strategy', 'scene_detection')  # Default to scene detection
        self.chunk_duration = config.get('chunk_duration', 5)
        self.min_chunk_duration = config.get('min_chunk_duration', 3)
        
        # Scene detection parameters
        self.scene_detection = config.get('scene_detection', {})
        self.scene_threshold = self.scene_detection.get('threshold', 0.3)
        self.scene_min_scene_length = self.scene_detection.get('min_scene_length', 2.0)
        self.scene_method = self.scene_detection.get('method', 'content')  # 'content' or 'edges'
        
        # Initialize scene detector
        self.scene_detector = SceneDetector(
            threshold=self.scene_threshold,
            min_scene_length=self.scene_min_scene_length,
            method=self.scene_method
        )
        
        self.codec_handler = codec_handler
        
        # Verify input file exists
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Input video not found: {self.input_path}")
        
        # Create output and chunks directories if they don't exist
        os.makedirs(self.output_directory, exist_ok=True)
        os.makedirs(self.hr_directory, exist_ok=True)
        os.makedirs(self.lr_directory, exist_ok=True)
        
        # Get video info for later use
        self.video_info = self.scene_detector.get_video_info(self.input_path)
    
    def split_video_by_duration(self) -> List[Tuple[str, str]]:
        """
        Split the input video into chunks of specified duration, ensuring clean cuts.
        
        Returns:
            List of tuples (hr_chunk_path, lr_chunk_path)
        """
        total_duration = self.video_info['duration']
        chunk_pairs = []
        
        # Clean HR and LR directories
        for directory in [self.hr_directory, self.lr_directory]:
            for file in os.listdir(directory):
                file_path = os.path.join(directory, file)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
        
        # Calculate number of chunks
        num_chunks = int(total_duration / self.chunk_duration) + (1 if total_duration % self.chunk_duration > 0 else 0)
        logger.info(f"Splitting video into {num_chunks} chunks of {self.chunk_duration} seconds each")
        
        # Split video into chunks
        for i in range(num_chunks):
            start_time = i * self.chunk_duration
            hr_chunk_path = os.path.join(self.hr_directory, f"chunk_{i:04d}.mp4")
            lr_chunk_path = os.path.join(self.lr_directory, f"chunk_{i:04d}.mp4")
            
            # Calculate duration for this chunk
            duration = min(self.chunk_duration, total_duration - start_time)
            
            # Skip if duration is too small
            if duration < 0.5:  # Skip chunks less than half a second
                continue
            
            try:
                logger.info(f"Creating chunk {i+1}/{num_chunks}: start={start_time}s, duration={duration:.2f}s")
                
                # Extract chunk using ffmpeg with precise seeking and segment options
                (
                    ffmpeg
                    .input(self.input_path, ss=start_time)
                    .output(hr_chunk_path, t=duration, c='copy', avoid_negative_ts='make_zero')
                    .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
                )
                
                chunk_pairs.append((hr_chunk_path, lr_chunk_path))
                logger.info(f"Created HR chunk {i+1}/{num_chunks}: {hr_chunk_path} (duration: {duration:.2f}s)")
            except ffmpeg.Error as e:
                logger.error(f"Error creating chunk {i+1}: {e.stderr.decode() if e.stderr else 'Unknown error'}")
        
        return chunk_pairs
    
    def split_video_by_scene(self) -> List[Tuple[str, str]]:
        """
        Split the input video by scene detection.
        
        Returns:
            List of tuples (hr_chunk_path, lr_chunk_path)
        """
        # Clean HR and LR directories
        for directory in [self.hr_directory, self.lr_directory]:
            for file in os.listdir(directory):
                file_path = os.path.join(directory, file)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
        
        # Detect scenes using the scene detector
        scene_times = self.scene_detector.detect_scenes(self.input_path)
        
        # Create chunks based on scene times
        chunk_pairs = []
        for i in range(len(scene_times) - 1):
            start_time = scene_times[i]
            end_time = scene_times[i + 1]
            duration = end_time - start_time
            
            # Skip if chunk is too short
            if duration < self.min_chunk_duration:
                logger.info(f"Skipping scene {i+1} (duration {duration:.2f}s is less than minimum {self.min_chunk_duration}s)")
                continue
            
            hr_chunk_path = os.path.join(self.hr_directory, f"chunk_{i:04d}.mp4")
            lr_chunk_path = os.path.join(self.lr_directory, f"chunk_{i:04d}.mp4")
            
            try:
                logger.info(f"Creating scene chunk {i+1}/{len(scene_times)-1}: start={start_time:.2f}s, duration={duration:.2f}s")
                
                # Extract chunk using ffmpeg with precise seeking
                (
                    ffmpeg
                    .input(self.input_path, ss=start_time)
                    .output(hr_chunk_path, t=duration, c='copy', avoid_negative_ts='make_zero')
                    .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
                )
                
                chunk_pairs.append((hr_chunk_path, lr_chunk_path))
                logger.info(f"Created HR scene chunk {i+1}/{len(scene_times)-1}: {hr_chunk_path} (duration: {duration:.2f}s)")
            except ffmpeg.Error as e:
                logger.error(f"Error creating scene chunk {i+1}: {e.stderr.decode() if e.stderr else 'Unknown error'}")
        
        return chunk_pairs
    
    def split_video(self) -> List[Tuple[str, str]]:
        """
        Split the input video into chunks based on the configured strategy.
        
        Returns:
            List of tuples (hr_chunk_path, lr_chunk_path)
        """
        if self.chunk_strategy == "duration":
            return self.split_video_by_duration()
        elif self.chunk_strategy == "scene_detection":
            return self.split_video_by_scene()
        else:
            raise ValueError(f"Unknown chunk strategy: {self.chunk_strategy}")
    
    def process_chunk(self, hr_chunk_path: str, lr_chunk_path: str) -> str:
        """
        Process a video chunk with a randomly selected codec and quality.
        
        Args:
            hr_chunk_path: Path to the input high-resolution chunk
            lr_chunk_path: Path to save the processed low-resolution chunk
            
        Returns:
            Path to the processed chunk
        """
        # Select random codec and quality
        codec, quality = self.codec_handler.get_random_encoding_config()
        
        try:
            logger.info(f"Processing chunk with codec {codec} at quality {quality}: {hr_chunk_path}")
            
            # Get the framerate and other parameters from the original video
            fps = self.video_info['fps']
            width = self.video_info['width']
            height = self.video_info['height']
            pix_fmt = self.video_info['pix_fmt']
            
            # Configure codec-specific parameters
            if codec == 'h264':
                # H.264 encoding with CRF quality
                (
                    ffmpeg
                    .input(hr_chunk_path)
                    .output(lr_chunk_path, vcodec='libx264', crf=quality, preset='medium', 
                           r=fps, vsync='cfr', video_bitrate=0, pix_fmt=pix_fmt,
                           g=int(fps*2))  # Set GOP size to 2 seconds
                    .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
                )
            elif codec == 'h265':
                # H.265/HEVC encoding with CRF quality
                (
                    ffmpeg
                    .input(hr_chunk_path)
                    .output(lr_chunk_path, vcodec='libx265', crf=quality, preset='medium', 
                           r=fps, vsync='cfr', video_bitrate=0, pix_fmt=pix_fmt,
                           g=int(fps*2))
                    .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
                )
            elif codec == 'vp9':
                # VP9 encoding with CRF quality
                (
                    ffmpeg
                    .input(hr_chunk_path)
                    .output(lr_chunk_path, vcodec='libvpx-vp9', crf=quality, b=0, 
                           r=fps, vsync='cfr', pix_fmt=pix_fmt,
                           g=int(fps*2))
                    .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
                )
            elif codec == 'av1':
                try:
                    # SVT-AV1 encoding with CRF quality
                    (
                        ffmpeg
                        .input(hr_chunk_path)
                        .output(lr_chunk_path, vcodec='libsvtav1', crf=quality, preset=7, 
                               r=fps, vsync='cfr', pix_fmt=pix_fmt,
                               g=int(fps*2))
                        .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
                    )
                except ffmpeg.Error as e:
                    # If SVT-AV1 fails, raise an error
                    error_message = e.stderr if isinstance(e.stderr, str) else e.stderr.decode() if e.stderr else 'Unknown error'
                    logger.error(f"SVT-AV1 encoding failed: {error_message}")
                    logger.error("Make sure SVT-AV1 is installed and properly configured.")
                    raise
            elif codec == 'mpeg1':
                # MPEG-1 encoding with qscale quality
                # For MPEG-1, qscale ranges from 1 (best) to 31 (worst)
                (
                    ffmpeg
                    .input(hr_chunk_path)
                    .output(lr_chunk_path, vcodec='mpeg1video', qscale=quality, 
                           r=fps, vsync='cfr', pix_fmt=pix_fmt,
                           g=int(fps*2))
                    .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
                )
            elif codec == 'mpeg2':
                # MPEG-2 encoding with qscale quality
                # For MPEG-2, qscale ranges from 1 (best) to 31 (worst)
                (
                    ffmpeg
                    .input(hr_chunk_path)
                    .output(lr_chunk_path, vcodec='mpeg2video', qscale=quality, 
                           r=fps, vsync='cfr', pix_fmt=pix_fmt,
                           g=int(fps*2))
                    .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
                )
            else:
                raise ValueError(f"Unsupported codec: {codec}")
            
            logger.info(f"Finished processing chunk with codec {codec} at quality {quality}: {lr_chunk_path}")
            return lr_chunk_path
            
        except ffmpeg.Error as e:
            error_message = e.stderr if isinstance(e.stderr, str) else e.stderr.decode() if e.stderr else 'Unknown error'
            logger.error(f"FFmpeg error: {error_message}")
            raise
        except Exception as e:
            logger.error(f"Error processing chunk: {str(e)}")
            raise
    
    def process_chunks(self, chunk_pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """
        Process all video chunks with random codecs and quality settings.
        
        Args:
            chunk_pairs: List of tuples (hr_chunk_path, lr_chunk_path)
            
        Returns:
            List of tuples (hr_chunk_path, lr_chunk_path) for successfully processed chunks
        """
        processed_pairs = []
        
        logger.info(f"Processing {len(chunk_pairs)} chunks with random codecs and quality settings")
        
        for i, (hr_path, lr_path) in enumerate(chunk_pairs):
            try:
                logger.info(f"Processing chunk {i+1}/{len(chunk_pairs)}: {hr_path}")
                self.process_chunk(hr_path, lr_path)
                processed_pairs.append((hr_path, lr_path))
            except Exception as e:
                logger.error(f"Failed to process chunk {i+1}/{len(chunk_pairs)}: {str(e)}")
        
        return processed_pairs
    
    def process_video(self) -> List[Tuple[str, str]]:
        """
        Process the input video by splitting it into chunks and applying random codecs.
        
        Returns:
            List of tuples (hr_chunk_path, lr_chunk_path) for successfully processed chunks
        """
        try:
            # Step 1: Split the video into chunks
            logger.info(f"Starting video processing for: {self.input_path}")
            logger.info("Step 1: Splitting video into chunks...")
            chunk_pairs = self.split_video()
            logger.info(f"Created {len(chunk_pairs)} chunks")
            
            # Step 2: Process each chunk with a random codec and quality
            logger.info("Step 2: Processing chunks with random codecs...")
            processed_pairs = self.process_chunks(chunk_pairs)
            logger.info(f"Processed {len(processed_pairs)} chunks")
            
            return processed_pairs
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            raise