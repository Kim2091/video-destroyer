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
    
    def __init__(self, config: Dict[str, Any], codec_handler, scene_detector=None):
        """
        Initialize the video processor.
        
        Args:
            config: Configuration dictionary
            codec_handler: CodecHandler instance for codec selection
            scene_detector: SceneDetector instance for scene detection
        """
        self.input_path = config['input_video']
        self.output_directory = config['output_directory']
        self.chunks_directory = os.path.join(config.get('chunks_directory', 'chunks'))
        self.hr_directory = os.path.join(self.chunks_directory, 'HR')
        self.lr_directory = os.path.join(self.chunks_directory, 'LR')
        self.chunk_strategy = config.get('chunk_strategy', 'scene_detection')
        self.chunk_duration = config.get('chunk_duration', 5)
        self.min_chunk_duration = config.get('min_chunk_duration', 3)
        
        # Scene detection parameters
        self.scene_detection = config.get('scene_detection', {})
        self.scene_threshold = self.scene_detection.get('threshold', 0.3)
        self.scene_min_scene_length = self.scene_detection.get('min_scene_length', 2.0)
        self.scene_method = self.scene_detection.get('method', 'content')
        
        # Initialize scene detector
        self.scene_detector = scene_detector or SceneDetector(
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
    
    def split_video(self) -> List[Tuple[str, str]]:
        """
        Split the input video into chunks based on the configured strategy.
        
        Returns:
            List of tuples (hr_chunk_path, lr_chunk_path)
        """
        # Use the enhanced SceneDetector to handle all splitting
        return self.scene_detector.split_video_with_output_pairs(
            self.input_path,
            self.hr_directory,
            self.lr_directory,
            chunk_strategy=self.chunk_strategy,
            chunk_duration=self.chunk_duration,
            min_chunk_duration=self.min_chunk_duration
        )
    
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

    def detect_scene_changes(self, video_path: str) -> List[int]:
        """
        Detect scene changes in a video using the SceneDetector.
        """
        logger.info(f"Detecting scene changes in: {video_path}")
        scene_times = self.scene_detector.detect_scenes(video_path)
        logger.info(f"Detected {len(scene_times)} scene changes")
        return scene_times
