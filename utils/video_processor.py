import os
import ffmpeg
from typing import Dict, Any, Tuple, List
import logging
from scenedetect.video_splitter import split_video_ffmpeg
from utils.scene_detector import SceneDetector
import subprocess
from .degradation_pipeline import DegradationPipeline
from .degradations.codec_degradation import CodecDegradation
from .logging_utils import DegradationLogger

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
        self.split_preset = config.get('split_preset', 'slow')
        self.strip_audio = config.get('strip_audio', True)
        
        # Initialize scene detector with config object
        self.scene_detector = scene_detector or SceneDetector(config=config)
        
        self.codec_handler = codec_handler
        
        # Initialize logger first
        self.logger = DegradationLogger(config)

        # Initialize degradation pipeline with logger
        self.degradation_pipeline = DegradationPipeline(config)

        # Add degradations with logger
        for degradation_config in config.get('degradations', []):
            if degradation_config.get('enabled', True):
                degradation_class = self.get_degradation_class(degradation_config['name'])
                if degradation_class:
                    self.degradation_pipeline.add_degradation(
                        degradation_class(degradation_config, self.logger)
                    )


        # Verify input file exists
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Input video not found: {self.input_path}")
        
        # Create output and chunks directories if they don't exist
        os.makedirs(self.output_directory, exist_ok=True)
        os.makedirs(self.hr_directory, exist_ok=True)
        os.makedirs(self.lr_directory, exist_ok=True)
        
        # Get video info for later use
        self.video_info = self.scene_detector.get_video_info(self.input_path)
    
    def _create_ffmpeg_split_command(self, start_frame, end_frame, output_file):
        """Helper method to create ffmpeg split command using frame numbers for precise cutting"""
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-i', self.input_path,
            '-r', str(self.video_info['fps']),  # Add input framerate
            '-vf', f'select=between(n\,{start_frame}\,{end_frame-1}),setpts=PTS-STARTPTS',
            '-c:v', 'hevc_nvenc',
            '-preset', self.split_preset,
            '-qp', '0',
            '-pix_fmt', 'yuv420p',
            '-fps_mode', 'cfr'
        ]
        
        if self.strip_audio:
            ffmpeg_cmd.extend(['-an', '-map', '0:v:0'])
        else:
            ffmpeg_cmd.extend(['-map', '0:v:0', '-map', '0:a?', '-map', '0:s?'])
        
        ffmpeg_cmd.append(output_file)
        return ffmpeg_cmd

    def get_degradation_class(self, name: str):
        """Get the degradation class by name"""
        degradation_classes = {
            'codec': CodecDegradation
            # Add other degradation classes here as they're implemented
        }
        return degradation_classes.get(name)

    def _create_chunk_pairs(self):
        """Helper method to create HR/LR pairs from HR chunks"""
        chunk_files = [os.path.join(self.hr_directory, f) for f in os.listdir(self.hr_directory) 
                      if f.endswith(".mp4")]
        chunk_files.sort()
        
        chunk_pairs = []
        for hr_chunk_path in chunk_files:
            basename = os.path.basename(hr_chunk_path)
            lr_chunk_path = os.path.join(self.lr_directory, basename)
            chunk_pairs.append((hr_chunk_path, lr_chunk_path))
        
        logger.info(f"Created {len(chunk_pairs)} HR/LR video chunk pairs")
        return chunk_pairs
    
    def split_video_by_scenes(self) -> List[Tuple[str, str]]:
        """
        Split the input video into chunks based on scene detection using frame numbers.
        
        Returns:
            List of tuples (hr_chunk_path, lr_chunk_path)
        """
        scene_list = self.scene_detector.detect_scenes(self.input_path)
        
        # Filter out scenes that are too short
        filtered_scene_list = []
        fps = self.video_info['fps']
        min_frames = int(self.min_chunk_duration * fps)
        
        for scene in scene_list:
            start_frame = scene[0].get_frames()
            end_frame = scene[1].get_frames()
            frame_count = end_frame - start_frame
            
            if frame_count >= min_frames:
                filtered_scene_list.append((start_frame, end_frame))
            else:
                logger.info(f"Skipping scene (frame count {frame_count} is less than minimum {min_frames})")
        
        # Process each scene
        for i, (start_frame, end_frame) in enumerate(filtered_scene_list):
            output_file = os.path.join(self.hr_directory, f'chunk_{i+1:04d}.mp4')
            
            ffmpeg_cmd = self._create_ffmpeg_split_command(
                start_frame, end_frame, output_file
            )
            
            logger.info(f"Splitting scene {i+1}/{len(filtered_scene_list)}: {output_file} (frames {start_frame}-{end_frame})")
            subprocess.run(ffmpeg_cmd, check=True)
        
        return self._create_chunk_pairs()    
    
    def split_video_by_duration(self) -> List[Tuple[str, str]]:
        """
        Split the input video into chunks of fixed duration.
        
        Returns:
            List of tuples (hr_chunk_path, lr_chunk_path)
        """
        logger.info(f"Splitting video by duration with chunk_duration={self.chunk_duration}s")
        
        # Get video info
        fps = self.video_info['fps']
        total_frames = self.video_info.get('nb_frames')
        
        if not total_frames:
            # If nb_frames isn't available, estimate from duration
            duration = self.video_info.get('duration')
            if not duration:
                raise ValueError("Cannot determine video duration")
            total_frames = int(float(duration) * fps)
        
        # Create frame-based chunks
        chunk_frames = int(self.chunk_duration * fps)
        scene_list = []
        
        for start_frame in range(0, total_frames, chunk_frames):
            end_frame = min(start_frame + chunk_frames, total_frames)
            
            output_file = os.path.join(self.hr_directory, f'chunk_{len(scene_list)+1:04d}.mp4')
            
            ffmpeg_cmd = self._create_ffmpeg_split_command(
                start_frame, end_frame, output_file
            )
            
            logger.info(f"Splitting chunk {len(scene_list)+1}: {output_file}")
            subprocess.run(ffmpeg_cmd, check=True)
            
            scene_list.append(output_file)
        
        return self._create_chunk_pairs()    
    
    def split_video(self) -> List[Tuple[str, str]]:
        """
        Split the input video into chunks based on the configured strategy.
        
        Returns:
            List of tuples (hr_chunk_path, lr_chunk_path)
        """
        # Clean directories
        for directory in [self.hr_directory, self.lr_directory]:
            for file in os.listdir(directory):
                file_path = os.path.join(directory, file)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
        
        if self.chunk_strategy == "scene_detection":
            return self.split_video_by_scenes()
        elif self.chunk_strategy == "duration":
            return self.split_video_by_duration()
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

        self.logger.log_chunk_start(hr_chunk_path)
        result = self.degradation_pipeline.process_video(hr_chunk_path, lr_chunk_path)
        self.logger.log_chunk_complete(lr_chunk_path)
        return result
    
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
                if os.path.exists(hr_path):
                    self.process_chunk(hr_path, lr_path)
                    processed_pairs.append((hr_path, lr_path))
                else:
                    logger.error(f"HR chunk file not found: {hr_path}")
            except Exception as e:
                logger.error(f"Failed to process chunk {i+1}/{len(chunk_pairs)}: {str(e)}")
        
        return processed_pairs
    
    def process_video(self) -> List[Tuple[str, str]]:
        """
        Process the input video by splitting it into chunks and applying random codecs.
        
        Returns:
            List of tuples (hr_chunk_path, lr_chunk_path) for successfully processed chunks
        """
        # Step 1: Split the video into chunks
        logger.info(f"Starting video processing for: {self.input_path}")
        logger.info("Step 1: Splitting video into chunks...")
        chunk_pairs = self.split_video()
        logger.info(f"Created {len(chunk_pairs)} chunks")
        
        # Step 2: Process each chunk with a random codec and quality
        logger.info("Step 2: Processing chunks with random codecs...")
        processed_pairs = self.process_chunks(chunk_pairs)
        
        # Print total frame count summary
        total_frames = 0
        for hr_path, _ in processed_pairs:
            try:
                probe = ffmpeg.probe(hr_path)
                stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
                frames = int(stream.get('nb_frames', 0))
                total_frames += frames
            except Exception as e:
                logger.error(f"Error getting frame count for {hr_path}: {str(e)}")
        
        logger.info(f"Processed {len(processed_pairs)} chunks with a total of {total_frames} frames")
        
        return processed_pairs

    def detect_scene_changes(self, video_path: str) -> List[int]:
        """
        Detect scene changes in a video using the SceneDetector.
        """
        logger.info(f"Detecting scene changes in: {video_path}")
        scene_times = self.scene_detector.detect_scenes(video_path)
        logger.info(f"Detected {len(scene_times)} scene changes")
        return scene_times
