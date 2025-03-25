import logging
import os
import subprocess
from typing import Any, Dict, List, Tuple
from functools import wraps

import ffmpeg
from tqdm import tqdm
from utils.scene_detector import SceneDetector
from .degradation_pipeline import DegradationPipeline
from .degradations.codec_degradation import CodecDegradation
from .degradations.resize_degradation import ResizeDegradation
from .degradations.halo_degradation import HaloDegradation
from .degradations.ghosting_degradation import GhostingDegradation
from .degradations.blur_degradation import BlurDegradation
from .logging_utils import DegradationLogger

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M'
)

def log_errors(func):
    """Decorator to handle and log errors"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise
    return wrapper

def check_ffmpeg_available():
    """Check if FFmpeg is available on the system"""
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        return True
    except (subprocess.SubProcessError, FileNotFoundError):
        logger.error("FFmpeg is not available on the system. Please install FFmpeg to continue.")
        return False

class VideoProcessor:
    """Handles video processing operations using FFmpeg."""
    
    def __init__(self, config: Dict[str, Any], codec_handler, scene_detector=None):
        """Initialize the video processor with configuration"""
        if not check_ffmpeg_available():
            raise RuntimeError("FFmpeg is not available on the system")
        # Store the full config and get params
        self.params = config.get('params', {})
        
        # Video paths and directories
        self.input_path = config['input_video']
        self.chunks_directory = os.path.join(config.get('chunks_directory', 'chunks'))
        self.hr_directory = os.path.join(self.chunks_directory, 'HR')
        self.lr_directory = os.path.join(self.chunks_directory, 'LR')
        
        # Chunk settings from config
        self.chunk_strategy = config.get('chunk_strategy', 'scene_detection')
        self.chunk_duration = config.get('chunk_duration', 10)
        self.frames_per_chunk = config.get('frames_per_chunk', 300)
        self.min_chunk_duration = config.get('min_chunk_duration', 1.0)
        
        # Processing settings
        self.split_preset = config.get('split_preset', 'slow')
        self.strip_audio = config.get('strip_audio', True)
        self.use_existing_chunks = config.get('use_existing_chunks', False)
        
        # Initialize components
        self.scene_detector = scene_detector or SceneDetector(config=config)
        self.codec_handler = codec_handler
        self.logger = DegradationLogger(config)
        self.degradation_pipeline = DegradationPipeline(config)
        
        # Setup
        self._verify_input_file()
        self._setup_directories()
        self._setup_degradations(config)
        self.video_info = self.scene_detector.get_video_info(self.input_path)

    def _verify_input_file(self):
        """Verify input file exists"""
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Input video not found: {self.input_path}")

    def _setup_directories(self):
        """Create necessary directories"""
        directories = [self.hr_directory, self.lr_directory]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def _setup_degradations(self, config):
        """Set up degradation pipeline"""
        for degradation_config in config.get('degradations', []):
            if not degradation_config.get('enabled', True):
                continue
            
            degradation_class = self.get_degradation_class(degradation_config['name'])
            if degradation_class:
                # Create degradation instance with logger
                degradation = degradation_class(degradation_config, self.logger)
                # Add to pipeline
                self.degradation_pipeline.add_degradation(degradation)


    def _create_ffmpeg_split_command(self, start_frame, end_frame, output_file):
        """Create FFmpeg command for splitting video"""
        start_time = start_frame / self.video_info['fps']
        duration = (end_frame - start_frame) / self.video_info['fps']
        seek_offset = max(0, start_time - 2)
        
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-hide_banner',
            '-loglevel', 'error',
            '-nostats',
            '-ss', str(seek_offset),
            '-i', self.input_path,
            '-ss', str(start_time - seek_offset),
            '-t', str(duration),
            '-c:v', 'hevc_nvenc',
            '-preset', self.split_preset,
            '-tune', 'lossless',
            '-qp', '0',
            '-pix_fmt', self.video_info['pix_fmt'],
            '-fps_mode', 'cfr',
            '-colorspace', 'bt709'
        ]
        
        if self.strip_audio:
            ffmpeg_cmd.extend(['-an', '-map', '0:v:0'])
        else:
            ffmpeg_cmd.extend(['-map', '0:v:0', '-map', '0:a?', '-map', '0:s?'])
        
        ffmpeg_cmd.append(output_file)
        return ffmpeg_cmd

    def _process_chunk_range(self, start_frame: int, end_frame: int, chunk_number: int) -> str:
        """Process a range of frames into a chunk"""
        output_file = os.path.join(self.hr_directory, f'chunk_{chunk_number:04d}.mp4')
        ffmpeg_cmd = self._create_ffmpeg_split_command(start_frame, end_frame, output_file)
        
        logger.debug(f"Splitting chunk {chunk_number}: {output_file} (frames {start_frame}-{end_frame})")
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return output_file

    def _get_frame_count(self, video_path: str) -> int:
        """Get frame count from video file"""
        try:
            probe = ffmpeg.probe(video_path)
            stream = next((stream for stream in probe['streams'] 
                         if stream['codec_type'] == 'video'), None)
            return int(stream.get('nb_frames', 0))
        except Exception as e:
            logger.error(f"Error getting frame count for {video_path}: {str(e)}")
            return 0

    def get_degradation_class(self, name: str):
        """Get the degradation class by name"""
        degradation_classes = {
            'codec': CodecDegradation,
            'resize': ResizeDegradation,
            'halo': HaloDegradation,
            'blur': BlurDegradation,
            'ghosting': GhostingDegradation
        }
        return degradation_classes.get(name)

    def _create_chunk_pairs(self):
        """Create HR/LR pairs from HR chunks"""
        chunk_files = sorted([f for f in os.listdir(self.hr_directory) if f.endswith(".mp4")])
        chunk_pairs = [(os.path.join(self.hr_directory, f), 
                       os.path.join(self.lr_directory, f)) for f in chunk_files]
        
        logger.info(f"Created {len(chunk_pairs)} HR/LR video chunk pairs")
        return chunk_pairs

    @log_errors
    def split_video_by_scenes(self) -> List[Tuple[str, str]]:
        """Split video into chunks based on scene detection"""
        scene_list = self.scene_detector.detect_scenes(self.input_path)
        min_frames = int(self.min_chunk_duration * self.video_info['fps'])
        
        # Filter scenes and prepare for processing
        filtered_scenes = []
        for i, scene in enumerate(scene_list):
            start_frame = scene[0].get_frames()
            end_frame = scene[1].get_frames()
            scene_duration = (end_frame - start_frame) / self.video_info['fps']
            if end_frame - start_frame >= min_frames:
                filtered_scenes.append((start_frame, end_frame))
            else:
                logger.info(f"Skipping scene {i+1} due to short duration: {scene_duration:.2f}s (minimum: {self.min_chunk_duration}s)")

        # Process scenes with progress bar
        logger.info(f"Saving {len(filtered_scenes)} detected scenes...")
        completed_files = []
        for i, (start_frame, end_frame) in enumerate(tqdm(filtered_scenes, desc="Saving scenes", unit="scene"), 1):
            try:
                output_file = self._process_chunk_range(start_frame, end_frame, i)
                completed_files.append(output_file)
            except Exception as e:
                logger.error(f"Failed to process scene {i}: {str(e)}")

        return self._create_chunk_pairs()

    @log_errors
    def split_video_by_duration(self) -> List[Tuple[str, str]]:
        """Split video into chunks of fixed duration"""
        logger.info(f"Splitting video by duration: {self.chunk_duration}s")
        
        fps = self.video_info['fps']
        total_frames = self.video_info.get('nb_frames')
        if not total_frames:
            duration = self.video_info.get('duration')
            if not duration:
                raise ValueError("Cannot determine video duration")
            total_frames = int(float(duration) * fps)
        
        chunk_frames = int(self.chunk_duration * fps)
        chunk_list = []
        
        for i, start_frame in enumerate(range(0, total_frames, chunk_frames), 1):
            end_frame = min(start_frame + chunk_frames, total_frames)
            output_file = self._process_chunk_range(start_frame, end_frame, i)
            chunk_list.append(output_file)
        
        return self._create_chunk_pairs()

    def split_video_by_frames(self) -> List[Tuple[str, str]]:
        """Split video into chunks with a fixed number of frames per chunk"""
        logger.info(f"Splitting video by frame count: {self.frames_per_chunk} frames")
        
        total_frames = self.video_info.get('nb_frames')
        if not total_frames:
            duration = self.video_info.get('duration')
            if not duration:
                raise ValueError("Cannot determine video duration")
            total_frames = int(float(duration) * self.video_info['fps'])
        
        chunk_list = []
        
        for i, start_frame in enumerate(range(0, total_frames, self.frames_per_chunk), 1):
            end_frame = min(start_frame + self.frames_per_chunk, total_frames)
            output_file = self._process_chunk_range(start_frame, end_frame, i)
            chunk_list.append(output_file)
        
        return self._create_chunk_pairs()

    def split_video(self) -> List[Tuple[str, str]]:
        """Split video based on configured strategy"""
        # Clean existing files
        for directory in [self.hr_directory, self.lr_directory]:
            for file in os.listdir(directory):
                os.unlink(os.path.join(directory, file))
        
        if self.chunk_strategy == "scene_detection":
            return self.split_video_by_scenes()
        elif self.chunk_strategy == "duration":
            return self.split_video_by_duration()
        elif self.chunk_strategy == "frame_count":
            return self.split_video_by_frames()
        else:
            raise ValueError(f"Unknown chunk strategy: {self.chunk_strategy}")

    @log_errors
    def process_chunk(self, hr_chunk_path: str, lr_chunk_path: str) -> str:
        """Process a video chunk with degradations"""
        try:
            # Log the start of chunk processing
            self.logger.log_chunk_start(hr_chunk_path)
            
            # Process the chunk using the pipeline
            result = self.degradation_pipeline.process_video(hr_chunk_path, lr_chunk_path)
            
            # Log successful completion
            self.logger.log_chunk_complete(lr_chunk_path)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing chunk {hr_chunk_path}: {str(e)}")
            raise

    def process_chunks(self, chunk_pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Process all chunks with degradations"""
        logger.info(f"Processing {len(chunk_pairs)} chunks")
        processed_pairs = []
        
        for i, (hr_path, lr_path) in enumerate(tqdm(chunk_pairs, desc="Processing chunks"), 1):
            try:
                if not os.path.exists(hr_path):
                    logger.error(f"HR chunk not found: {hr_path}")
                    continue
                    
                # Process the chunk
                self.process_chunk(hr_path, lr_path)
                processed_pairs.append((hr_path, lr_path))
                
            except Exception as e:
                logger.exception(f"Failed to process chunk {i}")  # This logs the full stack trace
                continue
        
        if not processed_pairs:
            logger.warning("No chunks were successfully processed!")
        
        return processed_pairs

    def process_existing_chunks(self) -> List[Tuple[str, str]]:
        """Process existing HR chunks without re-splitting the video"""
        logger.info("Processing existing HR chunks...")
        
        # Get existing HR chunks
        chunk_files = sorted([f for f in os.listdir(self.hr_directory) if f.endswith(".mp4")])
        if not chunk_files:
            raise ValueError("No HR chunks found in the HR directory")
            
        # Create chunk pairs
        chunk_pairs = [(os.path.join(self.hr_directory, f), 
                    os.path.join(self.lr_directory, f)) for f in chunk_files]
        
        # Process the chunks
        logger.info("Processing chunks with degradations...")
        processed_pairs = self.process_chunks(chunk_pairs)
        
        # Summarize results
        total_frames = sum(self._get_frame_count(hr_path) for hr_path, _ in processed_pairs)
        logger.info(f"Processed {len(processed_pairs)} existing chunks ({total_frames} frames)")
        
        return processed_pairs

    def process_video(self) -> List[Tuple[str, str]]:
        """Process video end-to-end"""
        logger.info(f"Processing video: {self.input_path}")
        
        if self.use_existing_chunks:
            return self.process_existing_chunks()
        
        # Original processing flow
        logger.info("Splitting video into chunks...")
        chunk_pairs = self.split_video()
        
        logger.info("Processing chunks with degradations...")
        processed_pairs = self.process_chunks(chunk_pairs)
        
        total_frames = sum(self._get_frame_count(hr_path) for hr_path, _ in processed_pairs)
        logger.info(f"Processed {len(processed_pairs)} chunks ({total_frames} frames)")
        
        return processed_pairs
