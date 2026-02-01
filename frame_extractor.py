import os
import argparse
import logging
import yaml
import ffmpeg
import tempfile
import shutil
from pathlib import Path
from tqdm import tqdm
from utils.scene_detector import SceneDetector
from utils.logging_utils import DegradationLogger
from typing import List, Tuple, Dict, Any, Optional
import glob

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M'
)
logger = logging.getLogger(__name__)

# Create a separate logger for progress information to avoid mixing with tqdm
progress_logger = logging.getLogger("progress")
progress_logger.setLevel(logging.INFO)
# Remove existing handlers to avoid duplicate output
for handler in progress_logger.handlers:
    progress_logger.removeHandler(handler)


class FrameSequenceExtractor:
    """
    Extracts sequences of frames from paired HR/LR video chunks.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the frame sequence extractor from configuration.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        # Extract configuration values with defaults
        self.chunks_directory = config.get('chunks_directory', 'chunks')
        self.hr_directory = os.path.join(self.chunks_directory, 'HR')
        self.lr_directory = os.path.join(self.chunks_directory, 'LR')
        
        frames_config = config.get('frame_extraction', {})
        self.output_directory = frames_config.get('output_directory', 'frames')
        self.sequence_length = frames_config.get('sequence_length', 10)
        self.use_scene_detection = frames_config.get('use_scene_detection', False)
        self.max_sequences = frames_config.get('max_sequences_per_chunk', None)
        self.frame_skip = frames_config.get('frame_skip', 0)
        self.time_gap = frames_config.get('time_gap', 3.0)
        self.skip_existing = frames_config.get('skip_existing', True)
        self.frame_format = frames_config.get('frame_format', 'png')
        self.extract_full_chunks = frames_config.get('extract_full_chunks', False)
        self.verbose_logging = frames_config.get('verbose_logging', False)
        
        # Create output directories
        self.hr_frames_dir = os.path.join(self.output_directory, 'HR')
        self.lr_frames_dir = os.path.join(self.output_directory, 'LR')
        os.makedirs(self.hr_frames_dir, exist_ok=True)
        os.makedirs(self.lr_frames_dir, exist_ok=True)
        
        # Global sequence counter - check existing files to determine the start value
        self.sequence_counter = self._get_next_sequence_id()
        
        # Count of extracted sequences
        self.extracted_sequences = 0

        # Initialize logger
        log_config = config.get('logging', {})
        if log_config:
            self.logger = DegradationLogger(config)
        else:
            self.logger = None
    
    def _get_next_sequence_id(self) -> int:
        """
        Get the next available sequence ID based on existing files.
        
        Returns:
            Next available sequence ID
        """
        hr_files = glob.glob(os.path.join(self.hr_frames_dir, f"show*_Frame*.{self.frame_format}"))
        lr_files = glob.glob(os.path.join(self.lr_frames_dir, f"show*_Frame*.{self.frame_format}"))
        
        # Extract sequence IDs from filenames
        sequence_ids = []
        for file_path in hr_files + lr_files:
            basename = os.path.basename(file_path)
            if basename.startswith("show") and "_Frame" in basename:
                try:
                    seq_id = int(basename.split("_")[0][4:])
                    sequence_ids.append(seq_id)
                except (ValueError, IndexError):
                    continue
        
        # Return next available ID (max + 1) or 1 if no files exist
        return max(sequence_ids) + 1 if sequence_ids else 1
    
    def get_chunk_pairs(self) -> List[Tuple[str, str]]:
        """
        Get pairs of HR and LR chunks.
    
        Returns:
            List of tuples (hr_chunk_path, lr_chunk_path)
        """
        hr_chunks = []
        for ext in ["*.mkv", "*.mp4"]:
            hr_chunks.extend(glob.glob(os.path.join(self.hr_directory, ext)))
        hr_chunks = sorted(hr_chunks)
    
        chunk_pairs = []
        for hr_chunk in hr_chunks:
            basename = os.path.basename(hr_chunk)
            lr_chunk = os.path.join(self.lr_directory, basename)
        
            if os.path.exists(lr_chunk):
                chunk_pairs.append((hr_chunk, lr_chunk))
            else:
                if self.verbose_logging:
                    logger.warning(f"Missing LR chunk for {basename}")
    
        return chunk_pairs
        
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """
        Get information about a video.
        
        Args:
            video_path: Path to the video
            
        Returns:
            Dictionary containing video information
        """
        try:
            probe = ffmpeg.probe(video_path)
            video_stream = next((stream for stream in probe['streams'] 
                                if stream['codec_type'] == 'video'), None)
            
            if video_stream is None:
                raise ValueError("No video stream found in input file")
            
            # Calculate fps safely
            r_frame_rate = video_stream['r_frame_rate']
            if '/' in r_frame_rate:
                num, den = map(int, r_frame_rate.split('/'))
                fps = num / den if den != 0 else 0
            else:
                fps = float(r_frame_rate)
            
            return {
                'duration': float(probe['format']['duration']),
                'width': int(video_stream['width']),
                'height': int(video_stream['height']),
                'fps': fps,
                'codec': video_stream['codec_name'],
                'nb_frames': int(video_stream.get('nb_frames', 0))
            }
        except ffmpeg.Error as e:
            logger.error(f"FFmpeg error: {e.stderr.decode() if e.stderr else 'Unknown error'}")
            raise
    
    def extract_frames_to_temp(self, video_path: str, temp_dir: str) -> int:
        """
        Extract all frames from a video to a temporary directory.
        
        Args:
            video_path: Path to the video
            temp_dir: Temporary directory to save frames
            
        Returns:
            Number of frames extracted
        """
        os.makedirs(temp_dir, exist_ok=True)
        output_pattern = os.path.join(temp_dir, f'frame_%05d.{self.frame_format}')
        
        try:
            # Extract all frames using ffmpeg (quiet mode for speed)
            # Use rgb24 for PNG to preserve full chroma, yuvj444p for JPEG
            # Use lanczos scaling for high-quality chroma upsampling
            if self.frame_format in ['jpg', 'jpeg']:
                output_args = {'q:v': 2, 'pix_fmt': 'yuvj444p', 'sws_flags': 'lanczos+accurate_rnd+full_chroma_int'}
            else:
                output_args = {'pix_fmt': 'rgb24', 'sws_flags': 'lanczos+accurate_rnd+full_chroma_int'}
            (
                ffmpeg
                .input(video_path)
                .output(output_pattern, **output_args)
                .run(capture_stdout=True, capture_stderr=True, overwrite_output=True, quiet=True)
            )
            
            # Count the number of frames
            frames = glob.glob(os.path.join(temp_dir, f'frame_*.{self.frame_format}'))
            return len(frames)
        
        except ffmpeg.Error as e:
            logger.error(f"Error extracting frames: {e.stderr.decode() if e.stderr else 'Unknown error'}")
            raise
    
    def extract_frames_direct(self, hr_path: str, lr_path: str, start_frame: int, sequence_id: int) -> bool:
        """
        Extract a sequence of frames directly from videos without temp directories.
        Much faster for simple sequential extraction.
        
        Args:
            hr_path: Path to HR video
            lr_path: Path to LR video  
            start_frame: Starting frame number (1-indexed)
            sequence_id: Sequence ID for naming
            
        Returns:
            True if successful, False otherwise
        """
        # Check if sequence already exists
        if self.skip_existing:
            first_frame_hr = os.path.join(self.hr_frames_dir, f"show{sequence_id:05d}_Frame00001.{self.frame_format}")
            if os.path.exists(first_frame_hr):
                return True
        
        try:
            # Get video info for FPS calculation
            video_info = self.get_video_info(hr_path)
            fps = video_info['fps']
            
            # Extract HR frames using select filter (simplified approach)
            for i in range(self.sequence_length):
                frame_num = start_frame + i - 1  # 0-indexed
                output_path = os.path.join(self.hr_frames_dir, f"show{sequence_id:05d}_Frame{i+1:05d}.{self.frame_format}")
                
                # Use select filter with proper syntax for ffmpeg-python
                # Use rgb24 for PNG to preserve full chroma, yuvj444p for JPEG
                # Use lanczos scaling for high-quality chroma upsampling
                if self.frame_format in ['jpg', 'jpeg']:
                    output_args = {'q:v': 2, 'pix_fmt': 'yuvj444p', 'sws_flags': 'lanczos+accurate_rnd+full_chroma_int'}
                else:
                    output_args = {'pix_fmt': 'rgb24', 'sws_flags': 'lanczos+accurate_rnd+full_chroma_int'}
                
                (
                    ffmpeg
                    .input(hr_path)
                    .filter('select', f'gte(n,{frame_num})')
                    .output(output_path, vframes=1, vsync=0, **output_args)
                    .overwrite_output()
                    .run(capture_stdout=True, capture_stderr=True, quiet=True)
                )
            
            # Extract LR frames
            for i in range(self.sequence_length):
                frame_num = start_frame + i - 1  # 0-indexed
                output_path = os.path.join(self.lr_frames_dir, f"show{sequence_id:05d}_Frame{i+1:05d}.{self.frame_format}")
                
                # Use rgb24 for PNG to preserve full chroma, yuvj444p for JPEG
                # Use lanczos scaling for high-quality chroma upsampling
                if self.frame_format in ['jpg', 'jpeg']:
                    output_args = {'q:v': 2, 'pix_fmt': 'yuvj444p', 'sws_flags': 'lanczos+accurate_rnd+full_chroma_int'}
                else:
                    output_args = {'pix_fmt': 'rgb24', 'sws_flags': 'lanczos+accurate_rnd+full_chroma_int'}
                
                (
                    ffmpeg
                    .input(lr_path)
                    .filter('select', f'gte(n,{frame_num})')
                    .output(output_path, vframes=1, vsync=0, **output_args)
                    .overwrite_output()
                    .run(capture_stdout=True, capture_stderr=True, quiet=True)
                )
            
            return True
            
        except ffmpeg.Error as e:
            if self.verbose_logging:
                logger.error(f"Error extracting frames directly: {e.stderr.decode() if e.stderr else 'Unknown error'}")
            # Clean up partial extractions
            cleanup_pattern_hr = os.path.join(self.hr_frames_dir, f"show{sequence_id:05d}_Frame*.{self.frame_format}")
            cleanup_pattern_lr = os.path.join(self.lr_frames_dir, f"show{sequence_id:05d}_Frame*.{self.frame_format}")
            for path in glob.glob(cleanup_pattern_hr) + glob.glob(cleanup_pattern_lr):
                try:
                    os.remove(path)
                except OSError:
                    pass
            return False
        except Exception as e:
            if self.verbose_logging:
                logger.error(f"Unexpected error in direct extraction: {str(e)}")
            return False
    
    def extract_frame_sequence(self, hr_temp_dir: str, lr_temp_dir: str, start_frame: int) -> bool:
        """
        Extract a sequence of frames from HR and LR temporary directories.
        
        Args:
            hr_temp_dir: Directory containing HR frames
            lr_temp_dir: Directory containing LR frames
            start_frame: Starting frame number (1-indexed as per ffmpeg output)
            
        Returns:
            True if sequence was successfully extracted, False otherwise
        """
        sequence_id = self.sequence_counter
        
        # Check if sequence already exists and we're skipping existing
        if self.skip_existing:
            first_frame_hr = os.path.join(self.hr_frames_dir, f"show{sequence_id:05d}_Frame00001.{self.frame_format}")
            if os.path.exists(first_frame_hr):
                if self.verbose_logging:
                    progress_logger.info(f"Sequence {sequence_id} already exists, skipping")
                self.sequence_counter += 1
                return True
        
        try:
            # Copy frames from temp directory to output directory
            for i in range(self.sequence_length):
                # Calculate the actual frame number to use
                # This ensures we get consecutive frames with no duplicates
                frame_num = start_frame + i
                
                # Source paths
                hr_src_path = os.path.join(hr_temp_dir, f'frame_{frame_num:05d}.{self.frame_format}')
                lr_src_path = os.path.join(lr_temp_dir, f'frame_{frame_num:05d}.{self.frame_format}')
                
                # Check if source frames exist
                if not os.path.exists(hr_src_path) or not os.path.exists(lr_src_path):
                    if self.verbose_logging:
                        progress_logger.warning(f"Frame {frame_num} not found in temp directories")
                    return False
                
                # Destination paths with updated naming format (5 digits for show, 5 digits for frame)
                hr_dst_path = os.path.join(self.hr_frames_dir, f"show{sequence_id:05d}_Frame{i+1:05d}.{self.frame_format}")
                lr_dst_path = os.path.join(self.lr_frames_dir, f"show{sequence_id:05d}_Frame{i+1:05d}.{self.frame_format}")
                
                # Copy frames (no need to preserve metadata, faster)
                shutil.copy(hr_src_path, hr_dst_path)
                shutil.copy(lr_src_path, lr_dst_path)
            
            # Increment sequence counter for next sequence
            self.sequence_counter += 1
            self.extracted_sequences += 1
            return True
            
        except Exception as e:
            if self.verbose_logging:
                logger.error(f"Error creating sequence: {str(e)}")
            # Clean up any partially extracted frames
            cleanup_pattern_hr = os.path.join(self.hr_frames_dir, f"show{sequence_id:05d}_Frame*.{self.frame_format}")
            cleanup_pattern_lr = os.path.join(self.lr_frames_dir, f"show{sequence_id:05d}_Frame*.{self.frame_format}")
            
            for path in glob.glob(cleanup_pattern_hr) + glob.glob(cleanup_pattern_lr):
                try:
                    os.remove(path)
                except OSError:
                    pass
            
            return False    
            
    def extract_sequences_from_chunk_pair(self, hr_path: str, lr_path: str, chunk_index: int, total_chunks: int) -> int:
        """
        Extract frame sequences from a pair of HR and LR chunks.
        
        Args:
            hr_path: Path to HR video chunk
            lr_path: Path to LR video chunk
            chunk_index: Index of current chunk
            total_chunks: Total number of chunks
            
        Returns:
            Number of sequences extracted
        """
        chunk_name = os.path.basename(hr_path)
        if self.verbose_logging:
            progress_logger.info(f"Extracting sequences from chunk pair: {chunk_name}")
        
        try:
            # Get video info to calculate frame skip and total frames
            video_info = self.get_video_info(hr_path)
            fps = video_info['fps']
            frame_count = video_info.get('nb_frames', 0)
            
            # If nb_frames is not available, estimate from duration
            if frame_count == 0:
                frame_count = int(video_info['duration'] * fps)
            
            # Calculate frames_to_skip based on time_gap or frame_skip
            if self.time_gap is not None and not self.use_scene_detection:
                frames_to_skip = int(self.time_gap * fps)
            else:
                frames_to_skip = self.frame_skip
            
            # Determine sequence start frames and extraction method
            start_frames = []
            use_direct_extraction = False
            
            if self.use_scene_detection:
                # Scene detection requires full frame extraction
                with tempfile.TemporaryDirectory() as temp_dir:
                    hr_temp_dir = os.path.join(temp_dir, 'hr')
                    lr_temp_dir = os.path.join(temp_dir, 'lr')
                    
                    if self.verbose_logging:
                        progress_logger.info("Using scene detection (full extraction required)...")
                    hr_frame_count = self.extract_frames_to_temp(hr_path, hr_temp_dir)
                    lr_frame_count = self.extract_frames_to_temp(lr_path, lr_temp_dir)
                    frame_count = min(hr_frame_count, lr_frame_count)
                    
                    scene_detector = SceneDetector()
                    scene_list = scene_detector.detect_scenes(hr_path)
                    
                    for scene in scene_list:
                        start_frame = scene[0].get_frames() + 1
                        if start_frame + self.sequence_length <= frame_count:
                            start_frames.append(start_frame)
                    
                    if self.max_sequences is not None and len(start_frames) > self.max_sequences:
                        if len(start_frames) > 1:
                            step = (len(start_frames) - 1) / (self.max_sequences - 1)
                            indices = [int(i * step) for i in range(self.max_sequences)]
                            start_frames = [start_frames[i] for i in indices]
                        else:
                            start_frames = start_frames[:self.max_sequences]
                    
                    return self._extract_with_temp_dirs(hr_temp_dir, lr_temp_dir, start_frames, chunk_index, total_chunks, hr_path)
            
            elif self.extract_full_chunks:
                # Full chunk extraction requires all frames
                with tempfile.TemporaryDirectory() as temp_dir:
                    hr_temp_dir = os.path.join(temp_dir, 'hr')
                    lr_temp_dir = os.path.join(temp_dir, 'lr')
                    
                    if self.verbose_logging:
                        progress_logger.info("Extracting full chunks (full extraction required)...")
                    hr_frame_count = self.extract_frames_to_temp(hr_path, hr_temp_dir)
                    lr_frame_count = self.extract_frames_to_temp(lr_path, lr_temp_dir)
                    frame_count = min(hr_frame_count, lr_frame_count)
                    
                    max_start_frame = frame_count - self.sequence_length + 1
                    start_frames = list(range(1, max_start_frame + 1, self.sequence_length))
                    
                    if self.max_sequences is not None and len(start_frames) > self.max_sequences:
                        if len(start_frames) > 1:
                            step = (len(start_frames) - 1) / (self.max_sequences - 1)
                            indices = [int(i * step) for i in range(self.max_sequences)]
                            start_frames = [start_frames[i] for i in indices]
                        else:
                            start_frames = start_frames[:self.max_sequences]
                    
                    return self._extract_with_temp_dirs(hr_temp_dir, lr_temp_dir, start_frames, chunk_index, total_chunks, hr_path)
            
            else:
                # Standard extraction with time-based gaps
                with tempfile.TemporaryDirectory() as temp_dir:
                    hr_temp_dir = os.path.join(temp_dir, 'hr')
                    lr_temp_dir = os.path.join(temp_dir, 'lr')
                    
                    if self.verbose_logging:
                        progress_logger.info("Extracting frames with time-based gaps...")
                    hr_frame_count = self.extract_frames_to_temp(hr_path, hr_temp_dir)
                    lr_frame_count = self.extract_frames_to_temp(lr_path, lr_temp_dir)
                    frame_count = min(hr_frame_count, lr_frame_count)
                    
                    max_start_frame = frame_count - self.sequence_length + 1
                    
                    if max_start_frame <= 0:
                        logger.warning(f"Video too short to extract sequences of length {self.sequence_length}")
                        return 0
                    
                    step = self.sequence_length + frames_to_skip
                    start_frames = list(range(1, max_start_frame + 1, step))
                    
                    if self.max_sequences is not None and len(start_frames) > self.max_sequences:
                        if len(start_frames) > 1:
                            step = (len(start_frames) - 1) / (self.max_sequences - 1)
                            indices = [int(i * step) for i in range(self.max_sequences)]
                            start_frames = [start_frames[i] for i in indices]
                        else:
                            start_frames = start_frames[:self.max_sequences]
                    
                    return self._extract_with_temp_dirs(hr_temp_dir, lr_temp_dir, start_frames, chunk_index, total_chunks, hr_path)
                
        except Exception as e:
            logger.error(f"Error during frame extraction for {chunk_name}: {str(e)}")
            return 0
    
    def _extract_with_temp_dirs(self, hr_temp_dir: str, lr_temp_dir: str, start_frames: list, 
                                 chunk_index: int, total_chunks: int, hr_path: str) -> int:
        """
        Helper method to extract sequences using temp directories (for scene detection and full chunks).
        """
        sequence_count = 0
        desc = f"Processing chunk {chunk_index}/{total_chunks} [{os.path.basename(hr_path)}]"
        
        if start_frames:
            for start_frame in tqdm(start_frames, desc=desc, leave=True):
                try:
                    success = self.extract_frame_sequence(hr_temp_dir, lr_temp_dir, start_frame)
                    if success:
                        sequence_count += 1
                    
                    if self.max_sequences is not None and sequence_count >= self.max_sequences:
                        break
                
                except Exception as e:
                    if self.verbose_logging:
                        logger.error(f"Error extracting sequence at frame {start_frame}: {str(e)}")
        else:
            tqdm.write(f"{desc}: No sequences to extract (video too short or empty)")
        
        if self.verbose_logging:
            progress_logger.info(f"Extracted {sequence_count} sequences from chunk")
        return sequence_count

    def extract_all_sequences(self) -> int:
        """
        Extract frame sequences from all chunk pairs.
        
        Returns:
            Total number of sequences extracted
        """
        # Get all chunk pairs
        chunk_pairs = self.get_chunk_pairs()
        
        if not chunk_pairs:
            logger.warning("No chunk pairs found. Make sure HR and LR directories contain matching video files.")
            return 0
            
        logger.info(f"Found {len(chunk_pairs)} chunk pairs")
        
        # Reset counter for this run
        self.extracted_sequences = 0
        
        # Extract sequences from each pair
        for i, (hr_path, lr_path) in enumerate(chunk_pairs, 1):
            sequences = self.extract_sequences_from_chunk_pair(hr_path, lr_path, i, len(chunk_pairs))
            
        # Final summary (only once at the end)
        logger.info(f"Extraction complete: {self.extracted_sequences} sequences extracted from {len(chunk_pairs)} chunks")
        return self.extracted_sequences


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    if not os.path.exists(config_path):
        logger.warning(f"Configuration file not found: {config_path}")
        return {}
    
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        return {}


def merge_configs(file_config: Dict[str, Any], arg_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge configuration from file and command-line arguments.
    Command-line arguments take precedence.
    
    Args:
        file_config: Configuration from file
        arg_config: Configuration from command-line arguments
        
    Returns:
        Merged configuration dictionary
    """
    # Create a deep copy of file_config
    config = {**file_config}
    
    # Create frame_extraction section if it doesn't exist
    if 'frame_extraction' not in config:
        config['frame_extraction'] = {}
    
    # Update with command-line arguments
    frame_config = config['frame_extraction']
    
    if arg_config.get('chunks_dir'):
        config['chunks_directory'] = arg_config['chunks_dir']
    
    if arg_config.get('output_dir'):
        frame_config['output_directory'] = arg_config['output_dir']
    
    if arg_config.get('sequence_length'):
        frame_config['sequence_length'] = arg_config['sequence_length']
    
    if arg_config.get('use_scene_detection') is not None:
        frame_config['use_scene_detection'] = arg_config['use_scene_detection']
    
    if arg_config.get('max_sequences') is not None:
        frame_config['max_sequences_per_chunk'] = arg_config['max_sequences']
    
    if arg_config.get('time_gap') is not None:
        frame_config['time_gap'] = arg_config['time_gap']
    
    if arg_config.get('frame_skip') is not None:
        frame_config['frame_skip'] = arg_config['frame_skip']
    
    if arg_config.get('frame_format') is not None:
        frame_config['frame_format'] = arg_config['frame_format']
    
    if arg_config.get('extract_full') is not None:
        frame_config['extract_full_chunks'] = arg_config['extract_full']
        
    if arg_config.get('verbose') is not None:
        frame_config['verbose_logging'] = arg_config['verbose']
    
    return config


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Extract frame sequences from video chunks')
    parser.add_argument('-c', '--chunks_dir', type=str, help='Directory containing HR and LR chunks')
    parser.add_argument('-o', '--output_dir', type=str, help='Directory to save extracted frames')
    parser.add_argument('-s', '--sequence_length', type=int, help='Number of frames in each sequence. Minimum of 5 for TSCUNet. 10-30 is good.')
    parser.add_argument('-d', '--use_scene_detection', action='store_true', help='Use scene detection to determine sequence start points')
    parser.add_argument('-m', '--max_sequences', type=int, help='Maximum number of sequences to extract per chunk pair')
    parser.add_argument('-t', '--time_gap', type=float, help='Time in seconds to skip between sequences (disabled when using scene detection)')
    parser.add_argument('-f', '--frame_skip', type=int, help='Number of frames to skip between sequences (alternative to time_gap)')
    parser.add_argument('--frame_format', choices=['png', 'jpg', 'jpeg'], help='Image format for extracted frames')
    parser.add_argument('--extract_full', action='store_true', help='Extract all possible sequences without overlap')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()
    
    try:
        # Load configuration from file
        file_config = load_config(args.config)
        
        # Convert args to dictionary
        arg_config = {k: v for k, v in vars(args).items() if v is not None and k != 'config'}
        
        # Merge configurations
        config = merge_configs(file_config, arg_config)
        
        # Initialize frame sequence extractor
        extractor = FrameSequenceExtractor(config)
        
        # Extract sequences
        total_sequences = extractor.extract_all_sequences()
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
