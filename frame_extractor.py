import os
import argparse
import logging
import ffmpeg
from utils.scene_detector import SceneDetector
from typing import List, Tuple, Dict, Any
import glob

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M'  # This matches your desired timestamp format
)
logger = logging.getLogger(__name__)


class FrameSequenceExtractor:
    """
    Extracts sequences of frames from paired HR/LR video chunks.
    """
    
    def __init__(self, chunks_directory: str, output_directory: str, sequence_length: int = 5,
                use_scene_detection: bool = False, max_sequences: int = None, time_gap: float = 3.0):
        """
        Initialize the frame sequence extractor.
        
        Args:
            chunks_directory: Directory containing HR and LR chunks
            output_directory: Directory to save extracted frame sequences
            sequence_length: Number of frames in each sequence
            use_scene_detection: Whether to use scene detection for sequence extraction
            max_sequences: Maximum number of sequences to extract (None for unlimited)
        """
        self.chunks_directory = chunks_directory
        self.hr_directory = os.path.join(chunks_directory, 'HR')
        self.lr_directory = os.path.join(chunks_directory, 'LR')
        self.output_directory = output_directory
        self.sequence_length = sequence_length
        self.use_scene_detection = use_scene_detection
        self.max_sequences = max_sequences
        
        # Create output directories
        self.hr_frames_dir = os.path.join(output_directory, 'HR')
        self.lr_frames_dir = os.path.join(output_directory, 'LR')
        os.makedirs(self.hr_frames_dir, exist_ok=True)
        os.makedirs(self.lr_frames_dir, exist_ok=True)
        
        # Global sequence counter
        self.sequence_counter = 1

        self.time_gap = time_gap
    
    def get_chunk_pairs(self) -> List[Tuple[str, str]]:
        """
        Get pairs of HR and LR chunks.
        
        Returns:
            List of tuples (hr_chunk_path, lr_chunk_path)
        """
        hr_chunks = sorted(glob.glob(os.path.join(self.hr_directory, "*.mp4")))
        
        chunk_pairs = []
        for hr_chunk in hr_chunks:
            basename = os.path.basename(hr_chunk)
            lr_chunk = os.path.join(self.lr_directory, basename)
            
            if os.path.exists(lr_chunk):
                chunk_pairs.append((hr_chunk, lr_chunk))
        
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
                fps = num / den
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
        output_pattern = os.path.join(temp_dir, 'frame_%05d.png')
        
        try:
            # Extract all frames using ffmpeg
            (
                ffmpeg
                .input(video_path)
                .output(output_pattern)
                .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
            )
            
            # Count the number of frames
            frames = glob.glob(os.path.join(temp_dir, 'frame_*.png'))
            return len(frames)
        
        except ffmpeg.Error as e:
            logger.error(f"Error extracting frames: {e.stderr.decode() if e.stderr else 'Unknown error'}")
            raise
    
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
        
        try:
            # Copy frames from temp directory to output directory
            for i in range(self.sequence_length):
                # Calculate the actual frame number to use
                # This ensures we get consecutive frames with no duplicates
                frame_num = start_frame + i
                
                # Source paths
                hr_src_path = os.path.join(hr_temp_dir, f'frame_{frame_num:05d}.png')
                lr_src_path = os.path.join(lr_temp_dir, f'frame_{frame_num:05d}.png')
                
                # Check if source frames exist
                if not os.path.exists(hr_src_path) or not os.path.exists(lr_src_path):
                    logger.warning(f"Frame {frame_num} not found in temp directories")
                    return False
                
                # Destination paths with updated naming format (5 digits for show, 5 digits for frame)
                hr_dst_path = os.path.join(self.hr_frames_dir, f"show{sequence_id:05d}_Frame{i+1:05d}.png")
                lr_dst_path = os.path.join(self.lr_frames_dir, f"show{sequence_id:05d}_Frame{i+1:05d}.png")
                
                
                # Copy frames
                import shutil
                shutil.copy2(hr_src_path, hr_dst_path)
                shutil.copy2(lr_src_path, lr_dst_path)
            
            # Increment sequence counter for next sequence
            self.sequence_counter += 1
            return True
            
        except Exception as e:
            logger.error(f"Error creating sequence: {str(e)}")
            # Clean up any partially extracted frames
            for i in range(self.sequence_length):
                hr_frame_path = os.path.join(self.hr_frames_dir, f"show{sequence_id}_Frame{i+1:04d}.png")
                lr_frame_path = os.path.join(self.lr_frames_dir, f"show{sequence_id}_Frame{i+1:04d}.png")
                
                if os.path.exists(hr_frame_path):
                    os.remove(hr_frame_path)
                if os.path.exists(lr_frame_path):
                    os.remove(lr_frame_path)
            
            return False    
            
    def extract_sequences_from_chunk_pair(self, hr_path: str, lr_path: str) -> int:
        """
        Extract frame sequences from a pair of HR and LR chunks.
        """
        logger.info(f"Extracting sequences from chunk pair: {os.path.basename(hr_path)}")
        
        # Create temporary directories for extracted frames
        hr_temp_dir = os.path.join(self.output_directory, 'temp_hr')
        lr_temp_dir = os.path.join(self.output_directory, 'temp_lr')
        
        try:
            # Extract all frames to temporary directories
            logger.info("Extracting all frames to temporary directories...")
            hr_frame_count = self.extract_frames_to_temp(hr_path, hr_temp_dir)
            lr_frame_count = self.extract_frames_to_temp(lr_path, lr_temp_dir)
            
            # Get video info to calculate frame skip
            video_info = self.get_video_info(hr_path)
            fps = video_info['fps']
            
            # Only calculate frames_to_skip if we're not using scene detection
            frames_to_skip = int(self.time_gap * fps) if self.time_gap is not None else 0
            
            logger.info(f"Extracted {hr_frame_count} HR frames and {lr_frame_count} LR frames")
            
            # Verify that both videos have the same number of frames
            if hr_frame_count != lr_frame_count:
                logger.warning(f"HR and LR videos have different frame counts: {hr_frame_count} vs {lr_frame_count}")
                frame_count = min(hr_frame_count, lr_frame_count)
            else:
                frame_count = hr_frame_count
            
            # Determine sequence start frames
            start_frames = []
            
            if self.use_scene_detection:
                # Initialize scene detector
                scene_detector = SceneDetector()
                # Get scene list
                scene_list = scene_detector.detect_scenes(hr_path)
                
                # Convert scene boundaries to start frames
                for scene in scene_list:
                    start_frame = scene[0].get_frames() + 1  # +1 because ffmpeg is 1-indexed
                    # Only add if there's enough frames left for a full sequence
                    if start_frame + self.sequence_length <= frame_count:
                        start_frames.append(start_frame)
            else:
                # Extract sequences with time-based gaps
                max_start_frame = frame_count - self.sequence_length + 1
                
                if max_start_frame <= 0:
                    logger.warning(f"Video too short to extract sequences of length {self.sequence_length}")
                    return 0
                
                # Step is sequence_length plus the frames we want to skip
                step = self.sequence_length + frames_to_skip
                
                # Generate start frames
                start_frames = list(range(1, max_start_frame + 1, step))
            
            # Check if we need to limit the number of sequences
            if self.max_sequences is not None and len(start_frames) > self.max_sequences:
                # Evenly distribute the sequences across the video
                if len(start_frames) > 1:
                    step = (len(start_frames) - 1) / (self.max_sequences - 1)
                    indices = [int(i * step) for i in range(self.max_sequences)]
                    start_frames = [start_frames[i] for i in indices]
                else:
                    start_frames = start_frames[:self.max_sequences]
            
            # Extract sequences at each start frame
            sequence_count = 0
            for start_frame in start_frames:
                try:
                    success = self.extract_frame_sequence(hr_temp_dir, lr_temp_dir, start_frame)
                    if success:
                        sequence_count += 1
                        logger.info(f"Extracted sequence {self.sequence_counter-1} starting at frame {start_frame}")
                    
                    # Check if we've reached the maximum number of sequences
                    if self.max_sequences is not None and sequence_count >= self.max_sequences:
                        break
                
                except Exception as e:
                    logger.error(f"Error extracting sequence at frame {start_frame}: {str(e)}")
            
            return sequence_count
            
        except Exception as e:
            logger.error(f"Error during frame extraction: {str(e)}")
            return 0
        
        finally:
            # Clean up temporary directories
            import shutil
            if os.path.exists(hr_temp_dir):
                shutil.rmtree(hr_temp_dir)
            if os.path.exists(lr_temp_dir):
                shutil.rmtree(lr_temp_dir)   


    def extract_all_sequences(self) -> int:
        """
        Extract frame sequences from all chunk pairs.
        
        Returns:
            Total number of sequences extracted
        """
        # Get all chunk pairs
        chunk_pairs = self.get_chunk_pairs()
        logger.info(f"Found {len(chunk_pairs)} chunk pairs")
        
        # Extract sequences from each pair
        total_sequences = 0
        for hr_path, lr_path in chunk_pairs:
            sequences = self.extract_sequences_from_chunk_pair(hr_path, lr_path)
            total_sequences += sequences
        
        logger.info(f"Extracted {total_sequences} sequences in total")
        return total_sequences


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Extract frame sequences from video chunks')
    parser.add_argument('-c', '--chunks_dir', type=str, default='chunks',
                        help='Directory containing HR and LR chunks')
    parser.add_argument('-o', '--output_dir', type=str, default='frames',
                        help='Directory to save extracted frames')
    parser.add_argument('-s', '--sequence_length', type=int, default=5,
                        help='Number of frames in each sequence')
    parser.add_argument('-d', '--use_scene_detection', action='store_true',
                        help='NOTE: Unnecessary if using scene detect in main.py || Use scene detection to determine sequence start points')
    parser.add_argument('-m', '--max_sequences', type=int, default=None,
                        help='Maximum number of sequences to extract per chunk pair')
    parser.add_argument('-t', '--time_gap', type=float, default=None,
                        help='Time in seconds to skip between sequences (default: 3.0, disabled when using scene detection)')
    args = parser.parse_args()
    
    try:
        # Set time_gap based on scene detection setting
        time_gap = None if args.use_scene_detection else (args.time_gap if args.time_gap is not None else 3.0)
        
        # Initialize frame sequence extractor
        extractor = FrameSequenceExtractor(
            args.chunks_dir, 
            args.output_dir, 
            args.sequence_length,
            args.use_scene_detection, 
            args.max_sequences,
            time_gap
        )
        
        # Extract sequences
        total_sequences = extractor.extract_all_sequences()
        
        logger.info(f"Frame sequence extraction complete. Extracted {total_sequences} sequences.")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
