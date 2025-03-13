import os
import logging
from typing import List, Dict, Any, Optional, Tuple

import ffmpeg
import scenedetect
from scenedetect import SceneManager, StatsManager
from scenedetect.detectors import ContentDetector, ThresholdDetector, AdaptiveDetector
from scenedetect.video_splitter import split_video_ffmpeg

logger = logging.getLogger(__name__)

class SceneDetector:
    """
    Class for detecting scene changes in videos using PySceneDetect.
    """
    
    def __init__(self, threshold: float = 32.0, min_scene_length: float = 0.8, method: str = 'content',
                 content_weights: List[float] = None, split_preset: str = 'slow', rate_factor: int = 17):
        """
        Initialize the scene detector.
        
        Args:
            threshold: Threshold for scene detection (interpretation depends on method)
            min_scene_length: Minimum length of a scene in seconds (default: 0.8s)
            method: Detection method ('content', 'threshold', or 'adaptive')
            content_weights: Weights for content detection algorithm [Y, Cb, Cr, delta_hue]
            split_preset: FFmpeg preset for splitting (fast, medium, slow)
            rate_factor: Quality factor for split video encoding (CRF/RF value)
        """
        self.threshold = threshold
        self.min_scene_length = min_scene_length
        self.method = method.lower()
        self.content_weights = content_weights or [1.0, 0.5, 1.0, 0.2]  # Default weights
        self.split_preset = split_preset
        self.rate_factor = rate_factor
        
        # Validate method
        valid_methods = ['content', 'threshold', 'adaptive']
        if self.method not in valid_methods:
            logger.warning(f"Invalid method '{method}'. Using 'content' instead.")
            self.method = 'content'
            
        logger.info(f"Initialized SceneDetector with threshold={threshold}, min_scene_length={min_scene_length}s, "
                   f"method={self.method}, content_weights={self.content_weights}, "
                   f"split_preset={self.split_preset}, rate_factor={self.rate_factor}")
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """
        Get video information using ffmpeg.probe.
        
        Args:
            video_path: Path to the video
            
        Returns:
            Dictionary with video information
        """
        try:
            probe = ffmpeg.probe(video_path)
            video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
            
            if video_stream is None:
                raise ValueError("No video stream found")
            
            # Get duration (in seconds)
            duration = float(probe['format']['duration'])
            
            # Get framerate
            fps_parts = video_stream.get('r_frame_rate', '').split('/')
            if len(fps_parts) == 2 and int(fps_parts[1]) != 0:
                fps = float(int(fps_parts[0]) / int(fps_parts[1]))
            else:
                fps = float(video_stream.get('avg_frame_rate', '30/1').split('/')[0])
            
            # Get resolution
            width = int(video_stream['width'])
            height = int(video_stream['height'])
            
            # Get pixel format
            pix_fmt = video_stream.get('pix_fmt', 'yuv420p')
            
            return {
                'duration': duration,
                'fps': fps,
                'width': width,
                'height': height,
                'pix_fmt': pix_fmt
            }
        
        except Exception as e:
            logger.error(f"Error getting video info: {str(e)}")
            # Return default values if probe fails
            return {
                'duration': 0,
                'fps': 30,
                'width': 1920,
                'height': 1080,
                'pix_fmt': 'yuv420p'
            }
    
    def detect_scenes(self, video_path: str) -> List[float]:
        """
        Detect scene changes in a video using PySceneDetect.
        
        Args:
            video_path: Path to the video
            
        Returns:
            List of timestamps (in seconds) where scene changes occur
        """
        logger.info(f"Detecting scenes in video: {video_path} using PySceneDetect with method: {self.method}")
        
        try:
            # Open the video using the recommended backend
            video = scenedetect.open_video(video_path)
            
            # Get the framerate for calculating min_scene_len
            fps = video.frame_rate
            min_frames = int(self.min_scene_length * fps)
            
            # Create a SceneManager and StatsManager
            stats_manager = StatsManager()
            scene_manager = SceneManager(stats_manager)
            
            # Add the appropriate detector based on the method
            if self.method == 'content':
                # ContentDetector is good for most videos with fast cuts
                detector = ContentDetector(
                    threshold=self.threshold, 
                    min_scene_len=min_frames,
                    weights=self.content_weights
                )
            elif self.method == 'threshold':
                # ThresholdDetector is good for videos with fades in/out to black
                detector = ThresholdDetector(
                    threshold=self.threshold, 
                    min_scene_len=min_frames
                )
            elif self.method == 'adaptive':
                # AdaptiveDetector is good for videos with varying lighting conditions
                detector = AdaptiveDetector(
                    min_scene_len=min_frames,
                    adaptive_threshold=self.threshold
                )
            
            # Add the detector to the SceneManager
            scene_manager.add_detector(detector)
            
            # Detect all scenes
            scene_manager.detect_scenes(video)
            
            # Get the scene list
            scene_list = scene_manager.get_scene_list()
            
            # Convert scene list to timestamps
            scene_times = [0.0]  # Start with 0.0 for the beginning of the video
            
            for scene in scene_list:
                # Convert frame number to timestamp
                scene_time = scene[0].get_seconds()
                scene_times.append(scene_time)
            
            # Get video duration and add it as the last scene time if needed
            video_info = self.get_video_info(video_path)
            if not scene_times or scene_times[-1] < video_info['duration'] - 1.0:
                scene_times.append(video_info['duration'])
            
            logger.info(f"Detected {len(scene_times)-1} scene changes")
            
            # Log the scene boundaries for debugging
            for i in range(len(scene_times) - 1):
                start = scene_times[i]
                end = scene_times[i+1]
                duration = end - start
                logger.info(f"Scene {i+1}: start={start:.2f}s, end={end:.2f}s, duration={duration:.2f}s")
            
            return scene_times
            
        except Exception as e:
            logger.error(f"Error in PySceneDetect: {str(e)}")
            logger.warning("Falling back to time-based segmentation")
            return self.generate_time_based_scenes(video_path)
    
    def split_video(self, video_path: str, output_dir: str, high_quality: bool = True) -> List[str]:
        """
        Split a video into chunks at scene boundaries using PySceneDetect's built-in splitter.
        
        Args:
            video_path: Path to the video
            output_dir: Directory to save the chunks
            high_quality: Whether to use high-quality mode for splitting
            
        Returns:
            List of paths to the created chunks
        """
        logger.info(f"Splitting video: {video_path} into chunks using PySceneDetect")
        
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Open the video
            video = scenedetect.open_video(video_path)
            
            # Get the framerate for calculating min_scene_len
            fps = video.frame_rate
            min_frames = int(self.min_scene_length * fps)
            
            # Create a SceneManager and StatsManager
            stats_manager = StatsManager()
            scene_manager = SceneManager(stats_manager)
            
            # Add the appropriate detector based on the method
            if self.method == 'content':
                detector = ContentDetector(
                    threshold=self.threshold, 
                    min_scene_len=min_frames,
                    weights=self.content_weights
                )
            elif self.method == 'threshold':
                detector = ThresholdDetector(
                    threshold=self.threshold, 
                    min_scene_len=min_frames
                )
            elif self.method == 'adaptive':
                detector = AdaptiveDetector(
                    min_scene_len=min_frames,
                    adaptive_threshold=self.threshold
                )
            
            # Add the detector to the SceneManager
            scene_manager.add_detector(detector)
            
            # Detect all scenes
            scene_manager.detect_scenes(video)
            
            # Get the scene list
            scene_list = scene_manager.get_scene_list()
            
            if not scene_list:
                logger.warning("No scenes detected. Falling back to time-based segmentation.")
                return self.split_video_by_duration(video_path, output_dir)
            
            # Get base filename without extension
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            
            # Split the video using PySceneDetect's built-in function
            output_file_template = os.path.join(output_dir, f"chunk_%04d.mp4")
            
            # Configure FFmpeg arguments for splitting
            ffmpeg_args = [
                '-preset', self.split_preset,
                '-crf', str(self.rate_factor)
            ]
            
            # Use high quality mode if specified
            split_video_ffmpeg(
                video_path, 
                scene_list, 
                output_file_template,
                video_name=base_name,
                high_quality=high_quality,
                show_progress=True,
                additional_ffmpeg_args=ffmpeg_args
            )
            
            # Get the list of created files
            chunk_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) 
                          if f.startswith("chunk_") and f.endswith(".mp4")]
            chunk_files.sort()
            
            logger.info(f"Created {len(chunk_files)} video chunks")
            
            return chunk_files
            
        except Exception as e:
            logger.error(f"Error splitting video: {str(e)}")
            # Fall back to duration-based splitting if PySceneDetect's splitter fails
            logger.warning("Falling back to duration-based video splitting")
            return self.split_video_by_duration(video_path, output_dir)
    
    def split_video_by_duration(self, video_path: str, output_dir: str) -> List[str]:
        """
        Split a video into chunks of equal duration.
        
        Args:
            video_path: Path to the video
            output_dir: Directory to save the chunks
            
        Returns:
            List of paths to the created chunks
        """
        logger.info(f"Splitting video by duration: {video_path}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate time-based scenes
        scene_times = self.generate_time_based_scenes(video_path)
        
        # Get video information
        video_info = self.get_video_info(video_path)
        fps = video_info['fps']
        
        # Convert time-based scenes to frame ranges for PySceneDetect
        scene_list = []
        for i in range(len(scene_times) - 1):
            start_frame = int(scene_times[i] * fps)
            end_frame = int(scene_times[i + 1] * fps)
            scene_list.append((
                scenedetect.FrameTimecode(start_frame, fps),
                scenedetect.FrameTimecode(end_frame, fps)
            ))
        
        # Get base filename without extension
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Split the video using PySceneDetect's built-in function
        output_file_template = os.path.join(output_dir, f"chunk_%04d.mp4")
        
        try:
            # Configure FFmpeg arguments for splitting
            ffmpeg_args = [
                '-preset', self.split_preset,
                '-crf', str(self.rate_factor)
            ]
            
            # Use high quality mode for better frame accuracy
            split_video_ffmpeg(
                video_path, 
                scene_list, 
                output_file_template,
                video_name=base_name,
                high_quality=True,
                show_progress=True,
                additional_ffmpeg_args=ffmpeg_args
            )
            
            # Get the list of created files
            chunk_files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) 
                          if f.startswith("chunk_") and f.endswith(".mp4")]
            chunk_files.sort()
            
            logger.info(f"Created {len(chunk_files)} video chunks by duration")
            
            return chunk_files
            
        except Exception as e:
            logger.error(f"Error splitting video by duration: {str(e)}")
            return []
    
    def generate_time_based_scenes(self, video_path: str) -> List[float]:
        """
        Generate time-based scenes if scene detection fails.
        
        Args:
            video_path: Path to the video
            
        Returns:
            List of timestamps (in seconds) for time-based segmentation
        """
        logger.info(f"Generating time-based scenes for video: {video_path}")
        
        try:
            # Get video duration
            video_info = self.get_video_info(video_path)
            duration = video_info['duration']
            
            # Generate scenes at regular intervals (use min_scene_length as the interval)
            scene_interval = max(5.0, self.min_scene_length)
            scene_times = [0.0]  # Start with 0.0
            
            current_time = scene_interval
            while current_time < duration:
                scene_times.append(current_time)
                current_time += scene_interval
            
            # Add the end time
            if scene_times[-1] < duration - 1.0:
                scene_times.append(duration)
            
            logger.info(f"Generated {len(scene_times)-1} time-based scenes")
            
            # Log the scene boundaries for debugging
            for i in range(len(scene_times) - 1):
                start = scene_times[i]
                end = scene_times[i+1]
                duration = end - start
                logger.info(f"Scene {i+1}: start={start:.2f}s, end={end:.2f}s, duration={duration:.2f}s")
            
            return scene_times
            
        except Exception as e:
            logger.error(f"Error generating time-based scenes: {str(e)}")
            # Return a simple split if all else fails
            return [0.0, video_info.get('duration', 60.0)]