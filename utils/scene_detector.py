import logging
from typing import Dict, Any, Callable, Optional
import yaml
import ffmpeg
from scenedetect import SceneManager, open_video, ContentDetector, FrameTimecode
import numpy

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class SceneDetector:
    """
    Class for detecting scene changes in videos using PySceneDetect.
    """
    
    def __init__(self, config_path='config.yaml', config=None):
        """
        Initialize the scene detector from config file or config object.
        
        Args:
            config_path: Path to the configuration file
            config: Configuration dictionary (alternative to config_path)
        """
        # Load configuration from either config object or file
        if config is not None:
            self.config = config
        else:
            # Load configuration from file
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        
        # Get scene detection settings
        scene_config = self.config.get('scene_detection', {})
        
        # Content detector parameters
        self.threshold = scene_config.get('threshold', 27.0)
        # Downscale factor for faster processing (0 = auto, 1 = disabled)
        self.downscale_factor = scene_config.get('downscale_factor', 0)
        # Maximum number of scenes to return (0 = no limit)
        self.max_scenes = scene_config.get('max_scenes', 0)
        
        logger.debug(f"Initialized SceneDetector with threshold={self.threshold}, downscale_factor={self.downscale_factor}, max_scenes={self.max_scenes}")
        
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
                # Fallback for variable frame rate or other formats
                avg_fps_str = video_stream.get('avg_frame_rate', '30/1')
                avg_fps_parts = avg_fps_str.split('/')
                if len(avg_fps_parts) == 2 and int(avg_fps_parts[1]) != 0:
                     fps = float(int(avg_fps_parts[0]) / int(avg_fps_parts[1]))
                else:
                     fps = 30.0 # Default if parsing fails
                     logger.warning(f"Could not parse FPS string: {avg_fps_str}. Using default 30.0")
            
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
            logger.error(f"Error getting video info for {video_path}: {str(e)}")
            # Return default values if probe fails
            return {
                'duration': 0,
                'fps': 30,
                'width': 1920,
                'height': 1080,
                'pix_fmt': 'yuv420p'
            }
    
    def detect_scenes(self, video_path: str):
        """
        Detect scene changes in a video using PySceneDetect.
        
        Args:
            video_path: Path to the video
            
        Returns:
            List of scene boundaries [(start_timecode, end_timecode)]
        """
        try:
            # Open video with the new API
            video = open_video(video_path)

            # Create scene manager
            scene_manager = SceneManager()
            # Set the downscale factor on the SceneManager instance
            # Explicitly disable auto_downscale to ensure manual factor is used
            scene_manager.auto_downscale = False
            scene_manager.downscale = self.downscale_factor 
            # Add the detector
            scene_manager.add_detector(ContentDetector(threshold=self.threshold))

            # --- Callback logic to stop detection early --- 
            detection_stopped_early = False
            detected_scene_count = 0  # Counter for detected scenes
            def scene_limit_callback(frame_img: numpy.ndarray, frame_num: int):
                nonlocal detection_stopped_early, detected_scene_count
                detected_scene_count += 1 # Increment counter on each detected scene event
                # Check if counter reached the limit
                if detected_scene_count >= self.max_scenes:
                    if not detection_stopped_early: # Prevent multiple stop calls/logs
                        logger.info(f"Scene limit ({self.max_scenes}) reached via callback count. Stopping detection early.")
                        scene_manager.stop()
                        detection_stopped_early = True
            # --- End callback logic ---

            effective_callback: Optional[Callable[[numpy.ndarray, int], None]] = None
            if self.max_scenes > 0:
                effective_callback = scene_limit_callback
            
            # A factor of 2 or higher downscales by that amount (e.g., 2 means 1/2 width & height).
            log_limit = f"limit: {self.max_scenes}" if self.max_scenes > 0 else "no limit"
            logger.info(f"Starting scene detection for {video_path} (downscale: {self.downscale_factor}, {log_limit})...")
            # Detect scenes, passing the callback if a limit is set
            # Disable progress bar if using the limit callback, as it might cause errors on stop()
            progress = False if self.max_scenes > 0 else True
            scene_manager.detect_scenes(
                video,
                callback=effective_callback,
                show_progress=progress
            )

            # Get the final list (might be shorter if stopped early)
            scene_list = scene_manager.get_scene_list()
            
            if detection_stopped_early:
                 logger.info(f"Detection stopped early. Found {len(scene_list)} scenes.")
            else:
                 logger.info(f"Detection completed. Found {len(scene_list)} scenes.")

            # Debug log scene information (after potential truncation)
            if logger.level <= logging.DEBUG:
                for i, scene in enumerate(scene_list):
                    start_frame = scene[0].get_frames()
                    end_frame = scene[1].get_frames()
                    logger.debug(f"Scene {i+1}: frames {start_frame} to {end_frame} (length: {end_frame - start_frame})")

            return scene_list

        except Exception as e:
            logger.error(f"Error during scene detection for {video_path}: {str(e)}")
            raise # Re-raise the exception after logging
