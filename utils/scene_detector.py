import logging
import yaml
from typing import Dict, Any
import ffmpeg
from scenedetect import SceneManager, open_video, ContentDetector

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
        
        logger.info(f"Initialized SceneDetector with threshold={self.threshold}")
        
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
    
    def detect_scenes(self, video_path: str):
        """
        Detect scene changes in a video using PySceneDetect.
        
        Args:
            video_path: Path to the video
            
        Returns:
            List of scene boundaries
        """
        logger.info("Detecting scenes...")

        # Open video with the new API
        video = open_video(video_path)

        # Create scene manager and add detector
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=self.threshold))

        # Detect scenes
        scene_manager.detect_scenes(video)

        # Get list of scenes
        scene_list = scene_manager.get_scene_list()  # Add this line
        logger.info(f"Detected {len(scene_list)} scenes")

        # Print scene information with simpler format
        for i, scene in enumerate(scene_list):
            start_frame = scene[0].get_frames()
            end_frame = scene[1].get_frames()
            logger.debug(f"Scene {i}: frames {start_frame} to {end_frame} (length: {end_frame - start_frame})")

        return scene_list
