import logging
import ffmpeg
from typing import Dict, Any, Tuple
from .base_degradation import BaseDegradation
from utils.codec_handler import CodecHandler

logger = logging.getLogger(__name__)

class CodecDegradation(BaseDegradation):
    def __init__(self, config: Dict[str, Any], logger=None, codec_handler=None):
        super().__init__(config, logger)
        # Use provided codec_handler or create new one from the codec config
        self.codec_handler = codec_handler or CodecHandler(config.get('params', {}))
        self.selected_params = None  # Initialize to None instead of empty dict

            
    @property
    def name(self) -> str:
        return "codec"
    
    def get_params(self) -> Dict[str, Any]:
        """Return the parameters used for this degradation"""
        if not self.selected_params:
            return None
            
        # If logger is in DEBUG mode, return all selected parameters
        if self.logger and self.logger.logger.level <= logging.DEBUG:
            return self.selected_params
            
        # Otherwise return the standard formatted parameters
        return {
            "codec": self.selected_params["codec"],
            "quality": self.selected_params["quality"]
        }
    
    def get_codec_params(self) -> Dict[str, Any]:
        """
        Get codec-specific parameters for encoding.
        This is called by the pipeline after should_apply() returns True.
        
        Returns:
            Dictionary of encoding parameters
        """
        # Select random codec and quality
        codec, quality = self.codec_handler.get_random_encoding_config()
        self.selected_params = {
            "codec": codec,
            "quality": quality
        }
        
        # Common parameters
        common_params = {
            'fps_mode': 'cfr',
            'pix_fmt': 'yuv420p',  # Default, will be overridden if needed
            'g': 60,  # Default GOP size
            'loglevel': 'error',
            'hide_banner': None,
            'colorspace': 'bt709'
        }
        
        # Codec-specific parameters
        codec_params = {
            'h264': {'vcodec': 'libx264', 'crf': quality, 'preset': 'medium'},
            'h265': {'vcodec': 'libx265', 'crf': quality, 'preset': 'medium'},
            'vp9': {'vcodec': 'libvpx-vp9', 'crf': quality, 'b': 0},
            'av1': {'vcodec': 'libsvtav1', 'crf': quality, 'preset': 7},
            'mpeg2': {'vcodec': 'mpeg2video', 'b:v': f'{quality}k'}
        }
        
        return {**common_params, **codec_params[codec]}
    
    def get_filter_expression(self, video_info):
        """
        Codec degradation doesn't use filter expressions as it's applied at output.
        This method exists for API compatibility.
        """
        return None
    
    def apply(self, input_path: str, output_path: str) -> str:
        """
        Direct file processing - kept for standalone usage
        """
        # Get video info
        probe = ffmpeg.probe(input_path)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)

        # Get encoding parameters (this will handle codec selection internally)
        output_params = self.get_codec_params()
        
        # Update GOP size based on video info if available
        if video_stream:
            fps = float(video_stream['r_frame_rate'].split('/')[0])
            output_params['g'] = int(fps * 2)
            output_params['pix_fmt'] = video_stream['pix_fmt']

        # Process the video
        (
            ffmpeg
            .input(input_path)
            .output(output_path, **output_params)
            .global_args('-hide_banner', '-loglevel', 'error', '-nostats')
            .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
        )
        
        return output_path
