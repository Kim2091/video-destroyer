import logging
import ffmpeg
from typing import Dict, Any, Tuple
from .base_degradation import BaseDegradation
from utils.codec_handler import CodecHandler

logger = logging.getLogger(__name__)

class CodecDegradation(BaseDegradation):
    def __init__(self, config: Dict[str, Any], logger=None, codec_handler=None):
        super().__init__(config, logger)
        # Use provided codec_handler or create new one
        self.codec_handler = codec_handler or CodecHandler(config['params'])
        self.selected_params = {}
        
    @property
    def name(self) -> str:
        return "codec"
    
    def get_params(self) -> Dict[str, Any]:
        """Return the parameters used for this degradation"""
        # If logger is in DEBUG mode, return all selected parameters
        if self.logger and self.logger.logger.level <= logging.DEBUG:
            return self.selected_params
        return {
            "codec": self.selected_params.get("codec"),
            "quality": self.selected_params.get("quality"),
            "probability": self.probability
        }
    
    def get_codec_params(self) -> Dict[str, Any]:
        """
        Get codec-specific parameters for encoding.
        This is called by the pipeline after should_apply() returns True.
        
        Returns:
            Dictionary of encoding parameters
        """
        # Select random codec and quality if not already selected
        if not self.selected_params:
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
            'h264': {'vcodec': 'libx264', 'crf': self.selected_params['quality'], 'preset': 'medium'},
            'h265': {'vcodec': 'libx265', 'crf': self.selected_params['quality'], 'preset': 'medium'},
            'vp9': {'vcodec': 'libvpx-vp9', 'crf': self.selected_params['quality'], 'b': 0},
            'av1': {'vcodec': 'libsvtav1', 'crf': self.selected_params['quality'], 'preset': 7},
            'mpeg2': {'vcodec': 'mpeg2video', 'qscale': self.selected_params['quality']}
        }
        
        return {**common_params, **codec_params[self.selected_params['codec']]}
    
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
        
        # Select random codec and quality
        codec, quality = self.codec_handler.get_random_encoding_config()
        self.selected_params = {
            "codec": codec,
            "quality": quality
        }
        
        # Get encoding parameters
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