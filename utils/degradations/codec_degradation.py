import ffmpeg
import logging
from typing import Dict, Any, Tuple
from .base_degradation import BaseDegradation
from utils.codec_handler import CodecHandler

logger = logging.getLogger(__name__)

class CodecDegradation(BaseDegradation):
    """Applies codec-based degradation to videos"""
    
    def __init__(self, config: Dict[str, Any], logger=None):
        super().__init__(config, logger)
        # Create a wrapper config that matches what CodecHandler expects
        codec_config = {'codecs': config['params']}
        self.codec_handler = CodecHandler(codec_config['codecs'])
        
    @property
    def name(self) -> str:
        return "codec"
    
    def get_params(self) -> Dict[str, Any]:
        """Return the parameters used for this degradation"""
        return {
            "codec": self.selected_params.get("codec"),
            "quality": self.selected_params.get("quality")
        }
    
    def get_codec_params(self, codec: str, quality: int, video_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get codec-specific parameters for encoding.
        
        Args:
            codec: Codec name (h264, h265, etc.)
            quality: Quality value for the codec
            video_info: Optional video information from probe
            
        Returns:
            Dictionary of encoding parameters
        """
        # Default values if video_info is not provided
        fps = 30
        pix_fmt = 'yuv420p'
        gop_size = 60
        
        # Extract video info if provided
        if video_info:
            fps = float(video_info['r_frame_rate'].split('/')[0])
            pix_fmt = video_info['pix_fmt']
            gop_size = int(fps * 2)
        
        # Common parameters
        common_params = {
            'fps_mode': 'cfr',
            'pix_fmt': pix_fmt,
            'g': gop_size,
            'loglevel': 'error',  # Add this
            'hide_banner': None   # Add this
        }
        
        # Codec-specific parameters
        codec_params = {
            'h264': {'vcodec': 'libx264', 'crf': quality, 'preset': 'medium'},
            'h265': {'vcodec': 'libx265', 'crf': quality, 'preset': 'medium'},
            'vp9': {'vcodec': 'libvpx-vp9', 'crf': quality, 'b': 0},
            'av1': {'vcodec': 'libsvtav1', 'crf': quality, 'preset': 7},
            'mpeg1': {'vcodec': 'mpeg1video', 'qscale': quality},
            'mpeg2': {'vcodec': 'mpeg2video', 'qscale': quality}
        }
        
        return {**common_params, **codec_params[codec]}
    
    def apply(self, input_path: str, output_path: str) -> str:
        """Apply codec degradation using file paths"""
        # Get video info
        probe = ffmpeg.probe(input_path)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        
        # Select random codec and quality
        codec, quality = self.codec_handler.get_random_encoding_config()
        
        # Get encoding parameters
        output_params = self.get_codec_params(codec, quality, video_stream)
        
        # Process the video
        (
            ffmpeg
            .input(input_path)
            .output(output_path, **output_params)
            .global_args('-hide_banner', '-loglevel', 'error', '-nostats')
            .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
        )
        
        return output_path
    
    def apply_piped(self, input_stream, video_info=None):
        """
        Apply codec degradation to a video stream and output to a file.
        
        Args:
            input_stream: FFmpeg input stream
            video_info: Optional video information from probe
            
        Returns:
            Tuple[stream, params]: The input stream and codec parameters
        """
        # Select random codec and quality ONCE
        codec, quality = self.codec_handler.get_random_encoding_config()
        
        # Store selected parameters for logging first
        self.selected_params = {
            "codec": codec,
            "quality": quality
        }
        
        # Get encoding parameters
        params = self.get_codec_params(codec, quality, video_info)
        
        # Return the stream and parameters separately
        return input_stream, params