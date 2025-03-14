import ffmpeg
import logging
from typing import Dict, Any
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
        codec, quality = self.codec_handler.get_random_encoding_config()
        return {
            "codec": codec,
            "quality": quality,
            "codec_probabilities": {
                name: config['probability']
                for name, config in self.config['params'].items()
            }
        }
    
    def apply(self, input_path: str, output_path: str) -> str:
        # Get video info
        probe = ffmpeg.probe(input_path)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        fps = float(video_stream['r_frame_rate'].split('/')[0])
        pix_fmt = video_stream['pix_fmt']
        gop_size = int(fps * 2)
        
        # Select random codec and quality
        codec, quality = self.codec_handler.get_random_encoding_config()
        
        # Common parameters
        common_params = {
            'fps_mode': 'cfr',
            'pix_fmt': pix_fmt,
            'g': gop_size
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
        
        output_params = {**common_params, **codec_params[codec]}
        
        # Process the video
        (
            ffmpeg
            .input(input_path)
            .output(output_path, **output_params)
            .run(capture_stdout=True, capture_stderr=True, overwrite_output=True)
        )
        
        return output_path