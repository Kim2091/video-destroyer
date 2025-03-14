import random
from typing import Dict, Any
import logging
from .base_degradation import BaseDegradation

logger = logging.getLogger(__name__)

class ResizeDegradation(BaseDegradation):
    """Applies resolution-based degradation to videos using FFmpeg"""
    
    def __init__(self, config: Dict[str, Any], logger=None):
        super().__init__(config, logger)
        self.params = config.get('params', {})
        self.fixed_scale = self.params.get('fixed_scale', 1.0)
        self.down_up = self.params.get('down_up', {})
        self.scaling_filters = self.params.get('scaling_filters', ['bicubic'])
        
        # Store selected parameters for logging
        self.selected_params = {}
        
    @property
    def name(self) -> str:
        return "resize"
        
    def _select_scaling_filter(self) -> str:
        """Randomly select a scaling filter from available options"""
        return random.choice(self.scaling_filters)
        
    def _select_down_up_scale(self) -> float:
        """Select random intermediate scale factor for down-up scaling"""
        if not self.down_up.get('enabled', False):
            return self.fixed_scale
            
        scale_range = self.down_up.get('range', {})
        min_scale = scale_range.get('min', 0.15)
        max_scale = scale_range.get('max', 0.8)
        
        # Ensure intermediate scale is not larger than target scale
        max_scale = min(max_scale, self.fixed_scale)
        return random.uniform(min_scale, max_scale)
    
    def get_params(self) -> Dict[str, Any]:
        """Return the parameters used for this degradation"""
        params = {
            'scale': self.fixed_scale,
            'down_filter': self.selected_params['down_filter']
        }
        
        if self.down_up.get('enabled', False):
            params.update({
                'down_up': 'enabled',
                'intermediate_scale': round(self.selected_params['intermediate_scale'], 3),
                'up_filter': self.selected_params['up_filter']
            })
        else:
            params['down_up'] = 'disabled'
            
        return params
    
    def apply(self, input_path: str, output_path: str) -> str:
        """Direct file processing - not used in pipeline"""
        raise NotImplementedError("Resize degradation only supports piped processing")
    
    def apply_piped(self, input_stream, video_info=None):
        """
        Apply resize degradation to video stream using FFmpeg.
        
        Args:
            input_stream: FFmpeg input stream
            video_info: Video information from probe
            
        Returns:
            FFmpeg output stream with resize filters applied
        """
        if not video_info:
            raise ValueError("Video info required for resize degradation")
            
        # Get original dimensions
        width = int(video_info['width'])
        height = int(video_info['height'])
        
        # Select parameters
        down_filter = self._select_scaling_filter()
        up_filter = self._select_scaling_filter()
        intermediate_scale = self._select_down_up_scale()
        
        # Store selected parameters for logging
        self.selected_params = {
            'fixed_scale': self.fixed_scale,
            'down_up_enabled': self.down_up.get('enabled', False),
            'down_filter': down_filter,
            'up_filter': up_filter if self.down_up.get('enabled', False) else None,
            'intermediate_scale': intermediate_scale if self.down_up.get('enabled', False) else None
        }
        
        # Calculate target dimensions
        target_width = int(width * self.fixed_scale)
        target_height = int(height * self.fixed_scale)
        
        if self.down_up.get('enabled', False):
            # Calculate intermediate dimensions
            inter_width = int(width * intermediate_scale)
            inter_height = int(height * intermediate_scale)
            
            # Apply down-up scaling
            stream = (
                input_stream
                .filter('scale', inter_width, inter_height, flags=down_filter)
                .filter('scale', target_width, target_height, flags=up_filter)
            )
        else:
            # Apply direct scaling
            stream = input_stream.filter(
                'scale', target_width, target_height, flags=down_filter
            )
            
        return stream