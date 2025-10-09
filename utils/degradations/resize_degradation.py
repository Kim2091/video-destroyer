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
        # Add selected_params initialization
        self.selected_params = {}
        
    @property
    def name(self) -> str:
        """Return the name of the degradation"""
        return "resize"
        
    def _select_scaling_filter(self) -> str:
        """Randomly select a scaling filter from available options"""
        return random.choice(self.scaling_filters)
        
    def _select_down_up_scale(self) -> float:
        """Select random intermediate scale factor for down-up scaling"""
        if not self.down_up.get('enabled', False):
            return self.fixed_scale
            
        scale_range = self.down_up.get('range', [0.15, 0.8])
        if not isinstance(scale_range, list) or len(scale_range) != 2:
            scale_range = [0.15, 0.8]  # Default range if invalid
            
        min_scale, max_scale = scale_range
        return random.uniform(min_scale, max_scale)
    
    def _should_apply_down_up(self) -> bool:
        """Determine if down-up scaling should be applied based on probability"""
        if not self.down_up.get('enabled', False):
            return False
        
        down_up_prob = self.down_up.get('probability', 1.0)
        return random.random() < down_up_prob
    
    def get_params(self) -> Dict[str, Any]:
        """Return the parameters used for this degradation"""
        # If logger is in DEBUG mode, return all selected parameters
        if self.logger and self.logger.logger.level <= logging.DEBUG:
            return self.selected_params

        params = {
            'scale': self.fixed_scale,
            'down_filter': self.selected_params['down_filter']
        }
        
        if self.selected_params.get('down_up_applied', False):
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
    
    def get_filter_expression(self, video_info):
        """
        Generate FFmpeg filter string for resize degradation.
        
        Args:
            video_info: Video information from probe
            
        Returns:
            String containing FFmpeg filter chain
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
        
        # Determine if down-up should be applied
        apply_down_up = self._should_apply_down_up()
        
        # Store selected parameters for logging
        self.selected_params = {
            'down_filter': down_filter,
            'up_filter': up_filter if apply_down_up else None,
            'intermediate_scale': intermediate_scale if apply_down_up else None,
            'down_up_applied': apply_down_up
        }
        
        # Calculate target dimensions
        target_width = int(width * self.fixed_scale)
        target_height = int(height * self.fixed_scale)
        
        if apply_down_up:
            # Calculate intermediate dimensions
            inter_width = int(width * intermediate_scale)
            inter_height = int(height * intermediate_scale)
            
            # Calculate target dimensions (using fixed_scale)
            target_width = int(width * self.fixed_scale)
            target_height = int(height * self.fixed_scale)
            
            # Create filter string for down-up scaling
            filter_expr = (
                f"scale={inter_width}:{inter_height}:flags={down_filter},"
                f"scale={target_width}:{target_height}:flags={up_filter}"
            )
        else:
            # Create filter string for direct scaling
            filter_expr = f"scale={target_width}:{target_height}:flags={down_filter}"
            
        return filter_expr