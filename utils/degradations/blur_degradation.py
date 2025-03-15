import random
from typing import Dict, Any, List, Tuple
import logging
from .base_degradation import BaseDegradation

logger = logging.getLogger(__name__)

class BlurDegradation(BaseDegradation):
    """Applies various blur effects to videos using FFmpeg filters"""
    
    # Define available blur types and their parameter ranges
    BLUR_TYPES = {
        'gaussian': {
            'sigma_range': (1.0, 5.0),
            'steps_range': (1, 3)  # Multiple passes for stronger effect
        },
        'box': {
            'radius_range': (1, 10),
            'power_range': (1, 2)  # Multiple passes
        },
        'motion': {
            'frames_range': (2, 5),  # Number of frames to mix
            'angle_range': (0, 360)  # Angle of motion blur
        }
    }
    
    def __init__(self, config: Dict[str, Any], logger=None):
        super().__init__(config, logger)
        self.params = config.get('params', {})
        
        # Get enabled blur types from config or use all by default
        self.enabled_blur_types = self.params.get('enabled_types', list(self.BLUR_TYPES.keys()))
        
        # Override default ranges with config values if provided
        for blur_type, ranges in self.BLUR_TYPES.items():
            # Check if this blur type has a nested config
            if blur_type in self.params:
                blur_config = self.params[blur_type]
                # If it's a dictionary, process the nested parameters
                if isinstance(blur_config, dict):
                    for param, default_range in ranges.items():
                        if param in blur_config:
                            ranges[param] = blur_config[param]
        
        # Store selected parameters for logging
        self.selected_params = {}
        
    @property
    def name(self) -> str:
        return "blur"
    
    def get_params(self) -> Dict[str, Any]:
        """Return the parameters used for this degradation"""
        # If logger is in DEBUG mode, return all selected parameters
        if self.logger and self.logger.logger.level <= logging.DEBUG:
            return self.selected_params
        
        # Otherwise return the standard formatted parameters
        return {
            'kernel_size': self.selected_params.get('kernel_size'),
            'sigma': round(self.selected_params.get('sigma', 0), 2),
            'blur_type': self.selected_params.get('blur_type', 'gaussian')
        }
    
    def _get_gaussian_blur_filter(self, sigma: float, steps: int) -> str:
        """Generate Gaussian blur filter string"""
        # Chain multiple gblur filters for steps
        filters = [f"gblur=sigma={sigma:.2f}" for _ in range(steps)]
        return ','.join(filters)
    
    def _get_box_blur_filter(self, radius: int, power: int) -> str:
        """Generate box blur filter string"""
        # Chain multiple boxblur filters for power
        filters = [
            f"boxblur=luma_radius={radius}:luma_power=1:chroma_radius={radius}:chroma_power=1"
            for _ in range(power)
        ]
        return ','.join(filters)
    
    def _get_motion_blur_filter(self, frames: int, angle: int) -> str:
        """Generate motion blur filter string"""
        # Use tblend filter for temporal blending
        filters = [f"tblend=all_mode=average:all_opacity=0.5" for _ in range(frames - 1)]
        return ','.join(filters)
    
    def apply(self, input_path: str, output_path: str) -> str:
        """Direct file processing - not used in pipeline"""
        raise NotImplementedError("Blur degradation only supports piped processing")
    
    def get_filter_expression(self, video_info):
        """
        Generate FFmpeg filter string for blur effects.
        
        Args:
            video_info: Video information from probe (not used for blur)
            
        Returns:
            String containing FFmpeg filter chain
        """
        # Select random blur type from enabled types
        blur_type = random.choice(self.enabled_blur_types)
        
        # Get parameter ranges for selected blur type
        params = self.BLUR_TYPES[blur_type]
        
        if blur_type == 'gaussian':
            sigma = random.uniform(*params['sigma_range'])
            steps = random.randint(*params['steps_range'])
            
            self.selected_params = {
                'type': 'gaussian',
                'sigma': round(sigma, 2),
                'steps': steps
            }
            
            return self._get_gaussian_blur_filter(sigma, steps)
            
        elif blur_type == 'box':
            radius = random.randint(*params['radius_range'])
            power = random.randint(*params['power_range'])
            
            self.selected_params = {
                'type': 'box',
                'radius': radius,
                'power': power
            }
            
            return self._get_box_blur_filter(radius, power)
            
        else:  # motion blur
            frames = random.randint(*params['frames_range'])
            angle = random.randint(*params['angle_range'])
            
            self.selected_params = {
                'type': 'motion',
                'frames': frames,
                'angle': angle
            }
            
            return self._get_motion_blur_filter(frames, angle)