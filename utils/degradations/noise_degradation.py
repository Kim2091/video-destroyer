import random
from typing import Dict, Any
import logging
from .base_degradation import BaseDegradation

logger = logging.getLogger(__name__)

class NoiseDegradation(BaseDegradation):
    """Applies noise degradation to videos using FFmpeg's native noise filter"""
    
    def __init__(self, config: Dict[str, Any], logger=None):
        super().__init__(config, logger)
        self.params = config.get('params', {})
        
        # Store selected parameters for this instance
        self.selected_params = {}
        
    @property
    def name(self) -> str:
        """Return the name of the degradation"""
        return "noise"
        
    def _select_noise_params(self) -> Dict[str, float]:
        """Select random noise parameters based on configuration"""
        # Get ranges for Y and UV noise strength
        y_range = self.params.get('y_strength_range', [1, 10])
        uv_range = self.params.get('uv_strength_range', [1, 10])
        
        # Validate ranges
        if not isinstance(y_range, list) or len(y_range) != 2:
            y_range = [1, 10]
        if not isinstance(uv_range, list) or len(uv_range) != 2:
            uv_range = [1, 10]
            
        # Select random strengths from ranges
        y_strength = random.uniform(y_range[0], y_range[1])
        uv_strength = random.uniform(uv_range[0], uv_range[1])
        
        # Select noise type
        noise_types = self.params.get('types', ['u'])  # u=uniform, t=temporal, a=averaged temporal
        noise_type = random.choice(noise_types)
        
        return {
            'y_strength': y_strength,
            'uv_strength': uv_strength,
            'type': noise_type
        }
    
    def get_params(self) -> Dict[str, Any]:
        """Return the parameters used for this degradation"""
        if self.logger and self.logger.logger.level <= logging.DEBUG:
            return self.selected_params
            
        # Return simplified params for non-debug mode
        return {
            'y_strength': round(self.selected_params.get('y_strength', 0), 2),
            'uv_strength': round(self.selected_params.get('uv_strength', 0), 2),
            'type': self.selected_params.get('type', 'u')
        }
    
    def apply(self, input_path: str, output_path: str) -> str:
        """Direct file processing - not used in pipeline"""
        raise NotImplementedError("Noise degradation only supports piped processing")
    
    def get_filter_expression(self, video_info):
        """
        Generate FFmpeg filter string for noise degradation.
        
        Args:
            video_info: Video information from probe
            
        Returns:
            String containing FFmpeg filter chain
        """
        if not video_info:
            raise ValueError("Video info required for noise degradation")
            
        # Select parameters for this instance
        params = self._select_noise_params()
        self.selected_params = params
        
        # Build the noise filter string
        # Use frame-based expressions for seeds to ensure per-frame randomization
        filter_expr = (
            f"noise=c0_seed=random(1)*100000:c0_strength={params['y_strength']}:"
            f"c1_seed=random(2)*100000:c1_strength={params['uv_strength']}:"
            f"c2_seed=random(3)*100000:c2_strength={params['uv_strength']}:"
            f"all_seed=random(4)*100000:allf={params['type']}"
        )
            
        return filter_expr
