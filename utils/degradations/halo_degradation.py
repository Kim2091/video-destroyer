import random
from typing import Dict, Any
import logging
from .base_degradation import BaseDegradation

logger = logging.getLogger(__name__)

class HaloDegradation(BaseDegradation):
    """Applies sharpening/halo artifacts to videos using FFmpeg's unsharp filter"""
    
    def __init__(self, config: Dict[str, Any], logger=None):
        super().__init__(config, logger)
        self.params = config.get('params', {})
        
        # Default parameter ranges if not specified in config
        self.luma_x = self.params.get('luma_x_range', (3, 7))  # Luma matrix size
        self.luma_y = self.params.get('luma_y_range', (3, 7))  # Luma matrix size
        self.luma_amount = self.params.get('luma_amount_range', (2.0, 5.0))  # Luma strength
        
        # Store selected parameters for logging
        self.selected_params = {}
        
    @property
    def name(self) -> str:
        return "halo"
    
    def get_params(self) -> Dict[str, Any]:
        """Return the parameters used for this degradation"""
        return self.selected_params
    
    def apply(self, input_path: str, output_path: str) -> str:
        """Direct file processing - not used in pipeline"""
        raise NotImplementedError("Halo degradation only supports piped processing")
    
    def get_filter_expression(self, video_info):
        """
        Generate FFmpeg filter string for halo/sharpening degradation.
        
        Args:
            video_info: Video information from probe (not used in this degradation)
            
        Returns:
            String containing FFmpeg filter chain
        """
        # Select random parameters within configured ranges
        luma_x = random.randint(self.luma_x[0], self.luma_x[1])
        luma_y = random.randint(self.luma_y[0], self.luma_y[1])
        luma_amount = random.uniform(self.luma_amount[0], self.luma_amount[1])
        
        # Ensure matrix sizes are odd numbers (required by unsharp filter)
        luma_x = luma_x if luma_x % 2 == 1 else luma_x + 1
        luma_y = luma_y if luma_y % 2 == 1 else luma_y + 1
        
        # Store selected parameters for logging
        self.selected_params = {
            'luma_x': luma_x,
            'luma_y': luma_y,
            'luma_amount': luma_amount
        }
        
        # Create unsharp filter string
        # Format: unsharp=lx:ly:la:cx:cy:ca
        filter_expr = (
            f"unsharp=lx={luma_x}:ly={luma_y}:la={luma_amount:.3f}"
            ":cx=3:cy=3:ca=0"  # Fixed chroma parameters
        )
        
        return filter_expr