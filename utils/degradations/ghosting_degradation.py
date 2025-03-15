import random
from typing import Dict, Any, List, Tuple
import logging
from .base_degradation import BaseDegradation

logger = logging.getLogger(__name__)

class GhostingDegradation(BaseDegradation):
    """Applies ghosting/trailing artifacts to videos using FFmpeg filters"""
    
    def __init__(self, config: Dict[str, Any], logger=None):
        super().__init__(config, logger)
        self.params = config.get('params', {})
        
        # Ghost trail parameters - adjusted for DVD-like artifacts
        self.num_ghosts_range = self.params.get('num_ghosts_range', (1, 2))  # Usually just 1-2 ghosts
        self.opacity_range = self.params.get('opacity_range', (0.2, 0.4))    # More visible than before
        self.delay_range = self.params.get('delay_range', (1, 2))            # Immediate trailing
        
        # Ghost offset parameters - DVD ghosting was mostly vertical
        self.offset_x_range = self.params.get('offset_x_range', (-1, 1))     # Minimal horizontal offset
        self.offset_y_range = self.params.get('offset_y_range', (-2, 2))     # Mainly vertical offset
        
        # Ghost color shift - DVDs often had chroma bleeding
        self.enable_color_shift = self.params.get('enable_color_shift', True)
        self.color_shift_range = self.params.get('color_shift_range', (0.85, 1.15))  # More pronounced color shift
        
        self.selected_params = {}
        
    @property
    def name(self) -> str:
        return "ghosting"
    
    def get_params(self) -> Dict[str, Any]:
        """Return the parameters used for this degradation"""
        # If logger is in DEBUG mode, return all selected parameters
        if self.logger and self.logger.logger.level <= logging.DEBUG:
            return self.selected_params
            
        return {
            'offset_x': self.selected_params.get('offset_x'),
            'offset_y': self.selected_params.get('offset_y'),
            'opacity': round(self.selected_params.get('opacity', 0), 2),
            'blend_mode': self.selected_params.get('blend_mode')
        }
    
    def apply(self, input_path: str, output_path: str) -> str:
        """Direct file processing - not used in pipeline"""
        raise NotImplementedError("Ghosting degradation only supports piped processing")
    
    def get_filter_expression(self, video_info):
        """
        Generate FFmpeg filter string for ghosting effects.
        
        Args:
            video_info: Video information from probe (not used for ghosting)
            
        Returns:
            String containing FFmpeg filter chain
        """
        # Select number of ghost trails
        num_ghosts = random.randint(*self.num_ghosts_range)
        
        # Store parameters for logging
        self.selected_params = {
            'num_ghosts': num_ghosts,
            'ghosts': []
        }
        
        # If no ghosts, return empty filter
        if num_ghosts == 0:
            return ""
        
        # List to store all filter components
        filter_parts = []
        
        # Create a more straightforward filter graph with proper labeling
        # Start with splitting the input for each ghost plus main
        filter_parts.append(f"split={num_ghosts + 1}[main]" + "".join(f"[copy{i}]" for i in range(num_ghosts)))
        
        # Process each ghost copy
        for i in range(num_ghosts):
            # Generate parameters for this ghost
            delay = random.randint(*self.delay_range)
            opacity = random.uniform(*self.opacity_range)
            offset_x = random.randint(*self.offset_x_range)
            offset_y = random.randint(*self.offset_y_range)
            
            # Generate color shifts if enabled
            color_shifts = None
            if self.enable_color_shift:
                color_shifts = tuple(
                    random.uniform(*self.color_shift_range)
                    for _ in range(3)  # RGB channels
                )
            
            # Store parameters for this ghost
            ghost_params = {
                'delay': delay,
                'opacity': round(opacity, 3),
                'offset_x': offset_x,
                'offset_y': offset_y
            }
            if color_shifts:
                ghost_params['color_shifts'] = [round(x, 3) for x in color_shifts]
            self.selected_params['ghosts'].append(ghost_params)
            
            # Process this ghost
            current = f"[copy{i}]"
            
            # Add median filter for delay effect
            filter_parts.append(f"{current}tmedian=radius={delay}[delayed{i}]")
            current = f"[delayed{i}]"
            
            # Add slight blur
            filter_parts.append(f"{current}boxblur=luma_radius=1:luma_power=1[blur{i}]")
            current = f"[blur{i}]"
            
            # Handle position offset
            if offset_x != 0 or offset_y != 0:
                pad_w = f"iw+abs({offset_x*2})"
                pad_h = f"ih+abs({offset_y*2})"
                pad_x = str(max(offset_x, 0))
                pad_y = str(max(offset_y, 0))
                crop_x = str(abs(min(offset_x, 0)))
                crop_y = str(abs(min(offset_y, 0)))
                
                filter_parts.append(
                    f"{current}pad=w={pad_w}:h={pad_h}:x={pad_x}:y={pad_y}[pad{i}]"
                )
                current = f"[pad{i}]"
                
                filter_parts.append(
                    f"{current}crop=w=iw-abs({offset_x*2}):h=ih-abs({offset_y*2}):"
                    f"x={crop_x}:y={crop_y}[pos{i}]"
                )
                current = f"[pos{i}]"
            
            # Add color shift
            if color_shifts:
                r_mult, g_mult, b_mult = color_shifts
                filter_parts.append(
                    f"{current}colorbalance=rh={r_mult-1.0}:gh={g_mult-1.0}:"
                    f"bh={b_mult-1.0}[color{i}]"
                )
                current = f"[color{i}]"
            
            # Add opacity and color bleeding
            filter_parts.append(
                f"{current}colorchannelmixer=aa={opacity}:rb=0.05:bg=0.05[ghost{i}]"
            )
        
        # Create overlay chain
        current = "[main]"
        for i in range(num_ghosts):
            if i == num_ghosts - 1:
                # Last overlay doesn't need an output label
                filter_parts.append(f"{current}[ghost{i}]overlay")
            else:
                filter_parts.append(f"{current}[ghost{i}]overlay[overlay{i}]")
                current = f"[overlay{i}]"
        
        return ";".join(filter_parts)