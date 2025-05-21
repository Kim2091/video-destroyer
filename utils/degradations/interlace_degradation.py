import random
from typing import Dict, Any, List
import logging
from .base_degradation import BaseDegradation

logger = logging.getLogger(__name__)

class InterlaceDegradation(BaseDegradation):
    """Applies interlacing to videos using FFmpeg's tinterlace filter."""

    def __init__(self, config: Dict[str, Any], logger=None):
        super().__init__(config, logger)
        self.params = config.get('params', {})
        self.selected_params = self._select_interlace_params()

    @property
    def name(self) -> str:
        """Return the name of the degradation."""
        return "interlace"

    def _select_interlace_params(self) -> Dict[str, Any]:
        """Select interlace parameters based on configuration."""
        # Default field orders: 'top' for top field first, 'bottom' for bottom field first
        field_orders = self.params.get('field_orders', ['top', 'bottom'])
        if not isinstance(field_orders, list) or not field_orders:
            field_orders = ['top', 'bottom']
            
        selected_field_order = random.choice(field_orders)
        
        # Select mode based on field order
        # Mode 4 = interleave_top (top field first)
        # Mode 5 = interleave_bottom (bottom field first)
        # Mode 7 = custom chroma-only interlacing
        if self.params.get('chroma_only', False):
            selected_mode = 7
        else:
            selected_mode = 4 if selected_field_order == 'top' else 5
        
        return {
            'mode': selected_mode,
            'field_order': selected_field_order,
        }

    def get_params(self) -> Dict[str, Any]:
        """Return the parameters used for this degradation."""
        return self.selected_params

    def apply(self, input_path: str, output_path: str) -> str:
        """Direct file processing - not typically used in the main pipeline."""
        raise NotImplementedError("Interlace degradation currently only supports piped processing as part of a filter chain.")

    def get_filter_expression(self, video_info: Dict[str, Any]) -> str:
        """
        Generate FFmpeg filter string for interlacing.

        Args:
            video_info: Dictionary containing video information (e.g., width, height, fps).

        Returns:
            String containing FFmpeg filter chain for interlacing.
        """
        if not video_info:
            pass  # Basic tinterlace doesn't strictly need video info

        mode = self.selected_params['mode']
        field_order = self.selected_params['field_order']
        
        # Special mode 7: Chroma-only interlacing
        if mode == 7:
            # For chroma-only, use mode 4 (top field) or 5 (bottom field) based on selected field order
            chroma_mode = 4 if field_order == 'top' else 5
            return (
                f"format=yuv420p,split=2[base][for_chroma];"
                f"[for_chroma]extractplanes=u+v[u][v];"
                f"[u]tinterlace=mode={chroma_mode}:flags=vlpf[u_int];"  # Interlace U plane
                f"[v]tinterlace=mode={chroma_mode}:flags=vlpf[v_int];"  # Interlace V plane
                f"[base]extractplanes=y[y];"      # Keep Y plane progressive
                f"[y][u_int][v_int]mergeplanes=0x001020:yuv420p"  # Merge back
            )
        
        # Standard interlacing modes
        # Add flags=vlpf for vertical low-pass filtering to reduce twitter and Moire patterns
        filter_expr = f"tinterlace=mode={mode}:flags=vlpf"
        
        if self.logger:
            self.logger.logger.debug(f"InterlaceDegradation: Applying tinterlace with mode {mode}, field order: {field_order}.")
            
        return filter_expr
