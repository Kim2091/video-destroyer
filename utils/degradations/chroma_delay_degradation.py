from typing import Dict, Any
import logging
from .base_degradation import BaseDegradation

logger = logging.getLogger(__name__)

class ChromaDelayDegradation(BaseDegradation):
    """Delays chroma channels (U and V) to create a bleeding artifact."""

    def __init__(self, config: Dict[str, Any], logger=None):
        super().__init__(config, logger)
        self.params = config.get('params', {})
        # No specific random parameters for now, but structure is ready
        self.selected_params = self._select_chroma_delay_params()

    @property
    def name(self) -> str:
        """Return the name of the degradation."""
        return "chroma_delay"

    def _select_chroma_delay_params(self) -> Dict[str, Any]:
        """Select chroma delay parameters based on configuration."""
        # Default delay is 1 frame. Configurable via 'delay_frames'.
        # Later, 'delay_fields' could be added if field-based processing is implemented.
        delay_frames = self.params.get('delay_frames', 1)
        if not isinstance(delay_frames, int) or delay_frames < 0:
            delay_frames = 1
            if self.logger:
                self.logger.logger.warning(f"Invalid 'delay_frames' for ChromaDelay, defaulting to 1.")
        
        return {
            'delay_frames': delay_frames
        }

    def get_params(self) -> Dict[str, Any]:
        """Return the parameters used for this degradation."""
        return self.selected_params

    def apply(self, input_path: str, output_path: str) -> str:
        """Direct file processing - not typically used in the main pipeline."""
        raise NotImplementedError("Chroma Delay degradation currently only supports piped processing.")

    def get_filter_expression(self, video_info: Dict[str, Any]) -> str:
        """
        Generate FFmpeg filter string for chroma delay.

        This filter separates Y, U, V planes. It delays U and V planes by a specified
        number of frames and then merges them back with the Y plane.

        Args:
            video_info: Dictionary containing video information.

        Returns:
            String containing FFmpeg filter chain for chroma delay.
        """
        delay_frames = self.selected_params['delay_frames']
        frame_rate = video_info.get('avg_frame_rate', '25')  # Default to 25 fps if not found
        
        try:
            num_str, den_str = frame_rate.split('/')
            fps = float(num_str) / float(den_str)
        except ValueError:
            fps = float(frame_rate)
        except ZeroDivisionError:
            fps = 25  # Fallback
            if self.logger:
                self.logger.logger.warning(f"Could not parse frame rate '{frame_rate}', defaulting to {fps} fps")

        delay_time = delay_frames / fps
        
        # Ensure input is in yuv420p format and then process
        return (f"format=yuv420p,split=2[base][for_chroma];"
                f"[for_chroma]format=yuv420p,extractplanes=u+v[u][v];"
                f"[u]setpts=PTS+{delay_time}/TB[u_delayed];"
                f"[v]setpts=PTS+{delay_time}/TB[v_delayed];"
                f"[base]format=yuv420p,extractplanes=y[y];"
                f"[y][u_delayed][v_delayed]mergeplanes=0x001020:yuv420p")
