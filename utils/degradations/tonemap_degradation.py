import random
from typing import Dict, Any
import logging
from .base_degradation import BaseDegradation

logger = logging.getLogger(__name__)

class TonemapDegradation(BaseDegradation):
    """Converts HDR video to SDR using tonemapping"""
    
    def __init__(self, config: Dict[str, Any], logger=None):
        super().__init__(config, logger)
        self.params = config.get('params', {})
        self.tonemap_algorithm = self.params.get('tonemap_algorithm', 'hable')
        self.desat = self.params.get('desat', 0.5)
        self.target_nits = self.params.get('target_nits', 100)
        self.auto_detect = self.params.get('auto_detect', True)
        
        # Valid tonemap algorithms in FFmpeg
        self.valid_algorithms = ['hable', 'mobius', 'reinhard', 'bt2390', 'linear']
        
        # Store selected parameters
        self.selected_params = {}
        
    @property
    def name(self) -> str:
        """Return the name of the degradation"""
        return "tonemap"
    
    def _select_tonemap_algorithm(self) -> str:
        """Select tonemapping algorithm"""
        # If tonemap_algorithm is a list, randomly select from it
        if isinstance(self.tonemap_algorithm, list):
            return random.choice(self.tonemap_algorithm)
        return self.tonemap_algorithm
    
    def _select_desat(self) -> float:
        """Select desaturation parameter"""
        # If desat is a range [min, max], randomly select from it
        if isinstance(self.desat, list) and len(self.desat) == 2:
            return random.uniform(self.desat[0], self.desat[1])
        return self.desat
    
    def get_params(self) -> Dict[str, Any]:
        """Return the parameters used for this degradation"""
        return self.selected_params
    
    def apply(self, input_path: str, output_path: str) -> str:
        """Direct file processing - not used in pipeline"""
        raise NotImplementedError("Tonemap degradation only supports piped processing")
    
    def get_filter_expression(self, video_info):
        """
        Generate FFmpeg filter string for HDR to SDR tonemapping.
        
        Args:
            video_info: Video information from probe
            
        Returns:
            String containing FFmpeg filter chain for tonemapping, or None if not HDR
        """
        if not video_info:
            raise ValueError("Video info required for tonemap degradation")
        
        # Check if video is HDR
        color_transfer = video_info.get('color_transfer', '')
        color_space = video_info.get('color_space', '')
        color_primaries = video_info.get('color_primaries', '')
        
        # HDR transfer characteristics
        is_hdr = (
            color_transfer in ['smpte2084', 'arib-std-b67'] or  # PQ (HDR10) or HLG
            'bt2020' in color_space or
            'bt2020' in color_primaries
        )
        
        # If auto_detect is enabled and video is not HDR, skip tonemapping
        if self.auto_detect and not is_hdr:
            logger.info("Input is SDR, skipping tonemapping")
            self.selected_params = {'skipped': True, 'reason': 'SDR input'}
            return None
        
        # Select parameters
        algorithm = self._select_tonemap_algorithm()
        desat_value = self._select_desat()
        
        # Store selected parameters for logging
        self.selected_params = {
            'algorithm': algorithm,
            'desat': round(desat_value, 3),
            'target_nits': self.target_nits,
            'detected_hdr': is_hdr,
            'color_transfer': color_transfer,
            'color_space': color_space
        }
        
        # Build the tonemap filter chain
        # zscale converts to linear light and appropriate primaries
        # tonemap performs the actual HDR->SDR conversion
        # zscale again converts back to bt709/sRGB for SDR
        filter_expr = (
            f"zscale=transfer=linear:primaries=bt709,"
            f"tonemap={algorithm}:desat={desat_value}:peak={self.target_nits},"
            f"zscale=transfer=bt709:primaries=bt709:matrix=bt709,"
            f"format=yuv420p"
        )
        
        logger.info(f"Applying HDR to SDR tonemapping: {algorithm} (desat={desat_value}, target={self.target_nits} nits)")
        
        return filter_expr
