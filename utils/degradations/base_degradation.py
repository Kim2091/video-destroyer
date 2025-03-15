from abc import ABC, abstractmethod
from typing import Dict, Any
import random

class BaseDegradation(ABC):
    """Base class for all video degradations"""
    
    def __init__(self, config: Dict[str, Any], logger=None):
        self.config = config
        self.probability = config.get('probability', 1.0)
        self.logger = logger
    
    
    def should_apply(self) -> bool:
        """Determine if this degradation should be applied based on probability."""
        return random.random() < self.probability
    
    def process(self, input_path: str, output_path: str) -> str:
        """Process the video, respecting probability settings"""
        try:
            should_apply = self.should_apply()
            
            # Log degradation attempt
            if self.logger:
                self.logger.log_degradation_applied(
                    degradation_name=self.name,
                    was_applied=should_apply,
                    probability=self.probability,
                    params=self.get_params() if should_apply else None
                )
            
            # Only apply the degradation if should_apply is True
            if should_apply:
                result = self.apply(input_path, output_path)
                return result
            else:
                return input_path
                
        except Exception as e:
            if self.logger:
                self.logger.logger.error(f"Error in {self.name} degradation: {str(e)}")
            raise
    
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Return the parameters used for this degradation"""
        pass
    
    @abstractmethod
    def apply(self, input_path: str, output_path: str) -> str:
        """
        Apply the degradation to a video.
        
        Args:
            input_path: Path to input video
            output_path: Path to save degraded video
            
        Returns:
            str: Path to the degraded video
        """
        pass
    
    @abstractmethod
    def get_filter_expression(self, video_info) -> str:
        """
        Generate FFmpeg filter string for this degradation.
        
        Args:
            video_info: Video information from probe
            
        Returns:
            str: FFmpeg filter string for this degradation, or None if no filter needed
        """
        pass
    
    @property
    def name(self) -> str:
        """Return the name of the degradation"""
        # Default implementation derives name from class name
        return self.__class__.__name__.replace('Degradation', '').lower()