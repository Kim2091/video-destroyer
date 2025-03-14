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
        """
        Determine if this degradation should be applied based on probability.
        
        Returns:
            bool: True if degradation should be applied, False otherwise
        """
        return random.random() < self.probability
    
    def process(self, input_path: str, output_path: str) -> str:
        """Process the video, respecting probability settings"""
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
            return self.apply(input_path, output_path)
        else:
            # Skip this degradation and return the input path unchanged
            return input_path
    
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
    def apply_piped(self, input_stream, video_info=None):
        """
        Apply the degradation to a video stream.
        
        Args:
            input_stream: FFmpeg input stream
            video_info: Optional video information from probe
            
        Returns:
            FFmpeg output stream
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the degradation"""
        pass