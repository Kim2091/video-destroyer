import random
from typing import Dict, Any, Tuple, List


class CodecHandler:
    """
    Handles codec selection and quality parameter generation based on configuration.
    """
    
    def __init__(self, codec_config: Dict[str, Any]):
        """
        Initialize the codec handler with configuration.
        
        Args:
            codec_config: Dictionary containing codec configurations
        """
        self.codec_config = codec_config
        self._validate_config()
        
    def _validate_config(self) -> None:
        """Validate codec configuration."""
        total_probability = sum(codec['probability'] for codec in self.codec_config.values())
        if not 0.99 <= total_probability <= 1.01:  # Allow small floating-point errors
            raise ValueError(f"Codec probabilities must sum to 1.0, got {total_probability}")
        
        for codec, config in self.codec_config.items():
            if 'quality_range' not in config:
                raise ValueError(f"Quality range missing for codec {codec}")
            if config['quality_range']['min'] > config['quality_range']['max']:
                raise ValueError(f"Min quality greater than max for codec {codec}")
    
    def select_random_codec(self) -> str:
        """
        Randomly select a codec based on configured probabilities.
        
        Returns:
            Selected codec name
        """
        r = random.random()
        cumulative = 0
        
        for codec, config in self.codec_config.items():
            cumulative += config['probability']
            if r <= cumulative:
                return codec
        
        # Fallback to the last codec if we have floating-point issues
        return list(self.codec_config.keys())[-1]
    
    def get_random_quality(self, codec: str) -> int:
        """
        Generate a random quality parameter within the configured range for a codec.
        
        Args:
            codec: The codec to generate quality parameter for
            
        Returns:
            Quality parameter value
        """
        if codec not in self.codec_config:
            raise ValueError(f"Unknown codec: {codec}")
        
        quality_range = self.codec_config[codec]['quality_range']
        return random.randint(quality_range['min'], quality_range['max'])
    
    def get_random_encoding_config(self) -> Tuple[str, int]:
        """
        Generate a random encoding configuration.
        
        Returns:
            Tuple of (codec, quality)
        """
        codec = self.select_random_codec()
        quality = self.get_random_quality(codec)
        return codec, quality
