import random
from typing import Dict, Any, Tuple


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
        
        # Normalize probabilities if they don't add up to 1
        if total_probability > 0 and not 0.99 <= total_probability <= 1.01:
            for codec in self.codec_config:
                self.codec_config[codec]['probability'] /= total_probability
            print(f"Warning: Codec probabilities summed to {total_probability:.4f}, normalized to 1.0")
        
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
    
    def get_random_preset(self, codec: str) -> Any:
        """
        Generate a random preset for a codec if configured.
        
        Args:
            codec: The codec to generate preset for
            
        Returns:
            Preset value (string or int) or None if not configured
        """
        if codec not in self.codec_config:
            raise ValueError(f"Unknown codec: {codec}")
        
        config = self.codec_config[codec]
        
        # Check for preset list (e.g., h264, h265)
        if 'presets' in config and config['presets']:
            return random.choice(config['presets'])
        
        # Check for preset range (e.g., av1)
        if 'preset_range' in config:
            preset_range = config['preset_range']
            # Handle both list [min, max] and dict {'min': x, 'max': y} formats
            if isinstance(preset_range, list):
                return random.randint(preset_range[0], preset_range[1])
            else:
                return random.randint(preset_range['min'], preset_range['max'])
        
        return None
    
    def get_random_encoding_config(self) -> Tuple[str, int, Any]:
        """
        Generate a random encoding configuration.
        
        Returns:
            Tuple of (codec, quality, preset)
        """
        codec = self.select_random_codec()
        quality = self.get_random_quality(codec)
        preset = self.get_random_preset(codec)
        return codec, quality, preset
