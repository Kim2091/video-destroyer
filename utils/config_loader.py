import os
import yaml
from typing import Dict, Any


def validate_degradation_config(degradation: Dict[str, Any]) -> None:
    """
    Validate a single degradation configuration.
    
    Args:
        degradation: Degradation configuration dictionary
    """
    required_fields = ['name', 'enabled', 'params']
    for field in required_fields:
        if field not in degradation:
            raise ValueError(f"Required field '{field}' missing in degradation configuration")
            
    # Validate specific degradation types
    if degradation['name'] == 'resize':
        params = degradation['params']
        if 'down_up' in params and 'range' in params['down_up']:
            range_val = params['down_up']['range']
            if not isinstance(range_val, list) or len(range_val) != 2:
                raise ValueError("resize down_up.range must be a list of [min, max]")
                
    elif degradation['name'] == 'codec':
        params = degradation['params']
        for codec in params.values():
            if 'quality_range' in codec:
                range_val = codec['quality_range']
                if not isinstance(range_val, list) or len(range_val) != 2:
                    raise ValueError("codec quality_range must be a list of [min, max]")


def convert_ranges_to_dict(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert array ranges to dictionary format for backward compatibility.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Updated configuration dictionary
    """
    for degradation in config.get('degradations', []):
        params = degradation.get('params', {})
        
        # Only convert codec ranges, not resize ranges
        if degradation['name'] == 'codec':
            for codec in params.values():
                if isinstance(codec.get('quality_range'), list):
                    range_list = codec['quality_range']
                    codec['quality_range'] = {
                        'min': range_list[0],
                        'max': range_list[1]
                    }
                    
    return config

def _normalize_path_values(config: Dict[str, Any]) -> None:
    """Normalize path-valued config keys so users can write forward slashes
    (e.g. ``C:/Videos/input.mp4``) without needing to escape backslashes.

    Only touches keys that are known to hold filesystem paths.
    """
    # Top-level path keys
    _TOP_LEVEL_PATHS = ('input', 'chunks_directory')
    for key in _TOP_LEVEL_PATHS:
        if key in config and isinstance(config[key], str):
            config[key] = os.path.normpath(config[key])

    # Nested path keys
    _NESTED_PATHS = {
        'frame_extraction': ('output_directory',),
        'logging': ('directory',),
    }
    for section, keys in _NESTED_PATHS.items():
        sub = config.get(section)
        if not isinstance(sub, dict):
            continue
        for key in keys:
            if key in sub and isinstance(sub[key], str):
                sub[key] = os.path.normpath(sub[key])


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Dictionary containing configuration parameters
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Normalize only the keys that are actual filesystem paths
    _normalize_path_values(config)
    
    # Validate required fields
    required_fields = [
        'chunks_directory', 
        'chunk_strategy',
        'degradations'
    ]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Required field '{field}' missing in configuration")
    
    # Validate degradations configuration
    degradations = config.get('degradations', [])
    if not isinstance(degradations, list):
        raise ValueError("'degradations' must be a list")
    
    # Validate each degradation
    for degradation in degradations:
        validate_degradation_config(degradation)
    
    # Convert array ranges to dictionary format for backward compatibility
    config = convert_ranges_to_dict(config)
        
    # For backward compatibility, create codecs config from degradations
    codec_degradation = next(
        (d for d in degradations if d['name'] == 'codec' and d['enabled']), 
        None
    )
    if codec_degradation:
        config['codecs'] = codec_degradation['params']

    # Validate chunk strategy
    valid_strategies = ["duration", "scene_detection", "frame_count"]
    if config.get('chunk_strategy') not in valid_strategies:
        raise ValueError(f"Invalid chunk_strategy. Must be one of: {valid_strategies}")

    # If using existing chunks, validate HR directory exists and contains files
    if config.get('use_existing_chunks', False):
        hr_dir = os.path.join(config['chunks_directory'], 'HR')
        if not os.path.exists(hr_dir):
            raise FileNotFoundError(f"HR chunks directory not found: {hr_dir}")
        if not any(f.endswith(('.mkv', '.mp4')) for f in os.listdir(hr_dir)):
            raise FileNotFoundError(f"No MKV or MP4 files found in HR chunks directory: {hr_dir}")
                
    # Create chunks directory structure
    chunks_dir = config['chunks_directory']
    hr_dir = os.path.join(chunks_dir, 'HR')
    lr_dir = os.path.join(chunks_dir, 'LR')
    
    os.makedirs(hr_dir, exist_ok=True)
    os.makedirs(lr_dir, exist_ok=True)
    
    return config
