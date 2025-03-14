import yaml
import os
from typing import Dict, Any, List


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
        
        if degradation['name'] == 'resize':
            if 'down_up' in params and isinstance(params['down_up'].get('range'), list):
                range_list = params['down_up']['range']
                params['down_up']['range'] = {
                    'min': range_list[0],
                    'max': range_list[1]
                }
                
        elif degradation['name'] == 'codec':
            for codec in params.values():
                if isinstance(codec.get('quality_range'), list):
                    range_list = codec['quality_range']
                    codec['quality_range'] = {
                        'min': range_list[0],
                        'max': range_list[1]
                    }
                    
    return config


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
    
    # Custom YAML loader to handle Windows paths
    class WindowsPathLoader(yaml.SafeLoader):
        pass
    
    def windows_path_constructor(loader, node):
        # Convert the scalar value to a string and normalize path
        scalar_value = loader.construct_scalar(node)
        return os.path.normpath(scalar_value)
    
    # Register the constructor for strings
    WindowsPathLoader.add_constructor('tag:yaml.org,2002:str', windows_path_constructor)
    
    with open(config_path, 'r') as file:
        config = yaml.load(file, Loader=WindowsPathLoader)
    
    # Validate required fields
    required_fields = [
        'input_video', 
        'output_directory', 
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
    
    # Validate input video exists
    if not os.path.exists(config['input_video']):
        raise FileNotFoundError(f"Input video not found: {config['input_video']}")
    
    # Create output directory if it doesn't exist
    os.makedirs(config['output_directory'], exist_ok=True)
    
    # Create chunks directory structure
    chunks_dir = config['chunks_directory']
    hr_dir = os.path.join(chunks_dir, 'HR')
    lr_dir = os.path.join(chunks_dir, 'LR')
    
    os.makedirs(hr_dir, exist_ok=True)
    os.makedirs(lr_dir, exist_ok=True)
    
    return config