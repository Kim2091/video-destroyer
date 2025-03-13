import yaml
import os
from typing import Dict, Any


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
    required_fields = ['input_video', 'output_directory', 'chunks_directory', 'codecs', 'chunk_strategy']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Required field '{field}' missing in configuration")
    
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
