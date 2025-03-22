import logging
import os
import json
from datetime import datetime
from typing import Dict, Any

class DegradationLogger:
    """Handles logging of degradation information"""
    
class DegradationLogger:
    """Handles logging of degradation information"""
    
    def __init__(self, config: Dict[str, Any]):
        log_config = config.get('logging', {})
        log_dir = log_config.get('directory', 'logs')
        log_file = log_config.get('filename', 'degradation_log.log')
        log_level = log_config.get('level', 'INFO').upper()
        
        os.makedirs(log_dir, exist_ok=True)
        
        self.logger = logging.getLogger('degradation_logger')
        self.logger.setLevel(getattr(logging, log_level))
        self.logger.propagate = False
        self.logger.handlers = []
        
        # Simplified timestamp format
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        log_path = os.path.join(log_dir, f"{timestamp}_{log_file}")
        
        # Create handlers with appropriate levels
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(getattr(logging, log_level))
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level))
        
        # Use more detailed format for DEBUG level
        if log_level == 'DEBUG':
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_formatter = logging.Formatter('%(levelname)s - %(message)s')
        else:
            file_formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M')
            console_formatter = logging.Formatter('%(message)s')
        
        
        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(console_formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        if log_level == 'DEBUG':
            self.logger.debug("Initialized DegradationLogger with DEBUG level")
    
    def log_chunk_start(self, chunk_path: str) -> None:
        """Log the start of chunk processing"""
        self.logger.info(f"\nProcessing: {os.path.basename(chunk_path)}")
    

    def _format_params(self, params: Dict[str, Any]) -> str:
        """Format parameters for display"""
        if not params:
            return ""
            
        # Add probability to string
        prob_str = f", prob={params.get('probability', 1.0)}"
            
        # For resize degradation
        if 'down_filter' in params:
            if params.get('up_filter'):  # Check if up_filter exists (indicates down-up scaling)
                return f"({params.get('fixed_scale', 1.0)}x, {params['down_filter']}->{params['up_filter']}, mid={params.get('intermediate_scale', 1.0):.2f}{prob_str})"
            return f"({params.get('fixed_scale', 1.0)}x, {params['down_filter']}{prob_str})"
            
        # For codec degradation
        if 'codec' in params:
            return f"({params['codec']}, q={params['quality']}{prob_str})"
            
        return f"({' '.join(f'{k}={v}' for k, v in params.items() if k not in ['codec_probabilities', 'probability'])}{prob_str})"




    def log_degradation_applied(self, 
                            degradation_name: str,
                            was_applied: bool,
                            probability: float,
                            params: Dict[str, Any] = None) -> None:
        """Log information about a degradation application"""
        if not was_applied and self.logger.level != logging.DEBUG:
            return  # Only log skipped degradations in DEBUG mode
            
        # Add probability to params
        if params is None:
            params = {}
        params['probability'] = probability
            
        # Console output - clean and concise
        status = "[+]" if was_applied else "[-]"
        param_str = self._format_params(params)
        message = f"  {status} {degradation_name.title()} {param_str}"
        
        if self.logger.level <= logging.DEBUG:
            # In DEBUG mode, log everything with extra details
            debug_info = {
                "degradation": degradation_name,
                "applied": was_applied,
                "probability": probability,
                "params": params,
                "timestamp": datetime.now().isoformat()
            }
            # Log the message and debug info separately
            self.logger.debug(message)
            self.logger.debug(f"Debug details: {json.dumps(debug_info, indent=2)}")
        else:
            # In normal mode, only log applied degradations
            if was_applied:
                self.logger.info(message)
                
    def log_chunk_complete(self, chunk_path: str) -> None:
        """Log the completion of chunk processing"""
        self.logger.info(f"Complete: {os.path.basename(chunk_path)}\n")

def setup_global_logging(config: Dict[str, Any]) -> None:
    """Setup global logging configuration"""
    log_config = config.get('logging', {})
    log_level = log_config.get('level', 'INFO').upper()
    
    # Reset any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers = []
    
    # Configure root logger
    root_logger.setLevel(getattr(logging, log_level))
    
    # Define formats based on log level
    if log_level == 'DEBUG':
        console_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        file_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    else:
        console_format = log_config.get('console_format', '%(message)s')
        file_format = log_config.get('file_format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(console_format))
    root_logger.addHandler(console_handler)
    
    # File handler
    if log_config.get('directory'):
        os.makedirs(log_config['directory'], exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        log_file = os.path.join(
            log_config['directory'], 
            f"{timestamp}_{log_config.get('filename', 'application.log')}"
        )
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(file_format))
        root_logger.addHandler(file_handler)
        
        if log_level == 'DEBUG':
            root_logger.debug("Global logging initialized with DEBUG level")
