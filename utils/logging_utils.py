import logging
import os
import json
from datetime import datetime
from typing import Dict, Any

class DegradationLogger:
    """Handles logging of degradation information"""
    
    def __init__(self, config: Dict[str, Any]):
        log_config = config.get('logging', {})
        log_dir = log_config.get('directory', 'logs')
        log_file = log_config.get('filename', 'degradation_log.log')
        log_level = log_config.get('level', 'INFO')
        
        # Create logs directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Set up logger
        self.logger = logging.getLogger('degradation_logger')
        self.logger.setLevel(getattr(logging, log_level))
        self.logger.propagate = False  # Prevent duplicate logging
        
        # Clear any existing handlers
        self.logger.handlers = []
        
        # Create file handler with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = os.path.join(log_dir, f"{timestamp}_{log_file}")
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(getattr(logging, log_level))
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level))
        
        # Create formatters
        file_formatter = logging.Formatter(
            log_config.get('file_format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        console_formatter = logging.Formatter(
            log_config.get('console_format', '%(message)s')
        )
        
        # Set formatters
        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(console_formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def log_chunk_start(self, chunk_path: str) -> None:
        """Log the start of chunk processing"""
        self.logger.info(f"\nProcessing chunk: {os.path.basename(chunk_path)}")
    
    # In the log_degradation_applied method, change the status symbols
    def log_degradation_applied(self, 
                            degradation_name: str,
                            was_applied: bool,
                            probability: float,
                            params: Dict[str, Any] = None) -> None:
        """Log information about a degradation application"""
        status = "[+]" if was_applied else "[-]"  # Using ASCII characters instead of ✓ and ✗
        message = f"  {status} {degradation_name.capitalize()} degradation"
        if was_applied and params:
            param_str = " ".join(f"{k}={v}" for k, v in params.items() if k != "codec_probabilities")
            message += f" ({param_str})"
        self.logger.info(message)
        
        # Log detailed parameters to file only
        if params:
            self.logger.debug(json.dumps({
                "degradation": degradation_name,
                "applied": was_applied,
                "probability": probability,
                "params": params
            }))
    
    def log_chunk_complete(self, chunk_path: str) -> None:
        """Log the completion of chunk processing"""
        self.logger.info(f"Completed processing: {os.path.basename(chunk_path)}\n")