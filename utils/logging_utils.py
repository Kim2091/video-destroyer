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
        
        # Set up file handler
        self.logger = logging.getLogger('degradation_logger')
        self.logger.setLevel(getattr(logging, log_level))
        
        # Create a file handler with timestamp in filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = os.path.join(log_dir, f"{timestamp}_{log_file}")
        
        handler = logging.FileHandler(log_path)
        handler.setLevel(getattr(logging, log_level))
        
        # Create a formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(handler)
    
    def log_chunk_start(self, chunk_path: str) -> None:
        """Log the start of chunk processing"""
        self.logger.info(f"\nProcessing chunk: {os.path.basename(chunk_path)}")
    
    def log_degradation_applied(self, 
                              degradation_name: str,
                              was_applied: bool,
                              probability: float,
                              params: Dict[str, Any] = None) -> None:
        """Log information about a degradation application"""
        status = "APPLIED" if was_applied else "SKIPPED"
        log_entry = {
            "degradation": degradation_name,
            "status": status,
            "probability": probability,
            "params": params or {}
        }
        self.logger.info(json.dumps(log_entry))
    
    def log_chunk_complete(self, chunk_path: str) -> None:
        """Log the completion of chunk processing"""
        self.logger.info(f"Completed processing: {os.path.basename(chunk_path)}\n")