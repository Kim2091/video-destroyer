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
        
        os.makedirs(log_dir, exist_ok=True)
        
        self.logger = logging.getLogger('degradation_logger')
        self.logger.setLevel(getattr(logging, log_level))
        self.logger.propagate = False
        self.logger.handlers = []
        
        # Simplified timestamp format
        timestamp = datetime.now().strftime('%H%M')
        log_path = os.path.join(log_dir, f"{timestamp}_{log_file}")
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(getattr(logging, log_level))
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level))
        
        # Update formatters for desired output
        file_formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M')
        console_formatter = logging.Formatter('%(message)s')
        
        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(console_formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
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
            if params.get('down_up') == 'enabled':
                return f"({params['scale']}x, {params['down_filter']}->{params['up_filter']}, mid={params['intermediate_scale']:.2f}{prob_str})"
            return f"({params['scale']}x, {params['down_filter']}{prob_str})"
            
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
        if not was_applied:
            return  # Don't log skipped degradations
            
        # Add probability to params
        if params is None:
            params = {}
        params['probability'] = probability
            
        # Console output - clean and concise
        status = "[+]" if was_applied else "[-]"
        param_str = self._format_params(params)
        message = f"  {status} {degradation_name.title()} {param_str}"
        self.logger.info(message)
        
        # Detailed JSON log to file only
        if params:
            self.logger.debug(json.dumps({
                "degradation": degradation_name,
                "applied": was_applied,
                "probability": probability,
                "params": params
            }))

    def log_chunk_complete(self, chunk_path: str) -> None:
        """Log the completion of chunk processing"""
        self.logger.info(f"Complete: {os.path.basename(chunk_path)}\n")