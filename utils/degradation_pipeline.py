from typing import List, Dict, Any
import logging
from pathlib import Path
import os
from .degradations.base_degradation import BaseDegradation
from .degradations.codec_degradation import CodecDegradation

logger = logging.getLogger(__name__)

class DegradationPipeline:
    """Manages a sequence of video degradations"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.degradations: List[BaseDegradation] = []
        self.codec_degradation = None
        
    def add_degradation(self, degradation: BaseDegradation) -> None:
        """Add a degradation to the pipeline"""
        # Store codec degradation separately
        if isinstance(degradation, CodecDegradation):
            self.codec_degradation = degradation
        else:
            self.degradations.append(degradation)
        
    def process_video(self, input_path: str, output_path: str) -> str:
        """
        Apply all degradations in sequence to the input video, with codec always last.
        
        Args:
            input_path: Path to input video
            output_path: Path to save final degraded video
            
        Returns:
            Path to the final degraded video
        """
        current_input = input_path
        
        # Create temporary directory for intermediate results
        temp_dir = os.path.join(os.path.dirname(output_path), "temp_degradations")
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # Apply each non-codec degradation in sequence
            for i, degradation in enumerate(self.degradations):
                # Create temporary output path for this degradation
                temp_output = os.path.join(
                    temp_dir, 
                    f"temp_{i}_{degradation.name}_{Path(output_path).name}"
                )
                
                # Always apply the degradation (ignore probability for piping)
                current_input = degradation.apply(current_input, temp_output)
                logger.info(f"Applied degradation {i+1}/{len(self.degradations)}: {degradation.name}")
            
            # Always apply codec degradation last if it exists
            if self.codec_degradation:
                # Create temporary output path for codec degradation
                codec_output = os.path.join(
                    temp_dir, 
                    f"temp_final_codec_{Path(output_path).name}"
                )
                
                # Apply codec degradation
                current_input = self.codec_degradation.apply(current_input, codec_output)
                logger.info(f"Applied final codec degradation: {self.codec_degradation.name}")
            
            # Move final result to output path
            if current_input != output_path:
                os.replace(current_input, output_path)
                
            return output_path
            
        finally:
            # Clean up temporary files
            if os.path.exists(temp_dir):
                for file in os.listdir(temp_dir):
                    try:
                        os.remove(os.path.join(temp_dir, file))
                    except:
                        pass
                os.rmdir(temp_dir)