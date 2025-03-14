from typing import List, Dict, Any
import logging
from pathlib import Path
import os
import ffmpeg
from .degradations.base_degradation import BaseDegradation
from .degradations.codec_degradation import CodecDegradation

logger = logging.getLogger(__name__)

class DegradationPipeline:
    """Manages a sequence of video degradations using direct piping"""
    
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
        Apply all degradations in sequence to the input video using direct piping.
        
        Args:
            input_path: Path to input video
            output_path: Path to save final degraded video
            
        Returns:
            Path to the final degraded video
        """
        # Get video info once for all degradations
        probe = ffmpeg.probe(input_path)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        
        # Start with the input file
        stream = ffmpeg.input(input_path)
        
        # Track which degradations will be applied (based on probability)
        applied_degradations = []
        skipped_degradations = []
        
        # Check which non-codec degradations should be applied
        for i, degradation in enumerate(self.degradations):
            if degradation.should_apply():
                applied_degradations.append((i, degradation))
            else:
                skipped_degradations.append((i, degradation))
        
        # Check if codec degradation should be applied
        codec_applied = False
        if self.codec_degradation and self.codec_degradation.should_apply():
            codec_applied = True
        
        # Log which degradations will be skipped
        for i, degradation in skipped_degradations:
            if degradation.logger:
                degradation.logger.log_degradation_applied(
                    degradation_name=degradation.name,
                    was_applied=False,
                    probability=degradation.probability,
                    params=None
                )
            logger.info(f"Skipped degradation {i+1}/{len(self.degradations)}: {degradation.name}")
        
        # Apply each selected degradation in sequence
        for i, degradation in applied_degradations:
            # Apply the degradation directly to the stream
            stream = degradation.apply_piped(stream, video_stream)
            
            # Log the application
            if degradation.logger:
                degradation.logger.log_degradation_applied(
                    degradation_name=degradation.name,
                    was_applied=True,
                    probability=degradation.probability,
                    params=degradation.get_params()
                )
            logger.info(f"Applied degradation {i+1}/{len(self.degradations)}: {degradation.name}")
        
        # Apply codec degradation last if it should be applied (FIXED INDENTATION)
        if codec_applied:
            # Apply the codec degradation - it returns the stream and codec params
            stream, codec_params = self.codec_degradation.apply_piped(stream, video_stream)
            
            # Create the final output with the codec parameters
            stream = ffmpeg.output(stream, output_path, **codec_params)
            
            # Log the application
            if self.codec_degradation.logger:
                self.codec_degradation.logger.log_degradation_applied(
                    degradation_name=self.codec_degradation.name,
                    was_applied=True,
                    probability=self.codec_degradation.probability,
                    params=self.codec_degradation.get_params()
                )
            logger.info(f"Applied final codec degradation: {self.codec_degradation.name}")
        elif self.codec_degradation:
            # Log that codec was skipped
            if self.codec_degradation.logger:
                self.codec_degradation.logger.log_degradation_applied(
                    degradation_name=self.codec_degradation.name,
                    was_applied=False,
                    probability=self.codec_degradation.probability,
                    params=None
                )
            logger.info(f"Skipped final codec degradation: {self.codec_degradation.name}")
            # If no codec degradation, we still need to output the stream
            stream = ffmpeg.output(stream, output_path)

        # Run the pipeline with visible output
        stream.run(overwrite_output=True)
        
        return output_path