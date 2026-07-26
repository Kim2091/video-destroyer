from typing import List, Dict, Any
import logging
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

    @staticmethod
    def _build_filter_graph(filter_expressions: List[str]) -> str:
        """Build an ordered filter graph without moving simple filters."""
        if not any(';' in expression for expression in filter_expressions):
            return ','.join(filter_expressions)

        graph_parts = []
        pending_simple_filters = []
        current_label = None
        chain_index = 0

        for index, expression in enumerate(filter_expressions):
            if ';' not in expression:
                pending_simple_filters.append(expression)
                continue

            prefix = f'[{current_label}]' if current_label else ''
            if pending_simple_filters:
                prefix += ','.join(pending_simple_filters) + ','

            graph = prefix + expression
            if index < len(filter_expressions) - 1:
                current_label = f'_chain_{chain_index}'
                graph += f'[{current_label}]'
                chain_index += 1
            else:
                current_label = None

            graph_parts.append(graph)
            pending_simple_filters = []

        if pending_simple_filters:
            prefix = f'[{current_label}]' if current_label else ''
            graph_parts.append(prefix + ','.join(pending_simple_filters))

        return ';'.join(graph_parts)

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
        
        # Track which degradations will be applied (based on probability)
        applied_degradations = []
        skipped_degradations = []
        filter_expressions = []
        
        # Check which non-codec degradations should be applied
        for i, degradation in enumerate(self.degradations):
            if degradation.should_apply():
                applied_degradations.append((i, degradation))
                # Get the filter expression and add it if not None
                filter_expr = degradation.get_filter_expression(video_stream)
                if filter_expr:
                    filter_expressions.append(filter_expr)
                    
                    # Log the application
                    if degradation.logger:
                        degradation.logger.log_degradation_applied(
                            degradation_name=degradation.name,
                            was_applied=True,
                            probability=degradation.probability,
                            params=degradation.get_params()
                        )
            else:
                skipped_degradations.append((i, degradation))
        
        # Log skipped degradations
        for i, degradation in skipped_degradations:
            if degradation.logger:
                degradation.logger.log_degradation_applied(
                    degradation_name=degradation.name,
                    was_applied=False,
                    probability=degradation.probability,
                    params=None
                )
        
        # Start building the ffmpeg command
        input_stream = ffmpeg.input(input_path)

        # Apply all filter expressions in their configured order.
        if filter_expressions:
            complex_filter = self._build_filter_graph(filter_expressions)
            logger.debug(f"Applied filter graph: {complex_filter}")
            output_args = {'filter_complex': complex_filter}
        else:
            output_args = {}
        
        # Handle codec degradation separately since it's applied at output.
        # Evaluate should_apply() once and reuse the result to avoid a
        # separate random roll from the non-codec degradations above.
        if self.codec_degradation:
            codec_should_apply = self.codec_degradation.should_apply()

            if codec_should_apply:
                # Get codec parameters
                codec_params = self.codec_degradation.get_codec_params()

                # Merge codec parameters with filter parameters
                output_args.update(codec_params)

                if self.codec_degradation.logger:
                    self.codec_degradation.logger.log_degradation_applied(
                        degradation_name=self.codec_degradation.name,
                        was_applied=True,
                        probability=self.codec_degradation.probability,
                        params=self.codec_degradation.get_params()
                    )
            else:
                if self.codec_degradation.logger:
                    self.codec_degradation.logger.log_degradation_applied(
                        degradation_name=self.codec_degradation.name,
                        was_applied=False,
                        probability=self.codec_degradation.probability,
                        params=None
                    )
        
        # Create the final output with all parameters
        stream = (
            ffmpeg.output(input_stream, output_path, **output_args)
            .global_args('-hide_banner', '-loglevel', 'error')
        )

        # Run the pipeline with detailed error logging
        try:
            stream.overwrite_output().run(capture_stdout=True, capture_stderr=True)
        except ffmpeg.Error as e:
            # Log the detailed error message
            stderr = e.stderr.decode('utf8')
            logger.error(f"FFmpeg error processing {input_path}:")
            logger.error(stderr)
            raise RuntimeError(f"FFmpeg error: {stderr}")
        
        return output_path
