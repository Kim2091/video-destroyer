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

        # Apply all filter expressions in one complex filter if we have any
        if filter_expressions:
            # Separate simple filters (no ';') from complex filter graphs (contain ';')
            simple_filters = []
            complex_filters = []
            for expr in filter_expressions:
                if ';' in expr:
                    complex_filters.append(expr)
                else:
                    simple_filters.append(expr)

            if complex_filters:
                # Chain multiple complex filter graphs through intermediate labels.
                # Each complex graph consumes the unnamed input stream; when there
                # are several, we label the output of graph N and feed it as the
                # input of graph N+1 so the unnamed input is only consumed once.
                chained_parts = []
                for idx, cf in enumerate(complex_filters):
                    is_last = (idx == len(complex_filters) - 1)

                    # If this is not the first complex graph, replace its unnamed
                    # input with the previous graph's output label.
                    if idx > 0:
                        cf = f'[_chain_{idx - 1}]' + cf

                    # If this is not the last complex graph, label its output so
                    # the next graph can consume it.
                    if not is_last:
                        cf = cf + f'[_chain_{idx}]'

                    chained_parts.append(cf)

                complex_filter = ';'.join(chained_parts)

                # Prepend simple filters before the first complex graph's input
                if simple_filters:
                    simple_chain = ','.join(simple_filters)
                    complex_filter = simple_chain + ',' + complex_filter

                logger.debug(f"Applied complex filter graph: {complex_filter}")
                output_args = {'filter_complex': complex_filter}
            else:
                # All filters are simple, join with commas as a linear chain
                complex_filter = ','.join(simple_filters)
                logger.debug(f"Applied filter chain: {complex_filter}")
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