import logging
import argparse
import os
from pathlib import Path
from utils.config_loader import load_config
from utils.codec_handler import CodecHandler
from utils.video_processor import VideoProcessor, check_ffmpeg_available
from utils.logging_utils import setup_global_logging
from frame_extractor import FrameSequenceExtractor
from utils.post_processor import PostProcessor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_video_files(folder_path, extensions):
    """Get all video files in a folder with supported extensions.
    
    Args:
        folder_path: Path to the folder containing videos
        extensions: List of supported video file extensions (e.g., ['.mp4', '.mov'])
        
    Returns:
        List of video file paths
    """
    video_files = []
    folder = Path(folder_path)
    
    if not folder.exists():
        logger.error(f"Folder not found: {folder_path}")
        return video_files
    
    # Get all files with matching extensions
    for ext in extensions:
        video_files.extend(folder.glob(f"*{ext}"))
        video_files.extend(folder.glob(f"*{ext.upper()}"))  # Also check uppercase
    
    return sorted([str(f) for f in video_files])


def get_video_name(video_path):
    """Get a clean name from video path for organizing output.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Clean video name without extension
    """
    return Path(video_path).stem


def process_single_video(config, video_path=None, skip_frame_extraction=False):
    """Process a single video with the given configuration.
    
    Args:
        config: Configuration dictionary
        video_path: Optional path to override config input_video
        skip_frame_extraction: If True, skip frame extraction and post-processing
        
    Returns:
        Tuple of (success: bool, error_message: str or None)
    """
    try:
        # Override input video if specified
        if video_path:
            config['input_video'] = video_path
            video_name = get_video_name(video_path)
            logger.info(f"\n{'='*80}")
            logger.info(f"Processing video: {video_name}")
            logger.info(f"{'='*80}")
        
        # Initialize codec handler
        codec_handler = CodecHandler(config['codecs'])
        
        # Initialize video processor
        video_processor = VideoProcessor(config, codec_handler)
        
        # Process the video
        processed_pairs = video_processor.process_video()
        
        logger.info(f"Video processing complete. Created {len(processed_pairs)} HR/LR chunk pairs.")
        
        # Skip frame extraction if requested (for batch processing)
        if skip_frame_extraction:
            return True, None
        
        # Check if automatic frame extraction is enabled
        frame_config = config.get('frame_extraction', {})
        auto_extract = frame_config.get('auto_extract_frames', False)
        
        if auto_extract:
            logger.info("Starting automatic frame extraction...")
            try:
                extractor = FrameSequenceExtractor(config)
                total_sequences = extractor.extract_all_sequences()
                logger.info(f"Frame extraction complete. Extracted {total_sequences} sequences.")
                
                # Check if automatic post-processing is enabled
                post_config = config.get('post_processing', {})
                post_enabled = post_config.get('enabled', False)
                
                if post_enabled:
                    logger.info("Starting automatic post-processing...")
                    try:
                        post_processor = PostProcessor(config)
                        post_processor.run()
                        logger.info("Post-processing complete.")
                    except Exception as e:
                        logger.error(f"Error during post-processing: {str(e)}")
                        return False, f"Post-processing failed: {str(e)}"
                        
            except Exception as e:
                logger.error(f"Error during frame extraction: {str(e)}")
                return False, f"Frame extraction failed: {str(e)}"
        
        return True, None
        
    except Exception as e:
        error_msg = f"Error processing video: {str(e)}"
        logger.error(error_msg)
        return False, error_msg


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Video destroyer: re-encode video chunks with random codecs')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file')
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Setup global logging
        setup_global_logging(config)
        logger.debug(f"Loading configuration from {args.config}")
        
        # Check FFmpeg availability
        if not check_ffmpeg_available():
            return 1

        # Get input path and auto-detect if it's a file or folder
        input_path = config.get('input')
        
        if not input_path:
            logger.error("No input specified in config. Please set 'input' to a video file or folder path.")
            return 1
        
        input_path_obj = Path(input_path)
        
        if not input_path_obj.exists():
            logger.error(f"Input path does not exist: {input_path}")
            return 1
        
        # Auto-detect: folder or file?
        if input_path_obj.is_dir():
            # Process folder of videos
            logger.info(f"Folder processing mode (auto-detected)")
            logger.info(f"Scanning folder: {input_path}")
            
            video_extensions = config.get('video_extensions', ['.mp4', '.mov', '.mkv', '.avi'])
            video_files = get_video_files(input_path, video_extensions)
            
            if not video_files:
                logger.error(f"No video files found in {input_path}")
                logger.info(f"Supported extensions: {', '.join(video_extensions)}")
                return 1
            
            logger.info(f"Found {len(video_files)} video files to process")
            
            # Process each video
            results = []
            base_chunks_dir = config.get('chunks_directory', 'chunks')
            
            for i, video_path in enumerate(video_files, 1):
                logger.info(f"\n{'='*80}")
                logger.info(f"Video {i}/{len(video_files)}: {Path(video_path).name}")
                logger.info(f"{'='*80}")
                
                # Create a copy of config for this video
                video_config = config.copy()
                
                # Process the video (skip frame extraction for batch mode)
                success, error = process_single_video(video_config, video_path, skip_frame_extraction=True)
                results.append((Path(video_path).name, success, error))
                
                if success:
                    logger.info(f"[SUCCESS] Successfully processed: {Path(video_path).name}")
                else:
                    logger.error(f"[FAILED] Failed to process: {Path(video_path).name}")
                    if error:
                        logger.error(f"  Error: {error}")
            
            # Summary
            logger.info(f"\n{'='*80}")
            logger.info("BATCH PROCESSING SUMMARY")
            logger.info(f"{'='*80}")
            
            successful = sum(1 for _, success, _ in results if success)
            failed = len(results) - successful
            
            logger.info(f"Total videos: {len(results)}")
            logger.info(f"Successful: {successful}")
            logger.info(f"Failed: {failed}")
            
            if failed > 0:
                logger.info("\nFailed videos:")
                for name, success, error in results:
                    if not success:
                        logger.info(f"  - {name}: {error}")
            
            logger.info(f"{'='*80}\n")
            
            # After all videos are processed, run frame extraction and post-processing
            if successful > 0:
                frame_config = config.get('frame_extraction', {})
                auto_extract = frame_config.get('auto_extract_frames', False)
                
                if auto_extract:
                    logger.info(f"\n{'='*80}")
                    logger.info("STARTING FRAME EXTRACTION FOR ALL VIDEOS")
                    logger.info(f"{'='*80}")
                    
                    try:
                        # Update config to use base chunks directory
                        extraction_config = config.copy()
                        extraction_config['chunks_directory'] = base_chunks_dir
                        
                        extractor = FrameSequenceExtractor(extraction_config)
                        total_sequences = extractor.extract_all_sequences()
                        logger.info(f"Frame extraction complete. Extracted {total_sequences} sequences from all videos.")
                        
                        # Check if automatic post-processing is enabled
                        post_config = config.get('post_processing', {})
                        post_enabled = post_config.get('enabled', False)
                        
                        if post_enabled:
                            logger.info(f"\n{'='*80}")
                            logger.info("STARTING POST-PROCESSING FOR ALL EXTRACTED FRAMES")
                            logger.info(f"{'='*80}")
                            
                            try:
                                post_processor = PostProcessor(extraction_config)
                                post_processor.run()
                                logger.info("Post-processing complete for all videos.")
                            except Exception as e:
                                logger.error(f"Error during post-processing: {str(e)}")
                                
                    except Exception as e:
                        logger.error(f"Error during frame extraction: {str(e)}")
            
            return 0 if failed == 0 else 1
        
        else:
            # Process single video file
            logger.info(f"Single video processing mode (auto-detected)")
            logger.info(f"Processing file: {input_path}")
            
            # Set input_video for compatibility with VideoProcessor
            config['input_video'] = input_path
            success, error = process_single_video(config)
            return 0 if success else 1
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
