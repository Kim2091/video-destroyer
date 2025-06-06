import logging
import argparse
from utils.config_loader import load_config
from utils.codec_handler import CodecHandler
from utils.video_processor import VideoProcessor, check_ffmpeg_available
from utils.logging_utils import setup_global_logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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

        # Initialize codec handler
        codec_handler = CodecHandler(config['codecs'])
        
        # Initialize video processor
        video_processor = VideoProcessor(config, codec_handler)
        
        # Process the video
        processed_pairs = video_processor.process_video()
        
        logger.info(f"Video processing complete. Created {len(processed_pairs)} HR/LR chunk pairs.")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
