# c:\Users\mystery\Pictures\Datasets\video-destroyer\post_process.py
# This script is optional, you can use this to run post processing on frames that you have already extracted
# By default the main script will run these steps automatically

import os
import logging
import argparse
import yaml
from pathlib import Path
from utils.post_processor import PostProcessor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        raise RuntimeError(f"Error loading configuration: {str(e)}")


def main():
    parser = argparse.ArgumentParser(
        description='Post-process extracted video frames: tile, detect motion, and sync folders',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config.yaml
  python post_process.py

  # Run with custom config
  python post_process.py --config my_config.yaml

  # Override frames directory
  python post_process.py --frames-dir ./my_frames

  # Disable specific steps
  python post_process.py --no-tiling
  python post_process.py --no-motion

  # Override tiling parameters
  python post_process.py --tile-size 256 --seed 42

  # Override motion detection thresholds
  python post_process.py --max-motion 20.0 --min-motion 0.5
        """
    )
    
    # Configuration file
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to configuration file (default: config.yaml)')
    
    # Directory overrides
    parser.add_argument('--frames-dir', type=str,
                        help='Override frames directory from config')
    
    # Enable/disable steps
    parser.add_argument('--no-tiling', action='store_true',
                        help='Disable tiling step')
    parser.add_argument('--no-motion', action='store_true',
                        help='Disable motion detection step')
    
    # Tiling parameters
    parser.add_argument('--tile-width', type=int,
                        help='Override tile width for HR frames')
    parser.add_argument('--tile-height', type=int,
                        help='Override tile height for HR frames')
    parser.add_argument('--seed', type=int,
                        help='Override random seed for tile selection')
    parser.add_argument('--workers', type=int,
                        help='Override number of parallel workers')
    
    # Motion detection parameters
    parser.add_argument('--min-motion', type=float,
                        help='Override minimum motion threshold percentage (-1 to disable)')
    parser.add_argument('--max-motion', type=float,
                        help='Override maximum motion threshold percentage (-1 to disable)')
    parser.add_argument('--motion-threshold', type=int,
                        help='Override pixel difference threshold (0-255)')
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)
        
        # Apply command-line overrides
        if 'post_processing' not in config:
            config['post_processing'] = {}
        
        post_config = config['post_processing']
        
        # Override frames directory
        if args.frames_dir:
            if 'frame_extraction' not in config:
                config['frame_extraction'] = {}
            config['frame_extraction']['output_directory'] = args.frames_dir
            logger.info(f"Overriding frames directory: {args.frames_dir}")
        
        # Override tiling settings
        if args.no_tiling:
            if 'tiling' not in post_config:
                post_config['tiling'] = {}
            post_config['tiling']['enabled'] = False
            logger.info("Tiling disabled via command-line")
        
        if 'tiling' not in post_config:
            post_config['tiling'] = {}
        
        if args.tile_width:
            post_config['tiling']['tile_width'] = args.tile_width
            logger.info(f"Overriding tile width: {args.tile_width}")
        
        if args.tile_height:
            post_config['tiling']['tile_height'] = args.tile_height
            logger.info(f"Overriding tile height: {args.tile_height}")
        
        if args.seed is not None:
            post_config['tiling']['seed'] = args.seed
            logger.info(f"Overriding seed: {args.seed}")
        
        if args.workers is not None:
            post_config['tiling']['workers'] = args.workers
            logger.info(f"Overriding workers: {args.workers}")
        
        # Override motion detection settings
        if args.no_motion:
            if 'motion_detection' not in post_config:
                post_config['motion_detection'] = {}
            post_config['motion_detection']['enabled'] = False
            logger.info("Motion detection disabled via command-line")
        
        if 'motion_detection' not in post_config:
            post_config['motion_detection'] = {}
        
        if args.min_motion is not None:
            post_config['motion_detection']['min_motion'] = args.min_motion
            logger.info(f"Overriding min motion: {args.min_motion}")
        
        if args.max_motion is not None:
            post_config['motion_detection']['max_motion'] = args.max_motion
            logger.info(f"Overriding max motion: {args.max_motion}")
        
        if args.motion_threshold is not None:
            post_config['motion_detection']['threshold'] = args.motion_threshold
            logger.info(f"Overriding motion threshold: {args.motion_threshold}")
        
        # Ensure post-processing is enabled
        post_config['enabled'] = True
        
        # Verify frames directory exists
        frame_config = config.get('frame_extraction', {})
        frames_dir = frame_config.get('output_directory', 'frames')
        hr_dir = os.path.join(frames_dir, 'HR')
        lr_dir = os.path.join(frames_dir, 'LR')
        
        if not os.path.exists(hr_dir) or not os.path.exists(lr_dir):
            logger.error(f"Frames directories not found: {hr_dir} and/or {lr_dir}")
            logger.error("Please run frame extraction first or specify correct --frames-dir")
            return 1
        
        # Initialize and run post-processor
        logger.info("Initializing post-processor...")
        post_processor = PostProcessor(config)
        
        # Run the pipeline
        post_processor.run()
        
        logger.info("Post-processing completed successfully!")
        return 0
        
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1
    except Exception as e:
        logger.error(f"Error during post-processing: {str(e)}")
        logger.exception("Full traceback:")
        return 1


if __name__ == "__main__":
    exit(main())