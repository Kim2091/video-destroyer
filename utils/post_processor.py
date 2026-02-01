import os
import logging
import shutil
import random
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from typing import Dict, Any, List, Tuple, Set
import re
from tqdm import tqdm

logger = logging.getLogger(__name__)


class PostProcessor:
    """
    Post-processing pipeline for extracted frames including:
    1. Tiling video frames with scale-aware sizing
    2. Blank frame detection to filter low-detail sequences
    3. Motion detection on HR frames
    4. Folder comparison to remove bad LR frames
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the post-processor with configuration.
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config
        post_config = config.get('post_processing', {})
        
        # Main toggle
        self.enabled = post_config.get('enabled', False)
        
        # Frame extraction config
        frame_config = config.get('frame_extraction', {})
        self.frames_directory = frame_config.get('output_directory', 'frames')
        self.hr_frames_dir = os.path.join(self.frames_directory, 'HR')
        self.lr_frames_dir = os.path.join(self.frames_directory, 'LR')
        
        # Tiling configuration
        tiling_config = post_config.get('tiling', {})
        self.tiling_enabled = tiling_config.get('enabled', False)
        self.tile_width = tiling_config.get('tile_width', 512)
        self.tile_height = tiling_config.get('tile_height', 512)
        self.tile_seed = tiling_config.get('seed', 1024)
        self.tile_workers = tiling_config.get('workers', None) or cpu_count()
        
        # Get scale factor from degradations config
        self.scale_factor = self._get_scale_factor(config)
        
        # Blank frame detection configuration
        blank_config = post_config.get('blank_detection', {})
        self.blank_enabled = blank_config.get('enabled', False)
        self.min_blank_frames = blank_config.get('min_blank_frames', 2)
        self.edge_threshold = blank_config.get('edge_threshold', 15.0)
        self.variance_threshold = blank_config.get('variance_threshold', 100.0)
        
        # Motion detection configuration
        motion_config = post_config.get('motion_detection', {})
        self.motion_enabled = motion_config.get('enabled', False)
        self.min_motion = motion_config.get('min_motion', -1)
        self.max_motion = motion_config.get('max_motion', 15.0)
        self.motion_threshold = motion_config.get('threshold', 30)
        
        # Sequence completeness check configuration
        completeness_config = post_config.get('sequence_completeness', {})
        self.completeness_enabled = completeness_config.get('enabled', False)
        
        # Output directories for tiled and bad frames
        self.hr_tiled_dir = os.path.join(self.frames_directory, 'hr_tiled')
        self.lr_tiled_dir = os.path.join(self.frames_directory, 'lr_tiled')
        self.hr_tiled_bad_dir = os.path.join(self.frames_directory, 'hr_tiled_bad')
        self.lr_tiled_bad_dir = os.path.join(self.frames_directory, 'lr_tiled_bad')
        
    def _get_scale_factor(self, config: Dict[str, Any]) -> float:
        """Extract scale factor from resize degradation config."""
        degradations = config.get('degradations', [])
        for deg in degradations:
            if deg.get('name') == 'resize' and deg.get('enabled', False):
                params = deg.get('params', {})
                return params.get('fixed_scale', 1.0)
        return 1.0
    
    def run(self):
        """Execute the full post-processing pipeline."""
        if not self.enabled:
            logger.info("Post-processing is disabled")
            return
        
        logger.info("=" * 60)
        logger.info("Starting post-processing pipeline")
        logger.info("=" * 60)
        
        # Step 1: Tile frames
        if self.tiling_enabled:
            logger.info("\n[1/5] Tiling video frames...")
            self._tile_frames()
        else:
            logger.info("\n[1/5] Tiling disabled, skipping...")
        
        # Step 2: Blank frame detection
        if self.blank_enabled:
            logger.info("\n[2/5] Running blank frame detection on HR frames...")
            self._detect_blank_frames()
        else:
            logger.info("\n[2/5] Blank frame detection disabled, skipping...")
        
        # Step 3: Motion detection
        if self.motion_enabled:
            logger.info("\n[3/5] Running motion detection on HR frames...")
            self._detect_motion()
        else:
            logger.info("\n[3/5] Motion detection disabled, skipping...")
        
        # Step 4: Sequence completeness check
        if self.completeness_enabled:
            logger.info("\n[4/5] Checking sequence completeness...")
            self._check_sequence_completeness()
        else:
            logger.info("\n[4/5] Sequence completeness check disabled, skipping...")
        
        # Step 5: Remove corresponding bad LR frames
        if self.motion_enabled or self.blank_enabled or self.completeness_enabled:
            logger.info("\n[5/5] Removing corresponding bad LR frames...")
            self._sync_lr_with_hr()
        else:
            logger.info("\n[5/5] Folder sync disabled, skipping...")
        
        logger.info("\n" + "=" * 60)
        logger.info("Post-processing complete!")
        logger.info("=" * 60)
    
    # ========== TILING FUNCTIONS ==========
    
    def _parse_filename(self, filename: str) -> Tuple[str, str, str, int]:
        """Parse filename to extract scene/show/sequence identifier and frame number."""
        match = re.match(r'([a-zA-Z]+)(\d+)_[fF]rame(\d+)', filename)
        if match:
            prefix = match.group(1)
            scene_num = match.group(2)
            frame_num = match.group(3)
            scene_id = f"{prefix}{scene_num}"
            return scene_id, frame_num, prefix, len(scene_num)
        return None, None, None, None
    
    def _group_frames_by_scene(self, input_dir: str, extensions=('.png', '.jpg', '.jpeg', '.bmp', '.tiff')) -> Tuple[Dict, str, int]:
        """Group frames by scene number."""
        scenes = defaultdict(list)
        prefix = None
        num_width = None
        
        for file in os.listdir(input_dir):
            if not any(file.lower().endswith(ext) for ext in extensions):
                continue
                
            scene, frame, file_prefix, file_num_width = self._parse_filename(file)
            if scene and frame:
                filepath = os.path.join(input_dir, file)
                scenes[scene].append((int(frame), filepath, file))
                if prefix is None:
                    prefix = file_prefix
                    num_width = file_num_width
        
        # Sort frames within each scene
        for scene in scenes:
            scenes[scene].sort(key=lambda x: x[0])
        
        return scenes, prefix, num_width
    
    def _get_tile_positions(self, img_width: int, img_height: int, tile_width: int, tile_height: int) -> List[Tuple[int, int]]:
        """Calculate all possible tile positions for an image."""
        positions = []
        for y in range(0, img_height - tile_height + 1, tile_height):
            for x in range(0, img_width - tile_width + 1, tile_width):
                positions.append((x, y))
        return positions
    
    def _process_tile_sequence(self, args_tuple):
        """Worker function to process a single tile from a sequence."""
        tile_idx, (x, y), sequence, output_dir, tile_width, tile_height, global_show_counter, prefix, num_width, sequence_length, extension = args_tuple
        
        show_num = global_show_counter + tile_idx
        
        for frame_idx in range(sequence_length):
            frame_num, filepath, filename = sequence[frame_idx]
            
            with Image.open(filepath) as img:
                tile = img.crop((x, y, x + tile_width, y + tile_height))
                
                output_filename = f"{prefix}{show_num:0{num_width}d}_Frame{frame_idx + 1:05d}{extension}"
                output_path = os.path.join(output_dir, output_filename)
                
                tile.save(output_path)
        
        return sequence_length
    
    def _process_scene_tiling(self, scene_name: str, scene_frames: List, output_dir: str, 
                             sequence_length: int, tile_width: int, tile_height: int, 
                             global_show_counter: int, prefix: str, num_width: int, pool) -> int:
        """Process all frames in a scene, extracting tiles from sequences."""
        if len(scene_frames) < sequence_length:
            logger.debug(f"Skipping {scene_name} with only {len(scene_frames)} frames (need {sequence_length})")
            return global_show_counter
        
        first_frame_path = scene_frames[0][1]
        with Image.open(first_frame_path) as img:
            img_width, img_height = img.size
        extension = Path(scene_frames[0][2]).suffix
        
        tile_positions = self._get_tile_positions(img_width, img_height, tile_width, tile_height)
        
        if not tile_positions:
            logger.warning(f"Image dimensions ({img_width}x{img_height}) are smaller than tile size ({tile_width}x{tile_height})")
            return global_show_counter
        
        random.shuffle(tile_positions)
        
        num_sequences = len(scene_frames) - sequence_length + 1
        all_task_args = []
        
        for seq_start_idx in range(num_sequences):
            sequence = scene_frames[seq_start_idx:seq_start_idx + sequence_length]
            
            for tile_idx, tile_pos in enumerate(tile_positions):
                all_task_args.append((
                    tile_idx, tile_pos, sequence, output_dir, 
                    tile_width, tile_height, global_show_counter, 
                    prefix, num_width, sequence_length, extension
                ))
            
            global_show_counter += len(tile_positions)
        
        pool.map(self._process_tile_sequence, all_task_args)
        
        logger.debug(f"  Processed {num_sequences} sequences: {len(tile_positions)} tiles × {sequence_length} frames each")
        
        return global_show_counter
    
    def _tile_directory(self, input_dir: str, output_dir: str, tile_width: int, tile_height: int, sequence_length: int):
        """Tile all frames in a directory."""
        os.makedirs(output_dir, exist_ok=True)
        
        random.seed(self.tile_seed)
        
        scenes, prefix, num_width = self._group_frames_by_scene(input_dir)
        
        if not scenes:
            logger.warning(f"No valid frame sequences found in {input_dir}")
            return
        
        if prefix is None:
            logger.error(f"Could not determine prefix from filenames in {input_dir}")
            return
        
        logger.info(f"  Found {len(scenes)} scene(s) with prefix '{prefix}'")
        
        with Pool(processes=self.tile_workers) as pool:
            global_show_counter = 1
            for scene_name in sorted(scenes.keys()):
                scene_frames = scenes[scene_name]
                global_show_counter = self._process_scene_tiling(
                    scene_name, scene_frames, output_dir, sequence_length,
                    tile_width, tile_height, global_show_counter, prefix, num_width, pool
                )
        
        logger.info(f"  Tiling complete for {input_dir}")
    
    def _tile_frames(self):
        """Tile both HR and LR frames with scale-aware sizing."""
        frame_config = self.config.get('frame_extraction', {})
        sequence_length = frame_config.get('sequence_length', 5)
        
        # Calculate LR tile size based on scale factor
        lr_tile_width = int(self.tile_width * self.scale_factor)
        lr_tile_height = int(self.tile_height * self.scale_factor)
        
        logger.info(f"Tiling HR frames: {self.tile_width}x{self.tile_height}")
        logger.info(f"Tiling LR frames: {lr_tile_width}x{lr_tile_height} (scale factor: {self.scale_factor})")
        logger.info(f"Sequence length: {sequence_length}, Seed: {self.tile_seed}, Workers: {self.tile_workers}")
        
        # Tile HR frames
        logger.info("Tiling HR frames...")
        self._tile_directory(self.hr_frames_dir, self.hr_tiled_dir, 
                           self.tile_width, self.tile_height, sequence_length)
        
        # Tile LR frames with adjusted size
        logger.info("Tiling LR frames...")
        self._tile_directory(self.lr_frames_dir, self.lr_tiled_dir, 
                           lr_tile_width, lr_tile_height, sequence_length)
    
    # ========== BLANK FRAME DETECTION FUNCTIONS ==========
    
    def _detect_black_borders(self, img: np.ndarray, threshold: int = 10) -> Tuple[float, int, int, int, int]:
        """
        Detect black borders (letterboxing/pillarboxing) in an image.
        Returns (black_border_percentage, top, bottom, left, right) border sizes.
        """
        height, width = img.shape
        
        # Find top border
        top = 0
        for i in range(height // 2):
            if np.mean(img[i, :]) > threshold:
                break
            top = i + 1
        
        # Find bottom border
        bottom = 0
        for i in range(height - 1, height // 2, -1):
            if np.mean(img[i, :]) > threshold:
                break
            bottom = height - i
        
        # Find left border
        left = 0
        for i in range(width // 2):
            if np.mean(img[:, i]) > threshold:
                break
            left = i + 1
        
        # Find right border
        right = 0
        for i in range(width - 1, width // 2, -1):
            if np.mean(img[:, i]) > threshold:
                break
            right = width - i
        
        # Calculate percentage of image that is black borders
        total_pixels = height * width
        border_pixels = (top + bottom) * width + (left + right) * (height - top - bottom)
        border_percentage = (border_pixels / total_pixels) * 100
        
        return border_percentage, top, bottom, left, right
    
    def _calculate_frame_detail(self, frame_path: str) -> Tuple[float, float, float, float]:
        """
        Calculate detail metrics for a frame using multiple methods.
        Returns (edge_density_percentage, variance, mean_brightness, black_border_percentage).
        """
        img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.warning(f"Could not load frame {frame_path}")
            return 0.0, 0.0, 0.0, 0.0
        
        # Detect black borders
        border_percentage, top, bottom, left, right = self._detect_black_borders(img)
        
        # Crop to content area (excluding borders) for detail analysis
        height, width = img.shape
        content_img = img[top:height-bottom, left:width-right] if border_percentage < 90 else img
        
        # If content area is too small, use full image
        if content_img.size < 100:
            content_img = img
        
        # Calculate variance (measure of overall contrast/detail)
        variance = np.var(content_img)
        
        # Calculate mean brightness
        mean_brightness = np.mean(content_img)
        
        # Calculate edge density using Sobel operator (more robust than Canny for blank detection)
        sobelx = cv2.Sobel(content_img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(content_img, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Count pixels with significant edge strength
        edge_threshold = 30  # Threshold for edge magnitude
        edge_pixels = np.count_nonzero(edge_magnitude > edge_threshold)
        total_pixels = content_img.size
        edge_density = (edge_pixels / total_pixels) * 100 if total_pixels > 0 else 0.0
        
        return edge_density, variance, mean_brightness, border_percentage
    
    def _is_blank_frame(self, frame_path: str) -> Tuple[bool, float, float]:
        """
        Determine if a frame is blank/low-detail based on multiple metrics.
        Returns (is_blank, edge_density, variance).
        
        A frame is considered blank if it has:
        - Large black borders (>40% of frame)
        - OR (Low edge density AND low variance in content area)
        """
        edge_density, variance, mean_brightness, border_percentage = self._calculate_frame_detail(frame_path)
        
        # Mark as blank if excessive black borders
        if border_percentage > 40.0:
            is_blank = True
        # Or if content area has low detail
        elif (edge_density < self.edge_threshold) and (variance < self.variance_threshold):
            is_blank = True
        else:
            is_blank = False
        
        return is_blank, edge_density, variance
    
    def _analyze_sequence_blanks(self, input_dir: str, sequence_name: str, frames: List, move_to_dir: str) -> bool:
        """
        Analyze a sequence for blank frames and move if too many are found.
        Returns True if sequence was moved, False otherwise.
        """
        blank_count = 0
        blank_frames = []
        frame_details = []
        
        for frame_num, filename in frames:
            frame_path = os.path.join(input_dir, filename)
            is_blank, edge_density, variance = self._is_blank_frame(frame_path)
            frame_details.append((filename, edge_density, variance, is_blank))
            if is_blank:
                blank_count += 1
                blank_frames.append(filename)
        
        # If we have too many blank frames, move the entire sequence
        if blank_count >= self.min_blank_frames:
            for frame_num, filename in frames:
                src = os.path.join(input_dir, filename)
                dst = os.path.join(move_to_dir, filename)
                shutil.move(src, dst)
            
            # Log details for first moved sequence to help with threshold tuning
            if blank_count == self.min_blank_frames:
                logger.info(f"\nExample moved sequence {sequence_name}: {blank_count}/{len(frames)} blank frames")
                for fname, edge, var, is_b in frame_details:
                    status = "BLANK" if is_b else "OK"
                    logger.info(f"  {fname}: edge={edge:.2f}%, var={var:.2f} [{status}]")
            
            return True
        
        return False
    
    def _detect_blank_frames(self):
        """
        Run blank frame detection on HR frames (tiled or original).
        """
        # Determine which directory to process
        if self.tiling_enabled:
            input_dir = self.hr_tiled_dir
            bad_dir = self.hr_tiled_bad_dir
        else:
            input_dir = self.hr_frames_dir
            bad_dir = os.path.join(self.frames_directory, 'hr_bad')
        
        os.makedirs(bad_dir, exist_ok=True)
        
        # Group frames by sequence
        sequences = defaultdict(list)
        for filename in os.listdir(input_dir):
            if filename.endswith('.png') or filename.endswith('.jpg'):
                seq_name, frame_num = self._parse_sequence_filename(filename)
                if seq_name and frame_num:
                    sequences[seq_name].append((frame_num, filename))
        
        # Sort frames within each sequence
        for seq_name in sequences:
            sequences[seq_name].sort(key=lambda x: x[0])
        
        if not sequences:
            logger.warning("No valid frame sequences found in HR tiled directory")
            return
        
        logger.info(f"Found {len(sequences)} sequence(s) to analyze")
        logger.info(f"Blank detection thresholds: edge_density >= {self.edge_threshold}%, variance >= {self.variance_threshold}")
        logger.info(f"Sequences with {self.min_blank_frames}+ blank frames will be moved")
        
        moved_count = 0
        for seq_name in tqdm(sorted(sequences.keys()), desc="Analyzing blank frames"):
            if self._analyze_sequence_blanks(input_dir, seq_name, sequences[seq_name], bad_dir):
                moved_count += 1
        
        logger.info(f"Moved {moved_count} sequences with blank frames to {bad_dir}")
    
    # ========== MOTION DETECTION FUNCTIONS ==========
    
    def _calculate_motion_score(self, frame1, frame2) -> float:
        """Calculate motion score between two frames using percentage of changed pixels."""
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        diff = cv2.absdiff(gray1, gray2)
        _, thresh = cv2.threshold(diff, self.motion_threshold, 255, cv2.THRESH_BINARY)
        
        total_pixels = thresh.size
        changed_pixels = np.count_nonzero(thresh)
        motion_score = (changed_pixels / total_pixels) * 100
        
        return motion_score
    
    def _analyze_sequence_motion(self, input_dir: str, sequence_name: str, frames: List, move_to_dir: str) -> bool:
        """Analyze motion in a sequence and move if issues found."""
        if len(frames) < 2:
            return False
        
        issues_found = False
        
        for i in range(len(frames) - 1):
            frame_num1, filename1 = frames[i]
            frame_num2, filename2 = frames[i + 1]
            
            img1_path = os.path.join(input_dir, filename1)
            img2_path = os.path.join(input_dir, filename2)
            
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)
            
            if img1 is None or img2 is None:
                logger.warning(f"Could not load frames {filename1} or {filename2}")
                continue
            
            motion_score = self._calculate_motion_score(img1, img2)
            
            if self.min_motion >= 0 and motion_score < self.min_motion:
                issues_found = True
                break
            elif self.max_motion >= 0 and motion_score > self.max_motion:
                issues_found = True
                break
        
        if issues_found:
            # Move entire sequence
            for frame_num, filename in frames:
                src = os.path.join(input_dir, filename)
                dst = os.path.join(move_to_dir, filename)
                shutil.move(src, dst)
            return True
        
        return False
    
    def _detect_motion(self):
        """Run motion detection on HR frames (tiled or original)."""
        # Determine which directory to process
        if self.tiling_enabled:
            input_dir = self.hr_tiled_dir
            bad_dir = self.hr_tiled_bad_dir
        else:
            input_dir = self.hr_frames_dir
            bad_dir = os.path.join(self.frames_directory, 'hr_bad')
        
        os.makedirs(bad_dir, exist_ok=True)
        
        # Group frames by sequence
        sequences = defaultdict(list)
        for filename in os.listdir(input_dir):
            if filename.endswith('.png') or filename.endswith('.jpg'):
                seq_name, frame_num = self._parse_sequence_filename(filename)
                if seq_name and frame_num:
                    sequences[seq_name].append((frame_num, filename))
        
        # Sort frames within each sequence
        for seq_name in sequences:
            sequences[seq_name].sort(key=lambda x: x[0])
        
        if not sequences:
            logger.warning("No valid frame sequences found in HR tiled directory")
            return
        
        logger.info(f"Found {len(sequences)} sequence(s) to analyze")
        
        threshold_parts = []
        if self.min_motion >= 0:
            threshold_parts.append(f"min: {self.min_motion}%")
        if self.max_motion >= 0:
            threshold_parts.append(f"max: {self.max_motion}%")
        
        if threshold_parts:
            logger.info(f"Motion thresholds: {', '.join(threshold_parts)}")
        
        moved_count = 0
        for seq_name in tqdm(sorted(sequences.keys()), desc="Analyzing motion"):
            if self._analyze_sequence_motion(input_dir, seq_name, sequences[seq_name], bad_dir):
                moved_count += 1
        
        logger.info(f"Moved {moved_count} sequences with motion issues to {bad_dir}")
    
    def _parse_sequence_filename(self, filename: str) -> Tuple[str, int]:
        """Parse filename to extract sequence name and frame number."""
        match = re.match(r'(.+)_Frame(\d+)\.(png|jpg|jpeg)', filename)
        if match:
            return match.group(1), int(match.group(2))
        return None, None
    
    # ========== SEQUENCE COMPLETENESS CHECK FUNCTIONS ==========
    
    def _check_sequence_completeness(self):
        """
        Check if sequences are complete (no missing frames).
        Move incomplete sequences to bad folder.
        """
        # Determine which directory to process
        if self.tiling_enabled:
            input_dir = self.hr_tiled_dir
            bad_dir = self.hr_tiled_bad_dir
        else:
            input_dir = self.hr_frames_dir
            bad_dir = os.path.join(self.frames_directory, 'hr_bad')
        
        os.makedirs(bad_dir, exist_ok=True)
        
        # Get expected sequence length
        frame_config = self.config.get('frame_extraction', {})
        expected_length = frame_config.get('sequence_length', 5)
        
        # Group frames by sequence
        sequences = defaultdict(list)
        for filename in os.listdir(input_dir):
            if filename.endswith('.png') or filename.endswith('.jpg'):
                seq_name, frame_num = self._parse_sequence_filename(filename)
                if seq_name and frame_num:
                    sequences[seq_name].append((frame_num, filename))
        
        # Sort frames within each sequence
        for seq_name in sequences:
            sequences[seq_name].sort(key=lambda x: x[0])
        
        if not sequences:
            logger.warning("No valid frame sequences found")
            return
        
        logger.info(f"Found {len(sequences)} sequence(s) to check")
        logger.info(f"Expected sequence length: {expected_length} frames")
        
        incomplete_count = 0
        moved_count = 0
        
        for seq_name in tqdm(sorted(sequences.keys()), desc="Checking completeness"):
            frames = sequences[seq_name]
            
            # Check if sequence has correct number of frames
            if len(frames) != expected_length:
                incomplete_count += 1
                # Move entire sequence to bad folder
                for frame_num, filename in frames:
                    src = os.path.join(input_dir, filename)
                    dst = os.path.join(bad_dir, filename)
                    shutil.move(src, dst)
                moved_count += 1
                continue
            
            # Check if frame numbers are consecutive
            frame_numbers = [f[0] for f in frames]
            expected_numbers = list(range(1, expected_length + 1))
            
            if frame_numbers != expected_numbers:
                incomplete_count += 1
                # Move entire sequence to bad folder
                for frame_num, filename in frames:
                    src = os.path.join(input_dir, filename)
                    dst = os.path.join(bad_dir, filename)
                    shutil.move(src, dst)
                moved_count += 1
                logger.debug(f"Sequence {seq_name} has non-consecutive frames: {frame_numbers}")
        
        logger.info(f"Found {incomplete_count} incomplete sequence(s)")
        logger.info(f"Moved {moved_count} incomplete sequences to {bad_dir}")
    
    # ========== FOLDER COMPARISON FUNCTIONS ==========
    
    def _get_relative_files(self, directory: str) -> Set[Path]:
        """Get all files in a directory with their relative paths."""
        directory = Path(directory)
        files = set()
        
        if not directory.exists():
            logger.warning(f"Directory '{directory}' does not exist")
            return files
        
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                full_path = Path(root) / filename
                relative_path = full_path.relative_to(directory)
                files.add(relative_path)
        
        return files
    
    def _move_files(self, source_dir: str, relative_paths: Set[Path], destination_dir: str) -> int:
        """Move files from source directory to destination directory."""
        source_dir = Path(source_dir)
        destination_dir = Path(destination_dir)
        
        moved_count = 0
        
        for rel_path in relative_paths:
            source_file = source_dir / rel_path
            dest_file = destination_dir / rel_path
            
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                shutil.move(str(source_file), str(dest_file))
                moved_count += 1
            except Exception as e:
                logger.error(f"Error moving {rel_path}: {e}")
        
        return moved_count
    
    def _sync_lr_with_hr(self):
        """Compare HR and LR folders and move missing LR frames to bad directory."""
        # Determine which directories to process
        if self.tiling_enabled:
            hr_dir = self.hr_tiled_dir
            lr_dir = self.lr_tiled_dir
            lr_bad_dir = self.lr_tiled_bad_dir
        else:
            hr_dir = self.hr_frames_dir
            lr_dir = self.lr_frames_dir
            lr_bad_dir = os.path.join(self.frames_directory, 'lr_bad')
        
        os.makedirs(lr_bad_dir, exist_ok=True)
        
        logger.info("Scanning directories...")
        hr_files = self._get_relative_files(hr_dir)
        lr_files = self._get_relative_files(lr_dir)
        
        logger.info(f"Files in HR tiled: {len(hr_files)}")
        logger.info(f"Files in LR tiled: {len(lr_files)}")
        
        # Find LR files that don't have corresponding HR files (HR was moved to bad)
        missing_from_hr = lr_files - hr_files
        
        logger.info(f"Files in LR but missing from HR: {len(missing_from_hr)}")
        
        if not missing_from_hr:
            logger.info("No missing files found. Directories are in sync!")
            return
        
        # Move the orphaned LR files
        logger.info(f"Moving {len(missing_from_hr)} LR files to bad directory...")
        count = self._move_files(lr_dir, missing_from_hr, lr_bad_dir)
        
        logger.info(f"Moved {count} files to {lr_bad_dir}")