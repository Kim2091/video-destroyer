import logging
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tests.helpers import load_tool_module, write_image, write_sequence
from utils.logging_utils import DegradationLogger, setup_global_logging


COMPARE = load_tool_module("tools/Compare Folders/compareFolders.py", "compare_folders_tool")
TILER = load_tool_module("tools/Tile Video Frames/tileVideoFrames.py", "tile_video_frames_tool")
MOTION = load_tool_module("tools/Video Frame Motion Detection/VideoFrameMotionDetect.py", "motion_detection_tool")


class FolderComparisonToolTests(unittest.TestCase):
    def test_compare_moves_only_asymmetric_files_and_preserves_relative_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline"
            secondary = root / "secondary"
            output = root / "output"
            (baseline / "shared").mkdir(parents=True)
            (secondary / "shared").mkdir(parents=True)
            (baseline / "shared" / "same.txt").write_text("same", encoding="utf-8")
            (secondary / "shared" / "same.txt").write_text("same", encoding="utf-8")
            (baseline / "only-baseline" / "item.txt").parent.mkdir(parents=True)
            (baseline / "only-baseline" / "item.txt").write_text("baseline", encoding="utf-8")
            (secondary / "only-secondary" / "item.txt").parent.mkdir(parents=True)
            (secondary / "only-secondary" / "item.txt").write_text("secondary", encoding="utf-8")
            COMPARE.compare_and_move_folders(baseline, secondary, output)
            self.assertTrue((output / "missing_from_secondary" / "only-baseline" / "item.txt").exists())
            self.assertTrue((output / "missing_from_baseline" / "only-secondary" / "item.txt").exists())
            self.assertTrue((baseline / "shared" / "same.txt").exists())
            self.assertTrue((secondary / "shared" / "same.txt").exists())

    def test_missing_folder_has_no_relative_files(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(set(), COMPARE.get_relative_files(Path(directory) / "missing"))


class TileToolTests(unittest.TestCase):
    def test_parsing_grouping_positions_and_real_tile_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            output_dir = root / "output"
            write_sequence(input_dir, 1, ["red", "blue"], size=(32, 24))
            self.assertEqual(("show00001", "00002", "show", 5), TILER.parse_filename("show00001_Frame00002.png"))
            self.assertEqual([(0, 0), (16, 0), (0, 12), (16, 12)], TILER.get_tile_positions(32, 24, 16, 12))
            scenes, prefix, width = TILER.group_frames_by_scene(input_dir)
            self.assertEqual("show", prefix)
            self.assertEqual(5, width)
            self.assertEqual([1, 2], [frame[0] for frame in scenes["show00001"]])
            output_dir.mkdir()
            sequence = scenes["show00001"]
            TILER.process_tile_sequence((0, (0, 0), sequence, str(output_dir), 16, 12, 1, prefix, width, 2, ".png"))
            with Image.open(output_dir / "show00001_Frame00001.png") as image:
                self.assertEqual((16, 12), image.size)


class MotionToolTests(unittest.TestCase):
    def test_grouping_motion_and_disabled_thresholds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_sequence(root, 1, ["red", "blue"])
            write_image(root / "ignored.png", "black")
            sequences = MOTION.group_frames_by_sequence(root)
            self.assertEqual([1, 2], [item[0] for item in sequences["show00001"]])
            red = __import__("cv2").imread(str(root / "show00001_Frame00001.png"))
            blue = __import__("cv2").imread(str(root / "show00001_Frame00002.png"))
            self.assertEqual(0.0, MOTION.calculate_motion_score(red, red))
            self.assertGreater(MOTION.calculate_motion_score(red, blue), 0.0)
            destination = root / "bad"
            destination.mkdir()
            MOTION.analyze_sequence_motion(root, "show00001", sequences["show00001"], None, None, destination)
            self.assertTrue((root / "show00001_Frame00001.png").exists())


class LoggingTests(unittest.TestCase):
    def test_logger_creates_a_log_file_and_global_setup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {"logging": {"directory": str(root), "filename": "run.log", "level": "DEBUG"}}
            logger = DegradationLogger(config)
            logger.log_chunk_start("chunk_0001.mkv")
            logger.log_degradation_applied("resize", True, 1.0, {"down_filter": "bilinear"})
            logger.log_chunk_complete("chunk_0001.mkv")
            self.assertEqual(2, len(logger.logger.handlers))
            self.assertEqual(1, len(list(root.glob("*_run.log"))))
            setup_global_logging(config)
            self.assertEqual(logging.DEBUG, logging.getLogger().level)
            logger.close()
            root_logger = logging.getLogger()
            for handler in root_logger.handlers[:]:
                handler.close()
                root_logger.removeHandler(handler)
