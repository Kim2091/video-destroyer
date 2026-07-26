import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from frame_extractor import FrameSequenceExtractor, merge_configs
from tests.helpers import make_video, minimal_config, write_image, write_sequence
from utils.post_processor import PostProcessor


class FrameExtractorTests(unittest.TestCase):
    def _paired_config(self, root):
        config = minimal_config(root)
        config["chunks_directory"] = str(root / "chunks")
        config["logging"] = {}
        config["frame_extraction"].update({"output_directory": str(root / "frames"), "auto_extract_frames": False})
        return config

    def _make_pair(self, root, source=None):
        source = source or make_video(root / "source.mp4")
        hr = root / "chunks" / "HR"
        lr = root / "chunks" / "LR"
        hr.mkdir(parents=True)
        lr.mkdir(parents=True)
        shutil.copy(source, hr / "chunk_0001.mp4")
        shutil.copy(source, lr / "chunk_0001.mp4")
        return hr / "chunk_0001.mp4", lr / "chunk_0001.mp4"

    def test_chunk_pairing_sequence_ids_and_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._paired_config(root)
            hr, lr = self._make_pair(root)
            write_image(root / "frames" / "HR" / "show00003_Frame00001.png", "red")
            extractor = FrameSequenceExtractor(config)
            self.assertEqual(4, extractor.sequence_counter)
            self.assertEqual([(str(hr), str(lr))], extractor.get_chunk_pairs())
            extractor.max_sequences = 1
            self.assertEqual([1], extractor._limit_start_frames([1, 4, 7]))
            extractor.max_sequences = 2
            self.assertEqual([1, 7], extractor._limit_start_frames([1, 4, 7]))
            extractor.max_sequences = 0
            self.assertEqual([], extractor._limit_start_frames([1, 4]))

    def test_extract_frames_and_sequences_from_real_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._paired_config(root)
            self._make_pair(root)
            extractor = FrameSequenceExtractor(config)
            self.assertEqual(5, extractor.extract_all_sequences())
            hr_frames = sorted((root / "frames" / "HR").glob("*.png"))
            lr_frames = sorted((root / "frames" / "LR").glob("*.png"))
            self.assertEqual(10, len(hr_frames))
            self.assertEqual([path.name for path in hr_frames], [path.name for path in lr_frames])
            with Image.open(hr_frames[0]) as image:
                self.assertEqual((64, 48), image.size)

    def test_rejects_mismatched_framerates_before_copying_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._paired_config(root)
            hr_source = make_video(root / "hr.mp4", rate=10)
            lr_source = make_video(root / "lr.mp4", rate=5)
            hr = root / "chunks" / "HR"
            lr = root / "chunks" / "LR"
            hr.mkdir(parents=True)
            lr.mkdir(parents=True)
            shutil.copy(hr_source, hr / "chunk_0001.mp4")
            shutil.copy(lr_source, lr / "chunk_0001.mp4")
            self.assertEqual(0, FrameSequenceExtractor(config).extract_all_sequences())
            self.assertFalse(any((root / "frames" / "HR").glob("*.png")))

    def test_extract_frame_sequence_rejects_missing_input_without_partial_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            extractor = FrameSequenceExtractor(self._paired_config(root))
            hr_temp = root / "temp_hr"
            lr_temp = root / "temp_lr"
            write_image(hr_temp / "frame_00001.png", "red")
            write_image(lr_temp / "frame_00001.png", "red")
            self.assertFalse(extractor.extract_frame_sequence(str(hr_temp), str(lr_temp), 1))
            self.assertFalse(any((root / "frames" / "HR").glob("*.png")))

    def test_merge_configs_preserves_yaml_boolean_values_without_flags(self):
        config = {"frame_extraction": {"use_scene_detection": True, "extract_full_chunks": True, "verbose_logging": True}}
        merged = merge_configs(config, {"use_scene_detection": None, "extract_full": None, "verbose": None})
        self.assertTrue(merged["frame_extraction"]["use_scene_detection"])
        self.assertTrue(merged["frame_extraction"]["extract_full_chunks"])
        self.assertTrue(merged["frame_extraction"]["verbose_logging"])


class PostProcessorTests(unittest.TestCase):
    def _config(self, root):
        config = minimal_config(root)
        config["frame_extraction"].update({"output_directory": str(root / "frames"), "sequence_length": 2})
        config["post_processing"] = {
            "enabled": True,
            "tiling": {"enabled": False, "tile_width": 16, "tile_height": 12, "workers": 1, "seed": 1},
            "blank_detection": {"enabled": False, "min_blank_frames": 1, "edge_threshold": 1, "variance_threshold": 1},
            "motion_detection": {"enabled": False, "min_motion": -1, "max_motion": -1, "threshold": 10},
            "sequence_completeness": {"enabled": False},
        }
        return config

    def test_actual_pair_scale_and_tiling_use_frame_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._config(root)
            write_sequence(root / "frames" / "HR", 1, ["red", "blue"], size=(64, 48))
            write_sequence(root / "frames" / "LR", 1, ["red", "blue"], size=(32, 24))
            processor = PostProcessor(config)
            self.assertEqual(0.5, processor._get_paired_scale_factor())
            config["post_processing"]["tiling"]["enabled"] = True
            processor = PostProcessor(config)
            processor._tile_frames()
            with Image.open(next((root / "frames" / "hr_tiled").glob("*.png"))) as image:
                self.assertEqual((16, 12), image.size)
            with Image.open(next((root / "frames" / "lr_tiled").glob("*.png"))) as image:
                self.assertEqual((8, 6), image.size)

    def test_blank_motion_completeness_and_sync_move_invalid_sequences(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._config(root)
            hr = root / "frames" / "HR"
            lr = root / "frames" / "LR"
            write_sequence(hr, 1, ["black", "black"])
            write_sequence(lr, 1, ["black", "black"])
            processor = PostProcessor(config)
            processor.blank_enabled = True
            processor._detect_blank_frames()
            self.assertTrue((root / "frames" / "hr_bad" / "show00001_Frame00001.png").exists())
            processor._sync_lr_with_hr()
            self.assertTrue((root / "frames" / "lr_bad" / "show00001_Frame00001.png").exists())

            write_sequence(hr, 2, ["red"])
            processor._check_sequence_completeness()
            self.assertTrue((root / "frames" / "hr_bad" / "show00002_Frame00001.png").exists())

    def test_motion_score_and_filename_parsing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            processor = PostProcessor(self._config(root))
            self.assertEqual(("show00001", 2), processor._parse_sequence_filename("show00001_Frame00002.png"))
            self.assertEqual((None, None), processor._parse_sequence_filename("invalid.png"))
            red = __import__("cv2").imread(str(write_image(root / "red.png", "red")))
            blue = __import__("cv2").imread(str(write_image(root / "blue.png", "blue")))
            self.assertEqual(0.0, processor._calculate_motion_score(red, red))
            self.assertGreater(processor._calculate_motion_score(red, blue), 0.0)
