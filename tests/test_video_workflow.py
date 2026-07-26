import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import ffmpeg

import main as application_main
from tests.helpers import ROOT, make_cut_video, make_video, minimal_config, write_yaml
from utils.codec_handler import CodecHandler
from utils.config_loader import load_config
from utils.scene_detector import SceneDetector
from utils.video_processor import VideoProcessor, check_ffmpeg_available


class SceneDetectorTests(unittest.TestCase):
    def test_get_video_info_and_no_cut_scene(self):
        with tempfile.TemporaryDirectory() as directory:
            source = make_video(Path(directory) / "source.mp4")
            detector = SceneDetector(config={"scene_detection": {"threshold": 10, "downscale_factor": 1}})
            info = detector.get_video_info(str(source))
            self.assertEqual((64, 48), (info["width"], info["height"]))
            self.assertEqual(10.0, info["fps"])
            self.assertEqual(10, info["nb_frames"])
            scenes = detector.detect_scenes(str(source))
            self.assertEqual(1, len(scenes))
            self.assertEqual(0, scenes[0][0].get_frames())

    def test_detects_a_hard_cut(self):
        with tempfile.TemporaryDirectory() as directory:
            source = make_cut_video(Path(directory) / "cut.mp4")
            detector = SceneDetector(config={"scene_detection": {"threshold": 5, "downscale_factor": 1}})
            self.assertGreaterEqual(len(detector.detect_scenes(str(source))), 2)


class VideoProcessorTests(unittest.TestCase):
    def _loaded_config(self, root, source):
        return load_config(write_yaml(root / "config.yaml", minimal_config(root, source)))

    def test_ffmpeg_is_available(self):
        self.assertTrue(check_ffmpeg_available())

    def test_chunk_helpers_and_cleanup_keep_unmanaged_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = make_video(root / "source.mp4")
            config = self._loaded_config(root, source)
            processor = VideoProcessor(config, CodecHandler(config["codecs"]))
            self.assertEqual(2, processor._round_to_even(1))
            self.assertTrue(processor._is_managed_chunk_file("chunk_0001.mkv"))
            self.assertFalse(processor._is_managed_chunk_file("notes.txt"))
            (Path(processor.hr_directory) / "notes.txt").write_text("keep", encoding="utf-8")
            (Path(processor.lr_directory) / "nested").mkdir()
            (Path(processor.lr_directory) / "chunk_0001.mkv").write_text("old", encoding="utf-8")
            processor.split_video()
            self.assertTrue((Path(processor.hr_directory) / "notes.txt").exists())
            self.assertTrue((Path(processor.lr_directory) / "nested").is_dir())
            self.assertEqual(2, len(list(Path(processor.hr_directory).glob("chunk_*"))))
            processor.logger.close()

    def test_process_video_creates_real_hr_lr_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = make_video(root / "source.mp4")
            config = self._loaded_config(root, source)
            processor = VideoProcessor(config, CodecHandler(config["codecs"]))
            pairs = processor.process_video()
            self.assertEqual(2, len(pairs))
            for hr_path, lr_path in pairs:
                self.assertTrue(Path(hr_path).exists())
                self.assertTrue(Path(lr_path).exists())
                hr_stream = next(stream for stream in ffmpeg.probe(hr_path)["streams"] if stream["codec_type"] == "video")
                lr_stream = next(stream for stream in ffmpeg.probe(lr_path)["streams"] if stream["codec_type"] == "video")
                self.assertEqual((64, 48), (hr_stream["width"], hr_stream["height"]))
                self.assertEqual((32, 24), (lr_stream["width"], lr_stream["height"]))
            processor.logger.close()

    def test_existing_chunks_process_without_source_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = make_video(root / "source.mp4")
            config = self._loaded_config(root, source)
            first = VideoProcessor(config, CodecHandler(config["codecs"]))
            pairs = first.process_video()
            config["use_existing_chunks"] = True
            config["input_video"] = ""
            reprocessor = VideoProcessor(config, CodecHandler(config["codecs"]))
            self.assertEqual(len(pairs), len(reprocessor.process_video()))
            reprocessor.logger.close()

    def test_split_command_has_expected_audio_and_resize_behavior(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = make_video(root / "source.mp4")
            config = self._loaded_config(root, source)
            config["scene_detection"]["hr_resize"] = {"enabled": True, "scale": 0.51, "filters": ["bilinear"]}
            processor = VideoProcessor(config, CodecHandler(config["codecs"]))
            command = processor._create_ffmpeg_split_command(0, 5, str(root / "piece.mkv"))
            self.assertIn("scale=32:24:sws_flags=bilinear", command)
            self.assertIn("-an", command)
            processor.logger.close()


class MainIntegrationTests(unittest.TestCase):
    def test_video_file_helpers_only_return_supported_top_level_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "clip.mp4").touch()
            (root / "CLIP.MKV").touch()
            (root / "ignore.txt").touch()
            (root / "nested").mkdir()
            (root / "nested" / "nested.mp4").touch()
            files = application_main.get_video_files(root, [".mp4", ".mkv"])
            self.assertEqual([str(root / "CLIP.MKV"), str(root / "clip.mp4")], files)
            self.assertEqual("archive.tar", application_main.get_video_name("archive.tar.mp4"))

    def test_batch_mode_isolates_each_video_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            input_dir.mkdir()
            make_video(input_dir / "first.mp4")
            make_video(input_dir / "second.mp4")
            config = minimal_config(root, input_dir)
            config["frame_extraction"]["auto_extract_frames"] = True
            config_path = write_yaml(root / "config.yaml", config)
            result = subprocess.run(["python", str(ROOT / "main.py"), "--config", str(config_path)], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(0, result.returncode, result.stderr)
            for name in ("first", "second"):
                self.assertGreater(len(list((root / "chunks" / name / "HR").glob("chunk_*"))), 0)
                self.assertGreater(len(list((root / "chunks" / name / "LR").glob("chunk_*"))), 0)
            self.assertGreater(len(list((root / "frames" / "HR").glob("*.png"))), 0)

    def test_existing_chunks_mode_does_not_require_input_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chunks_hr = root / "chunks" / "HR"
            chunks_hr.mkdir(parents=True)
            make_video(chunks_hr / "chunk_0001.mp4")
            config = minimal_config(root)
            config["input"] = ""
            config["use_existing_chunks"] = True
            result = subprocess.run(["python", str(ROOT / "main.py"), "--config", str(write_yaml(root / "existing.yaml", config))], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertTrue((root / "chunks" / "LR" / "chunk_0001.mp4").exists())
