import copy
import tempfile
import unittest
from pathlib import Path

from tests.helpers import minimal_config, write_yaml
from utils.codec_handler import CodecHandler
from utils.config_loader import (
    _normalize_path_values,
    convert_ranges_to_dict,
    load_config,
    validate_degradation_config,
)


class ConfigLoaderTests(unittest.TestCase):
    def test_validate_degradation_config_rejects_missing_and_invalid_ranges(self):
        with self.assertRaises(ValueError):
            validate_degradation_config({"name": "resize"})
        with self.assertRaises(ValueError):
            validate_degradation_config({"name": "resize", "enabled": True, "params": {"down_up": {"range": [1]}}})
        with self.assertRaises(ValueError):
            validate_degradation_config({"name": "codec", "enabled": True, "params": {"h264": {"quality_range": 10}}})

    def test_convert_ranges_only_changes_codec_ranges(self):
        config = {
            "degradations": [
                {"name": "resize", "params": {"down_up": {"range": [0.25, 0.75]}}},
                {"name": "codec", "params": {"h264": {"quality_range": [20, 30]}}},
            ]
        }
        converted = convert_ranges_to_dict(config)
        self.assertEqual([0.25, 0.75], converted["degradations"][0]["params"]["down_up"]["range"])
        self.assertEqual({"min": 20, "max": 30}, converted["degradations"][1]["params"]["h264"]["quality_range"])

    def test_normalize_path_values_leaves_non_paths_untouched(self):
        config = {"input": "a/b", "chunks_directory": "c/d", "frame_extraction": {"output_directory": "e/f"}, "logging": {"directory": "g/h"}, "label": "a/b"}
        _normalize_path_values(config)
        self.assertEqual("a/b", config["label"])
        self.assertEqual(str(Path("a/b")), config["input"])

    def test_load_config_creates_chunk_directories_and_normalizes_codecs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = minimal_config(root)
            loaded = load_config(write_yaml(root / "config.yaml", config))
            self.assertTrue((root / "chunks" / "HR").is_dir())
            self.assertTrue((root / "chunks" / "LR").is_dir())
            self.assertEqual({"min": 28, "max": 28}, loaded["codecs"]["h264"]["quality_range"])

    def test_load_config_rejects_invalid_top_level_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = minimal_config(root)
            del config["chunk_strategy"]
            with self.assertRaises(ValueError):
                load_config(write_yaml(root / "invalid.yaml", config))

    def test_load_config_requires_existing_hr_chunks_in_existing_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = minimal_config(root)
            config["use_existing_chunks"] = True
            with self.assertRaises(FileNotFoundError):
                load_config(write_yaml(root / "existing.yaml", config))


class CodecHandlerTests(unittest.TestCase):
    def test_normalizes_probabilities_and_selects_valid_values(self):
        config = {
            "h264": {"probability": 2.0, "quality_range": {"min": 20, "max": 20}, "presets": ["fast"]},
            "av1": {"probability": 2.0, "quality_range": {"min": 30, "max": 30}, "preset_range": [4, 4]},
        }
        handler = CodecHandler(config)
        self.assertEqual(1.0, sum(item["probability"] for item in config.values()))
        for _ in range(20):
            codec, quality, preset = handler.get_random_encoding_config()
            self.assertIn(codec, config)
            self.assertEqual(config[codec]["quality_range"]["min"], quality)
            self.assertIn(preset, ("fast", 4))

    def test_rejects_missing_or_reversed_quality_ranges(self):
        with self.assertRaises(ValueError):
            CodecHandler({"h264": {"probability": 1}})
        with self.assertRaises(ValueError):
            CodecHandler({"h264": {"probability": 1, "quality_range": {"min": 30, "max": 20}}})

    def test_unknown_codec_is_rejected(self):
        handler = CodecHandler({"h264": {"probability": 1, "quality_range": {"min": 20, "max": 20}}})
        with self.assertRaises(ValueError):
            handler.get_random_quality("missing")
        with self.assertRaises(ValueError):
            handler.get_random_preset("missing")
