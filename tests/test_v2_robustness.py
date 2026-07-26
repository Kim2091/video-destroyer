"""Recovery and input-validation behaviour of the version 2 package."""

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tests.helpers import make_video
from video_destroyer.config import ConfigError, load_config
from video_destroyer.curation import curate_sequences
from video_destroyer.models import SequenceRecord
from video_destroyer.pairing import PairingError, discover_import_pairs


def _frame(path, colour):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), colour).save(path)


class CurationRejectionTests(unittest.TestCase):
    CONFIG = {"tiling": {"enabled": False}, "blank_detection": {}, "motion_detection": {}}

    def _curate(self, directory, sequence, retain_rejected):
        work = Path(directory) / "work"
        (work / "frames" / "hr").mkdir(parents=True)
        (work / "frames" / "lr").mkdir(parents=True)
        return curate_sequences([sequence], work, self.CONFIG, retain_rejected, Path(directory) / "rejected"), work

    def test_retaining_an_unreadable_sequence_rejects_it_instead_of_failing_the_run(self):
        with tempfile.TemporaryDirectory() as directory:
            sequence = SequenceRecord("s1", "p1", 1, 1, ["missing.png"], ["missing.png"])

            curated, _ = self._curate(directory, sequence, retain_rejected=True)

            self.assertEqual("rejected", curated[0].status)
            self.assertIn("not retained", curated[0].rejection_reason)

    def test_a_readable_rejected_sequence_is_still_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            config = dict(self.CONFIG, blank_detection={"enabled": True})
            work = Path(directory) / "work"
            _frame(work / "frames" / "hr" / "f.png", "black")
            _frame(work / "frames" / "lr" / "f.png", "black")
            rejected_root = Path(directory) / "rejected"
            sequence = SequenceRecord("s1", "p1", 1, 1, ["f.png"], ["f.png"])

            curated = curate_sequences([sequence], work, config, True, rejected_root)

            self.assertEqual("rejected", curated[0].status)
            self.assertEqual("blank or low-detail frame", curated[0].rejection_reason)
            self.assertTrue((rejected_root / "hr" / "f.png").is_file())
            self.assertTrue((rejected_root / "lr" / "f.png").is_file())

    def test_curation_does_not_hold_frames_open(self):
        with tempfile.TemporaryDirectory() as directory:
            config = dict(self.CONFIG, blank_detection={"enabled": True}, motion_detection={"enabled": True})
            work = Path(directory) / "work"
            for side in ("hr", "lr"):
                _frame(work / "frames" / side / "a.png", "black")
                _frame(work / "frames" / side / "b.png", "black")
            sequence = SequenceRecord("s1", "p1", 1, 2, ["a.png", "b.png"], ["a.png", "b.png"])

            curate_sequences([sequence], work, config, False, Path(directory) / "rejected")

            # Windows refuses to unlink a file that is still open.
            for side in ("hr", "lr"):
                for name in ("a.png", "b.png"):
                    (work / "frames" / side / name).unlink()


class MaterializationRecoveryTests(unittest.TestCase):
    def _roots(self, directory):
        root = Path(directory)
        hr, lr, run = root / "hr", root / "lr", root / "run"
        for path in (hr, lr, run):
            path.mkdir()
        make_video(hr / "clip.mp4")
        make_video(lr / "clip.mp4")
        return hr, lr, run

    def test_hardlink_discovery_can_be_re_run_after_an_interrupted_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            hr, lr, run = self._roots(directory)
            first = discover_import_pairs(hr, lr, "hardlink", run)

            second = discover_import_pairs(hr, lr, "hardlink", run)

            self.assertEqual([record.to_dict() for record in first], [record.to_dict() for record in second])
            self.assertTrue((run / ".work" / "clips" / "hr" / "clip.mp4").is_file())

    def test_copy_discovery_can_be_re_run_after_an_interrupted_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            hr, lr, run = self._roots(directory)
            discover_import_pairs(hr, lr, "copy", run)

            discover_import_pairs(hr, lr, "copy", run)

            self.assertEqual((hr / "clip.mp4").read_bytes(), (run / ".work" / "clips" / "hr" / "clip.mp4").read_bytes())

    def test_unknown_materialization_mode_is_rejected_before_touching_the_run(self):
        with tempfile.TemporaryDirectory() as directory:
            hr, lr, run = self._roots(directory)

            with self.assertRaisesRegex(PairingError, "Unknown materialization mode"):
                discover_import_pairs(hr, lr, "symlink", run)

            self.assertFalse((run / ".work").exists())


class ExtractConfigValidationTests(unittest.TestCase):
    def _load(self, body):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text("version: 2\n" + body, encoding="utf-8")
            return load_config(path, "import-pairs")

    def test_null_frame_format_is_a_configuration_error(self):
        with self.assertRaisesRegex(ConfigError, "frame_format"):
            self._load("extract:\n  frame_format: null\n")

    def test_negative_gap_frames_is_a_configuration_error(self):
        with self.assertRaisesRegex(ConfigError, "gap_frames"):
            self._load("extract:\n  mode: gapped\n  gap_frames: -5\n")

    def test_boolean_sequence_length_is_a_configuration_error(self):
        with self.assertRaisesRegex(ConfigError, "sequence_length"):
            self._load("extract:\n  sequence_length: true\n")

    def test_valid_gapped_settings_are_accepted(self):
        config = self._load("extract:\n  mode: gapped\n  gap_frames: 3\n")

        self.assertEqual(3, config["extract"]["gap_frames"])
