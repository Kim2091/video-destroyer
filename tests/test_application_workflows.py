import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from tests.helpers import ROOT, make_video
from video_destroyer.cli import build_parser
from video_destroyer.pairing import PairingError, discover_import_pairs


class ImportWorkflowIntegrationTests(unittest.TestCase):
    def _run(self, *arguments):
        return subprocess.run(
            ["python", "-m", "video_destroyer", *arguments], cwd=ROOT,
            capture_output=True, text=True,
        )

    def test_gui_command_is_available_without_loading_gui_dependencies(self):
        self.assertEqual("gui", build_parser().parse_args(["gui"]).command)

    def test_import_reference_mode_builds_atomic_validated_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hr, lr = root / "hr", root / "lr"
            (hr / "show").mkdir(parents=True)
            (lr / "show").mkdir(parents=True)
            make_video(hr / "show" / "scene.mkv")
            make_video(lr / "show" / "scene.mp4")
            source_bytes = (hr / "show" / "scene.mkv").read_bytes()
            run = root / "run"
            result = self._run("import-pairs", "--hr", str(hr), "--lr", str(lr), "--output", str(run))
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertIn("Dataset ready:", result.stdout)
            self.assertEqual(source_bytes, (hr / "show" / "scene.mkv").read_bytes())
            self.assertEqual(
                sorted(path.name for path in (run / "dataset" / "hr").glob("*.png")),
                sorted(path.name for path in (run / "dataset" / "lr").glob("*.png")),
            )
            state = json.loads((run / "state.json").read_text(encoding="utf-8"))
            self.assertEqual("completed", state["status"])
            self.assertTrue((run / "reports" / "dataset-validation.json").is_file())

    def test_copy_materialization_is_run_owned_and_existing_output_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hr, lr = root / "hr", root / "lr"
            hr.mkdir()
            lr.mkdir()
            make_video(hr / "clip.mp4")
            make_video(lr / "clip.mp4")
            run = root / "run"
            result = self._run("import-pairs", "--hr", str(hr), "--lr", str(lr), "--output", str(run), "--materialize", "copy")
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertTrue(any((run / ".work" / "clips").rglob("*.mp4")))
            again = self._run("import-pairs", "--hr", str(hr), "--lr", str(lr), "--output", str(run))
            self.assertEqual(1, again.returncode)
            self.assertIn("Use resume instead", again.stderr)

    def test_unmatched_nested_paths_are_rejected_before_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "hr" / "one").mkdir(parents=True)
            (root / "lr" / "two").mkdir(parents=True)
            make_video(root / "hr" / "one" / "scene.mp4")
            make_video(root / "lr" / "two" / "scene.mp4")
            with self.assertRaises(PairingError):
                discover_import_pairs(root / "hr", root / "lr")

    def test_duplicate_and_case_folded_pair_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for side in ("hr", "lr"):
                (root / side).mkdir()
            make_video(root / "hr" / "scene.mp4")
            make_video(root / "hr" / "scene.mkv")
            make_video(root / "lr" / "scene.mp4")
            with self.assertRaises(PairingError):
                discover_import_pairs(root / "hr", root / "lr")

            (root / "hr" / "scene.mkv").unlink()
            (root / "hr" / "scene.mp4").rename(root / "hr" / "Scene.mp4")
            make_video(root / "hr" / "scene.mp4")
            with self.assertRaises(PairingError):
                discover_import_pairs(root / "hr", root / "lr")

    def test_fail_on_rejection_writes_completed_run_and_returns_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hr, lr = root / "hr", root / "lr"
            hr.mkdir()
            lr.mkdir()
            make_video(hr / "valid.mp4", rate=10)
            make_video(lr / "valid.mp4", rate=10)
            make_video(hr / "mismatch.mp4", rate=10)
            make_video(lr / "mismatch.mp4", rate=5)
            run = root / "run"
            result = self._run("import-pairs", "--hr", str(hr), "--lr", str(lr), "--output", str(run), "--fail-on-rejection")
            self.assertEqual(1, result.returncode, result.stderr)
            self.assertIn("Report:", result.stdout)
            state = json.loads((run / "state.json").read_text(encoding="utf-8"))
            self.assertEqual("completed_with_rejections", state["status"])
            pairs = [json.loads(line) for line in (run / "pairs.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(1, sum(pair["status"] == "rejected" for pair in pairs))

    def test_resume_recovers_after_dataset_rename_before_state_update(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hr, lr = root / "hr", root / "lr"
            hr.mkdir()
            lr.mkdir()
            make_video(hr / "clip.mp4")
            make_video(lr / "clip.mp4")
            run = root / "run"
            initial = self._run("import-pairs", "--hr", str(hr), "--lr", str(lr), "--output", str(run))
            self.assertEqual(0, initial.returncode, initial.stderr)
            state_path = run / "state.json"
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state["status"] = "interrupted"
            state["ended_at"] = None
            state["stages"]["finalization"]["status"] = "running"
            state_path.write_text(json.dumps(state), encoding="utf-8")

            resumed = self._run("resume", str(run))
            self.assertEqual(0, resumed.returncode, resumed.stderr)
            self.assertEqual("completed", json.loads(state_path.read_text(encoding="utf-8"))["status"])

    def test_invalid_v2_config_returns_usage_exit_code(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "invalid.yaml"
            config.write_text("version: 1\n", encoding="utf-8")
            result = self._run("import-pairs", "--hr", str(root), "--lr", str(root), "--output", str(root / "run"), "--config", str(config))
            self.assertEqual(2, result.returncode)

    def test_create_from_one_source_produces_a_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = make_video(root / "source.mp4")
            config = root / "create.yaml"
            config.write_text(
                """version: 2
create:
  chunking:
    strategy: frame_count
    frames: 5
    minimum_seconds: 0.1
  degradations:
    - name: resize
      probability: 1.0
      params:
        fixed_scale: 0.5
    - name: codec
      probability: 1.0
      params:
        h264:
          probability: 1.0
          quality_range: [28, 28]
          presets: [ultrafast]
extract:
  sequence_length: 2
""",
                encoding="utf-8",
            )
            run = root / "run"
            result = self._run("create", str(source), "--output", str(run), "--config", str(config))
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertTrue(any((run / "dataset" / "hr").glob("*.png")))

    def _presplit_config(self, path):
        path.write_text(
            """version: 2
create:
  chunking:
    strategy: none
  degradations:
    - name: resize
      probability: 1.0
      params:
        fixed_scale: 0.5
    - name: codec
      probability: 1.0
      params:
        h264:
          probability: 1.0
          quality_range: [28, 28]
          presets: [ultrafast]
extract:
  sequence_length: 2
""",
            encoding="utf-8",
        )
        return path

    def test_create_degrades_presplit_clips_without_splitting_them(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clips = root / "clips"
            (clips / "show").mkdir(parents=True)
            make_video(clips / "show" / "a.mkv")
            make_video(clips / "b.mp4")
            untouched = (clips / "b.mp4").read_bytes()
            run = root / "run"

            result = self._run("create", str(clips), "--output", str(run),
                               "--config", str(self._presplit_config(root / "presplit.yaml")))

            self.assertEqual(0, result.returncode, result.stderr)
            # One HR/LR pair per supplied clip: the clips are never re-split.
            pairs = [json.loads(line) for line in (run / "pairs.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(["b", "show/a"], sorted(pair["key"] for pair in pairs))
            self.assertTrue(all(pair["status"] == "validated" for pair in pairs), pairs)
            # HR is read in place; only the LR side is generated into the run.
            self.assertEqual(untouched, (clips / "b.mp4").read_bytes())
            self.assertTrue((run / ".work" / "clips" / "presplit" / "lr" / "b.mkv").is_file())
            self.assertTrue((run / ".work" / "clips" / "presplit" / "lr" / "show" / "a.mkv").is_file())
            self.assertTrue(any((run / "dataset" / "hr").glob("*.png")))
            self.assertEqual(
                sorted(path.name for path in (run / "dataset" / "hr").glob("*.png")),
                sorted(path.name for path in (run / "dataset" / "lr").glob("*.png")),
            )
            state = json.loads((run / "state.json").read_text(encoding="utf-8"))
            self.assertEqual("completed", state["status"])

    def test_presplit_run_resumes_from_its_recorded_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clips = root / "clips"
            clips.mkdir()
            make_video(clips / "a.mp4")
            run = root / "run"
            self.assertEqual(0, self._run("create", str(clips), "--output", str(run),
                                          "--config", str(self._presplit_config(root / "presplit.yaml"))).returncode)

            result = self._run("resume", str(run))

            self.assertEqual(0, result.returncode, result.stderr)
            self.assertTrue(any((run / "dataset" / "hr").glob("*.png")))

    def test_clips_differing_only_by_extension_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clips = root / "clips"
            clips.mkdir()
            make_video(clips / "scene.mp4")
            make_video(clips / "scene.mkv")

            result = self._run("create", str(clips), "--output", str(root / "run"),
                               "--config", str(self._presplit_config(root / "presplit.yaml")))

            self.assertEqual(1, result.returncode)
            self.assertIn("differ only by extension", result.stdout + result.stderr)

    def test_unknown_chunking_strategy_is_a_configuration_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "bad.yaml"
            config.write_text(
                "version: 2\ncreate:\n  chunking:\n    strategy: nope\n"
                "  degradations:\n    - name: codec\n      params:\n        h264: {probability: 1.0}\n",
                encoding="utf-8",
            )
            source = make_video(root / "source.mp4")

            result = self._run("create", str(source), "--output", str(root / "run"), "--config", str(config))

            self.assertEqual(2, result.returncode)
            self.assertIn("create.chunking.strategy", result.stderr)
