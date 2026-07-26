import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

import yaml

from tests.helpers import make_video


PYSIDE_AVAILABLE = importlib.util.find_spec("PySide6") is not None
if PYSIDE_AVAILABLE:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QMainWindow


@unittest.skipUnless(PYSIDE_AVAILABLE, "PySide6 is not installed")
class GuiIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def test_import_form_runs_the_canonical_workflow(self):
        from video_destroyer import gui

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hr, lr, output = root / "hr", root / "lr", root / "run"
            hr.mkdir()
            lr.mkdir()
            make_video(hr / "clip.mp4")
            make_video(lr / "clip.mp4")
            result = {}
            window = None

            def exercise():
                nonlocal window
                window = next(widget for widget in self.application.topLevelWidgets() if isinstance(widget, QMainWindow))
                window.import_hr.setText(str(hr))
                window.import_lr.setText(str(lr))
                window.import_output.setText(str(output))
                window.process.finished.connect(finished)
                window._start_import()

            def finished(exit_code, _exit_status):
                result["exit_code"] = exit_code
                QTimer.singleShot(0, self.application.quit)

            def timed_out():
                if "exit_code" not in result:
                    result["timed_out"] = True
                    if window is not None:
                        window._cancel()
                    self.application.quit()

            QTimer.singleShot(0, exercise)
            QTimer.singleShot(30000, timed_out)
            self.assertEqual(0, gui.main())
            if window is not None:
                window.close()
            self.assertNotIn("timed_out", result)
            self.assertEqual(0, result.get("exit_code"))
            self.assertTrue((output / "dataset" / "hr").is_dir())
            self.assertTrue((output / "dataset" / "lr").is_dir())

    def test_create_form_applies_the_pipeline_editor_configuration(self):
        from video_destroyer import gui

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output, base = root / "source.mp4", root / "run", root / "base.yaml"
            make_video(source)
            base.write_text(
                "version: 2\ncreate:\n  chunking:\n    strategy: frame_count\n    frames: 5\n    minimum_seconds: 0.1\nextract:\n  sequence_length: 2\n",
                encoding="utf-8",
            )
            result = {}
            window = None

            def exercise():
                nonlocal window
                window = next(widget for widget in self.application.topLevelWidgets() if isinstance(widget, QMainWindow) and widget.isVisible())
                window.create_input.setText(str(source))
                window.create_output.setText(str(output))
                window.create_config.setText(str(base))
                window.process.finished.connect(finished)
                window._start_create()

            def finished(exit_code, _exit_status):
                result["exit_code"] = exit_code
                QTimer.singleShot(0, self.application.quit)

            QTimer.singleShot(0, exercise)
            QTimer.singleShot(30000, self.application.quit)
            self.assertEqual(0, gui.main())
            if window is not None:
                window.close()
            self.assertEqual(0, result.get("exit_code"))
            resolved = yaml.safe_load((output / "run.yaml").read_text(encoding="utf-8"))
            self.assertEqual(["resize", "noise", "halo", "blur", "ghosting", "codec"], [stage["name"] for stage in resolved["config"]["create"]["degradations"]])
            self.assertTrue((output / "dataset" / "hr").is_dir())
