import tempfile
import unittest
from pathlib import Path

from video_destroyer.gui_config import build_create_config, default_stages, write_temp_create_config


class GuiConfigTests(unittest.TestCase):
    def test_pipeline_order_and_selection_are_written_to_create_config(self):
        stages = default_stages()
        stages[0], stages[3] = stages[3], stages[0]
        stages[0]["enabled"] = True
        stages[0]["probability"] = 0.75

        config = build_create_config(stages)

        pipeline = config["create"]["degradations"]
        self.assertEqual(["blur", "noise", "halo", "resize", "ghosting", "codec"], [item["name"] for item in pipeline])
        self.assertTrue(pipeline[0]["enabled"])
        self.assertEqual(0.75, pipeline[0]["probability"])
        self.assertEqual(1.0, pipeline[-1]["probability"])

    def test_base_config_is_preserved_while_pipeline_is_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "base.yaml"
            base.write_text("version: 2\nextract:\n  sequence_length: 8\n", encoding="utf-8")

            config = build_create_config(default_stages(), base)

            self.assertEqual(8, config["extract"]["sequence_length"])
            self.assertEqual("resize", config["create"]["degradations"][0]["name"])

    def test_codec_cannot_be_disabled_or_omitted(self):
        disabled = default_stages()
        disabled[-1]["enabled"] = False
        with self.assertRaisesRegex(ValueError, "Codec encode must stay enabled"):
            build_create_config(disabled)

        with self.assertRaisesRegex(ValueError, "Codec encode is required"):
            build_create_config(default_stages()[:-1])

    def test_temp_config_is_valid_yaml_file(self):
        path = write_temp_create_config(default_stages())
        try:
            self.assertTrue(path.is_file())
            self.assertIn("degradations:", path.read_text(encoding="utf-8"))
        finally:
            path.unlink(missing_ok=True)
