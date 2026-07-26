import random
import tempfile
import unittest
from pathlib import Path

import ffmpeg

from tests.helpers import make_video
from utils.codec_handler import CodecHandler
from utils.degradation_pipeline import DegradationPipeline
from utils.degradations.blur_degradation import BlurDegradation
from utils.degradations.base_degradation import BaseDegradation
from utils.degradations.chroma_delay_degradation import ChromaDelayDegradation
from utils.degradations.codec_degradation import CodecDegradation
from utils.degradations.ghosting_degradation import GhostingDegradation
from utils.degradations.halo_degradation import HaloDegradation
from utils.degradations.interlace_degradation import InterlaceDegradation
from utils.degradations.noise_degradation import NoiseDegradation
from utils.degradations.resize_degradation import ResizeDegradation
from utils.degradations.tonemap_degradation import TonemapDegradation


VIDEO_INFO = {"width": 64, "height": 48, "avg_frame_rate": "10/1"}


class ConcreteDegradation(BaseDegradation):
    @property
    def name(self):
        return "concrete"

    def get_params(self):
        return {}

    def apply(self, input_path, output_path):
        Path(output_path).write_text("applied", encoding="utf-8")
        return output_path

    def get_filter_expression(self, video_info):
        return "null"


class BaseDegradationTests(unittest.TestCase):
    def test_probability_controls_real_process_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            skipped = ConcreteDegradation({"probability": 0})
            self.assertEqual("input", skipped.process("input", str(root / "skipped")))
            self.assertFalse((root / "skipped").exists())
            applied = ConcreteDegradation({"probability": 1})
            output = root / "applied"
            self.assertEqual(str(output), applied.process("input", str(output)))
            self.assertEqual("applied", output.read_text(encoding="utf-8"))


class DegradationFilterTests(unittest.TestCase):
    def setUp(self):
        random.seed(1234)

    def test_resize_generates_even_dimensions_and_down_up_chain(self):
        direct = ResizeDegradation({"probability": 1, "params": {"fixed_scale": 0.5, "scaling_filters": ["bilinear"]}})
        self.assertEqual("scale=32:24:sws_flags=bilinear", direct.get_filter_expression(VIDEO_INFO))
        down_up = ResizeDegradation({"probability": 1, "params": {"fixed_scale": 0.5, "scaling_filters": ["bilinear"], "down_up": {"enabled": True, "probability": 1, "range": [0.75, 0.75]}}})
        self.assertEqual("scale=48:36:sws_flags=bilinear,scale=32:24:sws_flags=bilinear", down_up.get_filter_expression(VIDEO_INFO))
        self.assertEqual(2, direct._round_to_even(1))
        with self.assertRaises(NotImplementedError):
            direct.apply("in", "out")

    def test_noise_halo_and_blur_filters_use_valid_parameters(self):
        noise = NoiseDegradation({"probability": 1, "params": {"y_strength_range": [2, 2], "uv_strength_range": [3, 3], "types": ["t"]}})
        expression = noise.get_filter_expression(VIDEO_INFO)
        self.assertIn("c0_strength=2.00", expression)
        self.assertIn("c1_strength=3.00", expression)
        self.assertIn("allf=t", expression)

        halo = HaloDegradation({"probability": 1, "params": {"luma_x_range": [4, 4], "luma_y_range": [6, 6], "luma_amount_range": [1, 1]}})
        self.assertIn("lx=5:ly=7", halo.get_filter_expression(VIDEO_INFO))

        gaussian = BlurDegradation({"probability": 1, "params": {"enabled_types": ["gaussian"], "gaussian": {"sigma_range": [1, 1], "steps_range": [2, 2]}}})
        self.assertEqual("gblur=sigma=1.00,gblur=sigma=1.00", gaussian.get_filter_expression(VIDEO_INFO))
        motion = BlurDegradation({"probability": 1, "params": {"enabled_types": ["motion"], "motion": {"frames_range": [3, 3], "angle_range": [0, 0]}}})
        self.assertEqual(2, motion.get_filter_expression(VIDEO_INFO).count("tblend="))

    def test_complex_degradations_return_valid_graphs(self):
        chroma = ChromaDelayDegradation({"probability": 1, "params": {"delay_frames": 2}})
        self.assertIn("PTS+0.2/TB", chroma.get_filter_expression(VIDEO_INFO))
        self.assertEqual(1, ChromaDelayDegradation({"probability": 1, "params": {"delay_frames": -1}}).get_params()["delay_frames"])

        ghost = GhostingDegradation({"probability": 1, "params": {"num_ghosts_range": [1, 1], "opacity_range": [0.1, 0.1], "delay_range": [1, 1], "offset_x_range": [0, 0], "offset_y_range": [0, 0], "enable_color_shift": False}})
        self.assertIn("overlay", ghost.get_filter_expression(VIDEO_INFO))

        interlace = InterlaceDegradation({"probability": 1, "params": {"field_orders": ["top"]}})
        self.assertEqual("tinterlace=mode=4:flags=vlpf", interlace.get_filter_expression(VIDEO_INFO))
        chroma_interlace = InterlaceDegradation({"probability": 1, "params": {"field_orders": ["bottom"], "chroma_only": True}})
        self.assertIn("mergeplanes", chroma_interlace.get_filter_expression(VIDEO_INFO))

    def test_tonemap_skips_sdr_and_handles_hdr(self):
        tonemap = TonemapDegradation({"probability": 1, "params": {"auto_detect": True}})
        self.assertIsNone(tonemap.get_filter_expression(VIDEO_INFO))
        self.assertTrue(tonemap.get_params()["skipped"])
        expression = tonemap.get_filter_expression({**VIDEO_INFO, "color_transfer": "smpte2084"})
        self.assertIn("tonemap=hable", expression)
        self.assertTrue(tonemap.get_params()["detected_hdr"])

    def test_codec_parameters_and_real_direct_encode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = make_video(root / "source.mp4")
            output = root / "encoded.mp4"
            codec = CodecDegradation(
                {"probability": 1, "params": {"h264": {"probability": 1, "quality_range": {"min": 30, "max": 30}, "presets": ["ultrafast"]}}}
            )
            self.assertEqual("libx264", codec.get_codec_params()["vcodec"])
            self.assertEqual(str(output), codec.apply(str(source), str(output)))
            self.assertTrue(output.exists())
            self.assertEqual("h264", ffmpeg.probe(str(output))["streams"][0]["codec_name"])


class PipelineTests(unittest.TestCase):
    def test_filter_graph_keeps_declared_order(self):
        graph = DegradationPipeline._build_filter_graph([
            "hflip",
            "split=2[a][b];[a][b]hstack",
            "eq=contrast=1.1",
        ])
        self.assertEqual("hflip,split=2[a][b];[a][b]hstack[_chain_0];[_chain_0]eq=contrast=1.1", graph)
        self.assertEqual("hflip,vflip", DegradationPipeline._build_filter_graph(["hflip", "vflip"]))

    def test_pipeline_processes_real_video_with_resize_and_codec(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = make_video(root / "source.mp4")
            output = root / "output.mp4"
            pipeline = DegradationPipeline({})
            pipeline.add_degradation(ResizeDegradation({"probability": 1, "params": {"fixed_scale": 0.5, "scaling_filters": ["bilinear"]}}))
            pipeline.add_degradation(CodecDegradation({"probability": 1, "params": {"h264": {"probability": 1, "quality_range": {"min": 28, "max": 28}, "presets": ["ultrafast"]}}}, codec_handler=CodecHandler({"h264": {"probability": 1, "quality_range": {"min": 28, "max": 28}, "presets": ["ultrafast"]}})))
            self.assertEqual(str(output), pipeline.process_video(str(source), str(output)))
            stream = next(item for item in ffmpeg.probe(str(output))["streams"] if item["codec_type"] == "video")
            self.assertEqual((32, 24), (stream["width"], stream["height"]))
