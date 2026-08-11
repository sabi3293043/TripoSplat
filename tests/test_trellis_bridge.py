import unittest
from pathlib import Path


class NativeTripoSplatProvenanceTests(unittest.TestCase):
    def test_ui_uses_triposplat_multiview_not_trellis_pipeline(self):
        root = Path(__file__).parents[1]
        ui = (root / "run_gradio.py").read_text(encoding="utf-8")
        self.assertIn("from multiview import run_multi_image", ui)
        self.assertIn("_load_tripo_pipe()", ui)
        self.assertNotIn("_trellis_request", ui)
        self.assertNotIn("TRELLIS_BACKEND_URL", ui)

    def test_multiview_sampler_extends_triposplat_sampler(self):
        root = Path(__file__).parents[1]
        source = (root / "multiview.py").read_text(encoding="utf-8")
        self.assertIn("from triposplat import FlowEulerCfgSampler", source)
        self.assertIn("class MultiViewFlowEulerCfgSampler(FlowEulerCfgSampler)", source)
        self.assertNotIn("import trellis", source)
        self.assertNotIn("from trellis", source)


if __name__ == "__main__":
    unittest.main()
