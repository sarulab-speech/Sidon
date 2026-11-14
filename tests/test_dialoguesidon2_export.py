"""torch.export programs read by the Space's standalone inference (spaces/dialoguesidon2/ds2_infer.py) give the
eager model's output, for one window and for the multi-window path."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dialoguesidon2_fixtures import tiny_separator  # noqa: E402

from sidon.dialoguesidon2 import infer  # noqa: E402
from sidon.dialoguesidon2.export import export  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def load_space_inference():
    spec = importlib.util.spec_from_file_location("ds2_infer", REPO / "spaces" / "dialoguesidon2" / "ds2_infer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ExportTests(unittest.TestCase):
    def test_exported_programs_match_eager_inference(self):
        import warnings

        warnings.filterwarnings("ignore")
        torch.set_num_threads(2)
        model = tiny_separator(seed=2).eval()
        ds2_infer = load_space_inference()
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "export"
            report = export(model, out, window_seconds=1.0, overlap_seconds=0.25, global_step=3,
                            check_seconds=(0.7, 2.3), example_seconds=1.5)
            self.assertLess(report["wrapper_vs_model_max_error"], 1e-5)
            self.assertEqual(sorted(p.name for p in out.iterdir()),
                             ["metadata.json", "separator.pt2", "vae_decoder.pt2"])
            meta = json.loads((out / "metadata.json").read_text())
            self.assertEqual((meta["hop"], meta["latent_dim"], meta["window_seconds"]), (480, 4, 1.0))
            programs = ds2_infer.load(out, "cpu")
            generator = torch.Generator().manual_seed(0)
            for seconds, rate in ((0.8, 24000), (3.1, 16000)):
                wave = 0.1 * torch.randn(2, int(seconds * rate), generator=generator)  # stereo, mixed to mono
                log = []
                released = ds2_infer.separate(wave, rate, programs, log=log)
                eager = infer.separate(model, wave, rate, window_seconds=1.0, overlap_seconds=0.25)
                self.assertEqual(released.shape, eager.shape)
                self.assertEqual(released.size(-1), round(seconds * 24000))
                torch.testing.assert_close(released, eager, atol=1e-4, rtol=1e-4)
                self.assertEqual(len(log), 0 if seconds < 1.0 else len(infer.window_starts(
                    released.size(-1), 24000, 6000)) - 1)
            # No paths of this machine in the programs' metadata.
            for name in ("separator.pt2", "vae_decoder.pt2"):
                self.assertNotIn(str(REPO).encode(), (out / name).read_bytes())


if __name__ == "__main__":
    unittest.main()
