import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from src.dataset import load_radioml_data
from src.preprocessing.iq_transforms import enrich_iq, normalize_iq
from src.models.conv1d_amc import Conv1DAMCNet


class RFContractTests(unittest.TestCase):
    def test_amplitude_phase_and_shared_scaling(self):
        iq = np.array([[[3., -3., 0.], [4., -4., 0.]]], dtype=np.float32)
        features = enrich_iq(iq)
        np.testing.assert_allclose(features[:, 2], [[5., 5., 0.]])
        np.testing.assert_allclose(features[:, 3], [[np.arctan2(4, 3), np.arctan2(-4, -3), 0.]])
        np.testing.assert_allclose(normalize_iq(iq * 7), normalize_iq(iq), atol=1e-6)
        np.testing.assert_allclose(np.mean(np.sum(normalize_iq(iq)**2, axis=1)), 1, atol=1e-6)
        self.assertTrue(np.isfinite(normalize_iq(np.zeros((1, 2, 128)))).all())

    def test_missing_and_small_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'missing.pkl'
            with self.assertRaises(FileNotFoundError):
                load_radioml_data(path)
            self.assertFalse(path.exists())
            path.write_bytes(b'not a real dataset')
            with self.assertRaisesRegex(ValueError, '50 MB'):
                load_radioml_data(path)

    def test_both_channel_modes_support_training(self):
        torch.set_num_threads(1)
        for channels in (2, 4):
            model = Conv1DAMCNet(channels)
            logits = model(torch.randn(3, channels, 128))
            self.assertEqual(logits.shape, (3, 11))
            loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 5, 10]))
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
