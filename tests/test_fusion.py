"""Tests for fusion/ modules: CrossModalTransformer, TemporalFusion, AttentionPooling, ModalityGate."""
import unittest

import torch
import torch.nn as nn


class TestCrossModalTransformer(unittest.TestCase):
    """Tests for CrossModalTransformer forward pass."""

    def setUp(self):
        from fusion.cross_modal_transformer import CrossModalTransformer

        self.model = CrossModalTransformer(
            visual_dim=768, audio_dim=512, text_dim=768, hidden_dim=512, n_heads=8, n_layers=2
        )
        self.model.eval()

    def test_forward_output_fused_shape(self):
        """CrossModalTransformer.forward returns FusedRepresentation with 512-dim fused tensor."""
        visual = torch.randn(4, 768)
        audio = torch.randn(4, 512)
        text = torch.randn(4, 768)

        with torch.no_grad():
            result = self.model(visual, audio, text)

        # fused is pooled to (1, 512) when inputs are 2D (treated as single sequence)
        self.assertEqual(result.fused.shape[-1], 512)
        self.assertEqual(result.fused.ndim, 2)

    def test_forward_gate_weights_are_valid(self):
        """Gate weights from CrossModalTransformer are non-negative (sigmoid output)."""
        visual = torch.randn(2, 768)
        audio = torch.randn(2, 512)
        text = torch.randn(2, 768)

        with torch.no_grad():
            result = self.model(visual, audio, text)

        for key, val in result.gate_weights.items():
            self.assertGreaterEqual(val, 0.0, f"Gate weight for {key} must be >= 0")
            self.assertLessEqual(val, 1.0, f"Gate weight for {key} must be <= 1")


class TestTemporalFusion(unittest.TestCase):
    """Tests for TemporalFusion module."""

    def test_sinusoidal_pe_shape(self):
        """TemporalFusion._build_sinusoidal_pe returns tensor of shape (1, max_len, d_model)."""
        from fusion.temporal_fusion import TemporalFusion

        pe = TemporalFusion._build_sinusoidal_pe(100, 256)
        self.assertEqual(pe.shape, (1, 100, 256))

    def test_align_audio_to_frames_empty_returns_zeros(self):
        """align_audio_to_frames with empty audio returns zero tensor of frame count."""
        from fusion.temporal_fusion import TemporalFusion

        model = TemporalFusion(hidden_dim=64)
        audio_embeddings = torch.zeros(0, 64)
        frame_timestamps = [0.1 * i for i in range(10)]
        result = model.align_audio_to_frames(audio_embeddings, [], frame_timestamps)
        self.assertEqual(result.shape, (10, 64))
        self.assertTrue(torch.all(result == 0))


class TestAttentionPooling(unittest.TestCase):
    """Tests for AttentionPooling module."""

    def test_output_shape(self):
        """AttentionPooling.forward returns pooled vector of (batch, input_dim)."""
        from fusion.attention_pooling import AttentionPooling

        pooler = AttentionPooling(input_dim=256, hidden_dim=128)
        seq = torch.randn(2, 10, 256)
        pooled, weights = pooler(seq)
        self.assertEqual(pooled.shape, (2, 256))
        self.assertEqual(weights.shape, (2, 10))

    def test_attention_weights_sum_to_one(self):
        """AttentionPooling attention weights sum to ~1 along sequence dimension."""
        from fusion.attention_pooling import AttentionPooling

        pooler = AttentionPooling(input_dim=128, hidden_dim=64)
        seq = torch.randn(3, 8, 128)
        _, weights = pooler(seq)
        row_sums = weights.sum(dim=-1)
        self.assertTrue(torch.allclose(row_sums, torch.ones(3), atol=1e-5))


class TestModalityGate(unittest.TestCase):
    """Tests for ModalityGate module."""

    def test_gate_output_shape(self):
        """ModalityGate.forward returns dict with gated tensors preserving input shape."""
        from fusion.modality_gate import ModalityGate

        gate = ModalityGate(hidden_dim=64)
        visual = torch.randn(4, 64)
        audio = torch.randn(4, 64)
        text = torch.randn(4, 64)

        with torch.no_grad():
            result = gate(visual, audio, text)

        self.assertIn("visual", result)
        self.assertIn("audio", result)
        self.assertIn("text", result)
        self.assertEqual(result["visual"].shape, visual.shape)
        self.assertEqual(result["audio"].shape, audio.shape)
        self.assertEqual(result["text"].shape, text.shape)

    def test_missing_modality_zeroes_gate(self):
        """ModalityGate zeros out a specified missing modality's contribution."""
        from fusion.modality_gate import ModalityGate

        gate = ModalityGate(hidden_dim=64)
        visual = torch.randn(2, 64)
        audio = torch.randn(2, 64)
        text = torch.randn(2, 64)

        with torch.no_grad():
            result = gate(visual, audio, text, missing_modalities=["audio"])

        # audio gate is zero, so gated audio contribution should be all-zero
        self.assertTrue(torch.all(result["audio"] == 0), "Audio gate should be zeroed when missing")


if __name__ == "__main__":
    unittest.main()
