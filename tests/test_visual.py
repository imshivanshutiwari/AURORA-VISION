"""Tests for visual/ modules: CLIP, Swin, TimeSformer, SceneDetector, ObjectTracker."""
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

# Stub the 'clip' module so visual.clip_encoder can be imported without OpenAI CLIP installed
_clip_stub = MagicMock()
_clip_stub.load.return_value = (MagicMock(), MagicMock())
sys.modules.setdefault("clip", _clip_stub)


class TestCLIPFrameEncoder(unittest.TestCase):
    """Tests for CLIPFrameEncoder."""

    def _make_encoder(self):
        """Return a CLIPFrameEncoder with a fully mocked CLIP model."""
        mock_model = MagicMock()
        mock_model.eval.return_value = mock_model
        mock_model.encode_image.return_value = torch.randn(3, 768)

        mock_preprocess = MagicMock(side_effect=lambda img: torch.randn(3, 224, 224))

        sys.modules["clip"].load.return_value = (mock_model, mock_preprocess)

        from visual.clip_encoder import CLIPFrameEncoder

        # Clear any cached module state
        enc = CLIPFrameEncoder.__new__(CLIPFrameEncoder)
        enc.device = "cpu"
        enc.model = mock_model
        enc.preprocess = mock_preprocess
        return enc, mock_model, mock_preprocess

    def test_encode_frames_returns_correct_shape(self):
        """CLIPFrameEncoder.encode_frames returns tensor with 768-dim embeddings."""
        enc, mock_model, mock_preprocess = self._make_encoder()
        mock_model.encode_image.return_value = torch.randn(3, 768)

        frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(3)]
        result = enc.encode_frames(frames)

        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape[1], 768)

    def test_encode_empty_frames_returns_zero_tensor(self):
        """CLIPFrameEncoder.encode_frames on empty list returns (0, 768) zero tensor."""
        enc, _, _ = self._make_encoder()
        result = enc.encode_frames([])

        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, torch.Size([0, 768]))
        self.assertTrue(torch.all(result == 0))


class TestSwinEncoder(unittest.TestCase):
    """Tests for SwinEncoder."""

    @patch("visual.swin_encoder.AutoFeatureExtractor")
    @patch("visual.swin_encoder.SwinModel")
    def test_swin_encoder_loads_correct_model_id(self, mock_swin, mock_extractor):
        """SwinEncoder loads microsoft/swin-tiny-patch4-window7-224 by default."""
        mock_m = MagicMock()
        mock_m.to.return_value = mock_m
        mock_m.eval.return_value = mock_m
        mock_swin.from_pretrained.return_value = mock_m
        mock_extractor.from_pretrained.return_value = MagicMock()

        from visual.swin_encoder import SWIN_MODEL_ID, SwinEncoder

        SwinEncoder(device="cpu")
        mock_swin.from_pretrained.assert_called_once_with(SWIN_MODEL_ID, output_attentions=True)

    @patch("visual.swin_encoder.AutoFeatureExtractor")
    @patch("visual.swin_encoder.SwinModel")
    def test_encode_frame_returns_tensor(self, mock_swin, mock_extractor):
        """SwinEncoder.encode_frame returns a 2D patch embedding tensor."""
        mock_model = MagicMock()
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(1, 49, 768)
        mock_model.return_value = mock_output
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_swin.from_pretrained.return_value = mock_model

        mock_extractor_instance = MagicMock()
        mock_extractor_instance.return_value = {"pixel_values": torch.randn(1, 3, 224, 224)}
        mock_extractor.from_pretrained.return_value = mock_extractor_instance

        from visual.swin_encoder import SwinEncoder

        encoder = SwinEncoder(device="cpu")
        frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        result = encoder.encode_frame(frame)

        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.ndim, 2)


class TestSceneDetector(unittest.TestCase):
    """Tests for SceneDetector."""

    def test_scene_detector_default_threshold(self):
        """SceneDetector initializes with threshold=30.0 by default."""
        from visual.scene_detector import SceneDetector

        detector = SceneDetector()
        self.assertEqual(detector.threshold, 30.0)

    def test_scene_detector_custom_threshold(self):
        """SceneDetector accepts a custom threshold."""
        from visual.scene_detector import SceneDetector

        detector = SceneDetector(threshold=15.0)
        self.assertEqual(detector.threshold, 15.0)


class TestObjectTrackerDataclass(unittest.TestCase):
    """Tests for ObjectTracker dataclasses."""

    def test_tracked_object_lifetime_frames(self):
        """TrackedObject.lifetime_frames returns correct frame count."""
        from visual.object_tracker import TrackedObject

        obj = TrackedObject(
            track_id=1,
            class_name="person",
            bboxes=[(0, 0, 10, 10), (1, 1, 11, 11), (2, 2, 12, 12)],
            frame_indices=[0, 1, 2],
            confidences=[0.9, 0.85, 0.88],
        )
        self.assertEqual(obj.lifetime_frames, 3)
        self.assertEqual(obj.last_bbox, (2, 2, 12, 12))


if __name__ == "__main__":
    unittest.main()
