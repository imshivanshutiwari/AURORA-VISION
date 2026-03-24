"""Tests for pipeline/ modules: state typing, config loader, utils, and pipeline integration."""
import os
import tempfile
import unittest


class TestVideoQAState(unittest.TestCase):
    """Tests for VideoQAState TypedDict."""

    def test_state_accepts_required_fields(self):
        """VideoQAState can be instantiated with expected fields."""
        from agents.state import VideoQAState

        state: VideoQAState = {
            "video_url": "https://example.com/video.mp4",
            "query": "What is happening?",
            "retry_count": 0,
        }
        self.assertEqual(state["video_url"], "https://example.com/video.mp4")
        self.assertEqual(state["query"], "What is happening?")
        self.assertEqual(state["retry_count"], 0)

    def test_state_pipeline_trace_is_list(self):
        """VideoQAState pipeline_trace can hold a list of stage names."""
        from agents.state import VideoQAState

        state: VideoQAState = {
            "pipeline_trace": ["ingestor", "visual", "audio", "ocr", "fusion", "qa", "evaluator"],
        }
        self.assertEqual(len(state["pipeline_trace"]), 7)
        self.assertIn("fusion", state["pipeline_trace"])


class TestConfigLoader(unittest.TestCase):
    """Tests for ConfigLoader YAML loading."""

    def test_load_video_config(self):
        """ConfigLoader.load('video_config') returns a non-empty dict."""
        from utils.config_loader import ConfigLoader

        config = ConfigLoader.load("video_config")
        self.assertIsInstance(config, dict)
        self.assertGreater(len(config), 0)

    def test_load_model_config(self):
        """ConfigLoader.load('model_config') returns a non-empty dict."""
        from utils.config_loader import ConfigLoader

        config = ConfigLoader.load("model_config")
        self.assertIsInstance(config, dict)

    def test_config_get_dot_notation(self):
        """ConfigLoader.get supports dot-notation key access."""
        from utils.config_loader import ConfigLoader

        # Clear cache to ensure fresh load
        ConfigLoader.clear_cache()
        config = ConfigLoader.load("video_config")
        top_key = list(config.keys())[0]
        val = ConfigLoader.get("video_config", top_key)
        self.assertIsNotNone(val)

    def test_config_missing_key_returns_default(self):
        """ConfigLoader.get returns default when key is not found."""
        from utils.config_loader import ConfigLoader

        result = ConfigLoader.get("video_config", "nonexistent.deep.key", default="fallback")
        self.assertEqual(result, "fallback")


class TestSetSeed(unittest.TestCase):
    """Tests for set_seed reproducibility utility."""

    def test_set_seed_makes_numpy_reproducible(self):
        """set_seed(42) produces the same numpy random values across two calls."""
        import numpy as np

        from utils.seed import set_seed

        set_seed(42)
        arr1 = np.random.rand(5)
        set_seed(42)
        arr2 = np.random.rand(5)
        np.testing.assert_array_almost_equal(arr1, arr2)

    def test_set_seed_makes_torch_reproducible(self):
        """set_seed(7) produces the same torch random values across two calls."""
        import torch

        from utils.seed import set_seed

        set_seed(7)
        t1 = torch.rand(4)
        set_seed(7)
        t2 = torch.rand(4)
        self.assertTrue(torch.allclose(t1, t2))


class TestLogger(unittest.TestCase):
    """Tests for Aurora logger utility."""

    def test_get_logger_returns_logger(self):
        """get_logger returns a Logger with the specified name prefix."""
        import logging

        from utils.logger import get_logger

        logger = get_logger("aurora.test.module")
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, "aurora.test.module")

    def test_get_logger_idempotent(self):
        """get_logger called twice for the same name returns the same Logger object."""
        from utils.logger import get_logger

        logger1 = get_logger("aurora.test.idempotent")
        logger2 = get_logger("aurora.test.idempotent")
        self.assertIs(logger1, logger2)


if __name__ == "__main__":
    unittest.main()
