"""
pytest configuration for AURORA-VISION test suite.

Eager-loads torch.distributed sub-modules at session start so that the
wait_tensor kernel is registered exactly once before any test runs.
Without this, test_evaluation.py's bert_score import triggers a partial
registration during its first test; when test_ocr.py later imports
transformers-backed OCR modules, a second registration attempt is made,
causing a RuntimeError in the local development environment.
"""
import pytest


@pytest.fixture(scope="session", autouse=True)
def preload_torch_distributed():
    """Pre-register torch.distributed kernels before any test runs."""
    try:
        import torch.distributed  # noqa: F401
        import torch.distributed._functional_collectives  # noqa: F401
        import torch.distributed.tensor  # noqa: F401
    except Exception:
        pass  # torch.distributed not available in this environment; skip
