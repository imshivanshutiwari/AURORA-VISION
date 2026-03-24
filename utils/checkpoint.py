import json
import os
from pathlib import Path
from typing import Any, Optional

import torch

from utils.logger import get_logger

logger = get_logger("aurora.checkpoint")


class CheckpointManager:
    """Manages model and pipeline state checkpoints."""

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_model(self, model: torch.nn.Module, name: str, metadata: dict = None) -> str:
        """Save a PyTorch model checkpoint."""
        path = self.checkpoint_dir / f"{name}.pt"
        checkpoint = {
            "state_dict": model.state_dict(),
            "metadata": metadata or {},
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved model checkpoint: {path}")
        return str(path)

    def load_model(self, model: torch.nn.Module, name: str) -> dict:
        """Load model weights from checkpoint."""
        path = self.checkpoint_dir / f"{name}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
        logger.info(f"Loaded model checkpoint: {path}")
        return checkpoint.get("metadata", {})

    def save_pipeline_state(self, state: dict, run_id: str) -> str:
        """Persist pipeline state for resumption."""
        path = self.checkpoint_dir / f"pipeline_{run_id}.json"
        serializable = {
            k: v if not isinstance(v, torch.Tensor) else v.tolist()
            for k, v in state.items()
            if not hasattr(v, "__iter__") or isinstance(v, (str, list, dict))
        }
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2, default=str)
        return str(path)

    def load_pipeline_state(self, run_id: str) -> Optional[dict]:
        """Load persisted pipeline state."""
        path = self.checkpoint_dir / f"pipeline_{run_id}.json"
        if not path.exists():
            return None
        with open(path, "r") as f:
            return json.load(f)

    def list_checkpoints(self) -> list[str]:
        return [f.stem for f in self.checkpoint_dir.glob("*.pt")]
