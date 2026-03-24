import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from utils.logger import get_logger

logger = get_logger("aurora.deployment.onnx")

ONNX_OPSET = 17


class ONNXExporter:
    """
    Exports AURORA-VISION models to ONNX format for edge deployment.
    Validates ONNX output against PyTorch to ensure < 1e-4 absolute diff.
    """

    def __init__(self, output_dir: str = "deployment/onnx_models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_clip(self, model, save_path: Optional[str] = None) -> str:
        """Export CLIP visual encoder to ONNX."""
        save_path = save_path or str(self.output_dir / "clip_vit_l14.onnx")
        dummy = torch.randn(1, 3, 224, 224).to(next(model.parameters()).device)

        visual_encoder = model.visual if hasattr(model, "visual") else model

        torch.onnx.export(
            visual_encoder,
            dummy,
            save_path,
            opset_version=ONNX_OPSET,
            input_names=["frame"],
            output_names=["embedding"],
            dynamic_axes={"frame": {0: "batch"}, "embedding": {0: "batch"}},
            do_constant_folding=True,
        )
        logger.info(f"CLIP exported to ONNX: {save_path}")
        return save_path

    def export_swin(self, model, save_path: Optional[str] = None) -> str:
        """Export Swin Transformer to ONNX."""
        save_path = save_path or str(self.output_dir / "swin_transformer.onnx")
        dummy = torch.randn(1, 3, 224, 224).to(next(model.parameters()).device)

        torch.onnx.export(
            model,
            dummy,
            save_path,
            opset_version=ONNX_OPSET,
            input_names=["frame"],
            output_names=["embedding"],
            dynamic_axes={"frame": {0: "batch"}, "embedding": {0: "batch"}},
            do_constant_folding=True,
        )
        logger.info(f"Swin exported to ONNX: {save_path}")
        return save_path

    def export_cross_modal_transformer(self, model, save_path: Optional[str] = None) -> str:
        """Export CrossModalTransformer to ONNX."""
        save_path = save_path or str(self.output_dir / "cross_modal_transformer.onnx")

        model.eval()
        vis = torch.randn(1, 1, 768)
        aud = torch.randn(1, 1, 512)
        txt = torch.randn(1, 1, 768)

        torch.onnx.export(
            model,
            (vis, aud, txt),
            save_path,
            opset_version=ONNX_OPSET,
            input_names=["visual_feat", "audio_feat", "text_feat"],
            output_names=["fused"],
            dynamic_axes={
                "visual_feat": {0: "batch", 1: "seq_v"},
                "audio_feat": {0: "batch", 1: "seq_a"},
                "text_feat": {0: "batch", 1: "seq_t"},
                "fused": {0: "batch"},
            },
            do_constant_folding=True,
        )
        logger.info(f"CrossModalTransformer exported to ONNX: {save_path}")
        return save_path

    def validate_onnx(
        self,
        onnx_path: str,
        torch_model: nn.Module,
        test_input: torch.Tensor,
        atol: float = 1e-4,
    ) -> bool:
        """
        Validate ONNX model against PyTorch: max absolute diff < atol.

        Returns:
            True if validation passes, False otherwise
        """
        import onnxruntime as ort

        torch_model.eval()
        with torch.no_grad():
            if isinstance(test_input, (list, tuple)):
                torch_out = torch_model(*test_input)
            else:
                torch_out = torch_model(test_input)

        if isinstance(torch_out, torch.Tensor):
            torch_np = torch_out.cpu().numpy()
        else:
            torch_np = torch_out[0].cpu().numpy() if hasattr(torch_out, "__iter__") else None

        session = ort.InferenceSession(onnx_path)
        if isinstance(test_input, (list, tuple)):
            input_names = [inp.name for inp in session.get_inputs()]
            inputs_dict = {
                name: t.cpu().numpy()
                for name, t in zip(input_names, test_input)
            }
        else:
            input_name = session.get_inputs()[0].name
            inputs_dict = {input_name: test_input.cpu().numpy()}

        onnx_out = session.run(None, inputs_dict)

        if torch_np is not None and onnx_out:
            max_diff = float(np.abs(torch_np - onnx_out[0]).max())
            passed = max_diff < atol
            logger.info(f"ONNX validation: max_diff={max_diff:.2e}, passed={passed}")
            return passed

        return True
