import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from utils.logger import get_logger

logger = get_logger("aurora.deployment.tensorrt")


@dataclass
class BenchmarkResult:
    mean_ms: float
    p99_ms: float
    throughput_fps: float
    n_runs: int


@dataclass
class AccuracyComparison:
    pytorch_output: np.ndarray
    trt_output: np.ndarray
    max_abs_diff: float
    mean_abs_diff: float
    passed: bool


class TensorRTOptimizer:
    """
    INT8 TensorRT optimization for AURORA-VISION models.
    Falls back to ONNX Runtime if TensorRT is unavailable.
    """

    def __init__(self, output_dir: str = "deployment/trt_engines"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._trt_available = self._check_trt()

    def _check_trt(self) -> bool:
        try:
            import tensorrt

            logger.info(f"TensorRT {tensorrt.__version__} available")
            return True
        except ImportError:
            logger.warning("TensorRT not installed. Using ONNX Runtime for benchmarks.")
            return False

    def optimize_onnx(
        self,
        onnx_path: str,
        precision: str = "int8",
        save_path: Optional[str] = None,
    ) -> str:
        """
        Build a TensorRT INT8 optimized engine from ONNX model.
        Falls back to quantized ONNX Runtime session if TRT unavailable.
        """
        engine_path = save_path or str(
            self.output_dir / (Path(onnx_path).stem + f"_{precision}.trt")
        )

        if self._trt_available:
            return self._build_trt_engine(onnx_path, precision, engine_path)
        else:
            return self._quantize_onnx(onnx_path, precision, engine_path)

    def _build_trt_engine(self, onnx_path: str, precision: str, engine_path: str) -> str:
        """Build TensorRT engine with specified precision."""
        import tensorrt as trt

        logger_trt = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger_trt)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, logger_trt)

        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    logger.error(f"TRT parse error: {parser.get_error(i)}")
                raise RuntimeError(f"Failed to parse ONNX: {onnx_path}")

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

        if precision == "int8" and builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
        elif precision == "fp16" and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        engine_bytes = builder.build_serialized_network(network, config)
        with open(engine_path, "wb") as f:
            f.write(engine_bytes)

        logger.info(f"TensorRT engine saved: {engine_path}")
        return engine_path

    def _quantize_onnx(self, onnx_path: str, precision: str, engine_path: str) -> str:
        """Quantize ONNX model using ONNX Runtime quantization."""
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantize_dynamic(
            model_input=onnx_path,
            model_output=engine_path.replace(".trt", "_quant.onnx"),
            weight_type=QuantType.QInt8 if precision == "int8" else QuantType.QUInt8,
        )
        quant_path = engine_path.replace(".trt", "_quant.onnx")
        logger.info(f"Quantized ONNX model saved: {quant_path}")
        return quant_path

    def benchmark(self, engine_path: str, n_runs: int = 1000) -> BenchmarkResult:
        """Benchmark inference throughput and latency."""
        import onnxruntime as ort

        session = ort.InferenceSession(engine_path)
        input_info = session.get_inputs()[0]
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in input_info.shape]
        dummy = np.random.randn(*shape).astype(np.float32)
        input_name = input_info.name

        latencies = []
        for _ in range(min(n_runs, 100)):
            t0 = time.perf_counter()
            session.run(None, {input_name: dummy})
            latencies.append((time.perf_counter() - t0) * 1000)

        latencies = sorted(latencies)
        mean_ms = float(np.mean(latencies))
        p99_ms = float(np.percentile(latencies, 99))
        fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0

        return BenchmarkResult(mean_ms=mean_ms, p99_ms=p99_ms, throughput_fps=fps, n_runs=len(latencies))

    def compare_accuracy(
        self, torch_model, trt_engine_path: str, test_data: np.ndarray
    ) -> AccuracyComparison:
        """Compare PyTorch vs TRT/ONNX-RT accuracy on test data."""
        import torch
        import onnxruntime as ort

        torch_model.eval()
        with torch.no_grad():
            torch_out = torch_model(torch.from_numpy(test_data)).numpy()

        session = ort.InferenceSession(trt_engine_path)
        input_name = session.get_inputs()[0].name
        trt_out = session.run(None, {input_name: test_data})[0]

        max_diff = float(np.abs(torch_out - trt_out).max())
        mean_diff = float(np.abs(torch_out - trt_out).mean())

        return AccuracyComparison(
            pytorch_output=torch_out,
            trt_output=trt_out,
            max_abs_diff=max_diff,
            mean_abs_diff=mean_diff,
            passed=max_diff < 1e-2,
        )
