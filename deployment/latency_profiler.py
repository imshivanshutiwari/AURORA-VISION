import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from utils.logger import get_logger

logger = get_logger("aurora.deployment.latency_profiler")


@dataclass
class StageLatency:
    name: str
    mean_ms: float
    std_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    n_runs: int
    measurements: List[float] = field(default_factory=list)


@dataclass
class PipelineProfile:
    stages: Dict[str, StageLatency]
    total_mean_ms: float
    total_p99_ms: float
    throughput_fps: float
    bottleneck: str


class LatencyProfiler:
    """
    End-to-end latency measurement for the AURORA-VISION pipeline.
    Profiles each stage independently and computes throughput.
    """

    def __init__(self):
        self._measurements: Dict[str, List[float]] = {}
        self._start_times: Dict[str, float] = {}

    @contextmanager
    def measure(self, stage: str):
        """Context manager for measuring a stage's latency."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - t0) * 1000
            self._measurements.setdefault(stage, []).append(elapsed)

    def start(self, stage: str) -> None:
        """Start timing a stage."""
        self._start_times[stage] = time.perf_counter()

    def stop(self, stage: str) -> float:
        """Stop timing a stage and record latency."""
        if stage not in self._start_times:
            return 0.0
        elapsed = (time.perf_counter() - self._start_times.pop(stage)) * 1000
        self._measurements.setdefault(stage, []).append(elapsed)
        return elapsed

    def profile_function(
        self, fn: Callable, *args, n_runs: int = 10, stage_name: str = "function", **kwargs
    ) -> StageLatency:
        """Profile a function over n_runs iterations."""
        measurements = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            fn(*args, **kwargs)
            measurements.append((time.perf_counter() - t0) * 1000)

        return self._compute_stats(stage_name, measurements)

    def _compute_stats(self, name: str, measurements: List[float]) -> StageLatency:
        arr = np.array(measurements)
        return StageLatency(
            name=name,
            mean_ms=float(np.mean(arr)),
            std_ms=float(np.std(arr)),
            p50_ms=float(np.percentile(arr, 50)),
            p95_ms=float(np.percentile(arr, 95)),
            p99_ms=float(np.percentile(arr, 99)),
            n_runs=len(measurements),
            measurements=measurements,
        )

    def get_profile(self) -> PipelineProfile:
        """Compute a full pipeline profile from all recorded measurements."""
        stages = {}
        for stage, times in self._measurements.items():
            stages[stage] = self._compute_stats(stage, times)

        if stages:
            total_mean = sum(s.mean_ms for s in stages.values())
            total_p99 = sum(s.p99_ms for s in stages.values())
            fps = 1000.0 / total_mean if total_mean > 0 else 0.0
            bottleneck = max(stages, key=lambda k: stages[k].mean_ms)
        else:
            total_mean = 0.0
            total_p99 = 0.0
            fps = 0.0
            bottleneck = "none"

        return PipelineProfile(
            stages=stages,
            total_mean_ms=total_mean,
            total_p99_ms=total_p99,
            throughput_fps=fps,
            bottleneck=bottleneck,
        )

    def reset(self) -> None:
        """Clear all recorded measurements."""
        self._measurements.clear()
        self._start_times.clear()
