"""In-memory monitoring counters for the POC backend."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock
from typing import Iterator


@dataclass
class MonitoringMetrics:
    """Thread-safe, process-local counters for lightweight POC monitoring."""

    request_count: int = 0
    request_error_count: int = 0
    request_latency_seconds_total: float = 0.0
    model_prediction_count: int = 0
    retrieval_request_count: int = 0
    retrieval_latency_seconds_total: float = 0.0
    _lock: Lock = field(default_factory=Lock, repr=False)

    def record_request(self, latency_seconds: float, error: bool = False) -> None:
        """Record one API request, its non-negative latency, and optional failure."""
        with self._lock:
            self.request_count += 1
            self.request_latency_seconds_total += max(0.0, latency_seconds)
            if error:
                self.request_error_count += 1

    def record_model_prediction(self, count: int = 1) -> None:
        """Increment the number of model predictions performed by the process."""
        with self._lock:
            self.model_prediction_count += count

    def record_retrieval(self, latency_seconds: float) -> None:
        """Record one retrieval operation and its non-negative latency."""
        with self._lock:
            self.retrieval_request_count += 1
            self.retrieval_latency_seconds_total += max(0.0, latency_seconds)

    @contextmanager
    def time_retrieval(self) -> Iterator[None]:
        """Measure and record a retrieval block even when the block raises."""
        started = time.perf_counter()
        try:
            yield
        finally:
            self.record_retrieval(time.perf_counter() - started)

    def snapshot(self) -> dict[str, float | int]:
        """Return an atomic snapshot including total and average latency values."""
        with self._lock:
            return {
                "request_count": self.request_count,
                "request_error_count": self.request_error_count,
                "request_latency_seconds_total": self.request_latency_seconds_total,
                "request_latency_seconds_average": (
                    self.request_latency_seconds_total / self.request_count if self.request_count else 0.0
                ),
                "model_prediction_count": self.model_prediction_count,
                "retrieval_request_count": self.retrieval_request_count,
                "retrieval_latency_seconds_total": self.retrieval_latency_seconds_total,
                "retrieval_latency_seconds_average": (
                    self.retrieval_latency_seconds_total / self.retrieval_request_count
                    if self.retrieval_request_count else 0.0
                ),
            }

    def reset(self) -> None:
        """Reset all process-local counters, primarily for deterministic tests."""
        with self._lock:
            self.request_count = 0
            self.request_error_count = 0
            self.request_latency_seconds_total = 0.0
            self.model_prediction_count = 0
            self.retrieval_request_count = 0
            self.retrieval_latency_seconds_total = 0.0


monitor = MonitoringMetrics()
