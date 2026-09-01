"""Lightweight aggregate timing for regression-tree construction."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import Iterator, Literal


BuildTimingCategory = Literal["sampling", "splitter", "model_refit"]


@dataclass
class BuildTimingProfile:
    """Accumulate a few coarse build-time categories."""

    sampling_seconds: float = 0.0
    splitter_seconds: float = 0.0
    optimizer_model_refit_seconds: float = 0.0

    @contextmanager
    def measure(self, category: BuildTimingCategory) -> Iterator[None]:
        """Add one operation's elapsed time to a category."""

        start = perf_counter()
        try:
            yield
        finally:
            elapsed = perf_counter() - start
            if category == "sampling":
                self.sampling_seconds += elapsed
            elif category == "splitter":
                self.splitter_seconds += elapsed
            else:
                self.optimizer_model_refit_seconds += elapsed

    def summary(self, total_build_seconds: float) -> dict[str, float]:
        """Return aggregate seconds and percentages of total build time."""

        total = max(float(total_build_seconds), 0.0)
        other = max(total - self.sampling_seconds - self.splitter_seconds, 0.0)

        def percentage(seconds: float) -> float:
            return 0.0 if total == 0.0 else 100.0 * seconds / total

        return {
            "total_build_seconds": total,
            "sampling_seconds": self.sampling_seconds,
            "sampling_percent": percentage(self.sampling_seconds),
            "splitter_seconds": self.splitter_seconds,
            "splitter_percent": percentage(self.splitter_seconds),
            "optimizer_model_refit_seconds": self.optimizer_model_refit_seconds,
            "optimizer_model_refit_percent": percentage(
                self.optimizer_model_refit_seconds
            ),
            "other_seconds": other,
            "other_percent": percentage(other),
        }
