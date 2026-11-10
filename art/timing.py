"""Lightweight aggregate timing for regression-tree construction."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import Iterator, Literal


BuildTimingCategory = Literal["sampling", "splitter", "model_refit"]
SplitterTimingCategory = Literal[
    "least_squares",
    "backtracking",
    "gradient",
    "sigmoid",
]


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


@dataclass
class SplitterTimingProfile:
    """Collect high-level and operation-level timings for one soft split."""

    least_squares_seconds: float = 0.0
    backtracking_seconds: float = 0.0
    gradient_seconds: float = 0.0
    sigmoid_seconds: float = 0.0
    least_squares_count: int = 0
    backtracking_count: int = 0
    gradient_count: int = 0
    sigmoid_count: int = 0
    line_search_candidate_count: int = 0

    @contextmanager
    def measure(
        self,
        category: SplitterTimingCategory,
        *,
        count: bool = False,
    ) -> Iterator[None]:
        """Add one operation's elapsed time and optionally increment its count."""

        start = perf_counter()
        try:
            yield
        finally:
            elapsed = perf_counter() - start
            seconds_field = f"{category}_seconds"
            setattr(self, seconds_field, getattr(self, seconds_field) + elapsed)
            if count:
                count_field = f"{category}_count"
                setattr(self, count_field, getattr(self, count_field) + 1)

    def count_line_search_candidate(self) -> None:
        """Record one candidate step evaluated inside backtracking."""

        self.line_search_candidate_count += 1

    def summary(self, total_seconds: float) -> dict[str, float | int]:
        """Return measured timings as fractions of total splitter wall time."""

        total = max(float(total_seconds), 0.0)
        phases = {
            "least_squares_seconds": self.least_squares_seconds,
            "backtracking_seconds": self.backtracking_seconds,
            "gradient_seconds": self.gradient_seconds,
            "sigmoid_seconds": self.sigmoid_seconds,
        }
        summary: dict[str, float | int] = {
            "total_seconds": total,
            **phases,
            "least_squares_count": self.least_squares_count,
            "backtracking_count": self.backtracking_count,
            "gradient_count": self.gradient_count,
            "sigmoid_count": self.sigmoid_count,
            "line_search_candidate_count": self.line_search_candidate_count,
        }
        for name, seconds in phases.items():
            summary[name.replace("_seconds", "_percent")] = (
                0.0 if total == 0.0 else 100.0 * seconds / total
            )
        return summary
