"""Measure throughput and resource usage of the real advertising pipeline."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.run_pipeline import PipelineError, run_pipeline

try:
    import resource
except ImportError:  # pragma: no cover - Windows has no POSIX resource module.
    resource = None


class PerformanceError(RuntimeError):
    """Raised when a benchmark cannot produce a trustworthy measurement."""


def _peak_rss_mb() -> float | None:
    """Read the current process maximum resident set size in megabytes."""
    if resource is None:
        raise PerformanceError("peak RSS metrics require a POSIX resource module")
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if value <= 0:
        return None
    # macOS reports bytes; Linux reports KiB.
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return value / divisor


def summarize_results(
    results: Sequence[Mapping[str, Any]],
    *,
    elapsed_seconds: float,
    peak_rss_mb: float | None,
) -> dict[str, Any]:
    """
    Aggregate per-variant metrics emitted by an actual pipeline run.

    @param results Per-task results returned by `run_pipeline`.
    @param elapsed_seconds Wall-clock duration of the benchmarked run.
    @param peak_rss_mb Maximum resident memory observed by the benchmark process.
    @returns JSON-compatible throughput and byte metrics.
    @raises PerformanceError If timing or metric values are invalid.
    """
    if not elapsed_seconds > 0:
        raise PerformanceError("benchmark elapsed_seconds must be positive")
    succeeded = 0
    failed = 0
    input_bytes = 0
    output_bytes = 0
    variant_seconds: list[float] = []
    for task in results:
        variants = task.get("variants", [])
        if not isinstance(variants, Sequence):
            raise PerformanceError("task variants must be an array")
        for variant in variants:
            if not isinstance(variant, Mapping):
                raise PerformanceError("variant result must be an object")
            if variant.get("status") == "succeeded":
                succeeded += 1
                duration = variant.get("elapsed_seconds")
                source_bytes = variant.get("input_bytes")
                rendered_bytes = variant.get("output_bytes")
                if not isinstance(duration, (int, float)) or not duration >= 0:
                    raise PerformanceError("successful variant lacks elapsed_seconds")
                if not isinstance(source_bytes, int) or source_bytes < 0:
                    raise PerformanceError("successful variant lacks input_bytes")
                if not isinstance(rendered_bytes, int) or rendered_bytes < 0:
                    raise PerformanceError("successful variant lacks output_bytes")
                variant_seconds.append(float(duration))
                input_bytes += source_bytes
                output_bytes += rendered_bytes
            elif variant.get("status") == "failed":
                failed += 1
            else:
                raise PerformanceError("variant status must be succeeded or failed")
    return {
        "elapsed_seconds": elapsed_seconds,
        "peak_rss_mb": peak_rss_mb,
        "task_count": len(results),
        "succeeded_variants": succeeded,
        "failed_variants": failed,
        "input_bytes": input_bytes,
        "output_bytes": output_bytes,
        "variants_per_second": succeeded / elapsed_seconds,
        "max_variant_seconds": max(variant_seconds, default=0.0),
    }


def benchmark_run(**run_kwargs: Any) -> dict[str, Any]:
    """
    Run the real pipeline once and return its measured throughput summary.

    @param run_kwargs Keyword arguments accepted by `run_pipeline`.
    @returns Summary containing task, byte, elapsed-time, and RSS metrics.
    @raises PipelineError If the pipeline fails before producing results.
    """
    started_at = time.perf_counter()
    results = run_pipeline(**run_kwargs)
    elapsed = time.perf_counter() - started_at
    return summarize_results(
        results,
        elapsed_seconds=elapsed,
        peak_rss_mb=_peak_rss_mb(),
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse benchmark arguments that mirror the pipeline runner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--batch-file", required=True, type=Path)
    parser.add_argument("--mpt-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument("--resource-root", type=Path, default=Path.cwd())
    parser.add_argument("--spec-dir", type=Path)
    parser.add_argument("--mpt-result", type=Path)
    parser.add_argument("--state-file", type=Path)
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--ffprobe", default="ffprobe")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one real pipeline benchmark and print JSON metrics."""
    args = _parse_args(argv)
    try:
        summary = benchmark_run(
            job_id=args.job_id,
            batch_path=args.batch_file,
            mpt_root=args.mpt_root,
            output_root=args.output_root,
            resource_root=args.resource_root,
            spec_dir=args.spec_dir,
            mpt_result_path=args.mpt_result,
            state_path=args.state_file,
            python_executable=args.python_executable,
            ffprobe=args.ffprobe,
        )
    except (PipelineError, PerformanceError) as exc:
        print(f"benchmark failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
