"""Contracts shared by the MPT batch runner and post-processing stage."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping


class PipelineContractError(ValueError):
    """Raised when an MPT batch result violates the external pipeline contract."""


@dataclass(frozen=True)
class BatchTaskResult:
    """Normalized result for one MPT batch task and all of its output variants."""

    job_id: str
    task_index: int
    mpt_task_id: str
    status: str
    videos: tuple[str, ...]
    failed_stage: str | None
    error: str | None


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    """Validate that a value is a mapping before reading contract fields."""
    if not isinstance(value, Mapping):
        raise PipelineContractError(f"{name} must be a JSON object")
    return value


def _require_int(value: Any, *, name: str, minimum: int = 0) -> int:
    """Validate an integer contract field without accepting booleans."""
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PipelineContractError(f"{name} must be an integer >= {minimum}")
    return value


def _normalize_task_entry(
    entry: Any,
    *,
    job_id: str,
) -> BatchTaskResult:
    """Normalize one raw CLI task entry into an immutable result record."""
    task = _require_mapping(entry, name="batch task")
    task_index = _require_int(task.get("index"), name="task.index", minimum=1)
    mpt_task_id = task.get("task_id")
    if not isinstance(mpt_task_id, str) or not mpt_task_id.strip():
        raise PipelineContractError(f"task {task_index} task_id must be non-empty")

    status = task.get("status")
    if status not in {"succeeded", "failed"}:
        raise PipelineContractError(
            f"task {task_index} status must be succeeded or failed"
        )

    error = task.get("error")
    if error is not None and (not isinstance(error, str) or not error.strip()):
        raise PipelineContractError(f"task {task_index} error must be a string or null")
    failed_stage = task.get("failed_stage")
    if failed_stage is not None and (
        not isinstance(failed_stage, str) or not failed_stage.strip()
    ):
        raise PipelineContractError(
            f"task {task_index} failed_stage must be a string or null"
        )

    if status == "failed":
        if not isinstance(error, str) or not error.strip():
            raise PipelineContractError(f"failed task {task_index} requires error")
        return BatchTaskResult(
            job_id=job_id,
            task_index=task_index,
            mpt_task_id=mpt_task_id.strip(),
            status=status,
            videos=(),
            failed_stage=failed_stage.strip() if failed_stage else None,
            error=error.strip(),
        )

    result = _require_mapping(task.get("result"), name=f"task {task_index}.result")
    raw_videos = result.get("videos")
    if not isinstance(raw_videos, list) or not raw_videos:
        raise PipelineContractError(
            f"succeeded task {task_index} must contain a non-empty result.videos list"
        )
    videos: list[str] = []
    for variant, raw_video in enumerate(raw_videos, start=1):
        if not isinstance(raw_video, str) or not raw_video.strip():
            raise PipelineContractError(
                f"task {task_index} video variant {variant} must be a non-empty path"
            )
        if "\x00" in raw_video:
            raise PipelineContractError(
                f"task {task_index} video variant {variant} contains a NUL byte"
            )
        videos.append(raw_video.strip())
    return BatchTaskResult(
        job_id=job_id,
        task_index=task_index,
        mpt_task_id=mpt_task_id.strip(),
        status=status,
        videos=tuple(videos),
        failed_stage=None,
        error=error.strip() if error else None,
    )


def normalize_batch_result(
    payload: Mapping[str, Any],
    *,
    job_id: str,
    expected_tasks: int,
) -> tuple[BatchTaskResult, ...]:
    """
    Normalize the current MPT CLI batch JSON and enforce task-index integrity.

    @param payload Parsed JSON object printed by the MPT batch CLI.
    @param job_id Externally-owned pipeline job identifier.
    @param expected_tasks Number of frozen batch entries expected in the result.
    @returns Results sorted by the original one-based task index.
    @raises PipelineContractError If counts, indices, statuses, or output paths are invalid.
    """
    root = _require_mapping(payload, name="batch result")
    if not isinstance(job_id, str) or not job_id.strip() or "\x00" in job_id:
        raise PipelineContractError("job_id must be a non-empty string without NUL")
    expected = _require_int(expected_tasks, name="expected_tasks", minimum=1)
    tasks = root.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != expected:
        raise PipelineContractError(
            f"batch result must contain exactly {expected} task entries"
        )
    total = _require_int(root.get("total"), name="batch.total", minimum=0)
    succeeded = _require_int(root.get("succeeded"), name="batch.succeeded", minimum=0)
    failed = _require_int(root.get("failed"), name="batch.failed", minimum=0)
    if total != expected or succeeded + failed != expected:
        raise PipelineContractError("batch counts do not match expected task count")

    normalized = tuple(
        _normalize_task_entry(entry, job_id=job_id.strip()) for entry in tasks
    )
    indices = [result.task_index for result in normalized]
    expected_indices = list(range(1, expected + 1))
    if sorted(indices) != expected_indices:
        raise PipelineContractError(
            "batch task indices must be unique and cover 1..expected_tasks"
        )
    actual_succeeded = sum(result.status == "succeeded" for result in normalized)
    if actual_succeeded != succeeded or expected - actual_succeeded != failed:
        raise PipelineContractError("batch status counts do not match task entries")
    return tuple(sorted(normalized, key=lambda result: result.task_index))


def _canonical_json(value: Any) -> str:
    """Serialize a JSON-compatible value deterministically for hashing."""
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise PipelineContractError(
            "render fingerprint input must be JSON-compatible"
        ) from exc


def compute_render_key(
    raw_sha256: str,
    *,
    spec: Mapping[str, Any],
    resource_hashes: Mapping[str, str],
    implementation_version: str,
    moviepy_version: str,
    ffmpeg_version: str,
) -> str:
    """
    Compute a stable render fingerprint from every input that affects pixels or encoding.

    @param raw_sha256 SHA-256 digest of the MPT raw video.
    @param spec Normalized task output and layer configuration.
    @param resource_hashes Hashes of enabled fonts, Logo, and other render resources.
    @param implementation_version Version of the post-processing implementation.
    @param moviepy_version Installed MoviePy version used for rendering.
    @param ffmpeg_version Installed FFmpeg version used for encoding/probing.
    @returns A lowercase SHA-256 render key.
    """
    values = {
        "raw_sha256": raw_sha256,
        "spec": spec,
        "resource_hashes": dict(resource_hashes),
        "implementation_version": implementation_version,
        "moviepy_version": moviepy_version,
        "ffmpeg_version": ffmpeg_version,
    }
    if any(
        not isinstance(values[name], str) or not values[name].strip()
        for name in (
            "raw_sha256",
            "implementation_version",
            "moviepy_version",
            "ffmpeg_version",
        )
    ):
        raise PipelineContractError(
            "render fingerprint version and digest fields are required"
        )
    for name, digest in resource_hashes.items():
        if (
            not isinstance(name, str)
            or not name.strip()
            or not isinstance(digest, str)
            or not digest.strip()
        ):
            raise PipelineContractError(
                "resource hashes must have non-empty string keys and values"
            )
    encoded = _canonical_json(values).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
