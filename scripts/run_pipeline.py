"""Run the MPT batch stage and the advertising-video post-processing stage."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from datetime import datetime, timezone
from fractions import Fraction
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.pipeline_contract import (
    PipelineContractError,
    compute_render_key,
    normalize_batch_result,
)
from scripts.asset_state import AssetStateError, commit_assets
from scripts.post_process import (
    ConfigValidationError,
    compose_video,
    load_post_process_spec,
)

moviepy_version = package_version("moviepy")


class PipelineError(RuntimeError):
    """Raised when a pipeline boundary or publication step cannot be completed."""


_MAX_BATCH_BYTES = 1 * 1024 * 1024
_IMPLEMENTATION_VERSION = "1"
_SAFE_JOB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _read_utf8(path: Path) -> str:
    """Read a required UTF-8 text file and expose decoding failures explicitly."""
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise PipelineError(f"cannot read UTF-8 file: {path}") from exc


def load_batch_entries(batch_path: Path) -> tuple[Mapping[str, Any], ...]:
    """
    Load and count an MPT JSON-array or JSONL batch manifest.

    @param batch_path Path to the frozen batch manifest.
    @returns Immutable tuple of JSON object entries in source order.
    @raises PipelineError If the file is missing, oversized, empty, or malformed.
    """
    try:
        if batch_path.stat().st_size > _MAX_BATCH_BYTES:
            raise PipelineError(f"batch manifest exceeds {_MAX_BATCH_BYTES} bytes")
    except OSError as exc:
        raise PipelineError(f"cannot stat batch manifest: {batch_path}") from exc
    content = _read_utf8(batch_path).strip()
    if not content:
        raise PipelineError("batch manifest must contain at least one task")
    try:
        decoded = json.loads(content)
    except json.JSONDecodeError:
        decoded = None
    if decoded is not None and isinstance(decoded, list):
        if not decoded:
            raise PipelineError("JSON batch manifest must be a non-empty array")
        entries = tuple(decoded)
    elif decoded is not None and isinstance(decoded, Mapping) and "\n" not in content:
        entries = (decoded,)
    else:
        try:
            entries = tuple(
                json.loads(line) for line in content.splitlines() if line.strip()
            )
        except json.JSONDecodeError as exc:
            raise PipelineError("JSONL batch manifest contains invalid JSON") from exc
        if not entries:
            raise PipelineError("JSONL batch manifest must contain at least one task")
    if any(not isinstance(entry, Mapping) for entry in entries):
        raise PipelineError("every batch task must be a JSON object")
    return entries


def parse_batch_stdout(stdout: str) -> Mapping[str, Any]:
    """
    Extract exactly one MPT batch-result JSON object from standard output.

    @param stdout Complete UTF-8 stdout captured from the MPT CLI.
    @returns The parsed result object containing `tasks`.
    @raises PipelineError If output contains no result or multiple candidates.
    """
    if not isinstance(stdout, str):
        raise PipelineError("MPT stdout must be text")
    candidates: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(stdout.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping) and {
            "total",
            "succeeded",
            "failed",
            "tasks",
        }.issubset(value):
            candidates.append(value)
        elif isinstance(value, Mapping):
            continue
        else:
            raise PipelineError(f"MPT stdout line {line_number} is not a JSON object")
    if len(candidates) != 1:
        raise PipelineError(
            f"expected exactly one MPT batch result in stdout, found {len(candidates)}"
        )
    return candidates[0]


def run_mpt_batch(
    batch_path: Path,
    *,
    mpt_root: Path,
    job_dir: Path,
    python_executable: str = sys.executable,
) -> tuple[int, Mapping[str, Any]]:
    """
    Execute the official MPT CLI without shell interpolation and persist its logs.

    @param batch_path Frozen JSON/JSONL manifest passed to MPT.
    @param mpt_root Directory containing the official `cli.py` and runtime storage.
    @param job_dir Job directory where stdout and stderr are recorded.
    @param python_executable Python executable for the MPT environment.
    @returns `(exit_code, parsed_batch_result)`; non-zero exit may still contain partial successes.
    @raises PipelineError If the command, decoding, or result extraction fails.
    """
    cli_path = (mpt_root / "cli.py").resolve()
    if not cli_path.is_file():
        raise PipelineError(f"MPT CLI does not exist: {cli_path}")
    job_dir.mkdir(parents=True, exist_ok=True)
    command = [
        python_executable,
        str(cli_path),
        "--batch-file",
        str(batch_path.resolve()),
        "--stop-at",
        "video",
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=str(mpt_root.resolve()),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError as exc:
        raise PipelineError("failed to start MPT CLI") from exc
    try:
        stdout = completed.stdout.decode("utf-8")
        stderr = completed.stderr.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PipelineError("MPT CLI output is not UTF-8") from exc
    (job_dir / "mpt.stdout.log").write_text(stdout, encoding="utf-8")
    (job_dir / "mpt.stderr.log").write_text(stderr, encoding="utf-8")
    return completed.returncode, parse_batch_stdout(stdout)


def sha256_file(path: Path) -> str:
    """
    Calculate a streaming SHA-256 digest for a regular file.

    @param path File whose content identity is required.
    @returns Lowercase SHA-256 hexadecimal digest.
    @raises PipelineError If the path is not a readable regular file.
    """
    if not path.is_file():
        raise PipelineError(f"file does not exist: {path}")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise PipelineError(f"cannot hash file: {path}") from exc
    return digest.hexdigest()


def _resolve_mpt_output(raw_path: str, *, mpt_root: Path) -> Path:
    """Resolve an MPT-reported output and require it to remain under the MPT root."""
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = mpt_root / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(mpt_root.resolve())
    except ValueError as exc:
        raise PipelineError(f"MPT output is outside mpt_root: {raw_path}") from exc
    if not resolved.is_file():
        raise PipelineError(f"MPT output does not exist: {resolved}")
    return resolved


def copy_raw_video(
    source_path: str,
    destination_path: Path,
    *,
    mpt_root: Path,
) -> str:
    """
    Copy one MPT output into the stable raw-output area without changing its content.

    @param source_path Path reported by the trusted MPT CLI result.
    @param destination_path Stable job-owned raw copy path.
    @param mpt_root Approved root for MPT-reported output files.
    @returns SHA-256 digest of the copied or already matching file.
    @raises PipelineError If the source escapes the root or destination conflicts.
    """
    source = _resolve_mpt_output(source_path, mpt_root=mpt_root)
    source_hash = sha256_file(source)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.exists():
        if sha256_file(destination_path) != source_hash:
            raise PipelineError(
                f"raw output already exists with different content: {destination_path}"
            )
        return source_hash
    temporary = destination_path.with_name(
        f".{destination_path.name}.{os.getpid()}.tmp"
    )
    try:
        shutil.copy2(source, temporary)
        if sha256_file(temporary) != source_hash:
            raise PipelineError(f"raw output changed while copying: {source}")
        publish_without_overwrite(temporary, destination_path)
    finally:
        temporary.unlink(missing_ok=True)
    return source_hash


def publish_without_overwrite(temporary_path: Path, target_path: Path) -> None:
    """
    Atomically publish a same-filesystem temporary file without replacing an existing target.

    @param temporary_path Completed temporary file in the target directory.
    @param target_path Final path that must not already exist.
    @returns None after the hard-link publication succeeds.
    @raises PipelineError If either path is invalid or the target already exists.
    """
    if not temporary_path.is_file():
        raise PipelineError(f"temporary output does not exist: {temporary_path}")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(temporary_path, target_path)
    except FileExistsError as exc:
        raise PipelineError(
            f"refusing to overwrite existing output: {target_path}"
        ) from exc
    except OSError as exc:
        raise PipelineError(f"cannot atomically publish output: {target_path}") from exc
    temporary_path.unlink()


def ffprobe_duration(path: Path, *, ffprobe: str = "ffprobe") -> float:
    """
    Read a positive media duration through ffprobe without invoking a shell.

    @param path Media file to inspect.
    @param ffprobe Executable name or approved absolute path.
    @returns Duration in seconds.
    @raises PipelineError If ffprobe fails or emits an invalid duration.
    """
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise PipelineError("failed to start ffprobe") from exc
    if completed.returncode != 0:
        raise PipelineError(f"ffprobe failed for media: {path}")
    try:
        duration = float(completed.stdout.strip())
    except ValueError as exc:
        raise PipelineError(
            f"ffprobe returned an invalid duration for: {path}"
        ) from exc
    if not duration > 0:
        raise PipelineError(f"media duration must be positive: {path}")
    return duration


def _ffprobe_streams(path: Path, *, ffprobe: str) -> Mapping[str, Any]:
    """Read stream metadata needed to validate a rendered output."""
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=codec_type,width,height,r_frame_rate,pix_fmt,codec_name",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise PipelineError("failed to start ffprobe") from exc
    if completed.returncode != 0:
        raise PipelineError(f"ffprobe failed for media: {path}")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise PipelineError(f"ffprobe returned invalid JSON for media: {path}") from exc
    if not isinstance(payload, Mapping) or not isinstance(payload.get("streams"), list):
        raise PipelineError(f"ffprobe metadata is incomplete for media: {path}")
    return payload


def validate_output(
    path: Path,
    *,
    output_spec: Any,
    expected_duration: float | None = None,
    ffprobe: str = "ffprobe",
) -> None:
    """
    Validate rendered dimensions, frame rate, pixel format, audio policy, and duration.

    @param path Rendered video path.
    @param output_spec Normalized `OutputSpec` containing target values.
    @param ffprobe ffprobe executable used for metadata inspection.
    @param expected_duration Optional source duration to compare within 0.1 seconds.
    @returns None when the output meets its explicit target contract.
    @raises PipelineError If streams or target properties do not match.
    """
    payload = _ffprobe_streams(path, ffprobe=ffprobe)
    video_stream = next(
        (
            stream
            for stream in payload["streams"]
            if isinstance(stream, Mapping) and stream.get("codec_type") == "video"
        ),
        None,
    )
    if not isinstance(video_stream, Mapping):
        raise PipelineError(f"rendered output has no video stream: {path}")
    if (video_stream.get("width"), video_stream.get("height")) != (
        output_spec.width,
        output_spec.height,
    ):
        raise PipelineError(f"rendered dimensions do not match output spec: {path}")
    pixel_format = video_stream.get("pix_fmt")
    if pixel_format != output_spec.pixel_format:
        raise PipelineError(f"rendered pixel format does not match output spec: {path}")
    frame_rate = video_stream.get("r_frame_rate")
    try:
        actual_fps = float(Fraction(frame_rate))
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise PipelineError(f"rendered frame rate is invalid: {path}") from exc
    if not math.isclose(actual_fps, output_spec.fps, rel_tol=0.0, abs_tol=0.01):
        raise PipelineError(f"rendered frame rate does not match output spec: {path}")
    audio_present = any(
        isinstance(stream, Mapping) and stream.get("codec_type") == "audio"
        for stream in payload["streams"]
    )
    if output_spec.audio_required and not audio_present:
        raise PipelineError(f"rendered output is missing required audio: {path}")
    format_data = payload.get("format", {})
    if not isinstance(format_data, Mapping):
        raise PipelineError(f"ffprobe format metadata is invalid: {path}")
    try:
        duration = float(format_data.get("duration"))
    except (TypeError, ValueError) as exc:
        raise PipelineError(f"rendered duration is invalid: {path}") from exc
    if not duration > 0:
        raise PipelineError(f"rendered duration must be positive: {path}")
    if expected_duration is not None and not math.isclose(
        duration, expected_duration, rel_tol=0.0, abs_tol=0.1
    ):
        raise PipelineError(f"rendered duration does not match source: {path}")


def _write_json_atomic(path: Path, payload: Any) -> None:
    """Write JSON to a sibling temporary file and publish it without partial content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        raise PipelineError(f"cannot write JSON state: {path}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _spec_for_task(spec_dir: Path, task_index: int) -> Path:
    """Return the only supported per-task specification path."""
    return spec_dir / f"{task_index}.json"


def _task_asset_ids(
    request_path: Path, *, expected_tasks: int
) -> tuple[tuple[str, ...], ...]:
    """Read optional selector asset IDs while preserving the frozen task order."""
    if not request_path.exists():
        return tuple(() for _ in range(expected_tasks))
    request = _read_utf8(request_path)
    try:
        payload = json.loads(request)
    except json.JSONDecodeError as exc:
        raise PipelineError(
            f"selector input snapshot is invalid JSON: {request_path}"
        ) from exc
    if not isinstance(payload, Mapping) or not isinstance(payload.get("tasks"), list):
        raise PipelineError("selector input snapshot must contain a tasks array")
    tasks = payload["tasks"]
    if len(tasks) != expected_tasks:
        raise PipelineError("selector input task count differs from batch manifest")
    asset_groups: list[tuple[str, ...]] = []
    for index, raw_task in enumerate(tasks, start=1):
        if not isinstance(raw_task, Mapping):
            raise PipelineError(f"selector task {index} must be an object")
        raw_ids = raw_task.get("video_material_ids", [])
        if raw_ids is None:
            raw_ids = []
        if not isinstance(raw_ids, list) or any(
            not isinstance(asset_id, str) for asset_id in raw_ids
        ):
            raise PipelineError(f"selector task {index} video_material_ids is invalid")
        asset_groups.append(tuple(dict.fromkeys(raw_ids)))
    return tuple(asset_groups)


def _commit_successful_assets(
    request_path: Path,
    *,
    state_path: Path,
    job_id: str,
    results: Sequence[Mapping[str, Any]],
) -> None:
    """Commit only task assets that have at least one successfully rendered variant."""
    groups = _task_asset_ids(request_path, expected_tasks=len(results))
    timestamp = datetime.now(timezone.utc).isoformat()
    for task_result, asset_ids in zip(results, groups):
        if not asset_ids:
            continue
        variants = task_result.get("variants", [])
        succeeded = any(
            isinstance(variant, Mapping) and variant.get("status") == "succeeded"
            for variant in variants
        )
        if not succeeded:
            continue
        task_index = task_result.get("task_index")
        if not isinstance(task_index, int) or task_index < 1:
            raise PipelineError("pipeline result contains an invalid task_index")
        try:
            commit_assets(
                state_path,
                asset_ids,
                owner=f"{job_id}:{task_index}",
                now=timestamp,
            )
        except AssetStateError as exc:
            raise PipelineError(
                f"asset commit failed for task {task_index}: {exc}"
            ) from exc


def _resource_hashes(spec) -> dict[str, str]:
    """Hash enabled font and Logo resources referenced by a normalized specification."""
    hashes: dict[str, str] = {}
    if spec.watermark.enabled:
        assert spec.watermark.logo_path is not None
        hashes["logo"] = sha256_file(spec.watermark.logo_path)
    text_specs = getattr(spec, "custom_texts", None)
    if text_specs is None:
        text_specs = getattr(spec, "text_layers", ())
    for index, text_spec in enumerate(text_specs):
        hashes[f"font:{index}"] = sha256_file(text_spec.font_path)
    image_specs = getattr(spec, "image_layers", ())
    for index, image_spec in enumerate(image_specs):
        hashes[f"image:{index}"] = sha256_file(image_spec.path)
    return hashes


def _spec_json(spec) -> Mapping[str, Any]:
    """Convert normalized dataclasses and Paths into canonical JSON-compatible data."""
    value = asdict(spec)
    return json.loads(json.dumps(value, default=str, ensure_ascii=False))


def _ffmpeg_version(ffprobe: str) -> str:
    """Return the first ffprobe version line used as part of the render fingerprint."""
    try:
        completed = subprocess.run(
            [ffprobe, "-version"], capture_output=True, text=True, check=False
        )
    except OSError as exc:
        raise PipelineError("failed to query ffprobe version") from exc
    if completed.returncode != 0 or not completed.stdout.strip():
        raise PipelineError("ffprobe version query failed")
    return completed.stdout.splitlines()[0].strip()


def _render_variant(
    raw_path: Path,
    *,
    spec_path: Path,
    resource_root: Path,
    task_index: int,
    variant: int,
    output_root: Path,
    status: dict[str, Any],
    ffprobe: str,
    raw_sha256: str,
    ffmpeg_version: str,
) -> dict[str, Any]:
    """Validate, fingerprint, render, probe, and record one raw-video variant."""
    started_at = time.perf_counter()
    duration = ffprobe_duration(raw_path, ffprobe=ffprobe)
    spec = load_post_process_spec(
        spec_path,
        task_index=task_index,
        video_duration=duration,
        resource_root=resource_root,
    )
    render_key = compute_render_key(
        raw_sha256,
        spec=_spec_json(spec),
        resource_hashes=_resource_hashes(spec),
        implementation_version=_IMPLEMENTATION_VERSION,
        moviepy_version=moviepy_version,
        ffmpeg_version=ffmpeg_version,
    )
    output_path = output_root / str(task_index) / render_key / f"{variant}.mp4"
    existing = status.get("outputs", {}).get(str(output_path))
    if output_path.exists():
        if (
            not isinstance(existing, Mapping)
            or existing.get("render_key") != render_key
            or existing.get("status") != "succeeded"
        ):
            raise PipelineError(
                f"output exists without a matching successful record: {output_path}"
            )
        ffprobe_duration(output_path, ffprobe=ffprobe)
        validate_output(
            output_path,
            output_spec=spec.output,
            expected_duration=duration,
            ffprobe=ffprobe,
        )
        output_hash = sha256_file(output_path)
        if existing.get("output_sha256") != output_hash:
            raise PipelineError(
                f"output content does not match its recorded hash: {output_path}"
            )
        return {
            "status": "succeeded",
            "output_path": str(output_path),
            "render_key": render_key,
            "output_sha256": output_hash,
            "reused": True,
            "elapsed_seconds": time.perf_counter() - started_at,
            "input_bytes": raw_path.stat().st_size,
            "output_bytes": output_path.stat().st_size,
        }
    temporary = output_path.with_name(f".{variant}.{os.getpid()}.tmp.mp4")
    try:
        compose_video(raw_path, spec, temporary)
        ffprobe_duration(temporary, ffprobe=ffprobe)
        validate_output(
            temporary,
            output_spec=spec.output,
            expected_duration=duration,
            ffprobe=ffprobe,
        )
        publish_without_overwrite(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    output_hash = sha256_file(output_path)
    return {
        "status": "succeeded",
        "output_path": str(output_path),
        "render_key": render_key,
        "output_sha256": output_hash,
        "reused": False,
        "elapsed_seconds": time.perf_counter() - started_at,
        "input_bytes": raw_path.stat().st_size,
        "output_bytes": output_path.stat().st_size,
    }


def _process_task_variants(
    task: Any,
    *,
    spec_dir: Path,
    resource_root: Path,
    raw_root: Path,
    output_root: Path,
    mpt_root: Path,
    ffprobe: str,
    ffmpeg_version: str,
    status: dict[str, Any],
) -> dict[str, Any]:
    """
    Process every raw-video variant for one normalized MPT task.

    @param task Normalized immutable MPT batch task result.
    @param spec_dir Per-task post-processing spec directory.
    @param resource_root Approved Logo/font resource root.
    @param raw_root Stable raw-copy root for this job.
    @param output_root Stable final-output root for this job.
    @param mpt_root Approved root for MPT-reported outputs.
    @param ffprobe ffprobe executable used by rendering checks.
    @param status Mutable in-memory output index used only for idempotency checks.
    @returns Per-task status dictionary with independent variant results.
    """
    task_status: dict[str, Any] = {
        "task_index": task.task_index,
        "mpt_task_id": task.mpt_task_id,
        "status": task.status,
        "failed_stage": task.failed_stage,
        "error": task.error,
        "variants": [],
    }
    if task.status == "failed":
        return task_status
    spec_path = _spec_for_task(spec_dir, task.task_index)
    for variant, source_path in enumerate(task.videos, start=1):
        try:
            raw_path = raw_root / str(task.task_index) / f"{variant}.mp4"
            raw_hash = copy_raw_video(source_path, raw_path, mpt_root=mpt_root)
            variant_status = _render_variant(
                raw_path,
                spec_path=spec_path,
                resource_root=resource_root,
                task_index=task.task_index,
                variant=variant,
                output_root=output_root,
                status=status,
                ffprobe=ffprobe,
                raw_sha256=raw_hash,
                ffmpeg_version=ffmpeg_version,
            )
            variant_status["raw_path"] = str(raw_path)
            variant_status["raw_sha256"] = raw_hash
            status["outputs"][variant_status["output_path"]] = {
                "render_key": variant_status["render_key"],
                "status": "succeeded",
                "output_sha256": variant_status["output_sha256"],
            }
        except (PipelineError, ConfigValidationError) as exc:
            variant_status = {"status": "failed", "error": str(exc)}
        task_status["variants"].append(variant_status)
    if task_status["variants"] and all(
        item["status"] == "succeeded" for item in task_status["variants"]
    ):
        task_status["status"] = "succeeded"
    elif task_status["variants"]:
        task_status["status"] = "failed"
    return task_status


def process_batch_results(
    payload: Mapping[str, Any],
    *,
    job_id: str,
    expected_tasks: int,
    mpt_root: Path,
    job_dir: Path,
    spec_dir: Path,
    resource_root: Path,
    raw_root: Path,
    output_root: Path,
    ffprobe: str = "ffprobe",
) -> tuple[dict[str, Any], ...]:
    """
    Copy successful MPT variants and process every task independently.

    @param payload Parsed MPT batch result.
    @param job_id External pipeline job identifier.
    @param expected_tasks Frozen number of batch entries.
    @param mpt_root Approved root for MPT output paths.
    @param job_dir Job state and log directory.
    @param spec_dir Directory containing `{task_index}.json` specs.
    @param resource_root Approved root for Logo and font files.
    @param raw_root Stable destination for copied MPT outputs.
    @param output_root Stable destination for final rendered outputs.
    @param ffprobe ffprobe executable used for duration and output checks.
    @returns Immutable tuple of per-task status dictionaries.
    @raises PipelineError If the batch contract or shared state is invalid.
    """
    try:
        tasks = normalize_batch_result(
            payload, job_id=job_id, expected_tasks=expected_tasks
        )
    except PipelineContractError as exc:
        raise PipelineError(str(exc)) from exc
    status_path = job_dir / "status.json"
    status: dict[str, Any] = {"job_id": job_id, "outputs": {}}
    if status_path.exists():
        try:
            saved_status = json.loads(_read_utf8(status_path))
        except json.JSONDecodeError as exc:
            raise PipelineError(
                f"saved pipeline status is invalid JSON: {status_path}"
            ) from exc
        if (
            not isinstance(saved_status, Mapping)
            or saved_status.get("job_id") != job_id
        ):
            raise PipelineError("saved pipeline status has a different job_id")
        saved_outputs = saved_status.get("outputs", {})
        if not isinstance(saved_outputs, Mapping):
            raise PipelineError("saved pipeline status outputs must be an object")
        status["outputs"] = dict(saved_outputs)
    results: list[dict[str, Any]] = []
    ffmpeg_version = (
        _ffmpeg_version(ffprobe)
        if any(task.status == "succeeded" for task in tasks)
        else "not-used"
    )
    for task in tasks:
        task_result = _process_task_variants(
            task,
            spec_dir=spec_dir,
            resource_root=resource_root,
            raw_root=raw_root,
            output_root=output_root,
            mpt_root=mpt_root,
            ffprobe=ffprobe,
            ffmpeg_version=ffmpeg_version,
            status=status,
        )
        results.append(task_result)
        _write_json_atomic(status_path, {**status, "tasks": results})
    _write_json_atomic(status_path, {**status, "tasks": results})
    return tuple(results)


def run_pipeline(
    *,
    job_id: str,
    batch_path: Path,
    mpt_root: Path,
    output_root: Path,
    resource_root: Path,
    python_executable: str = sys.executable,
    mpt_result_path: Path | None = None,
    spec_dir: Path | None = None,
    state_path: Path | None = None,
    ffprobe: str = "ffprobe",
) -> tuple[dict[str, Any], ...]:
    """
    Run MPT (or consume a saved result) and then post-process all successful variants.

    @param job_id Stable external identifier for this pipeline run.
    @param batch_path Frozen MPT JSON/JSONL batch manifest.
    @param mpt_root Official MPT root containing `cli.py`.
    @param output_root Root for job state, raw copies, and final outputs.
    @param resource_root Approved Logo/font resource root.
    @param python_executable Python interpreter used for MPT CLI.
    @param mpt_result_path Optional previously saved MPT result to skip stage two.
    @param spec_dir Optional per-task spec directory; defaults to `<job>/post_process_specs`.
    @param state_path Optional asset state path; defaults to `<output-parent>/state/assets.json`.
    @param ffprobe ffprobe executable used for validation.
    @returns Immutable per-task result records; failures are recorded per variant.
    @raises PipelineError If stage-two result parsing or shared contract validation fails.
    """
    if not isinstance(job_id, str) or not _SAFE_JOB_ID.fullmatch(job_id):
        raise PipelineError("job_id must be 1..128 safe filename characters")
    entries = load_batch_entries(batch_path)
    job_dir = output_root / "jobs" / job_id
    selected_spec_dir = spec_dir or (job_dir / "post_process_specs")
    if mpt_result_path is None:
        exit_code, payload = run_mpt_batch(
            batch_path,
            mpt_root=mpt_root,
            job_dir=job_dir,
            python_executable=python_executable,
        )
    else:
        try:
            payload = json.loads(_read_utf8(mpt_result_path))
        except json.JSONDecodeError as exc:
            raise PipelineError(
                f"saved MPT result is invalid JSON: {mpt_result_path}"
            ) from exc
        exit_code = 0
    _write_json_atomic(job_dir / "mpt_result.json", payload)
    results = process_batch_results(
        payload,
        job_id=job_id,
        expected_tasks=len(entries),
        mpt_root=mpt_root,
        job_dir=job_dir,
        spec_dir=selected_spec_dir,
        resource_root=resource_root,
        raw_root=output_root / "mpt_raw" / job_id,
        output_root=output_root / "final" / job_id,
        ffprobe=ffprobe,
    )
    selected_state_path = state_path or output_root.parent / "state" / "assets.json"
    _commit_successful_assets(
        job_dir / "input.json",
        state_path=selected_state_path,
        job_id=job_id,
        results=results,
    )
    pipeline_summary = {
        "job_id": job_id,
        "mpt_exit_code": exit_code,
        "tasks": results,
    }
    _write_json_atomic(job_dir / "pipeline.json", pipeline_summary)
    if exit_code != 0 and all(task["status"] == "succeeded" for task in results):
        raise PipelineError(
            f"MPT CLI exited with status {exit_code} despite successful task records"
        )
    return results


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the standalone pipeline command-line arguments."""
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
    """
    Run the pipeline CLI and return a process exit status.

    @param argv Optional argument sequence; defaults to `sys.argv[1:]`.
    @returns 0 when every task/variant succeeds, otherwise 1 or 2 for explicit errors.
    """
    args = _parse_args(argv)
    try:
        results = run_pipeline(
            job_id=args.job_id,
            batch_path=args.batch_file,
            mpt_root=args.mpt_root,
            output_root=args.output_root,
            resource_root=args.resource_root,
            python_executable=args.python_executable,
            mpt_result_path=args.mpt_result,
            spec_dir=args.spec_dir,
            state_path=args.state_file,
            ffprobe=args.ffprobe,
        )
    except (PipelineError, ConfigValidationError) as exc:
        print(f"pipeline failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"tasks": results}, ensure_ascii=False))
    return 0 if all(task["status"] == "succeeded" for task in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
