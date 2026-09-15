"""Bridge WebUI task state with the standalone advertising post-processor."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.output_profiles import (
    BUILTIN_PROFILES,
    ProfileError,
    profile_dict,
    resolve_profile,
)
from scripts.run_pipeline import process_batch_results


class WebUIAdapterError(RuntimeError):
    """Raised when WebUI post-processing configuration or task output is invalid."""


_SAFE_TASK_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _relative_project_path(
    project_root: Path, raw_path: Any, *, description: str
) -> str:
    """Convert an existing resource path to a project-relative safe path."""
    if not isinstance(raw_path, (str, Path)):
        raise WebUIAdapterError(f"{description} path is required")
    root = project_root.resolve()
    candidate = (
        (root / raw_path).resolve()
        if not Path(raw_path).is_absolute()
        else Path(raw_path).resolve()
    )
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise WebUIAdapterError(
            f"{description} path is outside the project root"
        ) from exc
    if not candidate.is_file():
        raise WebUIAdapterError(f"{description} file does not exist")
    return relative.as_posix()


def _font_spec_path(raw_name: Any) -> str:
    """Normalize a font dropdown value into the approved resource/fonts namespace."""
    if not isinstance(raw_name, str) or not raw_name.strip():
        raise WebUIAdapterError("font name is required")
    path = Path(raw_name)
    if path.is_absolute() or ".." in path.parts:
        raise WebUIAdapterError(
            "font path must be selected from the managed font directory"
        )
    if len(path.parts) == 1:
        return f"resource/fonts/{path.name}"
    if path.parts[:2] != ("resource", "fonts"):
        raise WebUIAdapterError("font path must be selected from resource/fonts")
    return path.as_posix()


def build_webui_spec(
    *,
    profile_id: str,
    watermark: Mapping[str, Any],
    custom_texts: Sequence[Mapping[str, Any]],
    project_root: Path,
    image_layers: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """
    Convert page controls into a complete task-level post-processing spec.

    @param profile_id Built-in output profile selected by the page.
    @param watermark Watermark control values, including an optional absolute Logo path.
    @param custom_texts Independent text-row values from the page.
    @param project_root Project root used to constrain Logo and font resources.
    @param image_layers Optional overlay image rows with project-resolvable paths.
    @returns JSON-compatible specification for task index 1.
    @raises WebUIAdapterError If profile, paths, or control structures are invalid.
    """
    if not isinstance(watermark, Mapping):
        raise WebUIAdapterError("watermark settings must be an object")
    try:
        profile = resolve_profile(profile_id, BUILTIN_PROFILES)
    except ProfileError as exc:
        raise WebUIAdapterError(f"unknown output profile: {profile_id}") from exc
    enabled = watermark.get("enabled", False)
    if not isinstance(enabled, bool):
        raise WebUIAdapterError("watermark enabled must be boolean")
    trim_transparent = watermark.get("trim_transparent", True)
    if not isinstance(trim_transparent, bool):
        raise WebUIAdapterError("watermark trim_transparent must be boolean")
    logo_path = watermark.get("logo_path")
    if enabled:
        logo_path = _relative_project_path(project_root, logo_path, description="logo")
    else:
        logo_path = "watermark/logo.png"
    normalized_texts: list[dict[str, Any]] = []
    for text in custom_texts:
        if not isinstance(text, Mapping):
            raise WebUIAdapterError("custom text settings must be objects")
        normalized = dict(text)
        normalized["font_name"] = _font_spec_path(text.get("font_name"))
        normalized_texts.append(normalized)
    normalized_images: list[dict[str, Any]] = []
    for image in image_layers:
        if not isinstance(image, Mapping):
            raise WebUIAdapterError("image layer settings must be objects")
        normalized_image = dict(image)
        normalized_image["path"] = _relative_project_path(
            project_root, image.get("path"), description="image layer"
        )
        normalized_images.append(normalized_image)
    normalized_watermark = {
        "enabled": enabled,
        "logo_path": logo_path,
        "trim_transparent": trim_transparent,
        "position": watermark.get("position", "top_right"),
        "custom_xy_ratio": watermark.get("custom_xy_ratio"),
        "margin_px": watermark.get("margin_px", 20),
        "opacity": watermark.get("opacity", 0.85),
        "height_ratio": watermark.get("height_ratio", 0.08),
    }
    return {
        "schema_version": 1,
        "task_index": 1,
        "output": profile_dict(profile),
        "watermark": normalized_watermark,
        "custom_texts": normalized_texts,
        "image_layers": normalized_images,
    }


def save_webui_spec(path: Path, payload: Mapping[str, Any]) -> None:
    """
    Atomically save a WebUI post-processing spec as UTF-8 JSON.

    @param path Task-owned spec path.
    @param payload JSON-compatible specification.
    @returns None after durable replacement.
    @raises WebUIAdapterError If the file cannot be written.
    """
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
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        raise WebUIAdapterError(f"cannot save post-processing spec: {path}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def process_webui_result(
    task_id: str,
    result: Mapping[str, Any],
    *,
    spec_path: Path,
    project_root: Path,
    output_root: Path,
) -> tuple[str, ...]:
    """
    Post-process all MPT outputs for one WebUI task and return final paths.

    @param task_id MPT/WebUI task identifier used for stable output directories.
    @param result Successful `tm.start` result containing its `videos` list.
    @param spec_path Task-level post-processing JSON path.
    @param project_root MPT project root containing task output files/resources.
    @param output_root Root for stable raw copies and final advertising outputs.
    @returns Ordered tuple of final rendered paths.
    @raises WebUIAdapterError If task outputs are missing or any variant fails.
    """
    if (
        not isinstance(task_id, str)
        or not _SAFE_TASK_ID.fullmatch(task_id)
        or not isinstance(result, Mapping)
    ):
        raise WebUIAdapterError("task_id and result are required")
    videos = result.get("videos")
    if (
        not isinstance(videos, list)
        or not videos
        or any(not isinstance(path, str) for path in videos)
    ):
        raise WebUIAdapterError("MPT result must contain a non-empty videos list")
    return process_webui_videos(
        task_id,
        videos,
        spec_path=spec_path,
        project_root=project_root,
        output_root=output_root,
    )


def process_webui_videos(
    task_id: str,
    videos: Sequence[str],
    *,
    spec_path: Path,
    project_root: Path,
    output_root: Path,
) -> tuple[str, ...]:
    """
    Post-process a concrete list of MPT final paths before cross-post scheduling.

    @param task_id MPT/WebUI task identifier used for stable output directories.
    @param videos Ordered raw paths returned by `generate_final_videos`.
    @param spec_path Task-level post-processing JSON path.
    @param project_root MPT project root containing task output files/resources.
    @param output_root Root for stable raw copies and final advertising outputs.
    @returns Ordered tuple of final rendered paths.
    @raises WebUIAdapterError If any variant fails post-processing.
    """
    if (
        not isinstance(task_id, str)
        or not _SAFE_TASK_ID.fullmatch(task_id)
        or not isinstance(videos, Sequence)
        or isinstance(videos, (str, bytes))
        or not videos
        or any(not isinstance(path, str) or not path.strip() for path in videos)
    ):
        raise WebUIAdapterError("MPT result must contain a non-empty videos list")
    payload = {
        "total": 1,
        "succeeded": 1,
        "failed": 0,
        "tasks": [
            {
                "index": 1,
                "task_id": task_id,
                "status": "succeeded",
                "result": {"videos": list(videos)},
            }
        ],
    }
    task_dir = project_root / "storage" / "tasks" / task_id
    statuses = process_batch_results(
        payload,
        job_id=task_id,
        expected_tasks=1,
        mpt_root=project_root,
        job_dir=task_dir,
        spec_dir=spec_path.parent,
        resource_root=project_root,
        raw_root=output_root / "mpt_raw" / task_id,
        output_root=output_root / "final" / task_id,
    )
    variants = statuses[0].get("variants", [])
    failed = [item for item in variants if item.get("status") != "succeeded"]
    if failed:
        raise WebUIAdapterError(
            f"post-processing failed for task {task_id}: {failed[0].get('error', 'unknown error')}"
        )
    return tuple(item["output_path"] for item in variants)
