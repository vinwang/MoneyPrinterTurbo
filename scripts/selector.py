"""Create frozen MPT batch entries and per-task post-processing specifications."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.models.schema import VideoParams
from scripts.asset_state import AssetStateError, release_assets, reserve_assets
from scripts.llm_classifier import ClassificationError, LLMClassifier
from scripts.output_profiles import (
    BUILTIN_PROFILES,
    OutputProfile,
    ProfileError,
    load_profiles,
    profile_dict,
    resolve_profile,
)


class SelectorError(RuntimeError):
    """Raised when a selector request cannot be resolved against the manifest."""


_REQUEST_KEYS = {"job_id", "tasks"}
_TASK_CONTROL_KEYS = {"video_material_ids", "bgm_id", "post_process"}
_JOB_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SAFE_ASSET_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_TIMELINE_EPSILON_SECONDS = 0.02
_DEFAULT_PROFILE_ID = "portrait-test-v1"
_DEFAULT_OUTPUT = profile_dict(BUILTIN_PROFILES[_DEFAULT_PROFILE_ID])
_DEFAULT_WATERMARK = {
    "enabled": False,
    "logo_path": "watermark/logo.png",
    "trim_transparent": True,
    "position": "top_right",
    "custom_xy_ratio": None,
    "margin_px": 20,
    "opacity": 0.85,
    "height_ratio": 0.08,
}


def _read_json(path: Path, *, description: str) -> Any:
    """Read a required UTF-8 JSON file for the selector stage."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SelectorError(f"cannot read {description}: {path}") from exc


def _write_text_atomic(path: Path, content: str) -> None:
    """Write text through a sibling temporary file and atomic replacement."""
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
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        raise SelectorError(f"cannot write selector output: {path}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _mapping(value: Any, *, description: str) -> Mapping[str, Any]:
    """Require a JSON object at a selector input boundary."""
    if not isinstance(value, Mapping):
        raise SelectorError(f"{description} must be an object")
    return value


def _merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Merge nested output/watermark defaults without mutating caller dictionaries."""
    result = dict(base)
    for key, value in override.items():
        if isinstance(result.get(key), Mapping) and isinstance(value, Mapping):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def _resolve_asset(
    manifest: Mapping[str, Any],
    *,
    collection: str,
    asset_id: str,
    root_key: str,
) -> tuple[Mapping[str, Any], Path]:
    """
    Resolve one manifest ID to a real file below its declared root.

    @param manifest Loaded manifest containing asset collections and root fields.
    @param collection Collection name, such as `videos` or `bgm`.
    @param asset_id Stable manifest asset identifier.
    @param root_key Manifest field containing the collection root.
    @returns `(record, resolved_path)` for the selected asset.
    @raises SelectorError If the ID is unknown or the path escapes its root.
    """
    assets = manifest.get(collection)
    if not isinstance(assets, list):
        raise SelectorError(f"manifest.{collection} must be an array")
    record = next(
        (
            item
            for item in assets
            if isinstance(item, Mapping) and item.get("id") == asset_id
        ),
        None,
    )
    if record is None:
        raise SelectorError(f"unknown {collection} asset id: {asset_id}")
    if not isinstance(asset_id, str) or not _SAFE_ASSET_ID.fullmatch(asset_id):
        raise SelectorError(
            f"{collection} asset id contains unsafe filename characters"
        )
    root_value = manifest.get(root_key)
    if not isinstance(root_value, str) or not root_value.strip():
        raise SelectorError(f"manifest.{root_key} must be a non-empty path")
    relative_path = record.get("path")
    if (
        not isinstance(relative_path, str)
        or not relative_path.strip()
        or "\x00" in relative_path
    ):
        raise SelectorError(
            f"manifest {collection} asset has an invalid path: {asset_id}"
        )
    root = Path(root_value).resolve()
    resolved = (root / relative_path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise SelectorError(f"manifest path escapes its root: {asset_id}") from exc
    if not resolved.is_file():
        raise SelectorError(f"manifest asset file does not exist: {resolved}")
    expected_hash = record.get("sha256")
    if not isinstance(expected_hash, str) or not expected_hash.strip():
        raise SelectorError(
            f"manifest {collection} asset is missing sha256: {asset_id}"
        )
    if _sha256_file(resolved) != expected_hash.lower():
        raise SelectorError(
            f"manifest {collection} asset hash does not match: {asset_id}"
        )
    return record, resolved


def _sha256_file(path: Path) -> str:
    """Hash a selected BGM file before copying it into an MPT-managed directory."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise SelectorError(f"cannot hash BGM asset: {path}") from exc
    return digest.hexdigest()


def _materialize_bgm(path: Path, *, asset_id: str, mpt_root: Path) -> Path:
    """
    Copy a selected BGM into MPT's managed storage with no-clobber semantics.

    @param path Source BGM path resolved from the manifest.
    @param asset_id Stable manifest ID used to avoid basename collisions.
    @param mpt_root Official MPT root containing `storage/bgm`.
    @returns Absolute path of the managed BGM file.
    @raises SelectorError If the destination conflicts or copying fails.
    """
    destination_root = (mpt_root / "storage" / "bgm").resolve()
    try:
        destination_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise SelectorError(
            f"cannot create managed BGM directory: {destination_root}"
        ) from exc
    destination = destination_root / f"{asset_id}{path.suffix.lower()}"
    source_hash = _sha256_file(path)
    if destination.exists():
        if not destination.is_file() or _sha256_file(destination) != source_hash:
            raise SelectorError(f"managed BGM destination conflicts: {destination}")
        return destination
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        shutil.copy2(path, temporary)
        if _sha256_file(temporary) != source_hash:
            raise SelectorError(f"BGM changed while copying: {path}")
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise SelectorError(
                f"managed BGM destination appeared during copy: {destination}"
            ) from exc
        temporary.unlink()
    except OSError as exc:
        raise SelectorError(f"cannot materialize BGM: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _materialize_video(path: Path, *, asset_id: str, mpt_root: Path) -> Path:
    """
    Copy a template video source into MPT-managed local storage.

    @param path Source video resolved and hashed from the manifest.
    @param asset_id Stable manifest ID used for a collision-safe destination.
    @param mpt_root Official MPT root containing `storage/local_videos`.
    @returns Absolute managed video path.
    @raises SelectorError If the destination conflicts or copying fails.
    """
    destination_root = (mpt_root / "storage" / "local_videos").resolve()
    try:
        destination_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise SelectorError(
            f"cannot create managed video directory: {destination_root}"
        ) from exc
    destination = destination_root / f"template-{asset_id}{path.suffix.lower()}"
    source_hash = _sha256_file(path)
    if destination.exists():
        if not destination.is_file() or _sha256_file(destination) != source_hash:
            raise SelectorError(f"managed video destination conflicts: {destination}")
        return destination
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        shutil.copy2(path, temporary)
        if _sha256_file(temporary) != source_hash:
            raise SelectorError(f"video changed while copying: {path}")
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise SelectorError(
                f"managed video destination appeared during copy: {destination}"
            ) from exc
        temporary.unlink()
    except OSError as exc:
        raise SelectorError(f"cannot materialize video: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _allowed_video_fields() -> set[str]:
    """Return the exact field names accepted by the current MPT VideoParams model."""
    return set(VideoParams.model_fields)


def _json_value(value: Any) -> Any:
    """Convert Pydantic/enum values to JSON without serializer warnings."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "__dict__"):
        return _json_value(vars(value))
    return value


def _resolve_video_materials(
    material_ids: Sequence[str],
    *,
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Resolve manifest video IDs into MPT local material records."""
    materials = []
    for asset_id in material_ids:
        if not isinstance(asset_id, str) or not asset_id.strip():
            raise SelectorError("video_material_ids must contain non-empty strings")
        record, path = _resolve_asset(
            manifest,
            collection="videos",
            asset_id=asset_id,
            root_key="videos_root",
        )
        materials.append(
            {
                "provider": "local",
                "url": str(path),
                "duration": int(record.get("duration", 0)),
            }
        )
    return materials


def _build_batch_entry(
    task: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    mpt_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Resolve one request task into an MPT entry and a raw post-process override.

    @param task Selector task containing MPT fields and optional asset IDs.
    @param manifest Validated local media manifest.
    @returns `(validated MPT JSON entry, post-process override)`.
    @raises SelectorError If fields, IDs, or MPT model validation fail.
    """
    unknown = set(task) - _allowed_video_fields() - _TASK_CONTROL_KEYS
    if unknown:
        raise SelectorError(
            f"task contains unknown selector fields: {', '.join(sorted(unknown))}"
        )
    entry = {key: value for key, value in task.items() if key not in _TASK_CONTROL_KEYS}
    material_ids = task.get("video_material_ids")
    if material_ids is not None:
        if not isinstance(material_ids, list) or not material_ids:
            raise SelectorError(
                "video_material_ids must be a non-empty array when provided"
            )
        entry = {
            **entry,
            "video_source": "local",
            "video_materials": _resolve_video_materials(material_ids, manifest=manifest),
        }
    bgm_id = task.get("bgm_id")
    if bgm_id is not None:
        if not isinstance(bgm_id, str) or not bgm_id.strip():
            raise SelectorError("bgm_id must be a non-empty string")
        _, bgm_path = _resolve_asset(
            manifest,
            collection="bgm",
            asset_id=bgm_id,
            root_key="bgm_root",
        )
        entry["bgm_type"] = "custom"
        entry["bgm_file"] = str(
            _materialize_bgm(bgm_path, asset_id=bgm_id, mpt_root=mpt_root)
        )
    try:
        validated = VideoParams.model_validate(entry)
    except Exception as exc:
        raise SelectorError(
            "task does not satisfy the current MPT VideoParams contract"
        ) from exc
    serialized = {
        key: _json_value(value)
        for key, value in vars(validated).items()
        if value is not None
    }
    override = dict(_mapping(task.get("post_process", {}), description="post_process"))
    return serialized, override


def _classify_task(
    task: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    classifier: LLMClassifier,
) -> dict[str, Any]:
    """
    Fill missing manifest selections by calling the configured LLM classifier.

    @param task Original selector task without mutating its caller-owned mapping.
    @param manifest Manifest containing category and asset allowlists.
    @param classifier Injected LLM classification adapter.
    @returns New task dictionary with validated model-selected IDs.
    @raises SelectorError If the classifier fails or conflicts with explicit IDs.
    """
    result = dict(task)
    videos = manifest.get("videos", [])
    bgm = manifest.get("bgm", [])
    if not isinstance(videos, list) or not isinstance(bgm, list):
        raise SelectorError("manifest videos and bgm must be arrays")
    categories = tuple(
        sorted(
            {
                item.get("category", "uncategorized")
                for item in videos + bgm
                if isinstance(item, Mapping)
                and isinstance(item.get("category", "uncategorized"), str)
            }
        )
    )
    video_ids = tuple(
        item["id"]
        for item in videos
        if isinstance(item, Mapping) and isinstance(item.get("id"), str)
    )
    bgm_ids = tuple(
        item["id"]
        for item in bgm
        if isinstance(item, Mapping) and isinstance(item.get("id"), str)
    )
    subject = result.get("video_subject") or result.get("video_script")
    try:
        selection = classifier.classify(
            subject=subject,
            categories=categories,
            video_ids=video_ids,
            bgm_ids=bgm_ids,
        )
    except ClassificationError as exc:
        raise SelectorError(f"LLM classification failed: {exc}") from exc
    if "video_material_ids" in result and result["video_material_ids"] != list(
        selection.video_ids
    ):
        raise SelectorError(
            "LLM classification conflicts with explicit video_material_ids"
        )
    if "bgm_id" in result and result["bgm_id"] != selection.bgm_id:
        raise SelectorError("LLM classification conflicts with explicit bgm_id")
    result["video_material_ids"] = list(selection.video_ids)
    if selection.bgm_id is not None:
        result["bgm_id"] = selection.bgm_id
    return result


def _default_post_process_spec(task_index: int) -> dict[str, Any]:
    """Create explicit post-processing defaults for one frozen task index."""
    return {
        "schema_version": 1,
        "task_index": task_index,
        "output": dict(_DEFAULT_OUTPUT),
        "watermark": dict(_DEFAULT_WATERMARK),
        "custom_texts": [],
    }


def _post_process_spec(
    task_index: int,
    override: Mapping[str, Any],
    profiles: Mapping[str, OutputProfile],
) -> dict[str, Any]:
    """
    Expand an optional profile ID into a complete task output specification.

    @param task_index Frozen one-based task index.
    @param override User or rule-generated post-processing override.
    @param profiles Built-in plus optional external output profiles.
    @returns Complete JSON-compatible task specification before runtime validation.
    @raises SelectorError If the output override or profile ID is invalid.
    """
    base_spec = _default_post_process_spec(task_index)
    output_override = override.get("output", {})
    if output_override is None:
        output_override = {}
    if not isinstance(output_override, Mapping):
        raise SelectorError("post_process.output must be an object")
    profile_id = output_override.get("profile_id", _DEFAULT_PROFILE_ID)
    if not isinstance(profile_id, str) or not profile_id.strip():
        raise SelectorError("post_process.output.profile_id must be a non-empty string")
    try:
        profile = resolve_profile(profile_id, profiles)
    except ProfileError as exc:
        raise SelectorError(str(exc)) from exc
    override_without_output = {
        key: value for key, value in override.items() if key != "output"
    }
    expanded = dict(profile_dict(profile))
    expanded.update(output_override)
    override_with_profile = {
        **override_without_output,
        "output": expanded,
    }
    return _merge(base_spec, override_with_profile)


def _write_job_artifacts(
    job_dir: Path,
    *,
    entries: Sequence[Mapping[str, Any]],
    overrides: Sequence[Mapping[str, Any]],
    request: Mapping[str, Any],
    profiles: Mapping[str, OutputProfile],
) -> list[str]:
    """
    Atomically write the request snapshot, batch JSONL, and task specifications.

    @param job_dir Job directory receiving frozen selector artifacts.
    @param entries Validated MPT batch entries in task order.
    @param overrides Per-task post-processing overrides.
    @param request Original selector request snapshot.
    @returns Generated post-processing spec paths in task order.
    """
    _write_text_atomic(
        job_dir / "input.json",
        json.dumps(request, ensure_ascii=False, indent=2) + "\n",
    )
    batch_content = "".join(
        json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n"
        for entry in entries
    )
    _write_text_atomic(job_dir / "batch.jsonl", batch_content)
    spec_paths: list[str] = []
    spec_dir = job_dir / "post_process_specs"
    for task_index, override in enumerate(overrides, start=1):
        spec = _post_process_spec(task_index, override, profiles)
        spec_path = spec_dir / f"{task_index}.json"
        _write_text_atomic(
            spec_path,
            json.dumps(spec, ensure_ascii=False, indent=2) + "\n",
        )
        spec_paths.append(str(spec_path))
    return spec_paths


def _reserve_job_assets(
    state_path: Path,
    *,
    job_id: str,
    selected_asset_ids: Sequence[tuple[int, tuple[str, ...]]],
) -> list[str]:
    """
    Reserve selected assets and roll back this call on a later conflict.

    @param state_path JSON state file path.
    @param job_id External job ID used in owner strings.
    @param selected_asset_ids Task-indexed asset IDs to reserve.
    @returns Flattened IDs reserved by this job.
    @raises SelectorError If reservation or rollback fails.
    """
    reserved: list[tuple[str, tuple[str, ...]]] = []
    timestamp = datetime.now(timezone.utc).isoformat()
    try:
        for task_index, asset_ids in selected_asset_ids:
            if not asset_ids:
                continue
            owner = f"{job_id}:{task_index}"
            reserve_assets(state_path, asset_ids, owner=owner, now=timestamp)
            reserved.append((owner, asset_ids))
    except AssetStateError as exc:
        for owner, asset_ids in reversed(reserved):
            try:
                release_assets(
                    state_path,
                    asset_ids,
                    owner=owner,
                    now=datetime.now(timezone.utc).isoformat(),
                )
            except AssetStateError as release_error:
                raise SelectorError(
                    f"asset reservation failed and cleanup failed: {release_error}"
                ) from release_error
        raise SelectorError(f"asset reservation failed: {exc}") from exc
    return [asset_id for _, asset_ids in reserved for asset_id in asset_ids]


def _build_job_entries(
    tasks: Sequence[Mapping[str, Any]],
    *,
    manifest: Mapping[str, Any],
    mpt_root: Path,
    llm_classify: bool,
    classifier: LLMClassifier,
) -> tuple[list[dict[str, Any]], list[Mapping[str, Any]], list[tuple[int, tuple[str, ...]]]]:
    """Build validated batch entries and task-indexed asset reservations."""
    entries: list[dict[str, Any]] = []
    overrides: list[Mapping[str, Any]] = []
    selected_asset_ids: list[tuple[int, tuple[str, ...]]] = []
    for task in tasks:
        task_mapping = _mapping(task, description="selector task")
        if (
            llm_classify
            and "video_material_ids" not in task_mapping
            and "video_materials" not in task_mapping
        ):
            task_mapping = _classify_task(
                task_mapping,
                manifest=manifest,
                classifier=classifier,
            )
        entry, override = _build_batch_entry(
            task_mapping,
            manifest=manifest,
            mpt_root=mpt_root,
        )
        entries.append(entry)
        overrides.append(override)
        selected_asset_ids.append(
            (len(entries), tuple(task_mapping.get("video_material_ids") or ()))
        )
    return entries, overrides, selected_asset_ids


def generate_job(
    request_path: Path,
    manifest_path: Path,
    output_root: Path,
    *,
    mpt_root: Path | None = None,
    state_path: Path | None = None,
    profiles_path: Path | None = None,
    llm_classify: bool = False,
    classifier: LLMClassifier | None = None,
) -> dict[str, Any]:
    """
    Freeze selector input into a job snapshot, MPT batch, and per-task specs.

    @param request_path JSON request containing `job_id` and ordered `tasks`.
    @param manifest_path JSON media manifest generated by `build_manifest.py`.
    @param output_root Root containing the `jobs/{job_id}` output directory.
    @returns Summary with job ID, task count, and generated file paths.
    @raises SelectorError If request, manifest, IDs, or MPT fields are invalid.
    """
    request = _mapping(
        _read_json(request_path, description="selector request"),
        description="selector request",
    )
    if set(request) - _REQUEST_KEYS:
        raise SelectorError("selector request contains unknown fields")
    job_id = request.get("job_id")
    if not isinstance(job_id, str) or not _JOB_ID.fullmatch(job_id):
        raise SelectorError("job_id must be 1..128 safe filename characters")
    tasks = request.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise SelectorError("selector request.tasks must be a non-empty array")
    manifest = _mapping(
        _read_json(manifest_path, description="media manifest"),
        description="media manifest",
    )
    if manifest.get("schema_version") != 1:
        raise SelectorError("media manifest schema_version must be 1")
    try:
        profiles = (
            BUILTIN_PROFILES if profiles_path is None else load_profiles(profiles_path)
        )
    except ProfileError as exc:
        raise SelectorError(str(exc)) from exc
    selected_mpt_root = (mpt_root or Path.cwd()).resolve()
    selected_classifier = classifier or LLMClassifier()
    entries, overrides, selected_asset_ids = _build_job_entries(
        tasks,
        manifest=manifest,
        mpt_root=selected_mpt_root,
        llm_classify=llm_classify,
        classifier=selected_classifier,
    )

    job_dir = output_root / "jobs" / job_id
    spec_paths = _write_job_artifacts(
        job_dir,
        entries=entries,
        overrides=overrides,
        request=request,
        profiles=profiles,
    )
    selected_state_path = state_path or output_root.parent / "state" / "assets.json"
    reserved_asset_ids = _reserve_job_assets(
        selected_state_path,
        job_id=job_id,
        selected_asset_ids=selected_asset_ids,
    )
    return {
        "job_id": job_id,
        "task_count": len(entries),
        "batch_path": str(job_dir / "batch.jsonl"),
        "spec_paths": spec_paths,
        "state_path": str(selected_state_path),
        "reserved_asset_ids": reserved_asset_ids,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse selector command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument("--mpt-root", type=Path, default=Path.cwd())
    parser.add_argument("--state-file", type=Path)
    parser.add_argument("--profiles", type=Path)
    parser.add_argument(
        "--llm-classify",
        action="store_true",
        help="ask the configured LLM to select manifest IDs for tasks without explicit IDs",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Freeze selector outputs and return a process exit status."""
    args = _parse_args(argv)
    try:
        summary = generate_job(
            args.request,
            args.manifest,
            args.output_root,
            mpt_root=args.mpt_root,
            state_path=args.state_file,
            profiles_path=args.profiles,
            llm_classify=args.llm_classify,
        )
    except SelectorError as exc:
        print(f"selector failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
