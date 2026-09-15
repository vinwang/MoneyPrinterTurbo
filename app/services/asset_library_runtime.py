"""Runtime operations for selected local-library assets."""

from __future__ import annotations

import os
import re
import shutil
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.services import asset_library as library
from app.utils import utils


AssetLibraryError = library.AssetLibraryError
_BGM_VALIDATION_TIMEOUT_SECONDS = 120
_STORYBOARD_SNAPSHOT_DIR = "storyboard_sources"
_SAFE_TASK_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _copy_verified_snapshot(source: Path, target: Path, expected_hash: str) -> None:
    """复制一个素材快照，并在复制前后验证内容哈希未变化。"""
    source_hash = library._sha256_file(source)
    if source_hash != expected_hash:
        raise AssetLibraryError(
            f"asset changed since indexing: {source.name}"
        )
    if target.exists():
        if library._sha256_file(target) != expected_hash:
            raise AssetLibraryError(f"storyboard snapshot has unexpected content: {target.name}")
        return

    temporary = None
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        shutil.copy2(source, temporary)
        if library._sha256_file(temporary) != expected_hash:
            raise AssetLibraryError(f"asset changed while snapshotting: {source.name}")
        os.replace(temporary, target)
        temporary = None
    except OSError as exc:
        raise AssetLibraryError(f"cannot snapshot storyboard asset: {source.name}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def materialize_storyboard_sources(
    task_id: str,
    plan: Sequence[Mapping[str, Any]],
    *,
    db_path: Path | None = None,
) -> tuple[dict[str, Any], ...]:
    """
    为冻结分镜计划建立任务级源文件快照。

    @param task_id 当前任务 ID，用于隔离快照目录。
    @param plan 已校验的分镜计划，包含 asset_id 和索引时的 sha256。
    @param db_path 可选素材库数据库路径。
    @returns 带有不可变 source_path/source_sha256 的分镜计划元组。
    @raises AssetLibraryError 素材缺失、哈希变化或快照写入失败。
    """
    if not isinstance(task_id, str) or not _SAFE_TASK_ID.fullmatch(task_id):
        raise AssetLibraryError("task ID is invalid")
    if not isinstance(plan, Sequence) or isinstance(plan, (str, bytes)) or not plan:
        raise AssetLibraryError("storyboard plan is required")
    snapshot_dir = Path(utils.task_dir(task_id)) / _STORYBOARD_SNAPSHOT_DIR
    snapshots: dict[str, Path] = {}
    materialized: list[dict[str, Any]] = []
    for item in plan:
        if not isinstance(item, Mapping):
            raise AssetLibraryError("storyboard plan entries must be objects")
        asset_id = str(item.get("asset_id", "")).strip()
        expected_hash = str(item.get("asset_sha256", "")).strip()
        if not _SAFE_TASK_ID.fullmatch(asset_id) or not expected_hash:
            raise AssetLibraryError("storyboard plan asset identity is incomplete")
        asset = library.get_asset(asset_id, db_path=db_path)
        if asset.kind != "video" or asset.analysis_status != "ready":
            raise AssetLibraryError(f"selected asset is not a ready video: {asset_id}")
        if asset.sha256 != expected_hash:
            raise AssetLibraryError(f"asset index hash changed: {asset_id}")
        source = library.resolve_asset_path(asset)
        target = snapshots.get(asset_id)
        if target is None:
            target = snapshot_dir / f"{asset_id}{source.suffix.lower()}"
            _copy_verified_snapshot(source, target, expected_hash)
            snapshots[asset_id] = target
        entry = dict(item)
        entry["source_path"] = str(target)
        entry["source_sha256"] = expected_hash
        entry["source_relative_path"] = asset.relative_path
        materialized.append(entry)
    return tuple(materialized)


def materialize_manifest_storyboard_sources(
    task_id: str,
    plan: Sequence[Mapping[str, Any]],
    *,
    managed_root: Path | None = None,
) -> tuple[dict[str, Any], ...]:
    """
    Snapshot storyboard sources copied by the selector from a manifest.

    @param task_id Current task ID used to isolate the snapshot directory.
    @param plan Manifest-backed plan containing source paths and SHA-256 hashes.
    @param managed_root Optional managed source directory for injected test/runtime roots.
    @returns Plan entries with task-owned `source_path` and verified hash fields.
    @raises AssetLibraryError If identity, path containment, hash, or snapshot validation fails.
    """
    if not isinstance(task_id, str) or not _SAFE_TASK_ID.fullmatch(task_id):
        raise AssetLibraryError("task ID is invalid")
    if not isinstance(plan, Sequence) or isinstance(plan, (str, bytes)) or not plan:
        raise AssetLibraryError("manifest storyboard plan is required")
    source_root = (
        Path(managed_root).resolve()
        if managed_root is not None
        else Path(utils.storage_dir("local_videos", create=True)).resolve()
    )
    snapshot_dir = Path(utils.task_dir(task_id)) / _STORYBOARD_SNAPSHOT_DIR
    materialized: list[dict[str, Any]] = []
    for index, item in enumerate(plan, start=1):
        if not isinstance(item, Mapping):
            raise AssetLibraryError(f"manifest storyboard entry {index} is invalid")
        if item.get("source_kind") != "manifest":
            raise AssetLibraryError(
                f"manifest storyboard entry {index} has an invalid source kind"
            )
        asset_id = str(item.get("asset_id", "")).strip()
        expected_hash = str(item.get("asset_sha256", "")).strip().lower()
        source_value = str(item.get("source_path", "")).strip()
        if (
            not _SAFE_TASK_ID.fullmatch(asset_id)
            or not re.fullmatch(r"[0-9a-f]{64}", expected_hash)
            or not source_value
        ):
            raise AssetLibraryError(
                f"manifest storyboard entry {index} has incomplete source identity"
            )
        source = Path(source_value).resolve()
        try:
            source.relative_to(source_root)
        except ValueError as exc:
            raise AssetLibraryError(
                f"manifest storyboard source escapes managed storage: {asset_id}"
            ) from exc
        if not source.is_file():
            raise AssetLibraryError(f"manifest storyboard source is missing: {asset_id}")
        target = snapshot_dir / f"manifest-{asset_id}{source.suffix.lower()}"
        _copy_verified_snapshot(source, target, expected_hash)
        entry = dict(item)
        entry["source_path"] = str(target)
        entry["source_sha256"] = expected_hash
        materialized.append(entry)
    return tuple(materialized)


def materialize_bgm(asset_id: str, *, db_path: Path | None = None) -> str:
    """
    将选中的外部 BGM 复制到 MPT 的受管 BGM 目录并返回文件名。

    @param asset_id 已匹配的 BGM 资产 ID。
    @param db_path 可选素材库数据库路径。
    @returns storage/bgm 中可由现有视频服务解析的文件名。
    @raises AssetLibraryError BGM 无效、缺失或复制失败。
    """
    asset = library.get_asset(asset_id, db_path=db_path)
    if asset.kind != "bgm" or asset.analysis_status != "ready":
        raise AssetLibraryError(f"selected asset is not a ready BGM: {asset_id}")
    source = library.resolve_asset_path(asset)
    target_dir = Path(library.bgm.uploaded_bgm_dir(create=True))
    target = target_dir / f"library-{asset.asset_id}{source.suffix.lower()}"
    if target.exists():
        if library._sha256_file(target) != asset.sha256:
            raise AssetLibraryError(
                f"managed BGM copy has unexpected content: {target.name}"
            )
        return target.name
    try:
        library.bgm.validate_audio_file(
            str(source), timeout_seconds=_BGM_VALIDATION_TIMEOUT_SECONDS
        )
    except (library.bgm.BgmUploadError, library.bgm.BgmServiceError) as exc:
        raise AssetLibraryError(
            f"selected BGM is not decodable: {asset.asset_id}"
        ) from exc
    temporary = None
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=target_dir,
            prefix=".library-bgm-",
            suffix=source.suffix.lower(),
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        shutil.copy2(source, temporary)
        os.replace(temporary, target)
        temporary = None
    except OSError as exc:
        raise AssetLibraryError(f"cannot prepare managed BGM: {target.name}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target.name


def record_usage(asset_ids: Sequence[str], *, db_path: Path | None = None) -> None:
    """
    记录一次已成功成片任务实际使用的素材。

    @param asset_ids 已使用的视频或 BGM 资产 ID。
    @param db_path 可选素材库数据库路径。
    @returns None after the usage counters are committed.
    @raises AssetLibraryError 数据库更新失败。
    """
    if (
        not isinstance(asset_ids, Sequence)
        or isinstance(asset_ids, (str, bytes))
        or any(
            not isinstance(asset_id, str) or not asset_id.strip()
            for asset_id in asset_ids
        )
    ):
        raise AssetLibraryError("asset IDs must be non-empty strings")
    unique_ids = tuple(dict.fromkeys(asset_ids))
    if not unique_ids:
        return
    now = datetime.now(timezone.utc).isoformat()
    connection = library._connect(db_path or library.default_db_path())
    try:
        library._ensure_schema(connection)
        placeholders = ", ".join("?" for _ in unique_ids)
        row = connection.execute(
            """
            SELECT COUNT(*) AS count FROM assets
            WHERE analysis_status = 'ready' AND asset_id IN ("""
            f"{placeholders})",
            list(unique_ids),
        ).fetchone()
        if row is None or int(row["count"]) != len(unique_ids):
            raise AssetLibraryError("cannot record usage for an unready local asset")
        connection.executemany(
            """
            UPDATE assets
            SET use_count = use_count + 1, last_used_at = ?
            WHERE asset_id = ? AND analysis_status = 'ready'
            """,
            [(now, asset_id) for asset_id in unique_ids],
        )
        connection.commit()
    except sqlite3.Error as exc:
        connection.rollback()
        raise AssetLibraryError("cannot record asset usage") from exc
    finally:
        connection.close()
