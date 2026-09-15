"""Persistent local video and BGM indexing for semantic WebUI selection."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import shutil
import sqlite3
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from loguru import logger
from PIL import Image, UnidentifiedImageError

from app.config import config
from app.services import asset_library_segments as segment_index
from app.services import asset_library_shots as shot_index
from app.services.asset_library_analysis import asset_id as _asset_id, default_vision_analysis as _default_vision_analysis, existing_location as _existing_location, parse_visual_analysis as _parse_visual_analysis, vision_analysis_model as _vision_analysis_model
from app.services import bgm, material_upload
from app.utils import file_security, utils


class AssetLibraryError(RuntimeError):
    """表示素材库配置、索引、视觉分析或选择失败。"""


@dataclass(frozen=True, slots=True)
class LibraryAsset:
    """索引中的一个本地视频或背景音乐资产。"""

    asset_id: str
    kind: str
    root_path: str
    relative_path: str
    category: str
    duration: float
    width: int | None
    height: int | None
    size_bytes: int
    sha256: str
    description: str
    tags: tuple[str, ...]
    analysis_status: str
    analysis_error: str
    use_count: int
    last_used_at: str | None

    @property
    def name(self) -> str:
        """返回用于 WebUI 展示的相对文件名。"""
        return Path(self.relative_path).name


LibrarySegment = segment_index.LibrarySegment


@dataclass(frozen=True, slots=True)
class ScanSummary:
    """一次素材库扫描的可展示统计。"""

    added: int
    updated: int
    unchanged: int
    analyzed: int
    failed: int
    missing: int
    errors: tuple[str, ...]


_VIDEO_EXTENSIONS = frozenset(material_upload.SUPPORTED_VIDEO_EXTENSIONS)
_BGM_EXTENSIONS = frozenset(bgm.SUPPORTED_BGM_EXTENSIONS)
_SCHEMA_VERSION = 2
_LEGACY_SCHEMA_VERSION = 1
_FRAME_FRACTIONS = (0.1, 0.5, 0.9)
_FRAME_WIDTH = 640
_FRAME_TIMEOUT_SECONDS = 45
_MAX_FRAME_BYTES = 2 * 1024 * 1024
_COPY_CHUNK_BYTES = 1024 * 1024
_FRAME_END_MARGIN_SECONDS = 0.05
ProbeFn = Callable[[Path], Mapping[str, Any]]
VisionFn = Callable[[Path, tuple[bytes, ...]], str]


def default_db_path() -> Path:
    """
    返回素材库 SQLite 文件路径。

    @returns 位于项目 storage 根目录下的素材库数据库路径。
    """
    return Path(utils.storage_dir()) / "asset_library.sqlite3"


def configured_video_root() -> Path | None:
    """
    读取并校验 WebUI 配置的视频素材库根目录。

    @returns 已解析的视频素材库目录；配置为空时返回 None。
    @raises AssetLibraryError 配置路径不是可读目录。
    """
    return _configured_root(
        config.app.get("local_video_library_directory", ""),
        "local video library",
    )


def configured_bgm_root() -> Path | None:
    """
    读取并校验 WebUI 配置的背景音乐库根目录。

    @returns 已解析的背景音乐库目录；配置为空时返回 None。
    @raises AssetLibraryError 配置路径不是可读目录。
    """
    return _configured_root(
        config.app.get("local_bgm_library_directory", ""),
        "local BGM library",
    )


def _configured_root(raw_value: Any, description: str) -> Path | None:
    value = str(raw_value or "").strip()
    if not value:
        return None
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise AssetLibraryError(f"{description} directory does not exist: {root}")
    return root


def _connect(db_path: Path) -> sqlite3.Connection:
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(db_path), timeout=30)
    except (OSError, sqlite3.Error) as exc:
        raise AssetLibraryError(f"cannot open asset library database: {db_path}") from exc
    connection.row_factory = sqlite3.Row
    return connection


def _ensure_schema(connection: sqlite3.Connection) -> None:
    try:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS assets (
                asset_id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                root_path TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                category TEXT NOT NULL,
                duration REAL NOT NULL,
                width INTEGER, height INTEGER,
                size_bytes INTEGER NOT NULL,
                mtime_ns INTEGER NOT NULL,
                sha256 TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                tags_json TEXT NOT NULL DEFAULT '[]',
                analysis_status TEXT NOT NULL,
                analysis_error TEXT NOT NULL DEFAULT '', analysis_model TEXT NOT NULL DEFAULT '',
                manual_description TEXT NOT NULL DEFAULT '',
                manual_tags_json TEXT NOT NULL DEFAULT '[]',
                segments_json TEXT NOT NULL DEFAULT '[]',
                use_count INTEGER NOT NULL DEFAULT 0,
                last_used_at TEXT
            );
            CREATE UNIQUE INDEX IF NOT EXISTS assets_location_idx
                ON assets(kind, root_path, relative_path);
            CREATE INDEX IF NOT EXISTS assets_match_idx
                ON assets(kind, analysis_status);
            CREATE TABLE IF NOT EXISTS segments (
                segment_id TEXT PRIMARY KEY,
                asset_id TEXT NOT NULL,
                source_start_seconds REAL NOT NULL,
                source_end_seconds REAL NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                tags_json TEXT NOT NULL DEFAULT '[]',
                analysis_status TEXT NOT NULL,
                analysis_error TEXT NOT NULL DEFAULT '',
                analysis_model TEXT NOT NULL DEFAULT '',
                UNIQUE(asset_id, source_start_seconds, source_end_seconds)
            );
            CREATE INDEX IF NOT EXISTS segments_asset_idx
                ON segments(asset_id, analysis_status, source_start_seconds);
            """
        )
        row = connection.execute(
            "SELECT value FROM metadata WHERE key = ?", ("schema_version",)
        ).fetchone()
        if row is None:
            connection.execute(
                "INSERT INTO metadata(key, value) VALUES (?, ?)",
                ("schema_version", str(_SCHEMA_VERSION)),
            )
        elif row["value"] == str(_LEGACY_SCHEMA_VERSION):
            connection.execute(
                "UPDATE metadata SET value = ? WHERE key = ?",
                (str(_SCHEMA_VERSION), "schema_version"),
            )
        elif row["value"] != str(_SCHEMA_VERSION):
            raise AssetLibraryError("unsupported asset library schema version")
        columns = {
            column["name"]
            for column in connection.execute("PRAGMA table_info(assets)").fetchall()
        }
        if "analysis_model" not in columns:
            connection.execute(
                "ALTER TABLE assets ADD COLUMN analysis_model TEXT NOT NULL DEFAULT ''"
            )
        if "manual_description" not in columns:
            connection.execute(
                "ALTER TABLE assets ADD COLUMN manual_description TEXT NOT NULL DEFAULT ''"
            )
        if "manual_tags_json" not in columns:
            connection.execute(
                "ALTER TABLE assets ADD COLUMN manual_tags_json TEXT NOT NULL DEFAULT '[]'"
            )
        if "segments_json" not in columns:
            connection.execute(
                "ALTER TABLE assets ADD COLUMN segments_json TEXT NOT NULL DEFAULT '[]'"
            )
        segment_index.backfill_segments(connection)
        connection.commit()
    except segment_index.SegmentIndexError as exc:
        raise AssetLibraryError(str(exc)) from exc
    except sqlite3.Error as exc:
        raise AssetLibraryError("cannot initialize asset library schema") from exc


def _iter_media(root: Path, extensions: frozenset[str]):
    if not root.is_dir():
        raise AssetLibraryError(f"asset root is not a directory: {root}")

    def raise_walk_error(error: OSError) -> None:
        raise AssetLibraryError(f"cannot read asset directory: {error.filename}") from error

    for directory, dirnames, filenames in os.walk(
        root,
        followlinks=False,
        onerror=raise_walk_error,
    ):
        dirnames[:] = sorted(
            name
            for name in dirnames
            if not Path(directory, name).is_symlink()
        )
        for filename in sorted(filenames, key=str.lower):
            candidate = Path(directory, filename)
            if candidate.suffix.lower() not in extensions or not candidate.is_file():
                continue
            resolved = candidate.resolve()
            try:
                resolved.relative_to(root)
            except ValueError as exc:
                raise AssetLibraryError(
                    f"asset symlink escapes configured root: {candidate}"
                ) from exc
            yield resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            while chunk := source.read(_COPY_CHUNK_BYTES):
                digest.update(chunk)
    except OSError as exc:
        raise AssetLibraryError(f"cannot hash asset: {path}") from exc
    return digest.hexdigest()


def _probe_media(path: Path) -> Mapping[str, Any]:
    ffprobe = shutil.which("ffprobe") or "ffprobe"
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=codec_type,width,height",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=_FRAME_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AssetLibraryError(f"ffprobe failed for asset: {path}") from exc
    if completed.returncode != 0:
        raise AssetLibraryError(f"ffprobe failed for asset: {path}")
    try:
        payload = json.loads(completed.stdout)
        format_data = payload["format"]
        duration = float(format_data.get("duration") or 0)
        streams = payload.get("streams", [])
        video_stream = next(
            (
                stream
                for stream in streams
                if isinstance(stream, Mapping) and stream.get("codec_type") == "video"
            ),
            {},
        )
        width = video_stream.get("width")
        height = video_stream.get("height")
    except (KeyError, TypeError, ValueError, StopIteration, json.JSONDecodeError) as exc:
        raise AssetLibraryError(f"ffprobe returned invalid metadata: {path}") from exc
    if not math.isfinite(duration) or duration < 0:
        raise AssetLibraryError(f"asset duration is invalid: {path}")
    if width is not None and height is not None:
        width = int(width)
        height = int(height)
        if width <= 0 or height <= 0:
            raise AssetLibraryError(f"asset dimensions are invalid: {path}")
    return {"duration": duration, "width": width, "height": height}


def _extract_preview_frames(
    path: Path,
    duration: float,
    *,
    frame_count: int = len(_FRAME_FRACTIONS),
) -> tuple[bytes, ...]:
    """
    从本地视频抽取压缩代表帧，不上传原始视频。

    @param path 已通过素材库根目录校验的媒体路径。
    @param duration 视频时长，用于计算抽帧位置。
    @param frame_count 需要抽取的代表帧数量。
    @returns JPEG 图片字节元组。
    @raises AssetLibraryError 所有代表帧都无法抽取。
    """
    if path.suffix.lower() in material_upload.SUPPORTED_IMAGE_EXTENSIONS:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(path) as image:
                    image.thumbnail((_FRAME_WIDTH, _FRAME_WIDTH))
                    preview = image.convert("RGB")
                    output = io.BytesIO()
                    preview.save(output, format="JPEG", quality=80)
                    payload = output.getvalue()
        except (
            Image.DecompressionBombError,
            Image.DecompressionBombWarning,
            UnidentifiedImageError,
            OSError,
            ValueError,
        ) as exc:
            raise AssetLibraryError(f"cannot read preview image: {path}") from exc
        if not payload or len(payload) > _MAX_FRAME_BYTES:
            raise AssetLibraryError(f"preview image exceeds analysis size: {path}")
        return (payload,)

    fractions = _FRAME_FRACTIONS[: max(1, min(frame_count, len(_FRAME_FRACTIONS)))]
    frames: list[bytes] = []
    ffmpeg = utils.get_ffmpeg_binary()
    for fraction in fractions:
        position = max(
            0.0,
            min(max(duration - _FRAME_END_MARGIN_SECONDS, 0.0), duration * fraction),
        )
        command = [
            ffmpeg,
            "-nostdin",
            "-v",
            "error",
            "-ss",
            f"{position:.3f}",
            "-i",
            str(path),
            "-frames:v",
            "1",
            "-vf",
            f"scale={_FRAME_WIDTH}:-2",
            "-f",
            "image2",
            "-vcodec",
            "mjpeg",
            "pipe:1",
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                timeout=_FRAME_TIMEOUT_SECONDS,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            logger.warning(
                f"failed to extract asset preview frame: path={path}, "
                f"error={type(exc).__name__}"
            )
            continue
        if completed.returncode == 0 and 0 < len(completed.stdout) <= _MAX_FRAME_BYTES:
            frames.append(completed.stdout)
    if not frames:
        raise AssetLibraryError(f"cannot extract preview frames: {path}")
    return tuple(frames)


def _reused_segments(existing: sqlite3.Row) -> tuple[Mapping[str, Any], ...]:
    """内容与分析都未变化时沿用已存的逐窗口分析，避免重复外发。"""
    if "segments_json" not in existing.keys():
        return ()
    try:
        entries = json.loads(existing["segments_json"] or "[]")
    except (TypeError, json.JSONDecodeError):
        return ()
    return tuple(entries) if isinstance(entries, list) else ()


def _analyze_video_windows(
    path: Path,
    *,
    duration: float,
    category: str,
    vision_fn: VisionFn | None,
    app_config: Mapping[str, Any] | None,
) -> tuple[str, tuple[str, ...], tuple[Mapping[str, Any], ...]]:
    """
    检测镜头窗口并逐窗口分析，返回资产级摘要与逐窗口结果。

    单帧素材（图片）没有镜头可分，沿用整条分析路径。

    @param path 已通过素材库根目录校验的媒体路径。
    @param duration 媒体时长。
    @param category 素材相对目录分类。
    @param vision_fn 可注入的视觉分析器；None 表示使用默认模型。
    @param app_config 传给视觉模型的配置快照。
    @returns 资产级描述、标签，以及逐窗口分析负载。
    @raises Exception 抽帧或分析失败时由调用方记为 failed。
    """
    if path.suffix.lower() in material_upload.SUPPORTED_IMAGE_EXTENSIONS:
        frames = _extract_preview_frames(path, duration)
        if vision_fn is None:
            description, tags = _default_vision_analysis(category, frames, app_config)
        else:
            description, tags = _parse_visual_analysis(vision_fn(path, frames))
        return description, tags, ()
    windows = shot_index.detect_shot_windows(path, duration)
    if not windows:
        raise AssetLibraryError(f"cannot determine shot windows: {path}")
    window_vision = (
        _default_window_vision()
        if vision_fn is None
        else _adapted_window_vision(path, vision_fn)
    )
    analyses, _calls = shot_index.analyze_windows(
        path,
        windows,
        category=category,
        vision_fn=window_vision,
        app_config=app_config,
    )
    description, tags = shot_index.aggregate_window_analysis(analyses)
    payload = tuple(
        {
            "source_start_seconds": start,
            "source_end_seconds": end,
            "description": analysis.description,
            "tags": list(analysis.tags),
        }
        for (start, end), analysis in zip(windows, analyses)
    )
    return description, tags, payload


def _default_window_vision():
    """默认逐窗口视觉调用：直接把提示词和多帧发给当前视觉模型。"""
    def call(prompt: str, frames, app_config=None) -> str:
        from app.services import llm

        return llm.generate_vision_response(
            prompt=prompt,
            image_bytes=frames,
            app_config=app_config,
        )

    return call


def _adapted_window_vision(path: Path, vision_fn: VisionFn):
    """把注入的 `(path, frames) -> str` 分析器适配到逐窗口调用签名。"""
    def call(prompt: str, frames, app_config=None) -> str:
        return vision_fn(path, tuple(frames))

    return call


def _prepare_asset_record(
    path: Path,
    *,
    kind: str,
    root: Path,
    existing: sqlite3.Row | None,
    analyze_visual: bool,
    force_reanalyze: bool,
    retry_failed: bool,
    analysis_model: str,
    probe_fn: ProbeFn,
    vision_fn: VisionFn | None,
    app_config: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], bool]:
    relative_path = path.relative_to(root).as_posix()
    stat = path.stat()
    content_hash = _sha256_file(path)
    metadata = probe_fn(path)
    duration = float(metadata.get("duration", 0))
    width = metadata.get("width")
    height = metadata.get("height")
    if not math.isfinite(duration) or duration < 0:
        raise AssetLibraryError(f"asset duration is invalid: {path}")
    if kind == "bgm" and duration <= 0:
        raise AssetLibraryError(f"BGM duration must be positive: {path}")
    asset_id = _asset_id(kind, relative_path, content_hash)
    same_content = existing is not None and existing["sha256"] == content_hash
    description = ""
    tags: tuple[str, ...] = ()
    status = "ready" if kind == "bgm" else "pending"
    error = ""
    stored_analysis_model = ""
    reuse_analysis = (
        same_content
        and existing is not None
        and not force_reanalyze
        and not (
            kind == "video"
            and analyze_visual
            and (
                existing["analysis_status"] == "pending"
                or (retry_failed and existing["analysis_status"] == "failed")
                or (
                    existing["analysis_status"] == "ready"
                    and existing["analysis_model"] != analysis_model
                )
            )
        )
    )
    segments_payload: tuple[Mapping[str, Any], ...] = ()
    if reuse_analysis:
        description = existing["description"]
        tags = tuple(json.loads(existing["tags_json"]))
        status = existing["analysis_status"]
        error = existing["analysis_error"]
        stored_analysis_model = existing["analysis_model"]
        segments_payload = _reused_segments(existing)
    elif kind == "video" and analyze_visual:
        try:
            description, tags, segments_payload = _analyze_video_windows(
                path,
                duration=duration,
                category=Path(relative_path).parent.as_posix(),
                vision_fn=vision_fn,
                app_config=app_config,
            )
            status = "ready"
        except Exception as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
            segments_payload = ()
        stored_analysis_model = analysis_model
    elif kind == "video":
        status = "pending"
    record = {
        "asset_id": asset_id,
        "kind": kind,
        "root_path": str(root),
        "relative_path": relative_path,
        "category": Path(relative_path).parent.as_posix()
        if Path(relative_path).parent.as_posix() != "."
        else "uncategorized",
        "duration": duration,
        "width": width,
        "height": height,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": content_hash,
        "description": description,
        "tags_json": json.dumps(tags, ensure_ascii=False),
        "analysis_status": status,
        "analysis_error": error,
        "analysis_model": stored_analysis_model,
        "manual_description": (
            existing["manual_description"] if same_content and existing else ""
        ),
        "manual_tags_json": (
            existing["manual_tags_json"]
            if same_content and existing and "manual_tags_json" in existing.keys()
            else "[]"
        ),
        "segments_json": json.dumps(list(segments_payload), ensure_ascii=False),
        "use_count": existing["use_count"] if same_content and existing else 0,
        "last_used_at": existing["last_used_at"] if same_content and existing else None,
    }
    analyzed = kind == "video" and status == "ready" and (
        not same_content
        or force_reanalyze
        or existing is None
        or existing["analysis_status"] != "ready"
        or existing["analysis_model"] != analysis_model
    )
    return record, analyzed


def _upsert_asset(connection: sqlite3.Connection, record: Mapping[str, Any]) -> None:
    previous_rows = connection.execute(
        """
        SELECT asset_id FROM assets
        WHERE kind = ? AND root_path = ? AND relative_path = ?
        """,
        (record["kind"], record["root_path"], record["relative_path"]),
    ).fetchall()
    if previous_rows:
        connection.executemany(
            "DELETE FROM segments WHERE asset_id = ?",
            [(row["asset_id"],) for row in previous_rows],
        )
    connection.execute(
        """
        DELETE FROM assets
        WHERE kind = ? AND root_path = ? AND relative_path = ? AND asset_id <> ?
        """,
        (
            record["kind"],
            record["root_path"],
            record["relative_path"],
            record["asset_id"],
        ),
    )
    connection.execute(
        """
        INSERT INTO assets(
            asset_id, kind, root_path, relative_path, category, duration,
            width, height, size_bytes, mtime_ns, sha256, description, tags_json,
            analysis_status, analysis_error, analysis_model, use_count, last_used_at,
            manual_description, manual_tags_json, segments_json
        ) VALUES(
            :asset_id, :kind, :root_path, :relative_path, :category, :duration,
            :width, :height, :size_bytes, :mtime_ns, :sha256, :description,
            :tags_json, :analysis_status, :analysis_error, :analysis_model,
            :use_count, :last_used_at, :manual_description, :manual_tags_json,
            :segments_json
        )
        ON CONFLICT(asset_id) DO UPDATE SET
            root_path=excluded.root_path,
            relative_path=excluded.relative_path,
            category=excluded.category,
            duration=excluded.duration,
            width=excluded.width,
            height=excluded.height,
            size_bytes=excluded.size_bytes,
            mtime_ns=excluded.mtime_ns,
            sha256=excluded.sha256,
            description=excluded.description,
            tags_json=excluded.tags_json,
            analysis_status=excluded.analysis_status,
            analysis_error=excluded.analysis_error,
            analysis_model=excluded.analysis_model,
            manual_description=excluded.manual_description,
            manual_tags_json=excluded.manual_tags_json,
            segments_json=excluded.segments_json,
            use_count=excluded.use_count,
            last_used_at=excluded.last_used_at
        """,
        record,
    )
    if record["kind"] == "video":
        segment_index.sync_segments(connection, record)


def _mark_missing(
    connection: sqlite3.Connection,
    *,
    kind: str,
    root: Path,
    seen: set[str],
) -> int:
    rows = connection.execute(
        "SELECT asset_id, relative_path FROM assets WHERE kind = ? AND root_path = ?",
        (kind, str(root)),
    ).fetchall()
    missing_ids = [row["asset_id"] for row in rows if row["relative_path"] not in seen]
    if missing_ids:
        connection.executemany(
            "UPDATE assets SET analysis_status = 'missing', analysis_error = ? WHERE asset_id = ?",
            [("file is missing from configured root", asset_id) for asset_id in missing_ids],
        )
        connection.executemany(
            """
            UPDATE segments
            SET analysis_status = 'missing', analysis_error = ?
            WHERE asset_id = ?
            """,
            [("file is missing from configured root", asset_id) for asset_id in missing_ids],
        )
    return len(missing_ids)


def _scan_kind(
    connection: sqlite3.Connection,
    root: Path | None,
    *,
    kind: str,
    extensions: frozenset[str],
    analyze_visual: bool,
    force_reanalyze: bool,
    retry_failed: bool,
    analysis_model: str,
    probe_fn: ProbeFn,
    vision_fn: VisionFn | None,
    app_config: Mapping[str, Any] | None,
) -> tuple[int, int, int, int, int, int, tuple[str, ...]]:
    if root is None:
        return 0, 0, 0, 0, 0, 0, ()
    added = updated = unchanged = analyzed = failed = 0
    errors: list[str] = []
    seen: set[str] = set()
    for path in _iter_media(root, extensions):
        relative_path = path.relative_to(root).as_posix()
        seen.add(relative_path)
        existing = _existing_location(connection, kind, str(root), relative_path)
        try:
            stat = path.stat()
            needs_analysis = (
                kind == "video"
                and analyze_visual
                and existing is not None
                and (
                    existing["analysis_status"] == "pending"
                    or (retry_failed and existing["analysis_status"] == "failed")
                    or (
                        existing["analysis_status"] == "ready"
                        and existing["analysis_model"] != analysis_model
                    )
                )
            )
            if (
                existing is not None
                and existing["size_bytes"] == stat.st_size
                and existing["mtime_ns"] == stat.st_mtime_ns
                and not force_reanalyze
                and not needs_analysis
            ):
                unchanged += 1
                continue
            record, did_analyze = _prepare_asset_record(
                path,
                kind=kind,
                root=root,
                existing=existing,
                analyze_visual=analyze_visual,
                force_reanalyze=force_reanalyze,
                retry_failed=retry_failed,
                analysis_model=analysis_model,
                probe_fn=probe_fn,
                vision_fn=vision_fn,
                app_config=app_config,
            )
            _upsert_asset(connection, record)
            if existing is None:
                added += 1
            else:
                updated += 1
            analyzed += int(did_analyze)
            if record["analysis_status"] == "failed":
                failed += 1
                errors.append(f"{relative_path}: {record['analysis_error']}")
        except Exception as exc:
            failed += 1
            detail = f"{relative_path}: {type(exc).__name__}: {exc}"
            errors.append(detail)
            logger.warning(f"asset library scan failed: root={root}, error={detail}")
            if existing is not None:
                connection.execute(
                    "UPDATE assets SET analysis_status = 'failed', analysis_error = ? WHERE asset_id = ?",
                    (str(exc), existing["asset_id"]),
                )
    missing = _mark_missing(connection, kind=kind, root=root, seen=seen)
    return added, updated, unchanged, analyzed, failed, missing, tuple(errors)


def scan_library(
    video_root: Path | str,
    bgm_root: Path | str | None = None,
    *,
    db_path: Path | None = None,
    analyze_visual: bool = True,
    force_reanalyze: bool = False,
    retry_failed: bool = False,
    probe_fn: ProbeFn | None = None,
    vision_fn: VisionFn | None = None,
    app_config: Mapping[str, Any] | None = None,
) -> ScanSummary:
    """
    扫描视频和 BGM 根目录并增量更新 SQLite 索引。

    @param video_root 视频素材库根目录。
    @param bgm_root 可选的背景音乐库根目录。
    @param db_path 索引数据库路径，默认使用项目 storage。
    @param analyze_visual 是否为新增/变化视频抽帧并调用视觉模型。
    @param force_reanalyze 是否重新分析已有视频。
    @param retry_failed 是否重试上次失败的视觉分析。
    @param probe_fn 可注入的媒体元数据探测器，供测试或替换工具链。
    @param vision_fn 可注入的视觉分析器，供测试或替换模型。
    @param app_config 传给视觉模型的配置快照。
    @returns 扫描统计和逐文件错误。
    @raises AssetLibraryError 根目录或数据库不可用。
    """
    resolved_video_root = _required_root(video_root, "local video library")
    resolved_bgm_root = _optional_root(bgm_root, "local BGM library")
    runtime_config = app_config or config.app
    analysis_model = _vision_analysis_model(runtime_config)
    connection = _connect(db_path or default_db_path())
    try:
        _ensure_schema(connection)
        effective_probe = probe_fn or _probe_media
        video_stats = _scan_kind(
            connection,
            resolved_video_root,
            kind="video",
            extensions=_VIDEO_EXTENSIONS,
            analyze_visual=analyze_visual,
            force_reanalyze=force_reanalyze,
            retry_failed=retry_failed,
            analysis_model=analysis_model,
            probe_fn=effective_probe,
            vision_fn=vision_fn,
            app_config=app_config,
        )
        bgm_stats = _scan_kind(
            connection,
            resolved_bgm_root,
            kind="bgm",
            extensions=_BGM_EXTENSIONS,
            analyze_visual=False,
            force_reanalyze=False,
            retry_failed=False,
            analysis_model="",
            probe_fn=effective_probe,
            vision_fn=None,
            app_config=None,
        )
        connection.commit()
    except sqlite3.Error as exc:
        connection.rollback()
        raise AssetLibraryError("cannot update asset library database") from exc
    finally:
        connection.close()
    stats = [video_stats, bgm_stats]
    return ScanSummary(
        added=sum(item[0] for item in stats),
        updated=sum(item[1] for item in stats),
        unchanged=sum(item[2] for item in stats),
        analyzed=sum(item[3] for item in stats),
        failed=sum(item[4] for item in stats),
        missing=sum(item[5] for item in stats),
        errors=tuple(error for item in stats for error in item[6]),
    )


def scan_configured_library(
    *,
    db_path: Path | None = None,
    app_config: Mapping[str, Any] | None = None,
    retry_failed: bool = False,
) -> ScanSummary:
    """
    使用 WebUI 中保存的素材库目录执行一次增量扫描。

    @param db_path 可选的测试数据库路径。
    @param app_config 传给视觉模型的配置快照。
    @param retry_failed 是否重试上次失败的视觉分析。
    @returns 扫描统计。
    @raises AssetLibraryError 未配置视频目录或目录不可用。
    """
    video_root = configured_video_root()
    if video_root is None:
        raise AssetLibraryError("local video library directory is not configured")
    return scan_library(
        video_root,
        configured_bgm_root(),
        db_path=db_path,
        app_config=app_config,
        retry_failed=retry_failed,
    )


def _required_root(raw_root: Path | str, description: str) -> Path:
    root = _optional_root(raw_root, description)
    if root is None:
        raise AssetLibraryError(f"{description} directory is required")
    return root


def _optional_root(raw_root: Path | str | None, description: str) -> Path | None:
    if raw_root is None or not str(raw_root).strip():
        return None
    root = Path(raw_root).expanduser().resolve()
    if not root.is_dir():
        raise AssetLibraryError(f"{description} directory does not exist: {root}")
    return root


def _row_to_asset(row: sqlite3.Row) -> LibraryAsset:
    try:
        auto_tags = json.loads(row["tags_json"])
        manual_tags = json.loads(row["manual_tags_json"])
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise AssetLibraryError(f"asset index contains invalid tags: {row['asset_id']}") from exc
    if not isinstance(auto_tags, list) or any(
        not isinstance(tag, str) or not tag.strip() for tag in auto_tags
    ):
        raise AssetLibraryError(f"asset index contains invalid tags: {row['asset_id']}")
    if not isinstance(manual_tags, list) or any(
        not isinstance(tag, str) or not tag.strip() for tag in manual_tags
    ):
        raise AssetLibraryError(f"asset index contains invalid tags: {row['asset_id']}")
    tags = tuple(dict.fromkeys([*auto_tags, *manual_tags]))
    manual_description = str(row["manual_description"] or "").strip()
    return LibraryAsset(
        asset_id=row["asset_id"],
        kind=row["kind"],
        root_path=row["root_path"],
        relative_path=row["relative_path"],
        category=row["category"],
        duration=float(row["duration"]),
        width=row["width"],
        height=row["height"],
        size_bytes=int(row["size_bytes"]),
        sha256=row["sha256"],
        description=manual_description or row["description"],
        tags=tags,
        analysis_status=row["analysis_status"],
        analysis_error=row["analysis_error"],
        use_count=int(row["use_count"]),
        last_used_at=row["last_used_at"],
    )


def list_assets(
    *,
    kind: str | None = None,
    analysis_status: str | None = None,
    db_path: Path | None = None,
) -> tuple[LibraryAsset, ...]:
    """
    读取素材库中符合状态的资产。

    @param kind `video`、`bgm` 或 None 表示全部类型。
    @param analysis_status 可选索引状态过滤器。
    @param db_path 可选数据库路径。
    @returns 按相对路径稳定排序的不可变资产元组。
    """
    connection = _connect(db_path or default_db_path())
    try:
        _ensure_schema(connection)
        clauses: list[str] = []
        values: list[str] = []
        if kind:
            clauses.append("kind = ?")
            values.append(kind)
        if analysis_status:
            clauses.append("analysis_status = ?")
            values.append(analysis_status)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = connection.execute(
            f"SELECT * FROM assets{where} ORDER BY relative_path COLLATE NOCASE",
            values,
        ).fetchall()
        return tuple(_row_to_asset(row) for row in rows)
    except sqlite3.Error as exc:
        raise AssetLibraryError("cannot read asset library database") from exc
    finally:
        connection.close()


def get_asset(asset_id: str, *, db_path: Path | None = None) -> LibraryAsset:
    """
    根据白名单 ID 读取一个资产。

    @param asset_id 素材库中记录的稳定资产 ID。
    @param db_path 可选数据库路径。
    @returns 对应资产。
    @raises AssetLibraryError ID 不存在或数据库不可用。
    """
    if not isinstance(asset_id, str) or not asset_id.strip():
        raise AssetLibraryError("asset ID is required")
    connection = _connect(db_path or default_db_path())
    try:
        _ensure_schema(connection)
        row = connection.execute(
            "SELECT * FROM assets WHERE asset_id = ?", (asset_id,)
        ).fetchone()
        if row is None:
            raise AssetLibraryError(f"asset is not indexed: {asset_id}")
        return _row_to_asset(row)
    finally:
        connection.close()


def list_segments(
    *,
    asset_ids: Sequence[str] | None = None,
    analysis_status: str | None = None,
    db_path: Path | None = None,
) -> tuple[LibrarySegment, ...]:
    """
    读取视频片段索引，按源文件和源起始时间稳定排序。

    @param asset_ids 可选的已索引视频资产 ID 白名单。
    @param analysis_status 可选片段分析状态过滤器。
    @param db_path 可选数据库路径。
    @returns 不可变的片段索引元组。
    @raises AssetLibraryError 片段数据损坏或数据库不可用。
    """
    connection = _connect(db_path or default_db_path())
    try:
        _ensure_schema(connection)
        return segment_index.list_segments(
            connection,
            asset_ids=asset_ids,
            analysis_status=analysis_status,
        )
    except segment_index.SegmentIndexError as exc:
        raise AssetLibraryError(str(exc)) from exc
    except sqlite3.Error as exc:
        raise AssetLibraryError("cannot read asset segment index") from exc
    finally:
        connection.close()


def update_asset_annotations(
    asset_id: str,
    *,
    description: str | None = None,
    tags: Sequence[str] | None = None,
    db_path: Path | None = None,
) -> None:
    """
    保存用户对素材的人工描述和标签，不覆盖自动分析字段。

    @param asset_id 已索引的视频或 BGM 资产 ID。
    @param description 可选人工描述；None 表示保留原值。
    @param tags 可选人工标签；None 表示保留原值。
    @param db_path 可选数据库路径。
    @returns None after the annotation is committed.
    @raises AssetLibraryError 资产不存在、字段非法或数据库不可用。
    """
    connection = _connect(db_path or default_db_path())
    try:
        _ensure_schema(connection)
        segment_index.update_annotations(
            connection,
            asset_id,
            description=description,
            tags=tags,
        )
        connection.commit()
    except segment_index.SegmentIndexError as exc:
        connection.rollback()
        raise AssetLibraryError(str(exc)) from exc
    except sqlite3.Error as exc:
        connection.rollback()
        raise AssetLibraryError("cannot update asset annotations") from exc
    finally:
        connection.close()


def resolve_asset_path(asset: LibraryAsset) -> Path:
    """
    在资产记录的根目录内解析真实文件路径。

    @param asset 已从 SQLite 白名单读取的资产。
    @returns 位于其配置根目录内的现有文件路径。
    @raises AssetLibraryError 文件缺失或路径逃逸。
    """
    try:
        return Path(
            file_security.resolve_path_within_directory(
                asset.root_path,
                asset.relative_path,
            )
        )
    except ValueError as exc:
        raise AssetLibraryError(
            f"asset path is unavailable or outside its root: {asset.asset_id}"
        ) from exc
