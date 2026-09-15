"""SQLite helpers for local-video segments and human annotations."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


class SegmentIndexError(ValueError):
    """Raised when the segment index or manual annotation is invalid."""


@dataclass(frozen=True, slots=True)
class LibrarySegment:
    """索引中的一个视频源片段及其可追溯的分析信息。"""

    segment_id: str
    asset_id: str
    source_start_seconds: float
    source_end_seconds: float
    description: str
    tags: tuple[str, ...]
    analysis_status: str
    analysis_error: str
    analysis_model: str


_SEGMENT_INDEX_VERSION = "mpt-local-segment-v2"
_MIN_SEGMENT_SECONDS = 3.0
_MAX_SEGMENT_SECONDS = 5.0
_MAX_DESCRIPTION_CHARS = 500
_MAX_TAG_CHARS = 64
_RANGE_EPSILON_SECONDS = 0.05


def _segment_windows(duration: float) -> tuple[tuple[float, float], ...]:
    """按 3–5 秒目标生成不重叠的可用源片段窗口。"""
    if not math.isfinite(duration) or duration <= 0:
        return ()
    if duration <= _MAX_SEGMENT_SECONDS:
        return ((0.0, duration),)
    segment_count = max(1, math.ceil(duration / _MAX_SEGMENT_SECONDS))
    while segment_count > 1 and duration / segment_count < _MIN_SEGMENT_SECONDS:
        segment_count -= 1
    segment_duration = duration / segment_count
    return tuple(
        (index * segment_duration, (index + 1) * segment_duration)
        for index in range(segment_count)
    )


def _segment_id(asset_id: str, start: float, end: float) -> str:
    identity = f"{asset_id}:{start:.6f}:{end:.6f}".encode("utf-8")
    return f"segment-{hashlib.sha256(identity).hexdigest()[:20]}"


def _value(record: Mapping[str, Any], key: str, default: Any = None) -> Any:
    try:
        return record[key]
    except (KeyError, IndexError):
        return default


def _analyzed_windows(
    record: Mapping[str, Any],
    asset_id: str,
    duration: float,
) -> tuple[dict[str, Any], ...]:
    """读取并校验逐窗口分析结果；没有时返回空元组以退回均分。"""
    raw = _value(record, "segments_json", "[]")
    try:
        entries = json.loads(str(raw or "[]"))
    except json.JSONDecodeError as exc:
        raise SegmentIndexError(f"invalid segment analysis for asset: {asset_id}") from exc
    if not isinstance(entries, list):
        raise SegmentIndexError(f"invalid segment analysis for asset: {asset_id}")
    if not entries:
        return ()
    normalized: list[dict[str, Any]] = []
    cursor = 0.0
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise SegmentIndexError(f"invalid segment analysis for asset: {asset_id}")
        try:
            start = float(entry["source_start_seconds"])
            end = float(entry["source_end_seconds"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SegmentIndexError(
                f"invalid segment range for asset: {asset_id}"
            ) from exc
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            raise SegmentIndexError(f"invalid segment range for asset: {asset_id}")
        # 窗口必须首尾相接：重叠或留空会让「旁白时间段→源片段」的裁切
        # 落到未描述的画面上。
        if abs(start - cursor) > _RANGE_EPSILON_SECONDS:
            raise SegmentIndexError(
                f"segment windows must be contiguous for asset: {asset_id}"
            )
        cursor = end
        segment_description = str(entry.get("description", "")).strip()
        if not segment_description:
            raise SegmentIndexError(
                f"segment description is required for asset: {asset_id}"
            )
        segment_tags = entry.get("tags", [])
        if isinstance(segment_tags, (str, bytes)) or not isinstance(
            segment_tags, Sequence
        ):
            raise SegmentIndexError(f"invalid segment tags for asset: {asset_id}")
        if any(
            not isinstance(tag, str) or not tag.strip() or len(tag) > _MAX_TAG_CHARS
            for tag in segment_tags
        ):
            raise SegmentIndexError(f"invalid segment tags for asset: {asset_id}")
        normalized.append(
            {
                "start": start,
                "end": end,
                "description": segment_description[:_MAX_DESCRIPTION_CHARS],
                "tags": tuple(dict.fromkeys(tag.strip() for tag in segment_tags)),
            }
        )
    if abs(cursor - duration) > _RANGE_EPSILON_SECONDS:
        raise SegmentIndexError(
            f"segment windows must cover the whole asset: {asset_id}"
        )
    return tuple(normalized)


def sync_segments(connection: sqlite3.Connection, record: Mapping[str, Any]) -> None:
    """
    为一个已分析视频重建确定性的源片段索引。

    优先使用逐窗口视觉分析的结果；没有时退回按时长均分并复用整条视频的
    描述，保持旧库和分析失败素材的行为不变。

    @param connection 已打开的 SQLite 连接。
    @param record 已规范化的资产索引记录。
    @returns None after the segment rows are replaced.
    @raises SegmentIndexError 逐窗口分析不连续、未覆盖全片或字段非法。
    """
    asset_id = str(_value(record, "asset_id"))
    connection.execute("DELETE FROM segments WHERE asset_id = ?", (asset_id,))
    description = str(_value(record, "description", ""))
    try:
        tags = json.loads(str(_value(record, "tags_json", "[]")))
    except json.JSONDecodeError as exc:
        raise SegmentIndexError(f"invalid tags for asset: {asset_id}") from exc
    if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
        raise SegmentIndexError(f"invalid tags for asset: {asset_id}")
    duration = float(_value(record, "duration", 0))
    analyzed = _analyzed_windows(record, asset_id, duration)
    analysis_status = (
        "ready"
        if _value(record, "analysis_status") == "ready"
        else _value(record, "analysis_status")
    )
    analysis_error = str(_value(record, "analysis_error", ""))
    analysis_model = f"{_SEGMENT_INDEX_VERSION}:{_value(record, 'analysis_model', '')}"
    if analyzed:
        windows_with_text = tuple(
            (item["start"], item["end"], item["description"], item["tags"])
            for item in analyzed
        )
    else:
        windows_with_text = tuple(
            (start, end, description, tuple(tags))
            for start, end in _segment_windows(duration)
        )
    rows = [
        (
            _segment_id(asset_id, start, end),
            asset_id,
            start,
            end,
            segment_description,
            json.dumps(list(segment_tags), ensure_ascii=False),
            analysis_status,
            analysis_error,
            analysis_model,
        )
        for start, end, segment_description, segment_tags in windows_with_text
    ]
    connection.executemany(
        """
        INSERT INTO segments(
            segment_id, asset_id, source_start_seconds, source_end_seconds,
            description, tags_json, analysis_status, analysis_error, analysis_model
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def backfill_segments(connection: sqlite3.Connection) -> None:
    """
    为旧版数据库中的已有视频资产补建片段窗口。

    @param connection 已打开的 SQLite 连接。
    @returns None after missing segment rows are backfilled.
    """
    rows = connection.execute(
        """
        SELECT assets.* FROM assets
        LEFT JOIN segments ON segments.asset_id = assets.asset_id
        WHERE assets.kind = 'video' AND segments.asset_id IS NULL
        """
    ).fetchall()
    for row in rows:
        if row["analysis_status"] in {"ready", "failed", "missing"}:
            sync_segments(connection, row)


def row_to_segment(row: sqlite3.Row) -> LibrarySegment:
    try:
        raw_tags = json.loads(row["tags_json"])
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SegmentIndexError(
            f"asset index contains invalid segment tags: {row['segment_id']}"
        ) from exc
    if not isinstance(raw_tags, list) or any(
        not isinstance(tag, str) or not tag.strip() for tag in raw_tags
    ):
        raise SegmentIndexError(
            f"asset index contains invalid segment tags: {row['segment_id']}"
        )
    try:
        source_start = float(row["source_start_seconds"])
        source_end = float(row["source_end_seconds"])
    except (TypeError, ValueError) as exc:
        raise SegmentIndexError(
            f"asset index contains invalid segment range: {row['segment_id']}"
        ) from exc
    if (
        not math.isfinite(source_start)
        or not math.isfinite(source_end)
        or source_start < 0
        or source_end <= source_start
    ):
        raise SegmentIndexError(
            f"asset index contains invalid segment range: {row['segment_id']}"
        )
    return LibrarySegment(
        segment_id=row["segment_id"],
        asset_id=row["asset_id"],
        source_start_seconds=source_start,
        source_end_seconds=source_end,
        description=row["description"],
        tags=tuple(raw_tags),
        analysis_status=row["analysis_status"],
        analysis_error=row["analysis_error"],
        analysis_model=row["analysis_model"],
    )


def list_segments(
    connection: sqlite3.Connection,
    *,
    asset_ids: Sequence[str] | None,
    analysis_status: str | None,
) -> tuple[LibrarySegment, ...]:
    """
    从已打开的连接读取稳定排序的片段索引。

    @param connection 已打开的 SQLite 连接。
    @param asset_ids 可选的资产 ID 白名单。
    @param analysis_status 可选的片段状态过滤器。
    @returns 按资产和源起始时间排序的片段元组。
    """
    clauses: list[str] = []
    values: list[str] = []
    if asset_ids is not None:
        normalized_ids = tuple(dict.fromkeys(asset_ids))
        if not normalized_ids:
            return ()
        placeholders = ", ".join("?" for _ in normalized_ids)
        clauses.append(f"asset_id IN ({placeholders})")
        values.extend(normalized_ids)
    if analysis_status:
        clauses.append("analysis_status = ?")
        values.append(analysis_status)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    rows = connection.execute(
        "SELECT * FROM segments"
        f"{where} ORDER BY asset_id, source_start_seconds",
        values,
    ).fetchall()
    return tuple(row_to_segment(row) for row in rows)


def update_annotations(
    connection: sqlite3.Connection,
    asset_id: str,
    *,
    description: str | None,
    tags: Sequence[str] | None,
) -> None:
    """
    在已打开的连接中更新人工描述和标签。

    @param connection 已打开的 SQLite 连接。
    @param asset_id 要更新的资产 ID。
    @param description 可选人工描述。
    @param tags 可选人工标签。
    @returns None; transaction commit is handled by the caller.
    """
    if not isinstance(asset_id, str) or not asset_id.strip():
        raise SegmentIndexError("asset ID is required")
    if description is not None and (
        not isinstance(description, str)
        or len(description.strip()) > _MAX_DESCRIPTION_CHARS
    ):
        raise SegmentIndexError("manual description is invalid")
    normalized_tags = None
    if tags is not None:
        if isinstance(tags, (str, bytes)):
            raise SegmentIndexError("manual tags must be a sequence")
        if any(not isinstance(tag, str) for tag in tags):
            raise SegmentIndexError("manual tags are invalid")
        normalized_tags = tuple(dict.fromkeys(tag.strip() for tag in tags))
        if any(not tag or len(tag) > _MAX_TAG_CHARS for tag in normalized_tags):
            raise SegmentIndexError("manual tags are invalid")
    row = connection.execute(
        "SELECT asset_id FROM assets WHERE asset_id = ?", (asset_id,)
    ).fetchone()
    if row is None:
        raise SegmentIndexError(f"asset is not indexed: {asset_id}")
    fields: list[str] = []
    values: list[Any] = []
    if description is not None:
        fields.append("manual_description = ?")
        values.append(description.strip())
    if normalized_tags is not None:
        fields.append("manual_tags_json = ?")
        values.append(json.dumps(normalized_tags, ensure_ascii=False))
    if fields:
        values.append(asset_id)
        connection.execute(
            f"UPDATE assets SET {', '.join(fields)} WHERE asset_id = ?",
            values,
        )
