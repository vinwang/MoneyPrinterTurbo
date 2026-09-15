"""Pure analysis and identity helpers for the local asset library."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any, Mapping

from app.services import llm


_MAX_DESCRIPTION_CHARS = 500
_MAX_TAG_CHARS = 64
# v2 起改为逐镜头窗口分析：描述按源时间段生成，不再整条视频共用一份。
# 版本变化会让已索引素材在下次扫描时重新分析，调用量按窗口数而非文件数计。
_VISION_ANALYSIS_VERSION = "mpt-local-vision-v2"


def vision_analysis_model(runtime_config: Mapping[str, Any]) -> str:
    """
    生成用于判断视觉标签是否需要迁移的版本标识。

    @param runtime_config 当前 LLM 配置快照。
    @returns Stable analysis-version identifier.
    """
    provider = str(runtime_config.get("llm_provider", "") or "").strip().lower()
    model = str(runtime_config.get(f"{provider}_model_name", "") or "").strip()
    return f"{_VISION_ANALYSIS_VERSION}:{provider}:{model}"


def asset_id(kind: str, relative_path: str, content_hash: str) -> str:
    """
    根据媒体类型、相对路径和内容哈希生成稳定资产 ID。

    @param kind 媒体类型。
    @param relative_path 素材根目录内的相对路径。
    @param content_hash 文件内容哈希。
    @returns Stable asset identifier.
    """
    identity = f"{kind}:{relative_path}:{content_hash}".encode("utf-8")
    return f"{kind}-{hashlib.sha256(identity).hexdigest()[:16]}"


def parse_visual_analysis(response: str) -> tuple[str, tuple[str, ...]]:
    """
    解析并严格校验视觉模型返回的描述、标签和情绪 JSON。

    @param response 视觉模型原始文本。
    @returns Normalized description and tags.
    """
    if not isinstance(response, str) or not response.strip():
        raise ValueError("visual analysis returned empty text")
    if response.startswith("Error: "):
        raise ValueError(response.removeprefix("Error: ").strip())
    try:
        value = json.loads(response)
    except json.JSONDecodeError as exc:
        raise ValueError("visual analysis must return one JSON object") from exc
    if not isinstance(value, dict) or set(value) != {"description", "tags", "mood"}:
        raise ValueError(
            "visual analysis must contain exactly description, tags, and mood"
        )
    description = value["description"]
    mood = value["mood"]
    tags = value["tags"]
    if (
        not isinstance(description, str)
        or not description.strip()
        or len(description) > _MAX_DESCRIPTION_CHARS
        or not isinstance(mood, str)
        or not mood.strip()
        or len(mood) > _MAX_TAG_CHARS
        or not isinstance(tags, list)
        or not tags
    ):
        raise ValueError("visual analysis returned invalid fields")
    normalized_tags: list[str] = []
    for tag in [*tags, mood]:
        if not isinstance(tag, str) or not tag.strip() or len(tag) > _MAX_TAG_CHARS:
            raise ValueError("visual analysis returned invalid tags")
        if tag.strip() not in normalized_tags:
            normalized_tags.append(tag.strip())
    return description.strip(), tuple(normalized_tags)


def default_vision_analysis(
    category: str,
    frames: tuple[bytes, ...],
    app_config: Mapping[str, Any] | None,
) -> tuple[str, tuple[str, ...]]:
    """
    调用当前视觉模型并返回规范化的本地素材描述。

    @param category 素材相对目录分类。
    @param frames 已抽取的代表帧 JPEG 字节。
    @param app_config 可选运行时配置快照。
    @returns Normalized description and tags.
    """
    prompt = json.dumps(
        {
            "instruction": (
                "Analyze the supplied representative frames from one video. "
                "Return only JSON with exactly description, tags, and mood. "
                "Use concise Chinese descriptions when the asset category is Chinese."
            ),
            "category": category,
        },
        ensure_ascii=False,
    )
    response = llm.generate_vision_response(
        prompt=prompt,
        image_bytes=frames,
        app_config=app_config,
    )
    return parse_visual_analysis(response)


def existing_location(
    connection: sqlite3.Connection,
    kind: str,
    root_path: str,
    relative_path: str,
) -> sqlite3.Row | None:
    """
    读取同一素材根目录和相对路径的已有索引记录。

    @param connection 已打开的 SQLite 连接。
    @param kind 媒体类型。
    @param root_path 素材根目录。
    @param relative_path 根目录内的相对路径。
    @returns Existing row or None.
    """
    return connection.execute(
        "SELECT * FROM assets WHERE kind = ? AND root_path = ? AND relative_path = ?",
        (kind, root_path, relative_path),
    ).fetchone()
