"""Shot-change detection for local video assets.

The scan needs usable source windows, not fixed time slices: a 60-second product
video normally contains an entrance shot, a table shot and a close-up, and
matching "掰开玉米" to second 0 of that file is wrong even though the file-level
description mentions it.

Detection uses ffmpeg's `select=gt(scene,...)` score, which is a frame-difference
heuristic — it finds hard cuts reliably and misses slow transitions. Detected cuts
are therefore treated as candidate boundaries and still normalized into the
existing 3–5 second target, so a bad detection degrades window placement rather
than producing unusable stubs. When detection cannot run at all, windows fall back
to even slicing so one broken file never fails the whole scan.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from loguru import logger

from app.utils import utils


class ShotDetectionError(RuntimeError):
    """Raised when scene detection cannot run for one asset."""


class WindowAnalysisError(ValueError):
    """Raised when a window batch cannot be analyzed even after retrying."""


@dataclass(frozen=True, slots=True)
class WindowAnalysis:
    """一个源窗口的画面描述与标签。"""

    description: str
    tags: tuple[str, ...]


_SHOT_DETECTION_VERSION = "mpt-local-shot-v1"
_MIN_SHOT_SECONDS = 3.0
_MAX_SHOT_SECONDS = 5.0
_SCENE_THRESHOLD = 0.3
_DETECTION_TIMEOUT_SECONDS = 120
_PTS_TIME_RE = re.compile(r"pts_time:(-?\d+(?:\.\d+)?)")
_FRAME_EPSILON_SECONDS = 0.01
_FRAME_WIDTH = 640
_FRAME_TIMEOUT_SECONDS = 45
_MAX_FRAME_BYTES = 2 * 1024 * 1024
# 单次请求最多带 6 帧：长视频的窗口数可达 14 个，一次全发会让请求体和
# 模型的注意力都被摊薄，逐窗口一次调用又把调用量放大到窗口数。
_MAX_FRAMES_PER_CALL = 6
# 一批返回不可解析时拆半重试，直到这个下限。模型偶发返回损坏 JSON 是常态，
# 若因一批失败就丢掉整个素材的逐窗口结果，真实素材库会出现大量假失败。
_MIN_FRAMES_PER_CALL = 1
_MAX_DESCRIPTION_CHARS = 500
_MAX_TAG_CHARS = 64
_WINDOW_FIELDS = frozenset({"index", "description", "tags", "mood"})
# 非 OpenAI 系 Provider 常把 JSON 包在 ```json 围栏里，即使提示词要求返回裸 JSON。
_CODE_FENCE_RE = re.compile(r"^```[a-zA-Z0-9]*\s*|\s*```$")
VisionFn = Callable[..., str]


def _strip_code_fence(text: str) -> str:
    """剥掉响应外层的 markdown 代码围栏。"""
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        stripped = _CODE_FENCE_RE.sub("", stripped)
    return stripped.strip()


def detection_version() -> str:
    """
    Return the identifier recorded with segments produced by this detector.

    @returns Stable shot-detection version identifier.
    """
    return f"{_SHOT_DETECTION_VERSION}:scene{_SCENE_THRESHOLD}"


def _run_scene_detection(path: Path, duration: float) -> str:
    """调用 ffmpeg 输出场景切换的 metadata 文本。"""
    command = [
        utils.get_ffmpeg_binary(),
        "-nostdin",
        "-v",
        "error",
        "-i",
        str(path),
        "-vf",
        f"select='gt(scene,{_SCENE_THRESHOLD})',metadata=print:file=-",
        "-an",
        "-f",
        "null",
        "-",
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            timeout=_DETECTION_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ShotDetectionError(f"{type(exc).__name__}: {exc}") from exc
    if completed.returncode != 0:
        raise ShotDetectionError(f"ffmpeg exited with {completed.returncode}")
    return completed.stdout.decode("utf-8", errors="replace")


def _parse_scene_boundaries(output: str, duration: float) -> tuple[float, ...]:
    """从 metadata 文本提取有序去重的切点时间，越界值丢弃。"""
    boundaries: set[float] = set()
    for match in _PTS_TIME_RE.finditer(output):
        try:
            value = float(match.group(1))
        except ValueError:
            continue
        if not math.isfinite(value) or value <= 0 or value >= duration:
            continue
        boundaries.add(value)
    return tuple(sorted(boundaries))


def _even_windows(start: float, end: float) -> tuple[tuple[float, float], ...]:
    """把一段过长的镜头按 5 秒上限均分，避免留下不可用的短尾。"""
    span = end - start
    if span <= _MAX_SHOT_SECONDS:
        return ((start, end),)
    count = max(1, math.ceil(span / _MAX_SHOT_SECONDS))
    while count > 1 and span / count < _MIN_SHOT_SECONDS:
        count -= 1
    step = span / count
    return tuple(
        (start + index * step, start + (index + 1) * step) for index in range(count)
    )


def build_shot_windows(
    duration: float,
    boundaries: Sequence[float],
) -> tuple[tuple[float, float], ...]:
    """
    Turn candidate cut points into contiguous, usable source windows.

    @param duration Asset duration in seconds.
    @param boundaries Candidate cut times inside the duration.
    @returns Contiguous windows covering the whole duration, each at least
        `_MIN_SHOT_SECONDS` long unless the asset itself is shorter.
    """
    if not math.isfinite(duration) or duration <= 0:
        return ()
    if duration <= _MIN_SHOT_SECONDS:
        return ((0.0, duration),)
    usable = sorted(
        {
            float(value)
            for value in boundaries
            if math.isfinite(float(value)) and 0 < float(value) < duration
        }
    )
    # 先按切点切成镜头，再把过短的镜头并进相邻镜头：短于 3 秒的片段裁不出
    # 可用画面，留着只会让匹配拿到一堆废候选。
    cuts = [0.0, *usable, float(duration)]
    merged: list[tuple[float, float]] = []
    for start, end in zip(cuts, cuts[1:]):
        if end - start < _MIN_SHOT_SECONDS and merged:
            merged[-1] = (merged[-1][0], end)
            continue
        merged.append((start, end))
    while len(merged) > 1 and merged[0][1] - merged[0][0] < _MIN_SHOT_SECONDS:
        first, second = merged[0], merged[1]
        merged[:2] = [(first[0], second[1])]
    return tuple(
        window for start, end in merged for window in _even_windows(start, end)
    )


def detect_shot_windows(path: Path, duration: float) -> tuple[tuple[float, float], ...]:
    """
    Detect shot changes for one asset and normalize them into source windows.

    @param path Media path already validated against a library root.
    @param duration Asset duration in seconds.
    @returns Contiguous source windows; even slices when detection cannot run.
    """
    if not math.isfinite(duration) or duration <= 0:
        return ()
    if duration <= _MIN_SHOT_SECONDS:
        return ((0.0, duration),)
    try:
        output = _run_scene_detection(path, duration)
    except ShotDetectionError as exc:
        # 单个文件检测失败不应让整库扫描失败；退回均分并记录原因。
        logger.warning(f"shot detection unavailable: path={path}, error={exc}")
        return build_shot_windows(duration, ())
    return build_shot_windows(duration, _parse_scene_boundaries(output, duration))


def window_frame_positions(
    windows: Sequence[tuple[float, float]],
) -> tuple[float, ...]:
    """
    Pick one representative frame position per window.

    @param windows Source windows produced by `build_shot_windows`.
    @returns Sampling position strictly inside each window.
    """
    positions: list[float] = []
    for start, end in windows:
        midpoint = start + (end - start) / 2
        # 极短窗口的中点可能因浮点取整落到边界上，夹回窗口内部。
        positions.append(
            min(max(midpoint, start + _FRAME_EPSILON_SECONDS), end - _FRAME_EPSILON_SECONDS)
            if end - start > 2 * _FRAME_EPSILON_SECONDS
            else midpoint
        )
    return tuple(positions)


def extract_frames_at(
    path: Path,
    positions: Sequence[float],
) -> tuple[bytes, ...]:
    """
    Extract one compressed frame per requested position.

    @param path Media path already validated against a library root.
    @param positions Sampling positions in seconds.
    @returns JPEG bytes in position order; positions that fail are skipped.
    """
    ffmpeg = utils.get_ffmpeg_binary()
    frames: list[bytes] = []
    for position in positions:
        command = [
            ffmpeg,
            "-nostdin",
            "-v",
            "error",
            "-ss",
            f"{max(0.0, float(position)):.3f}",
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
                f"failed to extract window frame: path={path}, "
                f"position={position:.3f}, error={type(exc).__name__}"
            )
            continue
        if completed.returncode == 0 and 0 < len(completed.stdout) <= _MAX_FRAME_BYTES:
            frames.append(completed.stdout)
    return tuple(frames)


def _window_prompt(category: str, count: int) -> str:
    """构造逐窗口分析提示词，不包含原始视频路径。"""
    return json.dumps(
        {
            "instruction": (
                "Each supplied frame represents one consecutive segment of the same "
                "video, in order. Describe each frame separately. Return only JSON "
                f"with a windows array of exactly {count} objects, each having "
                "index (1-based), description, tags, and mood. Describe only what "
                "the frame shows; do not invent product claims. Use concise Chinese "
                "when the asset category is Chinese."
            ),
            "category": category,
            "window_count": count,
        },
        ensure_ascii=False,
    )


def parse_window_analysis(response: str, *, window_count: int) -> tuple[WindowAnalysis, ...]:
    """
    Parse and strictly validate a per-window visual analysis response.

    @param response Raw model text.
    @param window_count Number of windows the request covered.
    @returns One analysis per window, in window order.
    @raises ValueError If the response is unusable or does not align 1:1.
    """
    if not isinstance(response, str) or not response.strip():
        raise ValueError("window analysis returned empty text")
    if response.startswith("Error: "):
        raise ValueError(response.removeprefix("Error: ").strip())
    try:
        value = json.loads(_strip_code_fence(response))
    except json.JSONDecodeError as exc:
        raise ValueError("window analysis must return one JSON object") from exc
    if not isinstance(value, dict) or set(value) != {"windows"}:
        raise ValueError("window analysis must contain exactly a windows array")
    entries = value["windows"]
    if not isinstance(entries, list) or len(entries) != window_count:
        raise ValueError("window analysis must return one entry per window")
    analyses: list[WindowAnalysis] = []
    for position, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict) or set(entry) != _WINDOW_FIELDS:
            raise ValueError(
                "each window must contain exactly index, description, tags, and mood"
            )
        if entry["index"] != position:
            raise ValueError("window index must count up from one without gaps")
        description = entry["description"]
        mood = entry["mood"]
        tags = entry["tags"]
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
            raise ValueError("window analysis returned invalid fields")
        normalized_tags: list[str] = []
        for tag in [*tags, mood]:
            if not isinstance(tag, str) or not tag.strip() or len(tag) > _MAX_TAG_CHARS:
                raise ValueError("window analysis returned invalid tags")
            if tag.strip() not in normalized_tags:
                normalized_tags.append(tag.strip())
        analyses.append(
            WindowAnalysis(
                description=description.strip(),
                tags=tuple(normalized_tags),
            )
        )
    return tuple(analyses)


def analyze_windows(
    path: Path,
    windows: Sequence[tuple[float, float]],
    *,
    category: str,
    vision_fn: VisionFn,
    app_config: Mapping[str, Any] | None,
) -> tuple[tuple[WindowAnalysis, ...], int]:
    """
    Analyze every source window and report the real model call volume.

    @param path Media path already validated against a library root.
    @param windows Source windows produced by `build_shot_windows`.
    @param category Asset directory category passed to the model as context.
    @param vision_fn Vision entry point accepting prompt, frames, and config.
    @param app_config Optional runtime configuration snapshot.
    @returns Per-window analyses in window order, and the number of model calls.
    @raises WindowAnalysisError If a frame is missing or a batch never parses.
    """
    if not windows:
        return (), 0
    chain = tuple(windows)
    analyses, calls = _analyze_chain(
        path,
        chain,
        category=category,
        vision_fn=vision_fn,
        app_config=app_config,
    )
    return analyses, calls


def _analyze_chain(
    path: Path,
    windows: tuple[tuple[float, float], ...],
    *,
    category: str,
    vision_fn: VisionFn,
    app_config: Mapping[str, Any] | None,
) -> tuple[tuple[WindowAnalysis, ...], int]:
    """按上限分批分析；一批解析失败就拆半重试，避免丢整个素材。"""
    analyses: list[WindowAnalysis] = []
    calls = 0
    for offset in range(0, len(windows), _MAX_FRAMES_PER_CALL):
        batch = windows[offset : offset + _MAX_FRAMES_PER_CALL]
        parsed, batch_calls = _analyze_batch(
            path,
            batch,
            category=category,
            vision_fn=vision_fn,
            app_config=app_config,
        )
        analyses.extend(parsed)
        calls += batch_calls
    return tuple(analyses), calls


def _analyze_batch(
    path: Path,
    batch: tuple[tuple[float, float], ...],
    *,
    category: str,
    vision_fn: VisionFn,
    app_config: Mapping[str, Any] | None,
) -> tuple[tuple[WindowAnalysis, ...], int]:
    """分析一批窗口；解析失败且仍可拆分时拆半重试。"""
    frames = extract_frames_at(path, window_frame_positions(batch))
    if len(frames) != len(batch):
        # 抽帧失败不是模型问题，重试同一个位置也不会变好。
        raise WindowAnalysisError(
            "window analysis needs exactly one frame per window"
        )
    response = vision_fn(
        _window_prompt(category, len(batch)),
        frames,
        app_config=app_config,
    )
    try:
        return parse_window_analysis(response, window_count=len(batch)), 1
    except ValueError as exc:
        if len(batch) <= _MIN_FRAMES_PER_CALL:
            raise WindowAnalysisError(str(exc)) from exc
        midpoint = len(batch) // 2
        logger.warning(
            f"window analysis batch failed, splitting: path={path}, "
            f"size={len(batch)}, error={exc}"
        )
        head, head_calls = _analyze_batch(
            path,
            batch[:midpoint],
            category=category,
            vision_fn=vision_fn,
            app_config=app_config,
        )
        tail, tail_calls = _analyze_batch(
            path,
            batch[midpoint:],
            category=category,
            vision_fn=vision_fn,
            app_config=app_config,
        )
        return (*head, *tail), 1 + head_calls + tail_calls


def aggregate_window_analysis(
    analyses: Sequence[WindowAnalysis],
) -> tuple[str, tuple[str, ...]]:
    """
    Derive the asset-level description and tags from per-window analyses.

    @param analyses Per-window analyses in window order.
    @returns Asset description within the stored limit, and de-duplicated tags.
    @raises ValueError If there is no analysis to aggregate.
    """
    if not analyses:
        raise ValueError("cannot aggregate an empty window analysis")
    description = " ".join(item.description for item in analyses).strip()
    tags: list[str] = []
    for item in analyses:
        for tag in item.tags:
            if tag not in tags:
                tags.append(tag)
    return description[:_MAX_DESCRIPTION_CHARS], tuple(tags)
