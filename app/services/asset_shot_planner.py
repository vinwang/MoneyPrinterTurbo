"""Let a text model split the script into shots with explicit visual needs.

Punctuation splitting cannot tell that "下班回家，终于可以放松一下" needs a home
scene rather than an office one, and it has no way to say "a shopping-cart
screenshot is wrong here". This module asks a text model for that judgement and
then verifies the answer against the original script.

The model never returns narration text: it returns character ranges into the
script, and shot text is sliced locally from those ranges. A model that tries to
rewrite or invent narration therefore cannot affect the output — the worst it can
do is propose ranges that fail validation, which falls back to punctuation
splitting rather than failing generation.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from loguru import logger


class ShotPlanError(ValueError):
    """Raised when a shot plan is missing, malformed, or does not fit the script."""


@dataclass(frozen=True, slots=True)
class PlannedShot:
    """模型给出的一个分镜：原文范围、画面需求与约束。"""

    index: int
    text: str
    script_start: int
    script_end: int
    visual_query: str
    must_match: tuple[str, ...]
    must_not_appear: tuple[str, ...]
    expression: str


_SHOT_FIELDS = frozenset(
    {
        "index",
        "script_start",
        "script_end",
        "visual_query",
        "must_match",
        "must_not_appear",
        "expression",
    }
)
# direct 表示画面直接展示这句话说的东西；ambience 表示这句话抽象或情绪化，
# 画面只需要氛围相符。区分两者是为了让「忙碌了一整天」不去硬找字面画面。
_EXPRESSIONS = frozenset({"direct", "ambience"})
_MAX_QUERY_CHARS = 200
_MAX_CONSTRAINT_CHARS = 64
_MAX_CONSTRAINTS = 8
# 非 OpenAI 系 Provider 常把 JSON 包在 ```json 围栏里，即使提示词要求返回裸 JSON。
_CODE_FENCE_RE = re.compile(r"^```[a-zA-Z0-9]*\s*|\s*```$")
LLMFn = Callable[..., str]


def _strip_code_fence(text: str) -> str:
    """剥掉响应外层的 markdown 代码围栏。"""
    stripped = (text or "").strip()
    if stripped.startswith("```"):
        stripped = _CODE_FENCE_RE.sub("", stripped)
    return stripped.strip()


def _constraint_list(value: Any, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ShotPlanError(f"{field} must be an array of strings")
    items = tuple(str(item).strip() for item in value)
    if any(not item or len(item) > _MAX_CONSTRAINT_CHARS for item in items):
        raise ShotPlanError(f"{field} contains an empty or oversized entry")
    if len(items) > _MAX_CONSTRAINTS:
        raise ShotPlanError(f"{field} has too many entries")
    return tuple(dict.fromkeys(items))


def parse_shot_plan(response: str, script: str) -> tuple[PlannedShot, ...]:
    """
    Parse and validate a shot plan against the script it claims to cover.

    @param response Raw model text.
    @param script The exact script the plan must cover.
    @returns Planned shots in script order.
    @raises ShotPlanError If the response is unusable or does not cover the script.
    """
    if not isinstance(response, str) or not response.strip():
        raise ShotPlanError("shot plan returned empty text")
    if response.startswith("Error: "):
        raise ShotPlanError(response.removeprefix("Error: ").strip())
    try:
        value = json.loads(_strip_code_fence(response))
    except json.JSONDecodeError as exc:
        raise ShotPlanError("shot plan must return one JSON object") from exc
    if not isinstance(value, dict) or set(value) != {"shots"}:
        raise ShotPlanError("shot plan must contain exactly a shots array")
    entries = value["shots"]
    if not isinstance(entries, list) or not entries:
        raise ShotPlanError("shot plan must contain at least one shot")
    normalized = script
    shots: list[PlannedShot] = []
    cursor = 0
    for position, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict) or set(entry) != _SHOT_FIELDS:
            raise ShotPlanError(
                "each shot must contain exactly index, script_start, script_end, "
                "visual_query, must_match, must_not_appear, and expression"
            )
        if entry["index"] != position:
            raise ShotPlanError("shot index must count up from one without gaps")
        try:
            start = int(entry["script_start"])
            end = int(entry["script_end"])
        except (TypeError, ValueError) as exc:
            raise ShotPlanError("shot script range must be integers") from exc
        if start < 0 or end > len(normalized) or end <= start:
            raise ShotPlanError(f"shot {position} has an out-of-range script range")
        if start != cursor:
            raise ShotPlanError("shot script ranges must be contiguous")
        cursor = end
        visual_query = str(entry["visual_query"]).strip()
        if not visual_query or len(visual_query) > _MAX_QUERY_CHARS:
            raise ShotPlanError(f"shot {position} has an invalid visual_query")
        expression = str(entry["expression"]).strip()
        if expression not in _EXPRESSIONS:
            raise ShotPlanError(
                f"shot {position} expression must be one of {sorted(_EXPRESSIONS)}"
            )
        shots.append(
            PlannedShot(
                index=position,
                text=normalized[start:end],
                script_start=start,
                script_end=end,
                visual_query=visual_query,
                must_match=_constraint_list(entry["must_match"], "must_match"),
                must_not_appear=_constraint_list(
                    entry["must_not_appear"], "must_not_appear"
                ),
                expression=expression,
            )
        )
    if cursor != len(normalized):
        raise ShotPlanError("shot plan must cover the whole script")
    return tuple(shots)


def _plan_prompt(script: str, clip_duration: float) -> str:
    """构造拆镜提示词；模型只返回原文下标，不返回旁白正文。"""
    return json.dumps(
        {
            "instruction": (
                "Split the narration script into consecutive shots for a short "
                "video. Return only JSON with a shots array.每个 shot 必须有 "
                "index (1-based), script_start 与 script_end（原文字符下标，"
                "左闭右开，必须首尾相接并覆盖全文），visual_query（这一镜需要的"
                "画面，用中文描述），must_match（画面必须出现的元素），"
                "must_not_appear（画面不应出现的元素），expression（direct 表示"
                "画面直接展示这句话的内容，ambience 表示这句话抽象或情绪化、"
                "画面只需氛围相符）。"
                "不要改写、翻译或补写旁白；不要编造产品卖点。"
                f"每镜目标约 {clip_duration:.0f} 秒，优先保持语义完整，"
                "不要为了凑时长把一句话拦腰切断。"
            ),
            "script": script,
            "script_length": len(script),
        },
        ensure_ascii=False,
    )


def plan_shots(
    script: str,
    *,
    clip_duration: float,
    llm_fn: LLMFn,
    app_config: Mapping[str, Any] | None,
) -> tuple[PlannedShot, ...] | None:
    """
    Ask the text model to split the script, or return None when unusable.

    @param script The final narration script.
    @param clip_duration Target shot duration in seconds.
    @param llm_fn Text entry point accepting the prompt and an optional config.
    @param app_config Optional runtime configuration snapshot.
    @returns Planned shots, or None when the model output cannot be trusted.
    @raises ShotPlanError If the script itself is empty.
    """
    normalized = str(script or "")
    if not normalized.strip():
        raise ShotPlanError("video script is required for shot planning")
    try:
        response = llm_fn(_plan_prompt(normalized, clip_duration), app_config=app_config)
    except Exception as exc:
        # 拆镜失败不应让整次生成失败：退回标点拆分仍能出片。
        logger.warning(f"shot planning call failed: {type(exc).__name__}: {exc}")
        return None
    try:
        return parse_shot_plan(response, normalized)
    except ShotPlanError as exc:
        logger.warning(f"shot planning rejected, falling back to punctuation: {exc}")
        return None
