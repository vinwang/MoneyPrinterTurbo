"""Semantic storyboard matching and executable local-video timelines."""

from __future__ import annotations

import math
import re
from array import array
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from itertools import chain
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.models.schema import MaterialInfo
from app.services import asset_library as library


AssetLibraryError = library.AssetLibraryError
LibraryAsset = library.LibraryAsset
LibrarySegment = library.LibrarySegment


@dataclass(frozen=True, slots=True)
class StoryboardShot:
    """一个分镜文本、目标时间和按相关性过滤后的候选视频素材。"""

    index: int
    text: str
    start_seconds: float
    duration_seconds: float
    candidates: tuple[LibraryAsset, ...]
    candidate_segments: tuple[LibrarySegment, ...] = ()
    candidate_scores: tuple[float, ...] = ()
    candidate_reasons: tuple[str, ...] = ()
    script_start: int = 0
    script_end: int = 0
    visual_query: str = ""
    must_match: tuple[str, ...] = ()
    must_not_appear: tuple[str, ...] = ()
    expression: str = "direct"


@dataclass(frozen=True, slots=True)
class StoryboardMatch:
    """一次文案匹配得到的分镜和 BGM 候选快照。"""

    shots: tuple[StoryboardShot, ...]
    bgm_candidates: tuple[LibraryAsset, ...]


_FRAME_CJK_CHARS_PER_SECOND = 4.2
_FRAME_LATIN_CHARS_PER_SECOND = 13.0
_MAX_SHOT_DURATION_MULTIPLIER = 1.5
_CANDIDATE_COUNT = 3
_MIN_RELEVANCE_SCORE = 0.12
_RECENT_USE_WINDOW_SECONDS = 7 * 24 * 60 * 60
_RECENT_USE_PENALTY = 0.08
_USE_COUNT_PENALTY = 0.01
_LATIN_WORD_CHARACTER_WEIGHT = 5
_CATEGORY_SCORE_WEIGHT = 0.05
_MUST_MATCH_SCORE_WEIGHT = 0.10
_MAX_CATEGORY_OVERLAPS = 2
_MAX_RECENT_USE_COUNT = 10
_TIMELINE_EPSILON_SECONDS = 0.02
_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]", re.IGNORECASE)
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_SRT_TIME_RE = re.compile(
    r"(?P<hours>\d+):(?P<minutes>\d+):(?P<seconds>\d+),(?P<millis>\d+)"
)


def _split_script_clauses(script: str) -> tuple[str, ...]:
    normalized = script.strip()
    if not normalized:
        raise AssetLibraryError("video script is required for local matching")
    clauses = tuple(
        part.strip()
        for part in re.findall(r"[^。！？!?；;，,：:\n]+[。！？!?；;，,：:\n]?", normalized)
        if part.strip()
    )
    return clauses or (normalized,)


def _estimate_duration(text: str) -> float:
    cjk_count = len(_CJK_RE.findall(text))
    latin_count = len(re.findall(r"[A-Za-z0-9]+", text))
    return max(
        1.0,
        cjk_count / _FRAME_CJK_CHARS_PER_SECOND
        + latin_count * _LATIN_WORD_CHARACTER_WEIGHT / _FRAME_LATIN_CHARS_PER_SECOND,
    )


def _make_shots(script: str, clip_duration: float) -> tuple[StoryboardShot, ...]:
    target = max(1.0, float(clip_duration))
    grouped: list[tuple[str, float]] = []
    current: list[str] = []
    current_duration = 0.0
    for clause in _split_script_clauses(script):
        duration = _estimate_duration(clause)
        would_overrun = (
            bool(current)
            and current_duration + duration > target * _MAX_SHOT_DURATION_MULTIPLIER
        )
        if would_overrun:
            grouped.append(("".join(current), current_duration))
            current = []
            current_duration = 0.0
        current.append(clause)
        current_duration += duration
        if current_duration >= target:
            grouped.append(("".join(current), current_duration))
            current = []
            current_duration = 0.0
    if current:
        grouped.append(("".join(current), current_duration))
    shots: list[StoryboardShot] = []
    start = 0.0
    normalized_script = script.strip()
    script_cursor = 0
    for index, (text, duration) in enumerate(grouped, start=1):
        text_start = normalized_script.find(text.strip(), script_cursor)
        if text_start < 0:
            text_start = script_cursor
        text_end = text_start + len(text.strip())
        script_cursor = text_end
        shots.append(
            StoryboardShot(
                index=index,
                text=text,
                start_seconds=start,
                duration_seconds=duration,
                candidates=(),
                script_start=text_start,
                script_end=text_end,
                visual_query=text.strip(),
            )
        )
        start += duration
    return tuple(shots)


def _tokens(text: str) -> set[str]:
    normalized = text.lower()
    tokens = set(_TOKEN_RE.findall(normalized))
    cjk_text = "".join(_CJK_RE.findall(normalized))
    tokens.update(cjk_text[index : index + 2] for index in range(len(cjk_text) - 1))
    return tokens


def _recent_penalty(asset: LibraryAsset) -> float:
    penalty = min(asset.use_count, _MAX_RECENT_USE_COUNT) * _USE_COUNT_PENALTY
    if not asset.last_used_at:
        return penalty
    try:
        last_used = datetime.fromisoformat(asset.last_used_at)
        age = (datetime.now(timezone.utc) - last_used).total_seconds()
    except (TypeError, ValueError):
        return penalty
    return penalty + (_RECENT_USE_PENALTY if age < _RECENT_USE_WINDOW_SECONDS else 0.0)


def _segment_score(
    query: str,
    segment: LibrarySegment,
    asset: LibraryAsset,
    *,
    query_tokens: frozenset[str] | set[str] | None = None,
    overlap: int | None = None,
    semantic_size: int | None = None,
) -> float:
    """
    只使用描述、标签和低权重目录分类计算匹配分，不使用文件名。

    `query_tokens`、`overlap`、`semantic_size` 允许调用方传入已经算好的
    查询 token、交集数和片段 token 数：一次匹配要给几千个片段打分，分词是
    最大的开销，倒排索引建好后就不该再逐片段重算。
    """
    if query_tokens is None:
        query_tokens = _tokens(query)
    if overlap is None or semantic_size is None:
        semantic_tokens = _tokens(_semantic_text(segment, asset))
        overlap = len(query_tokens & semantic_tokens)
        semantic_size = len(semantic_tokens)
    if not query_tokens or not semantic_size:
        return -_recent_penalty(asset)
    score = overlap / math.sqrt(len(query_tokens) * semantic_size)
    category_overlap = len(query_tokens & _tokens(asset.category))
    score += min(category_overlap, _MAX_CATEGORY_OVERLAPS) * _CATEGORY_SCORE_WEIGHT
    return score - _recent_penalty(asset)


@dataclass(frozen=True, slots=True)
class RecallIndex:
    """
    片段语义 token 的倒排索引，用于把打分范围从全库收窄到有交集的片段。

    只在 `require_relevance=True` 时用于缩小候选：语义分是 token 交集除以
    长度的几何平均，没有交集的片段语义分恒为 0，必定低于最低相关性阈值，
    所以「只打分共享至少一个 token 的片段」与全表扫描结果完全一致。
    BGM 走 `require_relevance=False`，零交集片段仍可能凭近期使用惩罚排序，
    因此不套用这条捷径。
    """

    postings: Mapping[str, array]
    segments: tuple[LibrarySegment, ...]
    token_counts: array


def build_recall_index(
    segments: Sequence[LibrarySegment],
    assets_by_id: Mapping[str, LibraryAsset],
) -> RecallIndex:
    """
    Build the stage-one recall index for one matching pass.

    @param segments Segments the match may draw from.
    @param assets_by_id Assets keyed by ID, used for BGM-only semantic fields.
    @returns Token postings, the segment tuple they index into, and each
        segment's semantic token count so scoring does not re-tokenize.
    """
    # 倒排表和 token 数都用紧凑数组：三万片段的 Python set 元组会占几百 MB，
    # 而打分只需要交集数和片段 token 数，前者可以从倒排表直接累加得到。
    postings: dict[str, array] = {}
    ordered = tuple(segments)
    token_counts = array("I")
    for position, segment in enumerate(ordered):
        asset = assets_by_id.get(segment.asset_id)
        tokens = () if asset is None else _tokens(_semantic_text(segment, asset))
        token_counts.append(len(tokens))
        for token in tokens:
            posting = postings.get(token)
            if posting is None:
                posting = postings[token] = array("I")
            posting.append(position)
    return RecallIndex(postings=postings, segments=ordered, token_counts=token_counts)


def _recalled_segments(
    index: RecallIndex,
    query_tokens: frozenset[str] | set[str],
) -> tuple[tuple[LibrarySegment, int, int], ...]:
    """取出与查询至少共享一个 token 的片段，附带交集数和片段 token 数。"""
    postings = index.postings
    overlaps = Counter(
        chain.from_iterable(postings[token] for token in query_tokens if token in postings)
    )
    segments = index.segments
    token_counts = index.token_counts
    return tuple(
        (segments[position], overlap, token_counts[position])
        for position, overlap in sorted(overlaps.items())
    )


def _semantic_text(segment: LibrarySegment, asset: LibraryAsset) -> str:
    """片段可被约束检索的全部语义文本。"""
    # 视频按片段打分：资产级描述是所有窗口的汇总，掺进来会让同一视频的
    # 每个片段都拿到整条视频的语义，逐窗口索引就失效了。旧库里片段描述
    # 本就是整条描述的副本，去掉汇总不会让旧数据变差。
    values = (segment.description, *segment.tags)
    if asset.kind == "bgm":
        # BGM 没有片段索引，资产级描述和目录分类是唯一语义来源。
        values = (*values, asset.description, *asset.tags, asset.category)
    return " ".join(values).lower()


def _violates_exclusion(
    segment: LibrarySegment,
    asset: LibraryAsset,
    must_not_appear: Sequence[str],
) -> bool:
    """
    判断片段是否命中「不应出现」约束。

    只有排除项作为硬过滤。`must_match` 不作硬过滤：模型给出的是它想象中的
    画面元素（「薄皮」「疲惫的人物状态」），这些词多半不会逐字出现在素材
    描述里，按字面要求全部命中会把整库过滤成空。实测 17 个分镜里有 13 个
    会因此变成缺口，连本该命中的美妆素材也被挡掉。方案里也写明「只有明确
    业务要求才作为硬约束」，模型猜测的画面元素不属于此列——它们改为通过
    `visual_query` 影响排序。
    """
    if not must_not_appear:
        return False
    haystack = _semantic_text(segment, asset)
    return any(
        str(item).strip().lower() in haystack for item in must_not_appear if str(item).strip()
    )


def _rank_segments(
    query: str,
    segments: Sequence[LibrarySegment],
    assets_by_id: Mapping[str, LibraryAsset],
    *,
    excluded_ids: set[str] | None = None,
    count: int = _CANDIDATE_COUNT,
    require_relevance: bool = True,
    must_match: Sequence[str] = (),
    must_not_appear: Sequence[str] = (),
    recall_index: RecallIndex | None = None,
) -> tuple[tuple[LibrarySegment, float], ...]:
    """返回去重后的高相关片段；候选不足时不回填已排除素材。"""
    excluded = excluded_ids or set()
    ranked: list[tuple[LibrarySegment, float]] = []
    normalized_must_match = tuple(
        str(item).strip().lower() for item in must_match if str(item).strip()
    )
    # 第一阶段召回：只在要求相关性时收窄，见 RecallIndex 的等价性说明。
    # 查询词只分一次；索引路径带回交集数和片段 token 数，全表路径逐条分词。
    query_tokens = frozenset(_tokens(query))
    candidates: Sequence[tuple[LibrarySegment, int | None, int | None]] = (
        _recalled_segments(recall_index, query_tokens)
        if recall_index is not None and require_relevance
        else tuple((segment, None, None) for segment in segments)
    )
    for segment, overlap, semantic_size in candidates:
        asset = assets_by_id.get(segment.asset_id)
        if asset is None or segment.asset_id in excluded:
            continue
        # 排除项是硬过滤：宁可留缺口，也不能把明确不该出现的画面塞进来。
        if _violates_exclusion(segment, asset, must_not_appear):
            continue
        score = _segment_score(
            query,
            segment,
            asset,
            query_tokens=query_tokens,
            overlap=overlap,
            semantic_size=semantic_size,
        )
        semantic_score = score + _recent_penalty(asset)
        if require_relevance and semantic_score < _MIN_RELEVANCE_SCORE:
            continue
        if normalized_must_match:
            # 必须匹配项作为加分：命中得越多排得越前，但不命中不至于出局。
            haystack = _semantic_text(segment, asset)
            hits = sum(1 for item in normalized_must_match if item in haystack)
            score += hits / len(normalized_must_match) * _MUST_MATCH_SCORE_WEIGHT
        ranked.append((segment, score))
    ranked.sort(
        key=lambda item: (
            -item[1],
            item[0].asset_id,
            item[0].source_start_seconds,
        )
    )
    selected: list[tuple[LibrarySegment, float]] = []
    selected_assets: set[str] = set()
    for segment, score in ranked:
        if segment.asset_id in selected_assets:
            continue
        selected.append((segment, score))
        selected_assets.add(segment.asset_id)
        if len(selected) >= count:
            break
    return tuple(selected)


def _rank_assets(
    query: str,
    assets: Sequence[LibraryAsset],
    *,
    excluded_ids: set[str] | None = None,
    count: int = _CANDIDATE_COUNT,
) -> tuple[LibraryAsset, ...]:
    """保留旧内部调用的资产排序入口，并沿用新的无回填过滤规则。"""
    segments = tuple(
        library.LibrarySegment(
            segment_id=f"legacy-{asset.asset_id}",
            asset_id=asset.asset_id,
            source_start_seconds=0.0,
            source_end_seconds=asset.duration,
            description=asset.description,
            tags=asset.tags,
            analysis_status=asset.analysis_status,
            analysis_error=asset.analysis_error,
            analysis_model="legacy-asset-window",
        )
        for asset in assets
    )
    ranked = _rank_segments(
        query,
        segments,
        {asset.asset_id: asset for asset in assets},
        excluded_ids=excluded_ids,
        count=count,
    )
    return tuple({asset.asset_id: asset for asset in assets}[segment.asset_id] for segment, _ in ranked)


def _assets_for_root(
    assets: Sequence[LibraryAsset],
    root: Path | str | None,
) -> tuple[LibraryAsset, ...]:
    if root is None:
        return tuple(assets)
    if not str(root).strip():
        return ()
    normalized_root = str(Path(root).expanduser().resolve())
    return tuple(asset for asset in assets if asset.root_path == normalized_root)


def _shots_from_plan(
    planned: Sequence[Any],
    clip_duration: float,
) -> tuple[StoryboardShot, ...]:
    """把模型给出的分镜计划转成带时长估算的分镜。"""
    shots: list[StoryboardShot] = []
    start = 0.0
    for item in planned:
        duration = _estimate_duration(item.text)
        shots.append(
            StoryboardShot(
                index=item.index,
                text=item.text,
                start_seconds=start,
                duration_seconds=duration,
                candidates=(),
                script_start=item.script_start,
                script_end=item.script_end,
                visual_query=item.visual_query,
                must_match=item.must_match,
                must_not_appear=item.must_not_appear,
                expression=item.expression,
            )
        )
        start += duration
    return tuple(shots)


def _match_video_shots(
    video_script: str,
    clip_duration: float,
    videos: Sequence[LibraryAsset],
    segments: Sequence[LibrarySegment],
    *,
    query_context: str | None = None,
    planned_shots: Sequence[Any] | None = None,
) -> tuple[StoryboardShot, ...]:
    """为每个语义分镜生成候选，并在首选层面避免同片重复。"""
    assets_by_id = {asset.asset_id: asset for asset in videos}
    from_plan = bool(planned_shots)
    script_shots = (
        _shots_from_plan(planned_shots, clip_duration)
        if from_plan
        else _make_shots(video_script, clip_duration)
    )
    if query_context is not None and not isinstance(query_context, str):
        raise AssetLibraryError("query_context must be a string")
    normalized_context = query_context.strip() if query_context else ""
    shots: list[StoryboardShot] = []
    used_ids: set[str] = set()
    # 倒排索引每次匹配只建一次：一条文案通常拆 2-4 镜，逐镜重建会把建索引的
    # 成本乘上分镜数，反而比全表打分更慢。
    recall_index = build_recall_index(segments, assets_by_id)
    for shot in script_shots:
        visual_query = shot.visual_query or shot.text
        # 主题/关键词默认补给每一镜。只有模型明确把这镜判为 direct 时才不补：
        # 直述镜自己已经说清要什么画面，再拼上下文会把「现在下单立减」这类
        # 短句的语义冲掉。没有模型计划时无从判断，沿用旧行为全都补。
        use_context = bool(normalized_context) and not (
            from_plan and shot.expression == "direct"
        )
        match_query = (
            f"{normalized_context} {visual_query}".strip()
            if use_context
            else visual_query
        )
        ranked = _rank_segments(
            match_query,
            segments,
            assets_by_id,
            excluded_ids=used_ids,
            must_match=shot.must_match,
            must_not_appear=shot.must_not_appear,
            recall_index=recall_index,
        )
        selected_segments = tuple(item[0] for item in ranked)
        scores = tuple(item[1] for item in ranked)
        candidates = tuple(assets_by_id[item.asset_id] for item in selected_segments)
        reasons = tuple(
            f"semantic description/tag match score={score:.3f}" for score in scores
        )
        if selected_segments:
            used_ids.add(selected_segments[0].asset_id)
        shots.append(
            StoryboardShot(
                index=shot.index,
                text=shot.text,
                start_seconds=shot.start_seconds,
                duration_seconds=shot.duration_seconds,
                candidates=candidates,
                candidate_segments=selected_segments,
                candidate_scores=scores,
                candidate_reasons=reasons,
                script_start=shot.script_start,
                script_end=shot.script_end,
                visual_query=match_query,
                must_match=shot.must_match,
                must_not_appear=shot.must_not_appear,
                expression=shot.expression,
            )
        )
    return tuple(shots)


def _match_bgm_candidates(
    video_script: str,
    bgm_assets: Sequence[LibraryAsset],
) -> tuple[LibraryAsset, ...]:
    """按音乐描述、标签和目录分类生成不超过三个 BGM 候选。"""
    bgm_segments = tuple(
        library.LibrarySegment(
            segment_id=f"bgm-{asset.asset_id}",
            asset_id=asset.asset_id,
            source_start_seconds=0.0,
            source_end_seconds=asset.duration,
            description=asset.description,
            tags=asset.tags,
            analysis_status=asset.analysis_status,
            analysis_error=asset.analysis_error,
            analysis_model="bgm-metadata",
        )
        for asset in bgm_assets
    )
    asset_by_id = {asset.asset_id: asset for asset in bgm_assets}
    ranked = _rank_segments(
        video_script,
        bgm_segments,
        asset_by_id,
        count=_CANDIDATE_COUNT,
        require_relevance=False,
    )
    return tuple(asset_by_id[segment.asset_id] for segment, _score in ranked)


def match_storyboard(
    video_script: str,
    *,
    clip_duration: float,
    query_context: str | None = None,
    planned_shots: Sequence[Any] | None = None,
    db_path=None,
    video_root: Path | str | None = None,
    bgm_root: Path | str | None = None,
) -> StoryboardMatch:
    """
    根据文案为每个分镜召回并过滤本地视频片段。

    @param video_script 已输入或生成的最终视频文案。
    @param clip_duration 分镜目标时长，用于拆分文案和候选窗口。
    @param query_context 可选的主题或视频关键词，仅用于补充氛围镜。
    @param planned_shots 可选的模型拆镜结果；None 表示按标点拆分。
    @param db_path 可选素材库数据库路径。
    @param video_root 当前任务允许使用的视频库根目录；None 表示不限定。
    @param bgm_root 当前任务允许使用的 BGM 库根目录；空字符串表示没有 BGM 库。
    @returns 包含分镜候选和 BGM 候选的不可变匹配结果。
    @raises AssetLibraryError 没有合格片段或索引不可用。
    """
    videos = _assets_for_root(
        library.list_assets(kind="video", analysis_status="ready", db_path=db_path),
        video_root,
    )
    if not videos:
        failed = _assets_for_root(
            library.list_assets(
                kind="video", analysis_status="failed", db_path=db_path
            ),
            video_root,
        )
        state = (
            "video analysis failed"
            if failed
            else "no analyzed video assets; visual analysis is required"
        )
        raise AssetLibraryError(state)
    segments = library.list_segments(
        asset_ids=tuple(asset.asset_id for asset in videos),
        analysis_status="ready",
        db_path=db_path,
    )
    if not segments:
        raise AssetLibraryError("no analyzed video segments; update the local asset index")
    shots = _match_video_shots(
        video_script,
        clip_duration,
        videos,
        segments,
        query_context=query_context,
        planned_shots=planned_shots,
    )
    bgm_assets = _assets_for_root(
        library.list_assets(kind="bgm", analysis_status="ready", db_path=db_path),
        bgm_root,
    )
    return StoryboardMatch(shots, _match_bgm_candidates(video_script, bgm_assets))


def _candidate_segment(shot: StoryboardShot, candidate_index: int) -> LibrarySegment:
    if shot.candidate_segments and len(shot.candidate_segments) == len(shot.candidates):
        return shot.candidate_segments[candidate_index]
    asset = shot.candidates[candidate_index]
    return LibrarySegment(
        segment_id=f"legacy-{asset.asset_id}",
        asset_id=asset.asset_id,
        source_start_seconds=0.0,
        source_end_seconds=asset.duration,
        description=asset.description,
        tags=asset.tags,
        analysis_status=asset.analysis_status,
        analysis_error=asset.analysis_error,
        analysis_model="legacy-asset-window",
    )


_MANUAL_SELECTION_MODEL = "manual-selection"
_MANUAL_SELECTION_REASON = "manually selected from the local library to fill a gap shot"


def _shot_sort_key(index: str) -> tuple[int, int | str]:
    """分镜序号通常是数字，按数值排序；非数字序号退回字面排序且排在后面。"""
    text = str(index)
    return (0, int(text)) if text.isdigit() else (1, text)


def with_manual_candidates(
    match: StoryboardMatch,
    assets_by_shot: Mapping[str, LibraryAsset],
) -> StoryboardMatch:
    """
    把人工为缺口镜选择的素材写回候选快照。

    匹配器宁可留缺口也不硬凑无关素材，缺口镜因此没有候选，冻结计划时会被
    判为「选择未覆盖全部分镜」。人工选择的素材写回快照后，既有校验（候选
    必须存在于快照中）无需放宽即可通过，快照也仍然如实记录用户看到的东西。

    @param match 当前候选快照。
    @param assets_by_shot 按分镜序号映射的人工选择素材，只允许填补缺口镜。
    @returns 缺口镜带上人工候选的新快照；无人工选择时返回原快照。
    @raises AssetLibraryError 序号不存在，或该镜本来就有匹配候选。
    """
    if not isinstance(assets_by_shot, Mapping):
        raise AssetLibraryError("manual storyboard selections must be an object")
    if not assets_by_shot:
        return match
    shots_by_index = {str(shot.index): shot for shot in match.shots}
    unknown = sorted(set(assets_by_shot) - set(shots_by_index))
    if unknown:
        raise AssetLibraryError(
            f"manual selection refers to unknown shot(s): {', '.join(unknown)}"
        )
    occupied = sorted(
        index for index in assets_by_shot if shots_by_index[index].candidates
    )
    if occupied:
        raise AssetLibraryError(
            f"shot {', '.join(occupied)} already has matched candidates; "
            "manual selection only fills gap shots"
        )
    shots: list[StoryboardShot] = []
    for shot in match.shots:
        asset = assets_by_shot.get(str(shot.index))
        if asset is None:
            shots.append(shot)
            continue
        # 人工素材没有逐窗口分析，整条可用；相关性分数留空，不伪造匹配得分。
        segment = LibrarySegment(
            segment_id=f"manual-{asset.asset_id}",
            asset_id=asset.asset_id,
            source_start_seconds=0.0,
            source_end_seconds=asset.duration,
            description=asset.description,
            tags=asset.tags,
            analysis_status=asset.analysis_status,
            analysis_error=asset.analysis_error,
            analysis_model=_MANUAL_SELECTION_MODEL,
        )
        shots.append(
            replace(
                shot,
                candidates=(asset,),
                candidate_segments=(segment,),
                candidate_scores=(),
                candidate_reasons=(_MANUAL_SELECTION_REASON,),
            )
        )
    return StoryboardMatch(tuple(shots), match.bgm_candidates)


def build_storyboard_plan(
    match: StoryboardMatch,
    selected_ids: Mapping[str, str],
) -> tuple[dict[str, Any], ...]:
    """
    将用户在 WebUI 选择的候选冻结为可执行分镜计划。

    @param match 当前文案对应的候选快照。
    @param selected_ids 按分镜序号映射的用户选择资产 ID。
    @returns 顺序固定、包含源时间和索引哈希的不可变计划元组。
    @raises AssetLibraryError 选择缺失、越界或候选快照不一致。
    """
    if not isinstance(selected_ids, Mapping):
        raise AssetLibraryError("storyboard selections must be an object")
    plan: list[dict[str, Any]] = []
    expected_indexes = {str(shot.index) for shot in match.shots}
    provided = {str(key) for key in selected_ids}
    if provided != expected_indexes:
        # 报错必须指出是哪一镜：缺口镜要么人工补素材，要么改文案，
        # 只说「未覆盖全部分镜」用户无从下手。
        missing = sorted(expected_indexes - provided, key=_shot_sort_key)
        unexpected = sorted(provided - expected_indexes, key=_shot_sort_key)
        details: list[str] = []
        if missing:
            details.append(f"shot {', '.join(missing)} has no selected material")
        if unexpected:
            details.append(f"unknown shot {', '.join(unexpected)}")
        raise AssetLibraryError(
            f"storyboard selections do not cover all shots: {'; '.join(details)}"
        )
    for shot in match.shots:
        selected_id = str(selected_ids.get(str(shot.index), "")).strip()
        candidate_index = next(
            (
                index
                for index, asset in enumerate(shot.candidates)
                if asset.asset_id == selected_id
            ),
            None,
        )
        if candidate_index is None:
            raise AssetLibraryError(
                f"selected storyboard candidate is not available: shot {shot.index}"
            )
        asset = shot.candidates[candidate_index]
        segment = _candidate_segment(shot, candidate_index)
        candidate_snapshot = []
        for index, candidate in enumerate(shot.candidates):
            candidate_segment = _candidate_segment(shot, index)
            candidate_snapshot.append(
                {
                    "asset_id": candidate.asset_id,
                    "asset_sha256": candidate.sha256,
                    "source_relative_path": candidate.relative_path,
                    "segment_id": candidate_segment.segment_id,
                    "source_start_seconds": candidate_segment.source_start_seconds,
                    "source_end_seconds": candidate_segment.source_end_seconds,
                    "relevance_score": (
                        shot.candidate_scores[index]
                        if len(shot.candidate_scores) == len(shot.candidates)
                        else None
                    ),
                    "selection_reason": (
                        shot.candidate_reasons[index]
                        if len(shot.candidate_reasons) == len(shot.candidates)
                        else "candidate from the current storyboard snapshot"
                    ),
                }
            )
        plan.append(
            {
                "shot_index": shot.index,
                "text": shot.text,
                "script_start": shot.script_start,
                "script_end": shot.script_end,
                "visual_query": shot.visual_query or shot.text,
                "must_match": list(shot.must_match),
                "must_not_appear": list(shot.must_not_appear),
                "expression": shot.expression,
                "asset_id": asset.asset_id,
                "asset_sha256": asset.sha256,
                "source_relative_path": asset.relative_path,
                "segment_id": segment.segment_id,
                "candidates": candidate_snapshot,
                "source_start_seconds": segment.source_start_seconds,
                "source_end_seconds": segment.source_end_seconds,
                "target_start_seconds": shot.start_seconds,
                "target_end_seconds": shot.start_seconds + shot.duration_seconds,
                "target_duration_seconds": shot.duration_seconds,
                "relevance_score": (
                    shot.candidate_scores[candidate_index]
                    if len(shot.candidate_scores) == len(shot.candidates)
                    else None
                ),
                "selection_reason": (
                    shot.candidate_reasons[candidate_index]
                    if len(shot.candidate_reasons) == len(shot.candidates)
                    else "selected by user from the candidate snapshot"
                ),
            }
        )
    return tuple(plan)


def _parse_srt_time(value: str) -> float:
    match = _SRT_TIME_RE.fullmatch(value.strip())
    if not match:
        raise ValueError("invalid subtitle timestamp")
    return (
        int(match["hours"]) * 3600
        + int(match["minutes"]) * 60
        + int(match["seconds"])
        + int(match["millis"]) / 1000
    )


def _subtitle_ranges(path: str | Path | None) -> tuple[tuple[float, float, str], ...]:
    if not path or not Path(path).is_file():
        return ()
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    ranges: list[tuple[float, float, str]] = []
    for line_index, line in enumerate(lines):
        if "-->" not in line:
            continue
        left, right = (part.strip() for part in line.split("-->", 1))
        try:
            start = _parse_srt_time(left)
            end = _parse_srt_time(right)
        except ValueError as exc:
            raise AssetLibraryError(
                f"invalid subtitle timestamp at line {line_index + 1}"
            ) from exc
        text_lines: list[str] = []
        for text_line in lines[line_index + 1 :]:
            if not text_line.strip():
                break
            text_lines.append(text_line.strip())
        if end <= start:
            raise AssetLibraryError("subtitle range must have positive duration")
        if ranges and start < ranges[-1][1]:
            raise AssetLibraryError("subtitle ranges must be ordered and non-overlapping")
        ranges.append((start, end, "".join(text_lines)))
    return tuple(ranges)


def _alignment_text(value: str) -> str:
    return "".join(_TOKEN_RE.findall(str(value).lower()))


def _subtitle_character_spans(
    subtitles: Sequence[tuple[float, float, str]],
) -> tuple[tuple[float, float, int, int], ...]:
    """给每条字幕标出它在归一化文本流里的字符区间。"""
    spans: list[tuple[float, float, int, int]] = []
    cursor = 0
    for start, end, text in subtitles:
        length = len(_alignment_text(text))
        if length <= 0:
            continue
        spans.append((start, end, cursor, cursor + length))
        cursor += length
    return tuple(spans)


def _time_at_character(
    spans: Sequence[tuple[float, float, int, int]],
    position: int,
) -> float:
    """把归一化文本流里的字符位置映射回真实旁白时间。"""
    for index, (start, end, char_start, char_end) in enumerate(spans):
        if position <= char_start:
            if index == 0:
                return start
            # 位置正好落在两条字幕之间，说明这里是停顿。切在停顿中点，
            # 画面在说话人换句的空档里切走，而不是压着上一句或下一句。
            previous_end = spans[index - 1][1]
            return previous_end + (start - previous_end) / 2
        if position < char_end:
            return start + (end - start) * (position - char_start) / (
                char_end - char_start
            )
    return spans[-1][1]


def _subtitle_target_ranges(
    plan: Sequence[Mapping[str, Any]],
    audio_duration: float,
    subtitles: Sequence[tuple[float, float, str]],
) -> tuple[tuple[float, float], ...]:
    """
    按字幕时间戳而不是字数比例切分每个分镜的成片时间段。

    @param plan 用户确认的分镜计划。
    @param audio_duration 实际旁白或上传音频时长。
    @param subtitles 已校验顺序且不重叠的字幕区间。
    @returns 覆盖 0 到 audio_duration 的连续成片时间段。
    @raises AssetLibraryError 分镜没有可对齐文本，或对齐结果不再递增。
    """
    spans = _subtitle_character_spans(subtitles)
    if not spans:
        raise AssetLibraryError("subtitle text does not cover storyboard script")
    if spans[0][0] < 0 or spans[-1][1] > audio_duration + _TIMELINE_EPSILON_SECONDS:
        raise AssetLibraryError("subtitle timestamps exceed audio duration")
    ranges: list[tuple[float, float]] = []
    character_cursor = 0
    time_cursor = 0.0
    for index, item in enumerate(plan):
        length = len(_alignment_text(str(item["text"])))
        if length <= 0:
            raise AssetLibraryError(
                f"storyboard shot {index + 1} has no alignable text for subtitles"
            )
        character_cursor += length
        # 片头静音归第一镜、片尾静音归末镜，时间轴不留空洞。
        end = (
            audio_duration
            if index == len(plan) - 1
            else _time_at_character(spans, character_cursor)
        )
        if end <= time_cursor:
            raise AssetLibraryError(
                f"subtitle alignment produced an empty range for storyboard shot {index + 1}"
            )
        ranges.append((time_cursor, end))
        time_cursor = end
    return tuple(ranges)


def _target_ranges(
    plan: Sequence[Mapping[str, Any]],
    audio_duration: float,
    subtitle_path: str | Path | None,
) -> tuple[tuple[float, float], ...]:
    subtitles = _subtitle_ranges(subtitle_path)
    if subtitles:
        script_text = _alignment_text("".join(str(item["text"]) for item in plan))
        subtitle_text = _alignment_text("".join(item[2] for item in subtitles))
        if script_text and script_text != subtitle_text:
            raise AssetLibraryError("subtitle text does not cover storyboard script")
        if script_text and not subtitle_text:
            raise AssetLibraryError("subtitle text does not cover storyboard script")
        if subtitle_text:
            return _subtitle_target_ranges(plan, audio_duration, subtitles)

    explicit_durations = []
    for item in plan:
        try:
            duration = float(item["target_duration_seconds"])
        except (KeyError, TypeError, ValueError):
            explicit_durations = []
            break
        if not math.isfinite(duration) or duration <= 0:
            explicit_durations = []
            break
        explicit_durations.append(duration)
    if len(explicit_durations) == len(plan):
        total_duration = sum(explicit_durations)
        ranges: list[tuple[float, float]] = []
        cursor = 0.0
        for index, duration in enumerate(explicit_durations):
            end = (
                audio_duration
                if index == len(explicit_durations) - 1
                else cursor + audio_duration * duration / total_duration
            )
            ranges.append((cursor, end))
            cursor = end
        return tuple(ranges)

    weights = [max(1, len(_alignment_text(str(item["text"])))) for item in plan]
    total_weight = sum(weights)
    ranges = []
    cursor = 0.0
    for index, weight in enumerate(weights):
        end = (
            audio_duration
            if index == len(weights) - 1
            else cursor + audio_duration * weight / total_weight
        )
        ranges.append((cursor, end))
        cursor = end
    return tuple(ranges)


def align_storyboard_plan(
    plan: Sequence[Mapping[str, Any]],
    *,
    audio_duration: float,
    subtitle_path: str | Path | None = None,
    clip_speed: float = 1.0,
) -> tuple[dict[str, Any], ...]:
    """
    按真实旁白时间轴冻结分镜，并校验源片段足够播放。

    @param plan 用户确认的分镜计划。
    @param audio_duration 实际旁白或上传音频时长。
    @param subtitle_path 可选带时间戳字幕，用于文本与音频对齐。
    @param clip_speed 视频播放速度；源窗口按该速度反推。
    @returns 带连续 target/source 时间范围的不可变计划元组。
    @raises AssetLibraryError 输入非法、字幕不一致或源片段不足。
    """
    if not math.isfinite(float(audio_duration)) or audio_duration <= 0:
        raise AssetLibraryError("audio duration must be positive for storyboard alignment")
    if not math.isfinite(float(clip_speed)) or clip_speed <= 0:
        raise AssetLibraryError("clip speed must be positive for storyboard alignment")
    if not plan:
        raise AssetLibraryError("storyboard plan is required for local timeline")
    for index, item in enumerate(plan, start=1):
        if not isinstance(item, Mapping) or not str(item.get("text", "")).strip():
            raise AssetLibraryError(f"storyboard shot {index} is missing original text")
        try:
            shot_index = int(item.get("shot_index", 0))
        except (TypeError, ValueError) as exc:
            raise AssetLibraryError("storyboard shot indexes must be consecutive") from exc
        if shot_index != index:
            raise AssetLibraryError("storyboard shot indexes must be consecutive")
    ranges = _target_ranges(plan, float(audio_duration), subtitle_path)
    if len(ranges) != len(plan):
        raise AssetLibraryError("storyboard alignment did not cover every shot")

    aligned: list[dict[str, Any]] = []
    for index, (item, (target_start, target_end)) in enumerate(
        zip(plan, ranges), start=1
    ):
        entry = dict(item)
        if int(entry.get("shot_index", 0)) != index:
            raise AssetLibraryError("storyboard shot indexes must be consecutive")
        source_start = float(entry.get("source_start_seconds", 0.0))
        source_end = float(entry.get("source_end_seconds", 0.0))
        if not math.isfinite(source_start) or not math.isfinite(source_end):
            raise AssetLibraryError(f"invalid source range for storyboard shot {index}")
        if source_start < 0 or source_end <= source_start:
            raise AssetLibraryError(f"invalid source range for storyboard shot {index}")
        target_duration = target_end - target_start
        required_source_duration = target_duration * float(clip_speed)
        if source_end - source_start + _TIMELINE_EPSILON_SECONDS < required_source_duration:
            raise AssetLibraryError(
                f"selected source segment is too short for storyboard shot {index}"
            )
        entry["source_end_seconds"] = source_start + required_source_duration
        entry["target_start_seconds"] = target_start
        entry["target_end_seconds"] = target_end
        aligned.append(entry)
    if abs(aligned[0]["target_start_seconds"]) > _TIMELINE_EPSILON_SECONDS:
        raise AssetLibraryError("storyboard timeline must start at zero")
    if abs(aligned[-1]["target_end_seconds"] - audio_duration) > _TIMELINE_EPSILON_SECONDS:
        raise AssetLibraryError("storyboard timeline does not cover the audio")
    return tuple(aligned)


def material_infos_for_ids(
    asset_ids: Sequence[str], *, db_path=None
) -> tuple[MaterialInfo, ...]:
    """
    将已选视频资产 ID 转换成 MPT 本地素材参数。

    @param asset_ids 已经过 UI 候选列表校验的资产 ID 顺序。
    @param db_path 可选素材库数据库路径。
    @returns MPT 可消费的本地素材参数元组。
    @raises AssetLibraryError ID 无效、类型错误或文件缺失。
    """
    infos: list[MaterialInfo] = []
    for asset_id in asset_ids:
        asset = library.get_asset(asset_id, db_path=db_path)
        if asset.kind != "video" or asset.analysis_status != "ready":
            raise AssetLibraryError(f"selected asset is not a ready video: {asset_id}")
        library.resolve_asset_path(asset)
        infos.append(
            MaterialInfo(
                provider="local",
                url=asset.relative_path,
                duration=int(max(0, round(asset.duration))),
                source_info={"library_id": asset.asset_id},
            )
        )
    return tuple(infos)
