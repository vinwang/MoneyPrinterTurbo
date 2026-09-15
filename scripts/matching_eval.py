"""Score local storyboard matching against a human-labelled eval set.

The eval set fixes the scripts and the acceptable assets per shot so that any
change to shot splitting, segment indexing, recall or re-ranking can be compared
against a recorded baseline. Metrics come from labels, never from the matcher's
own confidence scores.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.services import asset_library, asset_matching


class EvalError(RuntimeError):
    """Raised when an eval set, a case result, or a baseline is unusable."""


_SCHEMA_VERSION = 1
_TOP_LEVEL_FIELDS = frozenset({"schema_version", "notes", "cases"})
_CASE_FIELDS = frozenset(
    {
        "case_id",
        "labels",
        "script",
        "clip_duration",
        "query_context",
        "expectations",
    }
)
_EXPECTATION_FIELDS = frozenset(
    {
        "text_contains",
        "acceptable_asset_ids",
        "forbidden_asset_ids",
        "expect_gap",
        "note",
    }
)
# 命中率越高越好，误报率越低越好。基线比较按方向判断回归。
_HIGHER_IS_BETTER = ("top1_hit_rate", "top3_available_rate", "expected_gap_honour_rate")
_LOWER_IS_BETTER = ("false_gap_rate", "forbidden_hit_rate")
_TRACKED_METRICS = (*_HIGHER_IS_BETTER, *_LOWER_IS_BETTER)


def _require_str(value: Any, field: str, case_id: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvalError(f"case {case_id}: {field} must be a non-empty string")
    return value.strip()


def _require_id_list(value: Any, field: str, case_id: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise EvalError(f"case {case_id}: {field} must be an array of asset IDs")
    ids = tuple(str(item).strip() for item in value)
    if any(not item for item in ids):
        raise EvalError(f"case {case_id}: {field} contains an empty asset ID")
    if len(set(ids)) != len(ids):
        raise EvalError(f"case {case_id}: {field} contains duplicate asset IDs")
    return ids


def _normalize_expectation(raw: Any, case_id: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise EvalError(f"case {case_id}: every expectation must be an object")
    unknown = set(raw) - _EXPECTATION_FIELDS
    if unknown:
        raise EvalError(
            f"case {case_id}: unknown expectation fields: {sorted(unknown)}"
        )
    text_contains = _require_str(raw.get("text_contains"), "text_contains", case_id)
    expect_gap = raw.get("expect_gap", False)
    if not isinstance(expect_gap, bool):
        raise EvalError(f"case {case_id}: expect_gap must be a boolean")
    acceptable = _require_id_list(
        raw.get("acceptable_asset_ids"), "acceptable_asset_ids", case_id
    )
    forbidden = _require_id_list(
        raw.get("forbidden_asset_ids"), "forbidden_asset_ids", case_id
    )
    if expect_gap and acceptable:
        raise EvalError(
            f"case {case_id}: expect_gap cannot be combined with acceptable_asset_ids"
        )
    if not expect_gap and not acceptable:
        raise EvalError(
            f"case {case_id}: expectation needs acceptable_asset_ids or expect_gap"
        )
    overlap = set(acceptable) & set(forbidden)
    if overlap:
        raise EvalError(
            f"case {case_id}: {sorted(overlap)} is both acceptable and forbidden"
        )
    return {
        "text_contains": text_contains,
        "acceptable_asset_ids": acceptable,
        "forbidden_asset_ids": forbidden,
        "expect_gap": expect_gap,
        "note": str(raw.get("note", "")),
    }


def _normalize_case(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise EvalError("every case must be an object")
    case_id = _require_str(raw.get("case_id"), "case_id", "<unknown>")
    unknown = set(raw) - _CASE_FIELDS
    if unknown:
        raise EvalError(f"case {case_id}: unknown case fields: {sorted(unknown)}")
    script = _require_str(raw.get("script"), "script", case_id)
    clip_duration = raw.get("clip_duration")
    if not isinstance(clip_duration, (int, float)) or clip_duration <= 0:
        raise EvalError(f"case {case_id}: clip_duration must be a positive number")
    labels = raw.get("labels", [])
    if isinstance(labels, (str, bytes)) or not isinstance(labels, Sequence):
        raise EvalError(f"case {case_id}: labels must be an array of strings")
    query_context = raw.get("query_context")
    if query_context is not None and not isinstance(query_context, str):
        raise EvalError(f"case {case_id}: query_context must be a string")
    expectations = raw.get("expectations")
    if not isinstance(expectations, Sequence) or not expectations:
        raise EvalError(f"case {case_id}: expectations must be a non-empty array")
    return {
        "case_id": case_id,
        "labels": tuple(str(label) for label in labels),
        "script": script,
        "clip_duration": float(clip_duration),
        "query_context": query_context,
        "expectations": tuple(
            _normalize_expectation(item, case_id) for item in expectations
        ),
    }


def load_eval_set(path: Path | str) -> tuple[dict[str, Any], ...]:
    """
    Load and validate the human-labelled eval set.

    @param path JSON eval-set path.
    @returns Normalized, immutable case tuple.
    @raises EvalError If the file, schema version, or any case is invalid.
    """
    resolved = Path(path)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvalError(f"cannot read eval set: {resolved}") from exc
    if not isinstance(payload, Mapping):
        raise EvalError("eval set must be an object")
    if payload.get("schema_version") != _SCHEMA_VERSION:
        raise EvalError(f"eval set schema_version must be {_SCHEMA_VERSION}")
    unknown_top_level = set(payload) - _TOP_LEVEL_FIELDS
    if unknown_top_level:
        raise EvalError(f"unknown eval set fields: {sorted(unknown_top_level)}")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, Sequence):
        raise EvalError("eval set cases must be an array")
    cases = tuple(_normalize_case(item) for item in raw_cases)
    seen: set[str] = set()
    for case in cases:
        if case["case_id"] in seen:
            raise EvalError(f"duplicate case_id: {case['case_id']}")
        seen.add(case["case_id"])
    return cases


def _shot_for_expectation(
    expectation: Mapping[str, Any],
    shots: Sequence[asset_matching.StoryboardShot],
    case_id: str,
) -> asset_matching.StoryboardShot | None:
    """按 text_contains 定位唯一分镜；歧义标注必须失败而不是随便挑一个。"""
    needle = expectation["text_contains"]
    matched = tuple(shot for shot in shots if needle in shot.text)
    if len(matched) > 1:
        raise EvalError(
            f"case {case_id}: text_contains {needle!r} matches {len(matched)} shots"
        )
    return matched[0] if matched else None


def score_case(
    case: Mapping[str, Any],
    match: asset_matching.StoryboardMatch,
) -> dict[str, Any]:
    """
    Score one case's match result against its human labels.

    @param case Eval case; validated here so the scorer never sees bad labels.
    @param match Storyboard match produced for that case's script.
    @returns Per-case counts used by `summarize_cases`.
    @raises EvalError If the case is invalid or a label is ambiguous.
    """
    normalized = _normalize_case(case)
    case_id = str(normalized["case_id"])
    shots = match.shots
    top1_hits = 0
    top3_hits = 0
    scored = 0
    expected_gaps = 0
    expected_gaps_honoured = 0
    false_gaps = 0
    forbidden_hits = 0
    unmatched: list[str] = []
    for expectation in normalized["expectations"]:
        shot = _shot_for_expectation(expectation, shots, case_id)
        if shot is None:
            unmatched.append(expectation["text_contains"])
            continue
        candidate_ids = tuple(asset.asset_id for asset in shot.candidates)
        forbidden = set(expectation["forbidden_asset_ids"])
        forbidden_hits += len(forbidden & set(candidate_ids))
        if expectation["expect_gap"]:
            expected_gaps += 1
            if not candidate_ids:
                expected_gaps_honoured += 1
            continue
        scored += 1
        acceptable = set(expectation["acceptable_asset_ids"])
        if not candidate_ids:
            false_gaps += 1
            continue
        if candidate_ids[0] in acceptable:
            top1_hits += 1
        if acceptable & set(candidate_ids):
            top3_hits += 1
    covered = "".join(shot.text for shot in shots)
    return {
        "case_id": case_id,
        "labels": tuple(normalized["labels"]),
        "shot_count": len(shots),
        "gap_shots": sum(1 for shot in shots if not shot.candidates),
        "scored_expectations": scored,
        "top1_hits": top1_hits,
        "top3_hits": top3_hits,
        "expected_gaps": expected_gaps,
        "expected_gaps_honoured": expected_gaps_honoured,
        "false_gaps": false_gaps,
        "forbidden_hits": forbidden_hits,
        "unmatched_expectations": tuple(unmatched),
        # 分镜必须完整覆盖原文，模型不得改写或漏掉旁白。
        "script_fully_covered": covered == str(normalized["script"]),
    }


def summarize_cases(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """
    Aggregate per-case counts into the rates the plan requires as a baseline.

    @param results Per-case results produced by `score_case`.
    @returns JSON-compatible summary with hit, gap, and forbidden rates.
    @raises EvalError If no expectation could be scored.
    """
    scored = sum(int(item["scored_expectations"]) for item in results)
    if scored <= 0:
        raise EvalError("eval run has no scored expectations")
    expected_gaps = sum(int(item["expected_gaps"]) for item in results)
    honoured = sum(int(item["expected_gaps_honoured"]) for item in results)
    return {
        "case_count": len(results),
        "shot_count": sum(int(item["shot_count"]) for item in results),
        "gap_shots": sum(int(item["gap_shots"]) for item in results),
        "scored_expectations": scored,
        "top1_hit_rate": sum(int(item["top1_hits"]) for item in results) / scored,
        "top3_available_rate": sum(int(item["top3_hits"]) for item in results) / scored,
        "false_gap_rate": sum(int(item["false_gaps"]) for item in results) / scored,
        "forbidden_hit_rate": sum(int(item["forbidden_hits"]) for item in results)
        / scored,
        "expected_gaps": expected_gaps,
        "expected_gap_honour_rate": (honoured / expected_gaps) if expected_gaps else 1.0,
        "unmatched_expectation_count": sum(
            len(item["unmatched_expectations"]) for item in results
        ),
        "cases_with_incomplete_script_coverage": tuple(
            str(item["case_id"])
            for item in results
            if not item["script_fully_covered"]
        ),
    }


def compare_to_baseline(
    summary: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Compare a summary against a recorded baseline on the tracked metrics.

    @param summary Current run summary.
    @param baseline Previously recorded summary.
    @returns Per-metric deltas and the sorted names that regressed.
    @raises EvalError If the baseline lacks a metric present in the summary.
    """
    deltas: dict[str, float] = {}
    regressions: list[str] = []
    for metric in _TRACKED_METRICS:
        if metric not in summary:
            continue
        if metric not in baseline:
            raise EvalError(f"baseline is missing metric: {metric}")
        delta = float(summary[metric]) - float(baseline[metric])
        deltas[metric] = delta
        if metric in _HIGHER_IS_BETTER and delta < 0:
            regressions.append(metric)
        if metric in _LOWER_IS_BETTER and delta > 0:
            regressions.append(metric)
    return {"deltas": deltas, "regressions": tuple(sorted(regressions))}


def run_eval(
    cases: Sequence[Mapping[str, Any]],
    *,
    db_path: Path | None = None,
    video_root: Path | str | None = None,
    bgm_root: Path | str | None = None,
) -> tuple[dict[str, Any], ...]:
    """
    Match every case against the real asset index and score it.

    @param cases Normalized eval cases.
    @param db_path Optional asset-library database path.
    @param video_root Optional video library root restricting candidates.
    @param bgm_root Optional BGM library root restricting candidates.
    @returns Per-case results in eval-set order.
    @raises EvalError If the asset index cannot serve a case.
    """
    normalized_cases = tuple(_normalize_case(case) for case in cases)
    indexed = {
        asset.asset_id
        for asset in asset_library.list_assets(kind="video", db_path=db_path)
    }
    # asset_id 含内容哈希，素材被重新编码后会变。此处显式报错，避免命中率
    # 静默掉到 0 却看不出是标注失效还是匹配变差。
    referenced = {
        asset_id
        for case in normalized_cases
        for expectation in case["expectations"]
        for asset_id in (
            *expectation["acceptable_asset_ids"],
            *expectation["forbidden_asset_ids"],
        )
    }
    stale = sorted(referenced - indexed)
    if stale:
        raise EvalError(f"eval set references assets missing from the index: {stale}")
    results: list[dict[str, Any]] = []
    for case in normalized_cases:
        try:
            match = asset_matching.match_storyboard(
                str(case["script"]),
                clip_duration=float(case["clip_duration"]),
                query_context=case.get("query_context"),
                db_path=db_path,
                video_root=video_root,
                bgm_root=bgm_root,
            )
        except asset_library.AssetLibraryError as exc:
            raise EvalError(f"case {case['case_id']}: matching failed: {exc}") from exc
        results.append(score_case(case, match))
    return tuple(results)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse eval arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-set", required=True, type=Path)
    parser.add_argument("--db-path", type=Path)
    parser.add_argument("--video-root", type=Path)
    parser.add_argument("--bgm-root", type=Path)
    parser.add_argument(
        "--baseline",
        type=Path,
        help="Previously recorded summary JSON to compare against.",
    )
    parser.add_argument(
        "--write-baseline",
        type=Path,
        help="Write this run's summary as the new baseline.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the eval set against the real index and print JSON metrics."""
    args = _parse_args(argv)
    try:
        cases = load_eval_set(args.eval_set)
        results = run_eval(
            cases,
            db_path=args.db_path,
            video_root=args.video_root,
            bgm_root=args.bgm_root,
        )
        summary = summarize_cases(results)
        report: dict[str, Any] = {"summary": summary, "cases": list(results)}
        if args.baseline is not None:
            try:
                baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise EvalError(f"cannot read baseline: {args.baseline}") from exc
            if not isinstance(baseline, Mapping):
                raise EvalError("baseline must be an object")
            report["comparison"] = compare_to_baseline(
                summary,
                baseline.get("summary", baseline),
            )
        if args.write_baseline is not None:
            args.write_baseline.write_text(
                json.dumps({"summary": summary}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    except EvalError as exc:
        print(f"eval failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    regressions = report.get("comparison", {}).get("regressions", ())
    return 1 if regressions else 0


if __name__ == "__main__":
    raise SystemExit(main())
