"""Measure asset-index and matching cost at a target library size.

The benchmark drives the real SQLite index, the real segment builder, and the
real matcher. Only the vision model is replaced by a deterministic stand-in, so
the reported `vision_calls` is the call volume a real run would spend while no
frames leave the machine. Model latency and model cost are therefore NOT part of
these numbers and must not be presented as measured.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.services import asset_library, asset_matching
from app.utils import utils
from scripts.performance import PerformanceError, _peak_rss_mb


class ScaleError(RuntimeError):
    """Raised when a scale benchmark cannot produce a trustworthy measurement."""


_CATEGORY_NAMES = (
    "玉米",
    "美妆",
    "服饰",
    "数码",
    "家居",
    "食品",
    "母婴",
    "运动",
)
_DESCRIPTION_TEMPLATES = (
    "金黄玉米在田间随风摆动，颗粒饱满。",
    "真空包装玉米摆拍，标注包邮到家。",
    "掰开玉米展示颗粒，晶莹软糯。",
    "水煮玉米搭配鸡蛋黄瓜的健康餐摆盘。",
    "促销价格与购物车截图，突出低价。",
    "口红小样与护肤水的开箱特写。",
)
_TAG_POOL = (
    "玉米",
    "农田",
    "美食",
    "真空包装",
    "促销",
    "优惠券",
    "美妆",
    "健康轻食",
)
_DEFAULT_QUERY_SCRIPTS = (
    "金黄玉米在田间随风摆动。现在下单立减。",
    "皮薄粒满，软糯香甜。掰开来看，颗粒晶莹。",
    "忙碌了一整天，也该好好对自己好一点。",
)
_SYNTHETIC_DURATION_SECONDS = 12.0
_SYNTHETIC_FRAME_RATE = 6
_SYNTHETIC_SIZE = "64x64"


def _synthetic_payload(root: Path, index: int, category: str) -> Path:
    """按索引确定性地生成一个极小 mp4，避免把真实素材写进仓库。"""
    directory = root / category
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"synthetic-{index:06d}.mp4"
    if path.exists():
        return path
    # 颜色由索引决定，使内容哈希稳定且各素材互不相同。
    digest = hashlib.sha256(f"{category}:{index}".encode("utf-8")).hexdigest()
    color = f"0x{digest[:6]}"
    command = [
        utils.get_ffmpeg_binary(),
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c={color}:s={_SYNTHETIC_SIZE}:"
        f"d={_SYNTHETIC_DURATION_SECONDS:.0f}:r={_SYNTHETIC_FRAME_RATE}",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, check=False)
    except OSError as exc:
        raise ScaleError(f"cannot generate synthetic asset: {path}") from exc
    if completed.returncode != 0 or not path.is_file():
        raise ScaleError(f"cannot generate synthetic asset: {path}")
    return path


def build_synthetic_library(
    root: Path,
    *,
    asset_count: int,
    category_count: int,
) -> tuple[Path, ...]:
    """
    Create a deterministic synthetic video library on disk.

    @param root Directory the synthetic library is written into.
    @param asset_count Number of video files to create.
    @param category_count Number of sub-directories to spread them across.
    @returns Created file paths in creation order.
    @raises ScaleError If the counts are invalid or ffmpeg cannot write a file.
    """
    if not isinstance(asset_count, int) or asset_count <= 0:
        raise ScaleError("asset_count must be a positive integer")
    if not isinstance(category_count, int) or category_count <= 0:
        raise ScaleError("category_count must be a positive integer")
    if category_count > len(_CATEGORY_NAMES):
        raise ScaleError(f"category_count cannot exceed {len(_CATEGORY_NAMES)}")
    root.mkdir(parents=True, exist_ok=True)
    return tuple(
        _synthetic_payload(root, index, _CATEGORY_NAMES[index % category_count])
        for index in range(asset_count)
    )


def _deterministic_analysis(description: str) -> tuple[Any, list[Path]]:
    """构造确定性的视觉分析替身，返回与真实模型同格式的逐窗口 JSON 文本。"""
    calls: list[Path] = []

    def vision_fn(path: Path, frames: Sequence[bytes]) -> str:
        if not frames:
            raise ScaleError(f"no preview frames extracted for {path}")
        calls.append(path)
        if not description:
            # 空描述用于验证「分析结果不可用时必须显式失败」。
            return ""
        windows = []
        for index in range(len(frames)):
            seed = int(
                hashlib.sha256(f"{path}:{index}".encode("utf-8")).hexdigest()[:8], 16
            )
            text = _DESCRIPTION_TEMPLATES[seed % len(_DESCRIPTION_TEMPLATES)]
            windows.append(
                {
                    "index": index + 1,
                    "description": f"{description}{text}",
                    "tags": sorted(
                        {
                            _TAG_POOL[(seed + offset) % len(_TAG_POOL)]
                            for offset in range(3)
                        }
                    ),
                    "mood": "中性",
                }
            )
        return json.dumps({"windows": windows}, ensure_ascii=False)

    return vision_fn, calls


def _scan_metrics(summary: asset_library.ScanSummary, vision_calls: int, elapsed: float):
    return {
        "elapsed_seconds": elapsed,
        "added": summary.added,
        "updated": summary.updated,
        "unchanged": summary.unchanged,
        "analyzed": summary.analyzed,
        "failed": summary.failed,
        "missing": summary.missing,
        "vision_calls": vision_calls,
        "errors": list(summary.errors[:5]),
    }


def measure_scale(
    library_root: Path,
    *,
    db_path: Path,
    asset_count: int,
    category_count: int,
    query_scripts: Sequence[str],
    clip_duration: float,
    rebuild_media: bool = True,
    vision_description: str = "合成素材：",
) -> dict[str, Any]:
    """
    Index a synthetic library twice and time matching queries against it.

    @param library_root Directory holding the synthetic video library.
    @param db_path SQLite index path used for the benchmark.
    @param asset_count Number of synthetic videos to index.
    @param category_count Number of sub-directories to spread them across.
    @param query_scripts Scripts to match once the index is warm.
    @param clip_duration Target shot duration passed to the matcher.
    @param rebuild_media Whether to (re)create the synthetic media files.
    @param vision_description Prefix for the stand-in analyser's description.
    @returns Raw per-phase measurements consumed by `summarize_scale`.
    @raises ScaleError If indexing produces no usable assets or segments.
    """
    if rebuild_media:
        build_synthetic_library(
            library_root,
            asset_count=asset_count,
            category_count=category_count,
        )
    cold_vision_fn, cold_calls = _deterministic_analysis(vision_description)
    started = time.perf_counter()
    cold_summary = asset_library.scan_library(
        library_root,
        db_path=db_path,
        vision_fn=cold_vision_fn,
        app_config={},
    )
    cold_elapsed = time.perf_counter() - started

    warm_vision_fn, warm_calls = _deterministic_analysis(vision_description)
    started = time.perf_counter()
    warm_summary = asset_library.scan_library(
        library_root,
        db_path=db_path,
        vision_fn=warm_vision_fn,
        app_config={},
    )
    warm_elapsed = time.perf_counter() - started

    ready_assets = asset_library.list_assets(
        kind="video",
        analysis_status="ready",
        db_path=db_path,
    )
    if not ready_assets:
        raise ScaleError("scale run indexed no analyzable video assets")
    segments = asset_library.list_segments(
        analysis_status="ready",
        db_path=db_path,
    )
    if not segments:
        raise ScaleError("scale run produced no ready segments")

    queries: list[dict[str, Any]] = []
    for script in query_scripts:
        started = time.perf_counter()
        try:
            match = asset_matching.match_storyboard(
                script,
                clip_duration=clip_duration,
                db_path=db_path,
                video_root=library_root,
                bgm_root="",
            )
        except asset_library.AssetLibraryError as exc:
            raise ScaleError(f"matching failed at scale: {exc}") from exc
        queries.append(
            {
                "script": script,
                "elapsed_seconds": time.perf_counter() - started,
                "shot_count": len(match.shots),
                "gap_shots": sum(1 for shot in match.shots if not shot.candidates),
            }
        )
    try:
        peak_rss_mb = _peak_rss_mb()
    except PerformanceError:
        peak_rss_mb = None
    return {
        "asset_count": len(ready_assets),
        "segment_count": len(segments),
        "peak_rss_mb": peak_rss_mb,
        "cold_scan": _scan_metrics(cold_summary, len(cold_calls), cold_elapsed),
        "warm_scan": _scan_metrics(warm_summary, len(warm_calls), warm_elapsed),
        "queries": queries,
    }


def summarize_scale(measurement: Mapping[str, Any]) -> dict[str, Any]:
    """
    Derive throughput figures from one scale measurement.

    @param measurement Raw measurement returned by `measure_scale`.
    @returns JSON-compatible throughput, call-volume, and latency summary.
    @raises ScaleError If a phase reports non-positive time or re-analyzes everything.
    """
    asset_count = int(measurement["asset_count"])
    cold = measurement["cold_scan"]
    warm = measurement["warm_scan"]
    cold_elapsed = float(cold["elapsed_seconds"])
    warm_elapsed = float(warm["elapsed_seconds"])
    if cold_elapsed <= 0 or warm_elapsed <= 0:
        raise ScaleError("scan elapsed_seconds must be positive")
    cold_calls = int(cold["vision_calls"])
    warm_calls = int(warm["vision_calls"])
    if warm_calls >= cold_calls and cold_calls > 0:
        raise ScaleError("unchanged media was re-analyzed on the second scan")
    query_seconds = [float(item["elapsed_seconds"]) for item in measurement["queries"]]
    return {
        "asset_count": asset_count,
        "segment_count": int(measurement["segment_count"]),
        "peak_rss_mb": measurement["peak_rss_mb"],
        "cold_scan_seconds": cold_elapsed,
        "rescan_seconds": warm_elapsed,
        "assets_indexed_per_second": asset_count / cold_elapsed,
        "rescan_assets_per_second": asset_count / warm_elapsed,
        "vision_calls": cold_calls,
        "vision_calls_per_asset": (cold_calls / asset_count) if asset_count else 0.0,
        "rescan_vision_calls": warm_calls,
        "query_count": len(query_seconds),
        "max_query_seconds": max(query_seconds, default=0.0),
        "mean_query_seconds": (
            sum(query_seconds) / len(query_seconds) if query_seconds else 0.0
        ),
        # 视觉模型用确定性替身，这里的耗时不含真实模型延迟与费用。
        "vision_model": "deterministic-stand-in",
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse scale-benchmark arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library-root", required=True, type=Path)
    parser.add_argument("--db-path", required=True, type=Path)
    parser.add_argument("--asset-count", type=int, default=10000)
    parser.add_argument("--category-count", type=int, default=8)
    parser.add_argument("--clip-duration", type=float, default=3.0)
    parser.add_argument(
        "--query-script",
        action="append",
        dest="query_scripts",
        help="Script to time against the warm index; repeatable.",
    )
    parser.add_argument(
        "--keep-media",
        action="store_true",
        help="Reuse existing synthetic media instead of regenerating it.",
    )
    parser.add_argument("--write-baseline", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one scale benchmark and print JSON metrics."""
    args = _parse_args(argv)
    try:
        measurement = measure_scale(
            args.library_root,
            db_path=args.db_path,
            asset_count=args.asset_count,
            category_count=args.category_count,
            query_scripts=tuple(args.query_scripts or _DEFAULT_QUERY_SCRIPTS),
            clip_duration=args.clip_duration,
            rebuild_media=not args.keep_media,
        )
        summary = summarize_scale(measurement)
        if args.write_baseline is not None:
            args.write_baseline.write_text(
                json.dumps({"summary": summary}, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    except (ScaleError, asset_library.AssetLibraryError) as exc:
        print(f"scale benchmark failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"summary": summary, "measurement": measurement}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
