"""Build labelled contact sheets so eval-set labels can be checked against pixels.

Labels must come from what a window actually shows, not from a description string.
This renders every indexed window as a numbered tile in one image per asset, with
the window's source range and stored description printed beside it, so a reviewer
can confirm or correct the stored text without opening a video editor.

Composition uses Pillow rather than ffmpeg's `drawtext` filter: the ffmpeg build
used here has no `drawtext`, and a missing filter inside a filter chain fails
silently, which would produce unlabelled sheets that look fine.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

from PIL import Image, ImageDraw, ImageFont

from app.services import asset_library
from app.utils import utils


class ReviewError(RuntimeError):
    """Raised when a contact sheet cannot be rendered."""


_TILE_WIDTH = 340
_TILE_HEIGHT = 450
_LABEL_LINES = 3
_LINE_HEIGHT = 16
_LABEL_PADDING = 6
_LABEL_WIDTH_CHARS = 24
_FONT_SIZE = 13
_FRAME_QUALITY = 88
_FONT_CANDIDATES = (
    "resource/fonts/MicrosoftYaHeiNormal.ttc",
    "resource/fonts/MicrosoftYaHeiBold.ttc",
    "resource/fonts/STHeitiLight.ttc",
    "resource/fonts/STHeitiMedium.ttc",
)

_LABEL_HEIGHT = _LABEL_LINES * _LINE_HEIGHT + _LABEL_PADDING * 2


def _font_path() -> str | None:
    for candidate in _FONT_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
    return None


def _load_font(size: int) -> ImageFont.ImageFont:
    """加载支持中文的字体；缺失时退回 PIL 默认字体而不是让整张图失败。"""
    path = _font_path()
    if path is None:
        return ImageFont.load_default()
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def _wrap(text: str, width: int) -> list[str]:
    """按字符数硬折行；中文近似等宽，粗略折行足够定位。"""
    stripped = " ".join(text.split())
    return [
        stripped[index : index + width] for index in range(0, len(stripped), width)
    ] or [""]


def _fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """等比缩放并居中裁切成固定瓦片尺寸。"""
    width, height = size
    scale = max(width / image.width, height / image.height)
    resized = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        getattr(Image, "LANCZOS", None) or Image.Resampling.LANCZOS,
    )
    left = (resized.width - width) // 2
    top = (resized.height - height) // 2
    return resized.crop((left, top, left + width, top + height))


def _extract_tile(source: Path, position: float) -> Image.Image:
    """抽取指定时间点的一帧并转成瓦片。"""
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / "frame.jpg"
        command = [
            utils.get_ffmpeg_binary(),
            "-nostdin",
            "-v",
            "error",
            "-ss",
            f"{position:.3f}",
            "-i",
            str(source),
            "-frames:v",
            "1",
            "-y",
            str(target),
        ]
        completed = subprocess.run(command, capture_output=True, check=False)
        if completed.returncode != 0 or not target.is_file():
            detail = completed.stderr.decode("utf-8", errors="replace")[:200]
            raise ReviewError(
                f"cannot extract window frame at {position:.1f}s: {detail}"
            )
        with Image.open(target) as opened:
            return _fit(opened.convert("RGB"), (_TILE_WIDTH, _TILE_HEIGHT))


def render_asset_sheet(
    asset: asset_library.LibraryAsset,
    segments: Sequence[asset_library.LibrarySegment],
    output_path: Path,
) -> Path:
    """
    Render one contact sheet with a numbered tile per window.

    @param asset Indexed video asset.
    @param segments That asset's windows, in source order.
    @param output_path Destination JPEG path.
    @returns The written path.
    @raises ReviewError If the asset path is unavailable or a frame cannot be read.
    """
    if not segments:
        raise ReviewError(f"asset has no indexed windows: {asset.asset_id}")
    try:
        source = asset_library.resolve_asset_path(asset)
    except asset_library.AssetLibraryError as exc:
        raise ReviewError(str(exc)) from exc
    ordered = sorted(segments, key=lambda item: item.source_start_seconds)
    columns = min(3, len(ordered))
    rows = (len(ordered) + columns - 1) // columns
    row_height = _TILE_HEIGHT + _LABEL_HEIGHT
    sheet = Image.new("RGB", (columns * _TILE_WIDTH, rows * row_height), "white")
    draw = ImageDraw.Draw(sheet)
    font = _load_font(_FONT_SIZE)
    for index, segment in enumerate(ordered):
        column = index % columns
        row = index // columns
        left = column * _TILE_WIDTH
        top = row * row_height
        midpoint = (segment.source_start_seconds + segment.source_end_seconds) / 2
        sheet.paste(_extract_tile(source, midpoint), (left, top))
        lines = [
            f"#{index + 1}  {segment.source_start_seconds:.1f}-"
            f"{segment.source_end_seconds:.1f}s",
            *_wrap(segment.description, _LABEL_WIDTH_CHARS)[:_LABEL_LINES - 1],
        ]
        for line_index, line in enumerate(lines):
            draw.text(
                (left + _LABEL_PADDING, top + _TILE_HEIGHT + _LABEL_PADDING
                 + line_index * _LINE_HEIGHT),
                line,
                fill="black",
                font=font,
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="JPEG", quality=_FRAME_QUALITY)
    return output_path


def render_library_sheets(
    *,
    db_path: Path,
    output_dir: Path,
    asset_ids: Sequence[str] | None = None,
) -> tuple[dict[str, Any], ...]:
    """
    Render one contact sheet per indexed video.

    @param db_path Asset-library database path.
    @param output_dir Directory the sheets are written into.
    @param asset_ids Optional subset of assets to render.
    @returns Per-asset record with the sheet path and its windows.
    @raises ReviewError If no asset can be rendered.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    assets = asset_library.list_assets(kind="video", db_path=db_path)
    if asset_ids is not None:
        wanted = set(asset_ids)
        assets = tuple(asset for asset in assets if asset.asset_id in wanted)
    segments = asset_library.list_segments(db_path=db_path)
    by_asset: dict[str, list[asset_library.LibrarySegment]] = {}
    for segment in segments:
        by_asset.setdefault(segment.asset_id, []).append(segment)
    sheets: list[dict[str, Any]] = []
    for asset in assets:
        rows = by_asset.get(asset.asset_id, [])
        if not rows:
            continue
        target = output_dir / f"{asset.asset_id}.jpg"
        render_asset_sheet(asset, rows, target)
        sheets.append(
            {
                "asset_id": asset.asset_id,
                "relative_path": asset.relative_path,
                "duration": asset.duration,
                "sheet": str(target),
                "windows": [
                    {
                        "index": index,
                        "source_start_seconds": item.source_start_seconds,
                        "source_end_seconds": item.source_end_seconds,
                        "description": item.description,
                        "tags": list(item.tags),
                    }
                    for index, item in enumerate(
                        sorted(rows, key=lambda value: value.source_start_seconds),
                        start=1,
                    )
                ],
            }
        )
    if not sheets:
        raise ReviewError("no asset windows to render")
    return tuple(sheets)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse contact-sheet arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", type=Path, default=asset_library.default_db_path())
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--asset-id", action="append", dest="asset_ids")
    parser.add_argument("--manifest", type=Path, help="Write the sheet index as JSON.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Render contact sheets and print where they were written."""
    args = _parse_args(argv)
    try:
        sheets = render_library_sheets(
            db_path=args.db_path,
            output_dir=args.output_dir,
            asset_ids=tuple(args.asset_ids) if args.asset_ids else None,
        )
    except ReviewError as exc:
        print(f"review sheets failed: {exc}", file=sys.stderr)
        return 2
    if args.manifest is not None:
        args.manifest.write_text(
            json.dumps(list(sheets), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    for sheet in sheets:
        print(f"{sheet['asset_id']}  {sheet['relative_path']}  ->  {sheet['sheet']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
