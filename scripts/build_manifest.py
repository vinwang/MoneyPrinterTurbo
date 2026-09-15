"""Build a deterministic, metadata-rich manifest for local video and audio assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.run_pipeline import PipelineError, sha256_file


class ManifestError(RuntimeError):
    """Raised when a media manifest cannot be built from the supplied asset roots."""


_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
_VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".mkv",
    ".webm",
    ".avi",
    ".flv",
    *_IMAGE_EXTENSIONS,
}
_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}


def _probe_media(path: Path, *, ffprobe: str = "ffprobe") -> Mapping[str, Any]:
    """
    Read duration and stream dimensions for one media file using ffprobe.

    @param path Media file to inspect.
    @param ffprobe ffprobe executable name or approved path.
    @returns Mapping containing positive duration and optional width/height.
    @raises ManifestError If ffprobe fails or returns malformed metadata.
    """
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
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise ManifestError("failed to start ffprobe") from exc
    if completed.returncode != 0:
        raise ManifestError(f"ffprobe failed for asset: {path}")
    try:
        payload = json.loads(completed.stdout)
        format_data = payload.get("format", {})
        if not isinstance(format_data, Mapping):
            raise TypeError("format metadata must be an object")
        duration_value = format_data.get("duration")
        if duration_value is None and path.suffix.lower() in _IMAGE_EXTENSIONS:
            duration = 0.0
        else:
            duration = float(duration_value)
        streams = payload.get("streams", [])
        if not isinstance(streams, list):
            raise TypeError("streams metadata must be an array")
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ManifestError(f"ffprobe returned invalid metadata for: {path}") from exc
    if not math.isfinite(duration) or duration < 0:
        raise ManifestError(f"asset duration must be non-negative: {path}")
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
    if (width is None) != (height is None):
        raise ManifestError(f"asset dimensions are incomplete: {path}")
    if width is not None and (
        not isinstance(width, int)
        or not isinstance(height, int)
        or width <= 0
        or height <= 0
    ):
        raise ManifestError(f"asset dimensions are invalid: {path}")
    return {"duration": duration, "width": width, "height": height}


def _asset_id(kind: str, relative_path: str, content_hash: str) -> str:
    """Create a deterministic ID that remains unique for same-content sibling files."""
    identity = f"{kind}:{relative_path}:{content_hash}".encode("utf-8")
    return f"{kind}-{hashlib.sha256(identity).hexdigest()[:16]}"


def _iter_assets(root: Path, extensions: set[str]) -> tuple[Path, ...]:
    """Find supported regular files below a root in stable relative-path order."""
    if not root.is_dir():
        raise ManifestError(f"asset root is not a directory: {root}")
    resolved_root = root.resolve()
    paths: list[Path] = []
    for candidate in root.rglob("*"):
        if candidate.suffix.lower() not in extensions or not candidate.is_file():
            continue
        resolved = candidate.resolve()
        try:
            resolved.relative_to(resolved_root)
        except ValueError as exc:
            raise ManifestError(f"asset symlink escapes root: {candidate}") from exc
        paths.append(resolved)
    return tuple(
        sorted(paths, key=lambda path: path.relative_to(resolved_root).as_posix())
    )


def scan_media(
    root: Path,
    *,
    kind: str,
    ffprobe: str = "ffprobe",
) -> tuple[dict[str, Any], ...]:
    """
    Scan one media root and return deterministic records with identity metadata.

    @param root Root directory containing media files.
    @param kind Either `video` or `bgm`.
    @param ffprobe ffprobe executable used to inspect each file.
    @returns Sorted immutable tuple of manifest records with relative paths.
    @raises ManifestError If kind, paths, metadata, or file hashing is invalid.
    """
    if kind not in {"video", "bgm"}:
        raise ManifestError("kind must be video or bgm")
    resolved_root = root.resolve()
    extensions = _VIDEO_EXTENSIONS if kind == "video" else _AUDIO_EXTENSIONS
    records: list[dict[str, Any]] = []
    for path in _iter_assets(resolved_root, extensions):
        relative_path = path.relative_to(resolved_root).as_posix()
        try:
            content_hash = sha256_file(path)
            metadata = _probe_media(path, ffprobe=ffprobe)
            if kind == "bgm" and metadata["duration"] <= 0:
                raise ManifestError(f"BGM duration must be positive: {path}")
            size_bytes = path.stat().st_size
        except (OSError, PipelineError, ManifestError) as exc:
            if isinstance(exc, ManifestError):
                raise
            raise ManifestError(f"cannot read asset metadata: {path}") from exc
        records.append(
            {
                "id": _asset_id(kind, relative_path, content_hash),
                "path": relative_path,
                "media_type": kind,
                "category": Path(relative_path).parent.as_posix()
                if Path(relative_path).parent.as_posix() != "."
                else "uncategorized",
                "duration": metadata["duration"],
                "width": metadata["width"],
                "height": metadata["height"],
                "size_bytes": size_bytes,
                "sha256": content_hash,
            }
        )
    return tuple(records)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a manifest through a sibling temporary file and atomic replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        raise ManifestError(f"cannot write manifest: {path}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def build_manifest(
    video_root: Path,
    bgm_root: Path,
    output_path: Path,
    *,
    ffprobe: str = "ffprobe",
) -> dict[str, Any]:
    """
    Scan video and BGM roots and persist one versioned manifest.

    @param video_root Local video-material root.
    @param bgm_root Local music root.
    @param output_path Destination JSON manifest path.
    @param ffprobe ffprobe executable used for media metadata.
    @returns JSON-compatible manifest dictionary.
    @raises ManifestError If either scan or atomic write fails.
    """
    videos = scan_media(video_root, kind="video", ffprobe=ffprobe)
    bgm = scan_media(bgm_root, kind="bgm", ffprobe=ffprobe)
    manifest = {
        "schema_version": 1,
        "videos_root": str(video_root.resolve()),
        "bgm_root": str(bgm_root.resolve()),
        "videos": list(videos),
        "bgm": list(bgm),
    }
    _write_json_atomic(output_path, manifest)
    return manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse build-manifest command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--bgm-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ffprobe", default="ffprobe")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Build a manifest from CLI arguments and return a process exit status."""
    args = _parse_args(argv)
    try:
        build_manifest(
            args.video_root, args.bgm_root, args.output, ffprobe=args.ffprobe
        )
    except ManifestError as exc:
        print(f"manifest failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
