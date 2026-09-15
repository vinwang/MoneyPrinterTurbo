"""One-file command-line entry point for the MoviePy advertising post-processor."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

from scripts.post_process import (
    ConfigValidationError,
    compose_video,
    load_post_process_spec,
)
from scripts.run_pipeline import (
    PipelineError,
    ffprobe_duration,
    publish_without_overwrite,
    validate_output,
)


def process_file(
    raw_video_path: Path,
    spec_path: Path,
    output_path: Path,
    *,
    resource_root: Path,
    ffprobe: str = "ffprobe",
) -> None:
    """
    Post-process one MPT video and publish it without overwriting an existing file.

    @param raw_video_path MPT-produced input video.
    @param spec_path Per-task post-processing JSON file with numeric filename.
    @param output_path Final output path; it must not already exist.
    @param resource_root Approved root for the Logo and font files.
    @param ffprobe Executable used to read duration before and after rendering.
    @returns None after atomic publication and output validation.
    @raises ConfigValidationError or PipelineError If validation/rendering fails.
    """
    duration = ffprobe_duration(raw_video_path, ffprobe=ffprobe)
    try:
        task_index = int(spec_path.stem)
    except ValueError as exc:
        raise ConfigValidationError(
            "direct post-processing requires a numeric spec filename"
        ) from exc
    spec = load_post_process_spec(
        spec_path,
        task_index=task_index,
        video_duration=duration,
        resource_root=resource_root,
    )
    if output_path.exists():
        raise PipelineError(f"refusing to overwrite existing output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp.mp4")
    try:
        compose_video(raw_video_path, spec, temporary)
        ffprobe_duration(temporary, ffprobe=ffprobe)
        validate_output(
            temporary,
            output_spec=spec.output,
            expected_duration=duration,
            ffprobe=ffprobe,
        )
        publish_without_overwrite(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse one-file post-processing command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resource-root", type=Path, required=True)
    parser.add_argument("--ffprobe", default="ffprobe")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """
    Run one-file post-processing from the command line.

    @param argv Optional argument sequence; defaults to `sys.argv[1:]`.
    @returns 0 after success, otherwise 2 with an explicit error on stderr.
    """
    args = _parse_args(argv)
    try:
        process_file(
            args.raw,
            args.spec,
            args.output,
            resource_root=args.resource_root,
            ffprobe=args.ffprobe,
        )
    except (ConfigValidationError, OSError, PipelineError) as exc:
        print(f"post-processing failed: {exc}", file=sys.stderr)
        return 2
    print(str(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
