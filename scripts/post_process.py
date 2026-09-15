"""MoviePy 2.2.1 post-processing primitives for advertising videos."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from moviepy import CompositeVideoClip, ImageClip, TextClip, VideoFileClip
from PIL import Image, ImageFont


class ConfigValidationError(ValueError):
    """Raised when a post-processing specification is invalid or unsafe."""


_POSITIONS = {"top_left", "top_right", "bottom_left", "bottom_right", "custom"}
_ANIMATIONS = {"none", "pop_spring", "letter_by_letter", "flicker_scale"}
_FIT_MODES = {"contain", "cover", "reject"}
_ASPECT_TOLERANCE = 1e-6
_SPRING_MIN_SCALE = 0.05
_SPRING_MAX_SCALE = 1.35
_FLICKER_AMPLITUDE = 0.6
_FONT_EXTENSIONS = {".ttf", ".ttc", ".otf"}
_HEX_COLOR = re.compile(r"^#[0-9A-Fa-f]{6}$")
_SPEC_KEYS = {
    "schema_version",
    "task_index",
    "output",
    "watermark",
    "custom_texts",
    "image_layers",
}
_OUTPUT_KEYS = {
    "profile_id",
    "width",
    "height",
    "fps",
    "fit_mode",
    "background_color",
    "video_codec",
    "pixel_format",
    "video_bitrate",
    "audio_required",
    "audio_codec",
    "audio_bitrate",
    "faststart",
}
_WATERMARK_KEYS = {
    "enabled",
    "logo_path",
    "trim_transparent",
    "position",
    "custom_xy_ratio",
    "margin_px",
    "opacity",
    "height_ratio",
}
_TEXT_KEYS = {
    "text",
    "start",
    "end",
    "font_name",
    "font_size_px",
    "color",
    "stroke_color",
    "stroke_width_px",
    "x_ratio",
    "y_ratio",
    "animation",
    "stagger",
    "pop_duration",
    "flicker_duration",
    "flicker_hz",
    "scale_from",
}
_IMAGE_LAYER_KEYS = {
    "path",
    "start",
    "end",
    "x_ratio",
    "y_ratio",
    "width_ratio",
    "height_ratio",
    "fit_mode",
    "opacity",
}
# 图片图层只做等比适配；output 的 reject 模式针对整片画幅，不适用于单个图层。
_IMAGE_FIT_MODES = {"contain", "cover"}
_MAX_IMAGE_BYTES = 20 * 1024 * 1024
_MAX_IMAGE_PIXELS = 16_000_000


@dataclass(frozen=True)
class OutputSpec:
    """Validated target encoding and canvas configuration."""

    profile_id: str
    width: int
    height: int
    fps: float
    fit_mode: str
    background_color: str
    video_codec: str
    pixel_format: str
    video_bitrate: str
    audio_required: bool
    audio_codec: str
    audio_bitrate: str
    faststart: bool


@dataclass(frozen=True)
class WatermarkSpec:
    """Validated Logo overlay configuration."""

    enabled: bool
    logo_path: Path | None
    trim_transparent: bool
    position: str
    custom_xy_ratio: tuple[float, float] | None
    margin_px: int
    opacity: float
    height_ratio: float


@dataclass(frozen=True)
class TextSpec:
    """Validated independent text layer configuration."""

    text: str
    start: float
    end: float
    font_path: Path
    font_size_px: int
    color: str
    stroke_color: str | None
    stroke_width_px: float
    x_ratio: float
    y_ratio: float
    animation: str
    stagger: float
    pop_duration: float
    flicker_duration: float
    flicker_hz: float
    scale_from: float


@dataclass(frozen=True)
class ImageLayerSpec:
    """Validated image overlay such as a product screenshot or offer card."""

    path: Path
    start: float
    end: float
    x_ratio: float
    y_ratio: float
    width_ratio: float
    height_ratio: float
    fit_mode: str
    opacity: float


@dataclass(frozen=True)
class PostProcessSpec:
    """Validated single-task post-processing configuration."""

    schema_version: int
    task_index: int
    output: OutputSpec
    watermark: WatermarkSpec
    custom_texts: tuple[TextSpec, ...]
    image_layers: tuple[ImageLayerSpec, ...] = ()


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    """Require a JSON object for a named configuration section."""
    if not isinstance(value, Mapping):
        raise ConfigValidationError(f"{name} must be an object")
    return value


def _keys(value: Mapping[str, Any], allowed: set[str], *, name: str) -> None:
    """Reject unknown configuration fields instead of silently ignoring them."""
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ConfigValidationError(
            f"{name} contains unknown fields: {', '.join(unknown)}"
        )


def _finite_number(value: Any, *, name: str, minimum: float | None = None) -> float:
    """Validate a finite numeric field and optional lower bound."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ConfigValidationError(f"{name} must be a finite number")
    number = float(value)
    if minimum is not None and number < minimum:
        raise ConfigValidationError(f"{name} must be >= {minimum}")
    return number


def _integer(value: Any, *, name: str, minimum: int = 0) -> int:
    """Validate an integer field without accepting booleans or fractional values."""
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigValidationError(f"{name} must be an integer >= {minimum}")
    return value


def _string(value: Any, *, name: str, non_empty: bool = True) -> str:
    """Validate a string field and optionally require non-empty content."""
    if not isinstance(value, str) or (non_empty and not value.strip()):
        raise ConfigValidationError(
            f"{name} must be a {'non-empty ' if non_empty else ''}string"
        )
    return value


def resolve_resource_path(
    resource_root: Path, raw_path: str, *, description: str
) -> Path:
    """
    Resolve a resource below an approved root without allowing traversal.

    @param resource_root Directory approved for this resource type.
    @param raw_path Relative resource path supplied by the normalized spec.
    @param description Human-readable resource name used in errors.
    @returns Existing regular-file path inside `resource_root`.
    @raises ConfigValidationError If the path escapes the root or is not a file.
    """
    if not isinstance(resource_root, Path):
        raise ConfigValidationError("resource_root must be a Path")
    if not isinstance(raw_path, str) or not raw_path.strip() or "\x00" in raw_path:
        raise ConfigValidationError(f"{description} path must be a non-empty string")
    root = resource_root.resolve()
    candidate = (root / raw_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ConfigValidationError(
            f"{description} path is outside the approved root"
        ) from exc
    if not candidate.is_file():
        raise ConfigValidationError(f"{description} file does not exist")
    return candidate


def _color(value: Any, *, name: str, allow_none: bool = False) -> str | None:
    """Validate a six-digit hexadecimal RGB color."""
    if value is None and allow_none:
        return None
    color = _string(value, name=name)
    if not _HEX_COLOR.fullmatch(color):
        raise ConfigValidationError(f"{name} must use #RRGGBB format")
    return color.upper()


def _rgb_color(value: str) -> tuple[int, int, int]:
    """Convert a validated hexadecimal color into an RGB tuple for MoviePy."""
    return tuple(int(value[index : index + 2], 16) for index in (1, 3, 5))


def _ratio(value: Any, *, name: str) -> float:
    """Validate a single ratio in the inclusive 0..1 range."""
    number = _finite_number(value, name=name)
    if not 0 <= number <= 1:
        raise ConfigValidationError(f"{name} must be between 0 and 1")
    return number


def _ratio_pair(
    value: Any, *, name: str, allow_none: bool = False
) -> tuple[float, float] | None:
    """Validate a two-dimensional ratio coordinate in the inclusive 0..1 range."""
    if value is None and allow_none:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ConfigValidationError(f"{name} must contain exactly two ratios")
    values = tuple(
        _finite_number(item, name=f"{name}[{index}]")
        for index, item in enumerate(value)
    )
    if not all(0 <= item <= 1 for item in values):
        raise ConfigValidationError(f"{name} values must be between 0 and 1")
    return values


def _output_spec(raw: Any) -> OutputSpec:
    """Validate and normalize target canvas and encoder settings."""
    data = _mapping(raw, name="output")
    _keys(data, _OUTPUT_KEYS, name="output")
    profile_id = _string(data.get("profile_id"), name="output.profile_id")
    width = _integer(data.get("width"), name="output.width", minimum=1)
    height = _integer(data.get("height"), name="output.height", minimum=1)
    fps = _finite_number(data.get("fps"), name="output.fps", minimum=1)
    fit_mode = _string(data.get("fit_mode"), name="output.fit_mode")
    if fit_mode not in _FIT_MODES:
        raise ConfigValidationError(
            f"output.fit_mode must be one of {sorted(_FIT_MODES)}"
        )
    background_color = _color(
        data.get("background_color"), name="output.background_color"
    )
    video_codec = _string(data.get("video_codec"), name="output.video_codec")
    pixel_format = _string(data.get("pixel_format"), name="output.pixel_format")
    video_bitrate = _string(data.get("video_bitrate"), name="output.video_bitrate")
    audio_required = data.get("audio_required")
    if not isinstance(audio_required, bool):
        raise ConfigValidationError("output.audio_required must be boolean")
    audio_codec = _string(data.get("audio_codec"), name="output.audio_codec")
    audio_bitrate = _string(data.get("audio_bitrate"), name="output.audio_bitrate")
    faststart = data.get("faststart")
    if not isinstance(faststart, bool):
        raise ConfigValidationError("output.faststart must be boolean")
    return OutputSpec(
        profile_id,
        width,
        height,
        fps,
        fit_mode,
        background_color,
        video_codec,
        pixel_format,
        video_bitrate,
        audio_required,
        audio_codec,
        audio_bitrate,
        faststart,
    )


def _watermark_spec(raw: Any, *, resource_root: Path | None) -> WatermarkSpec:
    """Validate watermark configuration and resolve its optional Logo file."""
    data = _mapping(raw, name="watermark")
    _keys(data, _WATERMARK_KEYS, name="watermark")
    enabled = data.get("enabled")
    if not isinstance(enabled, bool):
        raise ConfigValidationError("watermark.enabled must be boolean")
    logo_path = None
    if enabled:
        if resource_root is None:
            raise ConfigValidationError(
                "resource_root is required when watermark is enabled"
            )
        logo_path = resolve_resource_path(
            resource_root,
            _string(data.get("logo_path"), name="watermark.logo_path"),
            description="logo",
        )
    trim_transparent = data.get("trim_transparent", True)
    if not isinstance(trim_transparent, bool):
        raise ConfigValidationError("watermark.trim_transparent must be boolean")
    position = _string(data.get("position"), name="watermark.position")
    if position not in _POSITIONS:
        raise ConfigValidationError(
            f"watermark.position must be one of {sorted(_POSITIONS)}"
        )
    custom_xy_ratio = _ratio_pair(
        data.get("custom_xy_ratio"), name="watermark.custom_xy_ratio", allow_none=True
    )
    if position == "custom" and custom_xy_ratio is None:
        raise ConfigValidationError(
            "watermark.custom_xy_ratio is required for custom position"
        )
    if position != "custom" and custom_xy_ratio is not None:
        raise ConfigValidationError(
            "watermark.custom_xy_ratio is only valid for custom position"
        )
    margin_px = _integer(data.get("margin_px"), name="watermark.margin_px", minimum=0)
    opacity = _finite_number(data.get("opacity"), name="watermark.opacity")
    if not 0 <= opacity <= 1:
        raise ConfigValidationError("watermark.opacity must be between 0 and 1")
    height_ratio = _finite_number(
        data.get("height_ratio"), name="watermark.height_ratio", minimum=0
    )
    if height_ratio <= 0 or height_ratio >= 1:
        raise ConfigValidationError(
            "watermark.height_ratio must be greater than 0 and less than 1"
        )
    return WatermarkSpec(
        enabled,
        logo_path,
        trim_transparent,
        position,
        custom_xy_ratio,
        margin_px,
        opacity,
        height_ratio,
    )


def _text_spec(
    raw: Any, *, video_duration: float, resource_root: Path | None
) -> TextSpec:
    """Validate one independent text layer and its animation parameters."""
    data = _mapping(raw, name="custom_text")
    _keys(data, _TEXT_KEYS, name="custom_text")
    text = _string(data.get("text"), name="custom_text.text")
    start = _finite_number(data.get("start"), name="custom_text.start", minimum=0)
    end = _finite_number(data.get("end"), name="custom_text.end", minimum=0)
    if not start < end <= video_duration:
        raise ConfigValidationError(
            "custom_text must satisfy 0 <= start < end <= video duration"
        )
    if resource_root is None:
        raise ConfigValidationError("resource_root is required for custom text fonts")
    font_path = resolve_resource_path(
        resource_root,
        _string(data.get("font_name"), name="custom_text.font_name"),
        description="font",
    )
    if font_path.suffix.lower() not in _FONT_EXTENSIONS:
        raise ConfigValidationError("custom_text.font_name must be a font file")
    _validate_font_coverage(font_path, text)
    font_size_px = _integer(
        data.get("font_size_px"), name="custom_text.font_size_px", minimum=1
    )
    color = _color(data.get("color"), name="custom_text.color")
    stroke_color = _color(
        data.get("stroke_color"), name="custom_text.stroke_color", allow_none=True
    )
    stroke_width_px = _finite_number(
        data.get("stroke_width_px"), name="custom_text.stroke_width_px", minimum=0
    )
    x_ratio = _finite_number(data.get("x_ratio"), name="custom_text.x_ratio")
    y_ratio = _finite_number(data.get("y_ratio"), name="custom_text.y_ratio")
    if not 0 <= x_ratio <= 1 or not 0 <= y_ratio <= 1:
        raise ConfigValidationError("custom_text coordinates must be between 0 and 1")
    animation = _string(data.get("animation"), name="custom_text.animation")
    if animation not in _ANIMATIONS:
        raise ConfigValidationError(
            f"custom_text.animation must be one of {sorted(_ANIMATIONS)}"
        )
    stagger = _finite_number(data.get("stagger"), name="custom_text.stagger", minimum=0)
    pop_duration = _finite_number(
        data.get("pop_duration"), name="custom_text.pop_duration", minimum=0
    )
    flicker_duration = _finite_number(
        data.get("flicker_duration"), name="custom_text.flicker_duration", minimum=0
    )
    flicker_hz = _finite_number(
        data.get("flicker_hz"), name="custom_text.flicker_hz", minimum=0
    )
    scale_from = _finite_number(data.get("scale_from"), name="custom_text.scale_from")
    if not 0 < scale_from <= 1:
        raise ConfigValidationError("custom_text.scale_from must be > 0 and <= 1")
    if animation in {"pop_spring", "letter_by_letter"} and pop_duration <= 0:
        raise ConfigValidationError(
            "pop_duration must be positive for spring animations"
        )
    if animation == "flicker_scale" and (flicker_duration <= 0 or flicker_hz <= 0):
        raise ConfigValidationError(
            "flicker_duration and flicker_hz must be positive for flicker animation"
        )
    if animation == "letter_by_letter":
        last_start = start + (len(_visual_characters(text)) - 1) * stagger
        if last_start >= end or last_start + pop_duration > end:
            raise ConfigValidationError(
                "letter_by_letter animation does not fit in its display interval"
            )
    if animation == "flicker_scale" and flicker_duration > end - start:
        raise ConfigValidationError(
            "flicker_duration cannot exceed the text display interval"
        )
    return TextSpec(
        text,
        start,
        end,
        font_path,
        font_size_px,
        color,
        stroke_color,
        stroke_width_px,
        x_ratio,
        y_ratio,
        animation,
        stagger,
        pop_duration,
        flicker_duration,
        flicker_hz,
        scale_from,
    )


def _validate_font_coverage(font_path: Path, text: str) -> None:
    """Reject text whose grapheme clusters resolve to a font replacement glyph."""
    try:
        font = ImageFont.truetype(str(font_path), 32)
        replacement = font.getmask("\ufffd")
    except OSError as exc:
        raise ConfigValidationError(f"cannot load font: {font_path}") from exc
    for cluster in _visual_characters(text):
        visible_parts = [
            character
            for character in cluster
            if character != "\u200d"
            and not unicodedata.combining(character)
            and not 0xFE00 <= ord(character) <= 0xFE0F
            and not 0xE0100 <= ord(character) <= 0xE01EF
        ]
        for character in visible_parts:
            mask = font.getmask(character)
            if mask.size == replacement.size and bytes(mask) == bytes(replacement):
                raise ConfigValidationError(
                    f"font {font_path.name} does not contain glyph for {cluster!r}"
                )


def _image_layer(
    raw: Any,
    *,
    video_duration: float,
    resource_root: Path | None,
) -> ImageLayerSpec:
    """
    Validate one image overlay against the approved resource root.

    @param raw Parsed JSON object describing a single image layer.
    @param video_duration Actual raw video duration used for timing bounds.
    @param resource_root Approved root for overlay image files.
    @returns Immutable normalized image layer specification.
    @raises ConfigValidationError If any field, timing, or image file is invalid.
    """
    if resource_root is None:
        raise ConfigValidationError("resource_root is required for image layers")
    data = _mapping(raw, name="image_layer")
    _keys(data, _IMAGE_LAYER_KEYS, name="image_layer")
    start = _finite_number(data.get("start"), name="image_layer.start", minimum=0)
    end = _finite_number(data.get("end"), name="image_layer.end", minimum=0)
    if not start < end <= video_duration:
        raise ConfigValidationError(
            "image_layer must satisfy 0 <= start < end <= video duration"
        )
    fit_mode = _string(data.get("fit_mode"), name="image_layer.fit_mode")
    if fit_mode not in _IMAGE_FIT_MODES:
        raise ConfigValidationError(f"image_layer.fit_mode is invalid: {fit_mode}")
    path = resolve_resource_path(
        resource_root,
        _string(data.get("path"), name="image_layer.path"),
        description="image layer",
    )
    try:
        if path.stat().st_size > _MAX_IMAGE_BYTES:
            raise ConfigValidationError(
                f"image_layer file size exceeds {_MAX_IMAGE_BYTES} bytes"
            )
        with Image.open(path) as image:
            if image.width <= 0 or image.height <= 0:
                raise ConfigValidationError(
                    "image_layer image dimensions must be positive"
                )
            if image.width * image.height > _MAX_IMAGE_PIXELS:
                raise ConfigValidationError(
                    f"image_layer pixels exceed {_MAX_IMAGE_PIXELS}"
                )
    except ConfigValidationError:
        raise
    except OSError as exc:
        raise ConfigValidationError(f"cannot read image layer: {path}") from exc
    width_ratio = _finite_number(
        data.get("width_ratio"), name="image_layer.width_ratio", minimum=0
    )
    height_ratio = _finite_number(
        data.get("height_ratio"), name="image_layer.height_ratio", minimum=0
    )
    if not 0 < width_ratio <= 1 or not 0 < height_ratio <= 1:
        raise ConfigValidationError("image_layer dimensions must be between 0 and 1")
    return ImageLayerSpec(
        path=path,
        start=start,
        end=end,
        x_ratio=_ratio(data.get("x_ratio"), name="image_layer.x_ratio"),
        y_ratio=_ratio(data.get("y_ratio"), name="image_layer.y_ratio"),
        width_ratio=width_ratio,
        height_ratio=height_ratio,
        fit_mode=fit_mode,
        opacity=_ratio(data.get("opacity"), name="image_layer.opacity"),
    )


def validate_post_process_spec(
    raw: Mapping[str, Any],
    *,
    task_index: int,
    video_duration: float,
    resource_root: Path | None,
) -> PostProcessSpec:
    """
    Validate and normalize one task's post-processing specification.

    @param raw Parsed JSON configuration for one task.
    @param task_index Frozen one-based index from the MPT batch manifest.
    @param video_duration Actual raw video duration measured by the post-processor.
    @param resource_root Approved root for Logo and font files.
    @returns Immutable normalized post-processing specification.
    @raises ConfigValidationError If any field or resource is invalid.
    """
    data = _mapping(raw, name="post_process_spec")
    _keys(data, _SPEC_KEYS, name="post_process_spec")
    if data.get("schema_version") != 1:
        raise ConfigValidationError("schema_version must be 1")
    spec_task_index = _integer(data.get("task_index"), name="task_index", minimum=1)
    if spec_task_index != task_index:
        raise ConfigValidationError(
            "post-processing task_index does not match MPT task index"
        )
    duration = _finite_number(video_duration, name="video_duration", minimum=0)
    if duration <= 0:
        raise ConfigValidationError("video_duration must be greater than 0")
    texts = raw.get("custom_texts")
    if not isinstance(texts, list):
        raise ConfigValidationError("custom_texts must be an array")
    images = raw.get("image_layers", [])
    if not isinstance(images, list):
        raise ConfigValidationError("image_layers must be an array")
    return PostProcessSpec(
        schema_version=1,
        task_index=task_index,
        output=_output_spec(raw.get("output")),
        watermark=_watermark_spec(raw.get("watermark"), resource_root=resource_root),
        custom_texts=tuple(
            _text_spec(item, video_duration=duration, resource_root=resource_root)
            for item in texts
        ),
        image_layers=tuple(
            _image_layer(item, video_duration=duration, resource_root=resource_root)
            for item in images
        ),
    )


def load_post_process_spec(
    spec_path: Path,
    *,
    task_index: int,
    video_duration: float,
    resource_root: Path | None,
) -> PostProcessSpec:
    """
    Read and validate a UTF-8 JSON post-processing file.

    @param spec_path JSON file path controlled by the pipeline job directory.
    @param task_index Frozen MPT task index expected by the caller.
    @param video_duration Actual duration of the raw input video.
    @param resource_root Approved resource root for Logo and fonts.
    @returns Immutable normalized specification.
    @raises ConfigValidationError If the file cannot be parsed or fails validation.
    """
    try:
        raw = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConfigValidationError(
            f"cannot read post-processing spec: {spec_path}"
        ) from exc
    return validate_post_process_spec(
        raw,
        task_index=task_index,
        video_duration=video_duration,
        resource_root=resource_root,
    )


def calculate_watermark_position(
    *,
    canvas: tuple[int, int],
    logo: tuple[int, int],
    position: str,
    margin_px: int,
    custom_xy_ratio: Sequence[float] | None = None,
) -> tuple[int, int]:
    """
    Calculate a complete Logo rectangle's top-left position on the target canvas.

    @param canvas Target `(width, height)` in pixels.
    @param logo Scaled Logo `(width, height)` in pixels.
    @param position Corner or custom center position.
    @param margin_px Corner margin in target pixels.
    @param custom_xy_ratio Custom center coordinate when `position` is `custom`.
    @returns Integer top-left `(x, y)` coordinates.
    @raises ConfigValidationError If the position would leave the canvas.
    """
    width, height = canvas
    logo_width, logo_height = logo
    if min(width, height, logo_width, logo_height) <= 0 or margin_px < 0:
        raise ConfigValidationError("canvas and logo dimensions must be positive")
    if position == "top_left":
        x, y = margin_px, margin_px
    elif position == "top_right":
        x, y = width - logo_width - margin_px, margin_px
    elif position == "bottom_left":
        x, y = margin_px, height - logo_height - margin_px
    elif position == "bottom_right":
        x, y = width - logo_width - margin_px, height - logo_height - margin_px
    elif position == "custom":
        ratios = _ratio_pair(custom_xy_ratio, name="custom_xy_ratio")
        assert ratios is not None
        x = width * ratios[0] - logo_width / 2
        y = height * ratios[1] - logo_height / 2
    else:
        raise ConfigValidationError(f"unknown watermark position: {position}")
    x_int, y_int = round(x), round(y)
    if (
        x_int < 0
        or y_int < 0
        or x_int + logo_width > width
        or y_int + logo_height > height
    ):
        raise ConfigValidationError("watermark would be outside the target canvas")
    return x_int, y_int


def calculate_animation_state(
    animation: str,
    *,
    local_time: float,
    duration: float,
    flicker_duration: float = 0.4,
    flicker_hz: float = 8,
    scale_from: float = 0.6,
) -> tuple[float, float]:
    """
    Compute scale and opacity multiplier from clip-local time.

    @param animation Animation name.
    @param local_time Seconds since this text clip starts, not global video time.
    @param duration Animation display duration.
    @param flicker_duration Duration of the flicker intro.
    @param flicker_hz Flicker frequency in Hz.
    @param scale_from Initial scale for scale-based animations.
    @returns `(scale, opacity_multiplier)` with an opaque normal state at the end.
    @raises ConfigValidationError If timing or animation parameters are invalid.
    """
    if duration <= 0 or not math.isfinite(local_time):
        raise ConfigValidationError(
            "animation duration must be positive and local_time finite"
        )
    if animation == "none":
        return 1.0, 1.0
    if local_time <= 0:
        if animation == "flicker_scale":
            return scale_from, 1 - _FLICKER_AMPLITUDE
        return _SPRING_MIN_SCALE, 1.0
    if animation == "flicker_scale":
        if flicker_duration <= 0 or flicker_hz < 0 or not 0 < scale_from <= 1:
            raise ConfigValidationError("invalid flicker animation parameters")
        if local_time >= flicker_duration:
            return 1.0, 1.0
        progress = local_time / flicker_duration
        scale = scale_from + (1 - scale_from) * progress
        opacity = 1 - _FLICKER_AMPLITUDE * (1 - progress) * abs(
            math.sin(2 * math.pi * flicker_hz * local_time + math.pi / 2)
        )
        return scale, opacity
    if local_time >= duration:
        return 1.0, 1.0
    progress = max(0.0, min(local_time / duration, 1.0))
    if animation == "pop_spring":
        return _spring_scale(progress), 1.0
    if animation == "letter_by_letter":
        return _spring_scale(progress), 1.0
    raise ConfigValidationError(f"unknown animation: {animation}")


def _spring_scale(progress: float) -> float:
    """Calculate the bounded spring scale used by pop and per-letter animations."""
    q = 1 - math.exp(-6 * progress) * math.cos(2.5 * math.pi * progress)
    q_end = 1 - math.exp(-6) * math.cos(2.5 * math.pi)
    return max(
        _SPRING_MIN_SCALE, _SPRING_MIN_SCALE + (1 - _SPRING_MIN_SCALE) * q / q_end
    )


def _resize_frame_on_canvas(
    frame: np.ndarray,
    *,
    scale: float,
    opacity: float,
) -> np.ndarray:
    """
    Scale a color frame or mask around its center while preserving its canvas size.

    @param frame RGB/RGBA uint8 color frame or 0..1 two-dimensional mask.
    @param scale Scale factor; content outside the canvas is clipped explicitly.
    @param opacity Mask multiplier; color frames keep their original values.
    @returns Same-shaped transformed frame.
    @raises ConfigValidationError If the frame shape or scale is invalid.
    """
    if not math.isfinite(scale) or scale <= 0 or not 0 <= opacity <= 1:
        raise ConfigValidationError("animation scale and opacity must be valid")
    if frame.ndim == 2:
        if not np.issubdtype(frame.dtype, np.floating):
            raise ConfigValidationError("mask frames must use a floating-point dtype")
        source = Image.fromarray(np.clip(frame * 255, 0, 255).astype(np.uint8))
        mode = "L"
    elif frame.ndim == 3 and frame.shape[2] in (3, 4):
        source = Image.fromarray(frame)
        mode = source.mode
    else:
        raise ConfigValidationError(
            "animation frame must be a mask, RGB, or RGBA array"
        )

    height, width = frame.shape[:2]
    scaled_size = (
        max(1, round(width * scale)),
        max(1, round(height * scale)),
    )
    resized = source.resize(scaled_size, Image.Resampling.LANCZOS)
    canvas = Image.new(mode, (width, height), 0)
    offset = ((width - scaled_size[0]) // 2, (height - scaled_size[1]) // 2)
    canvas.paste(resized, offset)
    transformed = np.asarray(canvas)
    if frame.ndim == 2:
        return (transformed.astype(np.float64) / 255 * opacity).astype(frame.dtype)
    return transformed.astype(frame.dtype, copy=False)


def _apply_animation(clip, text_spec: TextSpec, *, animation_duration: float):
    """
    Apply a validated animation to a clip using clip-local time and its mask.

    @param clip Text clip whose duration already equals its visible interval.
    @param text_spec Validated style and animation parameters.
    @param animation_duration Duration used by the intro animation.
    @returns Animated clip with the original canvas dimensions.
    """
    if text_spec.animation == "none":
        return clip

    def transform_frame(get_frame, local_time):
        """Transform one color frame or mask using clip-local animation time."""
        scale, opacity = calculate_animation_state(
            text_spec.animation,
            local_time=local_time,
            duration=animation_duration,
            flicker_duration=text_spec.flicker_duration,
            flicker_hz=text_spec.flicker_hz,
            scale_from=text_spec.scale_from,
        )
        return _resize_frame_on_canvas(
            get_frame(local_time), scale=scale, opacity=opacity
        )

    return clip.transform(transform_frame, apply_to=["mask"])


def _visual_characters(text: str) -> tuple[str, ...]:
    """
    Split text into Unicode grapheme-like clusters for letter animation.

    @param text Text to split; line breaks are unsupported in the single-line renderer.
    @returns Tuple containing base characters, combining marks, ZWJ sequences, flags,
        keycaps, and emoji modifiers as visual groups.
    @raises ConfigValidationError If unsupported control characters occur.
    """
    if "\n" in text or "\r" in text:
        raise ConfigValidationError("letter_by_letter does not support line breaks")

    def is_extend(character: str) -> bool:
        """Return whether a code point extends the preceding grapheme cluster."""
        codepoint = ord(character)
        return (
            unicodedata.category(character).startswith("M")
            or 0x1F3FB <= codepoint <= 0x1F3FF
            or 0xFE00 <= codepoint <= 0xFE0F
            or 0xE0100 <= codepoint <= 0xE01EF
            or 0xE0020 <= codepoint <= 0xE007F
        )

    def is_regional_indicator(character: str) -> bool:
        """Return whether a code point is a regional-indicator flag symbol."""
        return 0x1F1E6 <= ord(character) <= 0x1F1FF

    def is_control(character: str) -> bool:
        """Return whether a code point is a disallowed control/format character."""
        return unicodedata.category(character) == "Cc" or (
            unicodedata.category(character) == "Cf"
            and character != "\u200d"
            and not 0xE0020 <= ord(character) <= 0xE007F
        )

    characters: list[str] = []
    index = 0
    while index < len(text):
        character = text[index]
        if is_control(character):
            raise ConfigValidationError(
                "letter_by_letter text contains unsupported control characters"
            )
        cluster = [character]
        index += 1
        if (
            is_regional_indicator(character)
            and index < len(text)
            and is_regional_indicator(text[index])
        ):
            cluster.append(text[index])
            index += 1
        while index < len(text) and is_extend(text[index]):
            cluster.append(text[index])
            index += 1
        while index < len(text) and text[index] == "\u200d":
            cluster.append(text[index])
            index += 1
            if index >= len(text) or is_control(text[index]):
                raise ConfigValidationError(
                    "letter_by_letter has an incomplete ZWJ sequence"
                )
            cluster.append(text[index])
            index += 1
            while index < len(text) and is_extend(text[index]):
                cluster.append(text[index])
                index += 1
        characters.append("".join(cluster))
    if not characters:
        raise ConfigValidationError("letter_by_letter text cannot be empty")
    return tuple(characters)


def _base_text_clip(text_spec: TextSpec, text: str | None = None):
    """Create a raw MoviePy text clip from validated style fields."""
    padding = round(text_spec.font_size_px * (_SPRING_MAX_SCALE - 1) / 2)
    return TextClip(
        font=str(text_spec.font_path),
        text=text_spec.text if text is None else text,
        font_size=text_spec.font_size_px,
        margin=(padding, padding),
        color=text_spec.color,
        stroke_color=text_spec.stroke_color,
        stroke_width=text_spec.stroke_width_px,
    )


def build_text_layers(video_clip: VideoFileClip, text_spec: TextSpec) -> tuple:
    """
    Build one or many independently positioned text layers for a target video.

    @param video_clip Target-sized video whose dimensions define the center anchor.
    @param text_spec Validated text and animation configuration.
    @returns Tuple of clips positioned at the configured center point.
    """
    center_x = video_clip.w * text_spec.x_ratio
    center_y = video_clip.h * text_spec.y_ratio
    if text_spec.animation != "letter_by_letter":
        clip = _base_text_clip(text_spec).with_duration(text_spec.end - text_spec.start)
        intro_duration = (
            text_spec.flicker_duration
            if text_spec.animation == "flicker_scale"
            else text_spec.pop_duration
        )
        clip = _apply_animation(clip, text_spec, animation_duration=intro_duration)
        clip = clip.with_start(text_spec.start)
        position = _center_position(
            center_x=center_x,
            center_y=center_y,
            layer_width=clip.w,
            layer_height=clip.h,
            canvas_width=video_clip.w,
            canvas_height=video_clip.h,
        )
        return (clip.with_position(position),)

    characters = _visual_characters(text_spec.text)
    raw_clips = tuple(_base_text_clip(text_spec, character) for character in characters)
    font = ImageFont.truetype(str(text_spec.font_path), text_spec.font_size_px)
    advances = tuple(font.getlength(character) for character in characters)
    total_width = sum(advances)
    cursor_x = center_x - total_width / 2
    layers = []
    for index, (character, raw_clip, advance) in enumerate(
        zip(characters, raw_clips, advances)
    ):
        character_start = text_spec.start + index * text_spec.stagger
        clip = raw_clip.with_duration(text_spec.end - character_start)
        clip = _apply_animation(
            clip,
            text_spec,
            animation_duration=text_spec.pop_duration,
        )
        clip = clip.with_start(character_start)
        position = (
            round(cursor_x + advance / 2 - clip.w / 2),
            round(center_y - clip.h / 2),
        )
        _ensure_inside_canvas(position, (clip.w, clip.h), (video_clip.w, video_clip.h))
        layers.append(clip.with_position(position))
        cursor_x += advance
    return tuple(layers)


def _ensure_inside_canvas(
    position: tuple[int, int],
    layer_size: tuple[int, int],
    canvas_size: tuple[int, int],
) -> None:
    """
    Require a positioned layer rectangle to remain entirely inside its canvas.

    @param position Layer top-left coordinate.
    @param layer_size Layer width and height.
    @param canvas_size Canvas width and height.
    @returns None when the rectangle is valid.
    @raises ConfigValidationError If any edge is outside the canvas.
    """
    x, y = position
    layer_width, layer_height = layer_size
    canvas_width, canvas_height = canvas_size
    if (
        x < 0
        or y < 0
        or x + layer_width > canvas_width
        or y + layer_height > canvas_height
    ):
        raise ConfigValidationError("text layer would be outside the target canvas")


def _center_position(
    *,
    center_x: float,
    center_y: float,
    layer_width: int,
    layer_height: int,
    canvas_width: int,
    canvas_height: int,
) -> tuple[int, int]:
    """
    Calculate a center-anchored position and validate its canvas boundaries.

    @param center_x Horizontal center coordinate in canvas pixels.
    @param center_y Vertical center coordinate in canvas pixels.
    @param layer_width Layer width in pixels.
    @param layer_height Layer height in pixels.
    @param canvas_width Canvas width in pixels.
    @param canvas_height Canvas height in pixels.
    @returns Validated integer top-left position.
    """
    position = (round(center_x - layer_width / 2), round(center_y - layer_height / 2))
    _ensure_inside_canvas(
        position, (layer_width, layer_height), (canvas_width, canvas_height)
    )
    return position


def _fit_video_clip(video: VideoFileClip, output: OutputSpec):
    """
    Fit a raw video to the target canvas without silently stretching its aspect ratio.

    @param video Raw MPT video clip.
    @param output Validated target canvas and fit policy.
    @returns A target-sized clip suitable for compositing.
    @raises ConfigValidationError If `reject` is selected for mismatched aspect ratios.
    """
    target_ratio = output.width / output.height
    source_ratio = video.w / video.h
    if output.fit_mode == "reject":
        if not math.isclose(
            source_ratio,
            target_ratio,
            rel_tol=_ASPECT_TOLERANCE,
            abs_tol=_ASPECT_TOLERANCE,
        ):
            raise ConfigValidationError("raw video aspect ratio does not match output")
        return video.resized(new_size=(output.width, output.height))
    scale = (
        min(output.width / video.w, output.height / video.h)
        if output.fit_mode == "contain"
        else max(output.width / video.w, output.height / video.h)
    )
    resized = video.resized(
        new_size=(max(1, round(video.w * scale)), max(1, round(video.h * scale)))
    )
    if output.fit_mode == "contain":
        return CompositeVideoClip(
            [resized.with_position(("center", "center"))],
            size=(output.width, output.height),
            bg_color=_rgb_color(output.background_color),
        )
    return resized.cropped(
        width=output.width,
        height=output.height,
        x_center=resized.w / 2,
        y_center=resized.h / 2,
    )


def _load_logo_image(path: Path, *, trim_transparent: bool) -> np.ndarray:
    """
    Load an RGBA Logo, optionally trimming only fully transparent outer pixels.

    @param path Approved Logo image path.
    @param trim_transparent Whether to crop the transparent outer bounding box.
    @returns RGBA uint8 image array with at least one visible pixel.
    @raises ConfigValidationError If the image is unreadable or fully transparent.
    """
    try:
        with Image.open(path) as source:
            rgba = source.convert("RGBA")
            alpha_box = rgba.getchannel("A").getbbox()
            if alpha_box is None:
                raise ConfigValidationError("watermark logo has no visible pixels")
            if trim_transparent:
                rgba = rgba.crop(alpha_box)
            return np.asarray(rgba).copy()
    except ConfigValidationError:
        raise
    except OSError as exc:
        raise ConfigValidationError(f"cannot read watermark logo: {path}") from exc


def add_watermark(video_clip: VideoFileClip, spec: WatermarkSpec) -> ImageClip | None:
    """
    Create a positioned, scaled Logo layer for a target video.

    @param video_clip Raw or target-sized video clip.
    @param spec Validated watermark configuration.
    @returns Positioned ImageClip, or `None` when the watermark is disabled.
    """
    if not spec.enabled:
        return None
    assert spec.logo_path is not None
    logo_height = max(1, round(video_clip.h * spec.height_ratio))
    logo_image = _load_logo_image(
        spec.logo_path,
        trim_transparent=spec.trim_transparent,
    )
    logo = ImageClip(logo_image, transparent=True)
    logo = logo.resized(height=logo_height).with_duration(video_clip.duration)
    logo = logo.with_opacity(spec.opacity)
    position = calculate_watermark_position(
        canvas=(video_clip.w, video_clip.h),
        logo=(logo.w, logo.h),
        position=spec.position,
        margin_px=spec.margin_px,
        custom_xy_ratio=spec.custom_xy_ratio,
    )
    return logo.with_position(position)


def _render_image_layer(spec: ImageLayerSpec, *, canvas: tuple[int, int]) -> ImageClip:
    """
    Build one positioned image overlay clip for the target canvas.

    @param spec Validated image layer specification.
    @param canvas Target canvas size as width and height.
    @returns Positioned MoviePy image clip that preserves transparency.
    @raises ConfigValidationError If the layer would fall outside the canvas.
    """
    canvas_width, canvas_height = canvas
    with Image.open(spec.path) as source:
        image = source.convert("RGBA")
    target_width = max(1, round(canvas_width * spec.width_ratio))
    target_height = max(1, round(canvas_height * spec.height_ratio))
    if spec.fit_mode == "contain":
        scale = min(target_width / image.width, target_height / image.height)
        resized = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
        canvas_image = Image.new("RGBA", (target_width, target_height), (0, 0, 0, 0))
        canvas_image.alpha_composite(
            resized,
            (
                (target_width - resized.width) // 2,
                (target_height - resized.height) // 2,
            ),
        )
    else:
        scale = max(target_width / image.width, target_height / image.height)
        resized = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
        left = max(0, (resized.width - target_width) // 2)
        top = max(0, (resized.height - target_height) // 2)
        canvas_image = resized.crop(
            (left, top, left + target_width, top + target_height)
        )
    x = round(canvas_width * spec.x_ratio - target_width / 2)
    y = round(canvas_height * spec.y_ratio - target_height / 2)
    _ensure_inside_canvas(
        (x, y),
        (target_width, target_height),
        (canvas_width, canvas_height),
    )
    clip = ImageClip(np.asarray(canvas_image).copy(), transparent=True)
    return (
        clip.with_duration(spec.end - spec.start)
        .with_start(spec.start)
        .with_opacity(spec.opacity)
        .with_position((x, y))
    )


def compose_video(
    raw_video_path: Path,
    spec: PostProcessSpec,
    output_path: Path,
) -> None:
    """
    Composite Logo, image, and text layers and write one target-encoded video.

    @param raw_video_path Existing MPT video to post-process.
    @param spec Validated task-level configuration.
    @param output_path Temporary or final output path selected by the caller.
    @returns None after successful encoding; raises on any rendering or encoding error.
    """
    if output_path.exists():
        raise ConfigValidationError(
            f"refusing to overwrite existing output: {output_path}"
        )
    video = VideoFileClip(str(raw_video_path))
    fitted = None
    watermark = None
    image_clips = []
    text_clips = []
    composite = None
    try:
        if spec.output.audio_required and video.audio is None:
            raise ConfigValidationError(
                "output.audio_required is true but raw video has no audio"
            )
        fitted = _fit_video_clip(video, spec.output)
        layers = [fitted]
        # 图层顺序固定为 图片 → 水印 → 指定文字，保证文字始终压在截图和 Logo 之上。
        image_clips = [
            _render_image_layer(image_spec, canvas=(fitted.w, fitted.h))
            for image_spec in spec.image_layers
        ]
        layers.extend(image_clips)
        watermark = add_watermark(fitted, spec.watermark)
        if watermark is not None:
            layers.append(watermark)
        text_clips = [
            layer
            for text_spec in spec.custom_texts
            for layer in build_text_layers(fitted, text_spec)
        ]
        layers.extend(text_clips)
        composite = CompositeVideoClip(
            layers,
            size=(spec.output.width, spec.output.height),
            bg_color=_rgb_color(spec.output.background_color),
        )
        ffmpeg_params = ["-movflags", "+faststart"] if spec.output.faststart else None
        output_path.parent.mkdir(parents=True, exist_ok=True)
        composite.write_videofile(
            str(output_path),
            fps=spec.output.fps,
            codec=spec.output.video_codec,
            bitrate=spec.output.video_bitrate,
            audio=video.audio is not None,
            audio_codec=spec.output.audio_codec,
            audio_bitrate=spec.output.audio_bitrate,
            pixel_format=spec.output.pixel_format,
            ffmpeg_params=ffmpeg_params,
            logger=None,
        )
    finally:
        if composite is not None:
            composite.close()
        for clip in text_clips:
            clip.close()
        for clip in image_clips:
            clip.close()
        if watermark is not None:
            watermark.close()
        if fitted is not None and fitted is not video:
            fitted.close()
        video.close()
