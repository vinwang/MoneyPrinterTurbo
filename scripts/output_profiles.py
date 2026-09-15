"""Built-in and file-loaded output profiles for common advertising placements."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


class ProfileError(ValueError):
    """Raised when an output profile file or ID is invalid."""


_HEX_COLOR = re.compile(r"^#[0-9A-Fa-f]{6}$")


@dataclass(frozen=True)
class OutputProfile:
    """Complete target canvas and encoding profile embedded into each task spec."""

    profile_id: str
    width: int
    height: int
    fps: int
    fit_mode: str
    background_color: str
    video_codec: str
    pixel_format: str
    video_bitrate: str
    audio_required: bool
    audio_codec: str
    audio_bitrate: str
    faststart: bool


def _profile(
    profile_id: str,
    width: int,
    height: int,
    *,
    fit_mode: str = "contain",
    video_bitrate: str = "1800k",
) -> OutputProfile:
    """Create a standard H.264/AAC profile with explicit target properties."""
    return OutputProfile(
        profile_id,
        width,
        height,
        30,
        fit_mode,
        "#000000",
        "libx264",
        "yuv420p",
        video_bitrate,
        True,
        "aac",
        "128k",
        True,
    )


BUILTIN_PROFILES: Mapping[str, OutputProfile] = {
    "portrait-1080x1920": _profile("portrait-1080x1920", 1080, 1920),
    "portrait-1080x1920-high": _profile(
        "portrait-1080x1920-high", 1080, 1920, video_bitrate="8000k"
    ),
    "portrait-720x1280": _profile(
        "portrait-720x1280", 720, 1280, video_bitrate="2500k"
    ),
    "landscape-1920x1080": _profile("landscape-1920x1080", 1920, 1080),
    "square-1080": _profile("square-1080", 1080, 1080),
    "portrait-test-v1": _profile("portrait-test-v1", 1080, 1920),
}


def _validate_profile(profile_id: str, raw: Mapping[str, Any]) -> OutputProfile:
    """Validate a complete profile mapping before making it available to selector."""
    required = set(OutputProfile.__dataclass_fields__)
    if set(raw) != required - {"profile_id"} and set(raw) != required:
        missing = sorted((required - {"profile_id"}) - set(raw))
        unknown = sorted(set(raw) - required)
        detail = f"missing={','.join(missing)} unknown={','.join(unknown)}"
        raise ProfileError(f"profile {profile_id} fields are invalid: {detail}")
    values = dict(raw)
    values["profile_id"] = profile_id
    integer_fields = ("width", "height", "fps")
    for field_name in integer_fields:
        field_value = values.get(field_name)
        if isinstance(field_value, bool) or not isinstance(field_value, int):
            raise ProfileError(f"profile {profile_id} {field_name} must be an integer")
    for field_name in (
        "fit_mode",
        "background_color",
        "video_codec",
        "pixel_format",
        "video_bitrate",
        "audio_codec",
        "audio_bitrate",
    ):
        if (
            not isinstance(values.get(field_name), str)
            or not values[field_name].strip()
        ):
            raise ProfileError(
                f"profile {profile_id} {field_name} must be a non-empty string"
            )
    for field_name in ("audio_required", "faststart"):
        if not isinstance(values.get(field_name), bool):
            raise ProfileError(f"profile {profile_id} {field_name} must be boolean")
    try:
        profile = OutputProfile(**values)
    except TypeError as exc:
        raise ProfileError(f"profile {profile_id} fields are invalid") from exc
    if profile.width <= 0 or profile.height <= 0 or profile.fps <= 0:
        raise ProfileError(f"profile {profile_id} dimensions and fps must be positive")
    if not _HEX_COLOR.fullmatch(profile.background_color):
        raise ProfileError(f"profile {profile_id} background_color must use #RRGGBB")
    if profile.fit_mode not in {"contain", "cover", "reject"}:
        raise ProfileError(f"profile {profile_id} fit_mode is invalid")
    return profile


def load_profiles(path: Path) -> dict[str, OutputProfile]:
    """
    Load complete custom output profiles from a version-one JSON file.

    @param path JSON file containing `{schema_version: 1, profiles: {...}}`.
    @returns Mapping of profile IDs to immutable profiles.
    @raises ProfileError If the file or any profile is malformed.
    """
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProfileError(f"cannot read profiles: {path}") from exc
    if not isinstance(value, Mapping) or value.get("schema_version") != 1:
        raise ProfileError("profiles schema_version must be 1")
    profiles = value.get("profiles")
    if not isinstance(profiles, Mapping) or not profiles:
        raise ProfileError("profiles must be a non-empty object")
    result = dict(BUILTIN_PROFILES)
    for profile_id, raw in profiles.items():
        if (
            not isinstance(profile_id, str)
            or not profile_id.strip()
            or not isinstance(raw, Mapping)
        ):
            raise ProfileError("profile IDs and values must be valid")
        result[profile_id] = _validate_profile(profile_id, raw)
    return result


def resolve_profile(
    profile_id: str, profiles: Mapping[str, OutputProfile] | None = None
) -> OutputProfile:
    """
    Resolve one profile ID from built-ins or a caller-supplied profile mapping.

    @param profile_id Profile identifier.
    @param profiles Optional mapping that replaces the built-in registry.
    @returns Complete immutable output profile.
    @raises ProfileError If profile_id is unknown.
    """
    registry = BUILTIN_PROFILES if profiles is None else profiles
    profile = registry.get(profile_id)
    if profile is None:
        raise ProfileError(f"unknown output profile: {profile_id}")
    return profile


def profile_dict(profile: OutputProfile) -> dict[str, Any]:
    """Convert a profile to a JSON-compatible dictionary for task snapshots."""
    return asdict(profile)
