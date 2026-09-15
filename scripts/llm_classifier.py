"""Strict LLM adapter for selecting manifest categories and asset IDs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from app.services.llm import _generate_response


class ClassificationError(ValueError):
    """Raised when an LLM classification response is unsafe or malformed."""


@dataclass(frozen=True)
class ClassificationResult:
    """Validated manifest selections returned by the classification adapter."""

    categories: tuple[str, ...]
    video_ids: tuple[str, ...]
    bgm_id: str | None


_KEYS = {"categories", "video_ids", "bgm_id"}
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _strict_json(response: str) -> dict[str, Any]:
    """Parse one unwrapped JSON object without regex recovery or silent defaults."""
    if not isinstance(response, str) or not response.strip():
        raise ClassificationError("LLM classifier returned empty text")
    try:
        value = json.loads(response)
    except json.JSONDecodeError as exc:
        raise ClassificationError("LLM classifier must return one JSON object") from exc
    if not isinstance(value, dict):
        raise ClassificationError("LLM classifier response must be a JSON object")
    unknown = sorted(set(value) - _KEYS)
    if unknown:
        raise ClassificationError(
            f"LLM classifier returned unknown fields: {', '.join(unknown)}"
        )
    return value


@dataclass(frozen=True)
class LLMClassifier:
    """Adapter that reuses the repository LLM request path with strict output validation."""

    response_fn: Callable[..., str] = _generate_response
    app_config: Any = None

    def classify(
        self,
        *,
        subject: str,
        categories: Sequence[str],
        video_ids: Sequence[str],
        bgm_ids: Sequence[str],
    ) -> ClassificationResult:
        """
        Ask the configured LLM to select only IDs present in a supplied manifest.

        @param subject Advertisement subject or campaign description.
        @param categories Allowed manifest category names.
        @param video_ids Allowed video asset IDs.
        @param bgm_ids Allowed BGM asset IDs.
        @returns Immutable validated classification result.
        @raises ClassificationError If the request fails or response contains invalid IDs.
        """
        if not isinstance(subject, str) or not subject.strip():
            raise ClassificationError("classification subject must be non-empty")
        allowed_categories = _normalize_allowlist(categories, "categories")
        allowed_videos = _normalize_allowlist(video_ids, "video_ids")
        allowed_bgm = _normalize_allowlist(bgm_ids, "bgm_ids")
        prompt = self._build_prompt(
            subject=subject,
            categories=allowed_categories,
            video_ids=allowed_videos,
            bgm_ids=allowed_bgm,
        )
        kwargs = {"prompt": prompt}
        if self.app_config is not None:
            kwargs["app_config"] = self.app_config
        response = self.response_fn(**kwargs)
        if isinstance(response, str) and response.startswith("Error: "):
            raise ClassificationError(response.removeprefix("Error: ").strip())
        value = _strict_json(response)
        result_categories = self._validate_ids(
            value.get("categories", []),
            allowed=allowed_categories,
            field_name="categories",
        )
        result_videos = self._validate_ids(
            value.get("video_ids", []),
            allowed=allowed_videos,
            field_name="video_ids",
        )
        bgm_id = value.get("bgm_id")
        if bgm_id is not None:
            if not isinstance(bgm_id, str) or bgm_id not in allowed_bgm:
                raise ClassificationError("LLM classifier returned an unknown bgm_id")
        if not result_videos:
            raise ClassificationError(
                "LLM classifier must select at least one video_id"
            )
        return ClassificationResult(result_categories, result_videos, bgm_id)

    def _build_prompt(
        self,
        *,
        subject: str,
        categories: Sequence[str],
        video_ids: Sequence[str],
        bgm_ids: Sequence[str],
    ) -> str:
        """Build a bounded JSON-only prompt from trusted manifest values."""
        return json.dumps(
            {
                "role": "advertising asset classifier",
                "instruction": "Return only JSON with categories, video_ids, and bgm_id. Select IDs exactly from the supplied arrays.",
                "subject": subject,
                "allowed_categories": list(categories),
                "allowed_video_ids": list(video_ids),
                "allowed_bgm_ids": list(bgm_ids),
            },
            ensure_ascii=False,
        )

    def _validate_ids(
        self,
        values: Any,
        *,
        allowed: Sequence[str],
        field_name: str,
    ) -> tuple[str, ...]:
        """Validate an array of safe, allow-listed IDs or category names."""
        if not isinstance(values, list):
            raise ClassificationError(
                f"LLM classifier field {field_name} must be an array"
            )
        allowed_set = set(allowed)
        result: list[str] = []
        for value in values:
            if (
                not isinstance(value, str)
                or not value.strip()
                or "\x00" in value
                or (field_name != "categories" and not _SAFE_ID.fullmatch(value))
                or value not in allowed_set
            ):
                raise ClassificationError(
                    f"LLM classifier returned an invalid {field_name} value"
                )
            if value not in result:
                result.append(value)
        return tuple(result)


def _normalize_allowlist(values: Sequence[str], field_name: str) -> tuple[str, ...]:
    """Validate and deduplicate trusted manifest values before placing them in a prompt."""
    if not isinstance(values, (list, tuple)):
        raise ClassificationError(f"classifier {field_name} allowlist must be an array")
    result: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip() or "\x00" in value:
            raise ClassificationError(
                f"classifier {field_name} contains an invalid value"
            )
        if field_name != "categories" and not _SAFE_ID.fullmatch(value):
            raise ClassificationError(f"classifier {field_name} contains an unsafe ID")
        if value not in result:
            result.append(value)
    return tuple(result)
