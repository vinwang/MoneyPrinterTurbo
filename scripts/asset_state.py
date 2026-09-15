"""Small atomic JSON state store for advertising-material reservations."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping


class AssetStateError(RuntimeError):
    """Raised when an asset reservation transition is invalid or cannot be persisted."""


_VALID_STATUSES = {"available", "reserved", "committed", "released"}


def _read_state(path: Path) -> dict[str, Any]:
    """Read an existing state file or return an empty version-one state."""
    if not path.exists():
        return {"schema_version": 1, "assets": {}}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AssetStateError(f"asset state is invalid: {path}") from exc
    if not isinstance(value, Mapping) or value.get("schema_version") != 1:
        raise AssetStateError("asset state schema_version must be 1")
    assets = value.get("assets")
    if not isinstance(assets, Mapping):
        raise AssetStateError("asset state assets must be an object")
    return {"schema_version": 1, "assets": dict(assets)}


def load_state(path: Path) -> dict[str, Any]:
    """
    Load and validate an asset state file.

    @param path JSON state path.
    @returns Copy of the version-one state object.
    @raises AssetStateError If the state cannot be parsed or has invalid entries.
    """
    state = _read_state(path)
    for asset_id, record in state["assets"].items():
        if not isinstance(asset_id, str) or not asset_id.strip():
            raise AssetStateError("asset IDs must be non-empty strings")
        if (
            not isinstance(record, Mapping)
            or record.get("status") not in _VALID_STATUSES
        ):
            raise AssetStateError(f"invalid state record for asset: {asset_id}")
    return state


def _write_state(path: Path, state: Mapping[str, Any]) -> None:
    """Persist state atomically as canonical UTF-8 JSON."""
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
            json.dump(state, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        raise AssetStateError(f"cannot write asset state: {path}") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _transition(
    path: Path,
    asset_ids: Iterable[str],
    *,
    owner: str,
    target_status: str,
    now: str,
) -> None:
    """Apply an owner-checked state transition to a unique set of asset IDs."""
    if target_status not in _VALID_STATUSES - {"available"}:
        raise AssetStateError(f"invalid transition target: {target_status}")
    if (
        not isinstance(owner, str)
        or not owner.strip()
        or not isinstance(now, str)
        or not now.strip()
    ):
        raise AssetStateError("asset transition owner and timestamp are required")
    unique_ids = tuple(dict.fromkeys(asset_ids))
    if any(
        not isinstance(asset_id, str) or not asset_id.strip() for asset_id in unique_ids
    ):
        raise AssetStateError("asset IDs must be non-empty strings")
    state = _read_state(path)
    records = state["assets"]
    for asset_id in unique_ids:
        record = records.get(asset_id)
        if record is None:
            record = {
                "status": "available",
                "owner": None,
                "updated_at": now,
                "reason": None,
            }
        if (
            not isinstance(record, Mapping)
            or record.get("status") not in _VALID_STATUSES
        ):
            raise AssetStateError(f"invalid state record for asset: {asset_id}")
        current_status = record["status"]
        current_owner = record.get("owner")
        if target_status == "reserved":
            if current_status == "committed" or (
                current_status == "reserved" and current_owner != owner
            ):
                raise AssetStateError(f"asset is unavailable: {asset_id}")
        elif (
            target_status == "committed"
            and current_status == "committed"
            and current_owner == owner
        ):
            continue
        elif current_status != "reserved" or current_owner != owner:
            raise AssetStateError(f"asset transition owner mismatch: {asset_id}")
        records[asset_id] = {
            "status": target_status,
            "owner": owner if target_status != "released" else None,
            "updated_at": now,
            "reason": None,
        }
    _write_state(path, state)


def reserve_assets(
    path: Path, asset_ids: Iterable[str], *, owner: str, now: str
) -> None:
    """
    Reserve assets for one job/task owner.

    @param path JSON state path.
    @param asset_ids Manifest IDs to reserve.
    @param owner Stable `job_id:task_index` owner string.
    @param now ISO timestamp recorded with the transition.
    @returns None after an atomic state update.
    """
    _transition(path, asset_ids, owner=owner, target_status="reserved", now=now)


def commit_assets(
    path: Path, asset_ids: Iterable[str], *, owner: str, now: str
) -> None:
    """
    Mark owner-reserved assets committed after a successful output.

    @param path JSON state path.
    @param asset_ids Manifest IDs to commit.
    @param owner Reservation owner that must match each record.
    @param now ISO timestamp recorded with the transition.
    @returns None after an atomic state update.
    """
    _transition(path, asset_ids, owner=owner, target_status="committed", now=now)


def release_assets(
    path: Path, asset_ids: Iterable[str], *, owner: str, now: str
) -> None:
    """
    Release owner-reserved assets after an explicitly abandoned job.

    @param path JSON state path.
    @param asset_ids Manifest IDs to release.
    @param owner Reservation owner that must match each record.
    @param now ISO timestamp recorded with the transition.
    @returns None after an atomic state update.
    """
    _transition(path, asset_ids, owner=owner, target_status="released", now=now)
