"""Explicit import of uploaded local video assets into the configured library."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from uuid import uuid4

from app.services import asset_library as library
from app.services import material_upload
from app.utils import file_security, utils


AssetLibraryError = library.AssetLibraryError
_SAFE_CATEGORY = re.compile(r"^[^/\\\x00]+$")
_IMPORT_ID_SUFFIX_LENGTH = 8


def import_video_file(
    source_path: str | Path,
    original_name: str,
    category: str,
    *,
    library_root: Path | None = None,
) -> Path:
    """
    将已校验的本地上传文件复制进视频素材库分类目录。

    @param source_path storage/local_videos 中的已校验源文件。
    @param original_name 浏览器提供的原始文件名。
    @param category 单层素材分类目录名。
    @param library_root 目标根目录，默认读取已配置的视频库目录。
    @returns 新写入的素材路径。
    @raises AssetLibraryError 路径、权限或文件名不合法。
    """
    root = library_root or library.configured_video_root()
    if root is None:
        raise AssetLibraryError("local video library directory is not configured")
    try:
        source = Path(
            file_security.resolve_path_within_directory(
                utils.storage_dir("local_videos"), str(source_path)
            )
        )
        safe_name = material_upload.sanitize_material_filename(original_name)
    except (ValueError, material_upload.MaterialUploadError) as exc:
        raise AssetLibraryError("uploaded material cannot be imported") from exc
    safe_category = str(category or "").strip()
    if not _SAFE_CATEGORY.fullmatch(safe_category) or safe_category in {".", ".."}:
        raise AssetLibraryError("material library category must be one directory name")
    target_dir = root / safe_category
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        candidate = target_dir / safe_name
        if candidate.exists():
            candidate = target_dir / (
                f"{Path(safe_name).stem}-{uuid4().hex[:_IMPORT_ID_SUFFIX_LENGTH]}"
                f"{Path(safe_name).suffix.lower()}"
            )
        shutil.copy2(source, candidate)
    except OSError as exc:
        raise AssetLibraryError("cannot import uploaded material into library") from exc
    return candidate
