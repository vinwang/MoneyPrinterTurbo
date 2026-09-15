import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.services import asset_library
from app.services import asset_library_import
from app.services import asset_library_runtime, asset_library_shots, asset_matching


def _frames_per_window(_path: Path, positions) -> tuple[bytes, ...]:
    """替身抽帧器：每个窗口返回一帧，数量必须与窗口一致。"""
    return tuple(b"frame" for _ in positions)


def _window_analysis(description: str, tags: list[str], frame_count: int) -> str:
    """构造逐窗口视觉分析响应，条目数与帧数一一对应。"""
    return json.dumps(
        {
            "windows": [
                {
                    "index": index + 1,
                    "description": f"{description} 第{index + 1}段",
                    "tags": tags,
                    "mood": "自然",
                }
                for index in range(frame_count)
            ]
        },
        ensure_ascii=False,
    )


class TestAssetLibrary(unittest.TestCase):
    """验证本地素材索引和分镜匹配的核心行为。"""

    def _probe(self, path: Path):
        if path.suffix == ".mp3":
            return {"duration": 12.0, "width": None, "height": None}
        return {"duration": 6.0, "width": 1080, "height": 1920}

    @staticmethod
    def _vision(path: Path, frames: tuple[bytes, ...]) -> str:
        category = path.parent.name
        return _window_analysis(
            f"{category} 场景素材",
            [category, path.stem],
            len(frames),
        )

    def test_scan_is_incremental_and_reanalyzes_changed_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video_root = root / "videos"
            bgm_root = root / "music"
            (video_root / "人物").mkdir(parents=True)
            bgm_root.mkdir()
            video_path = video_root / "人物" / "worker.mp4"
            video_path.write_bytes(b"video-v1")
            (bgm_root / "warm.mp3").write_bytes(b"music")
            db_path = root / "library.sqlite3"

            with patch.object(
                asset_library_shots,
                "extract_frames_at",
                side_effect=_frames_per_window,
            ) as extract_frames:
                first = asset_library.scan_library(
                    video_root,
                    bgm_root,
                    db_path=db_path,
                    probe_fn=self._probe,
                    vision_fn=self._vision,
                    app_config={
                        "llm_provider": "openai",
                        "openai_model_name": "vision-v1",
                    },
                )
                second = asset_library.scan_library(
                    video_root,
                    bgm_root,
                    db_path=db_path,
                    probe_fn=self._probe,
                    vision_fn=self._vision,
                    app_config={
                        "llm_provider": "openai",
                        "openai_model_name": "vision-v1",
                    },
                )
                migrated = asset_library.scan_library(
                    video_root,
                    bgm_root,
                    db_path=db_path,
                    probe_fn=self._probe,
                    vision_fn=self._vision,
                    app_config={
                        "llm_provider": "openai",
                        "openai_model_name": "vision-v2",
                    },
                )
                video_path.write_bytes(b"video-v2")
                third = asset_library.scan_library(
                    video_root,
                    bgm_root,
                    db_path=db_path,
                    probe_fn=self._probe,
                    vision_fn=self._vision,
                )

            self.assertEqual(first.added, 2)
            self.assertEqual(first.analyzed, 1)
            self.assertEqual(second.unchanged, 2)
            self.assertEqual(second.analyzed, 0)
            self.assertEqual(migrated.analyzed, 1)
            self.assertEqual(third.updated, 1)
            self.assertEqual(third.analyzed, 1)
            self.assertEqual(extract_frames.call_count, 3)

    def test_match_storyboard_uses_candidates_without_repeating_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video_root = root / "videos"
            bgm_root = root / "music"
            for category, filename in (
                ("人物", "worker.mp4"),
                ("城市", "street.mp4"),
                ("产品", "product.mp4"),
            ):
                category_dir = video_root / category
                category_dir.mkdir(parents=True)
                (category_dir / filename).write_bytes(filename.encode())
            bgm_root.mkdir()
            (bgm_root / "轻快" ).mkdir()
            (bgm_root / "轻快" / "up.mp3").write_bytes(b"music")
            db_path = root / "library.sqlite3"

            with patch.object(
                asset_library_shots,
                "extract_frames_at",
                side_effect=_frames_per_window,
            ):
                asset_library.scan_library(
                    video_root,
                    bgm_root,
                    db_path=db_path,
                    probe_fn=self._probe,
                    vision_fn=self._vision,
                )

            result = asset_matching.match_storyboard(
                "人物在城市中展示产品。",
                clip_duration=3,
                db_path=db_path,
            )

            self.assertGreaterEqual(len(result.shots), 1)
            selected_ids = [shot.candidates[0].asset_id for shot in result.shots]
            self.assertEqual(len(selected_ids), len(set(selected_ids)))
            self.assertEqual(len(result.bgm_candidates), 1)
            self.assertEqual(result.bgm_candidates[0].category, "轻快")

    def test_scan_rejects_missing_root_and_resolves_only_indexed_asset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(asset_library.AssetLibraryError):
                asset_library.scan_library(
                    root / "missing-videos",
                    root / "missing-music",
                    db_path=root / "library.sqlite3",
                    probe_fn=self._probe,
                )

    def test_failed_visual_analysis_is_explicitly_excluded_from_matching(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video_root = root / "videos"
            bgm_root = root / "music"
            video_root.mkdir()
            bgm_root.mkdir()
            (video_root / "clip.mp4").write_bytes(b"video")
            db_path = root / "library.sqlite3"

            with patch.object(
                asset_library_shots,
                "extract_frames_at",
                side_effect=_frames_per_window,
            ):
                summary = asset_library.scan_library(
                    video_root,
                    bgm_root,
                    db_path=db_path,
                    probe_fn=self._probe,
                    vision_fn=lambda _path, _frames: "Error: vision unavailable",
                )

            self.assertEqual(summary.failed, 1)
            with self.assertRaisesRegex(asset_library.AssetLibraryError, "analysis"):
                asset_matching.match_storyboard(
                    "展示产品",
                    clip_duration=3,
                    db_path=db_path,
                )

            with patch.object(
                asset_library_shots,
                "extract_frames_at",
                side_effect=_frames_per_window,
            ):
                retried = asset_library.scan_library(
                    video_root,
                    bgm_root,
                    db_path=db_path,
                    probe_fn=self._probe,
                    vision_fn=self._vision,
                    retry_failed=True,
                )
            self.assertEqual(retried.analyzed, 1)
            self.assertEqual(
                len(
                    asset_library.list_assets(
                        kind="video",
                        analysis_status="ready",
                        db_path=db_path,
                    )
                ),
                1,
            )

    def test_materialize_bgm_uses_stable_managed_copy_and_records_usage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video_root = root / "videos"
            music_root = root / "music"
            video_root.mkdir()
            music_root.mkdir()
            music_path = music_root / "calm.wav"
            music_path.write_bytes(b"music")
            db_path = root / "library.sqlite3"
            asset_library.scan_library(
                video_root,
                music_root,
                db_path=db_path,
                probe_fn=self._probe,
                analyze_visual=False,
            )
            music_asset = asset_library.list_assets(kind="bgm", db_path=db_path)[0]
            managed_dir = root / "managed-bgm"

            with patch.object(
                asset_library.bgm,
                "uploaded_bgm_dir",
                return_value=str(managed_dir),
            ), patch.object(asset_library.bgm, "validate_audio_file"):
                managed_name = asset_library_runtime.materialize_bgm(
                    music_asset.asset_id,
                    db_path=db_path,
                )
                asset_library_runtime.record_usage(
                    [music_asset.asset_id],
                    db_path=db_path,
                )

            self.assertEqual((managed_dir / managed_name).read_bytes(), b"music")
            used_asset = asset_library.get_asset(music_asset.asset_id, db_path=db_path)
            self.assertEqual(used_asset.use_count, 1)
            self.assertTrue(used_asset.last_used_at)

    def test_import_uploaded_video_requires_safe_category_and_copies_to_library(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upload_root = root / "uploads"
            library_root = root / "library"
            upload_root.mkdir()
            library_root.mkdir()
            source = upload_root / "stored-video.mp4"
            source.write_bytes(b"validated video")

            with (
                patch.object(
                    asset_library_import.library,
                    "configured_video_root",
                    return_value=library_root,
                ),
                patch.object(
                    asset_library_import.utils,
                    "storage_dir",
                    return_value=str(upload_root),
                ),
            ):
                imported = asset_library_import.import_video_file(
                    source,
                    "new.mp4",
                    "人物",
                )
                with self.assertRaisesRegex(
                    asset_library.AssetLibraryError,
                    "category",
                ):
                    asset_library_import.import_video_file(source, "new.mp4", "../outside")

            self.assertEqual(imported.read_bytes(), b"validated video")
            self.assertEqual(imported.parent, library_root / "人物")
