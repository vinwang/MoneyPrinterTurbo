import tempfile
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from app.config import config
from app.services import asset_library
from app.services import asset_matching


ROOT_DIR = Path(__file__).parent.parent.parent
WEBUI_MAIN = ROOT_DIR / "webui" / "Main.py"


def test_local_source_renders_automatic_storyboard_preview():
    """本地来源和文案存在时，WebUI 应展示匹配分镜而不改变人声入口。"""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        video_path = root / "人物" / "worker.mp4"
        video_path.parent.mkdir()
        video_path.write_bytes(b"video")
        asset = asset_library.LibraryAsset(
            asset_id="video-worker",
            kind="video",
            root_path=str(root),
            relative_path="人物/worker.mp4",
            category="人物",
            duration=4.0,
            width=1080,
            height=1920,
            size_bytes=5,
            sha256="hash",
            description="人物工作场景",
            tags=("人物", "工作"),
            analysis_status="ready",
            analysis_error="",
            use_count=0,
            last_used_at=None,
        )
        match = asset_matching.StoryboardMatch(
            shots=(
                asset_matching.StoryboardShot(
                    index=1,
                    text="人物正在工作。",
                    start_seconds=0.0,
                    duration_seconds=3.0,
                    candidates=(asset,),
                ),
            ),
            bgm_candidates=(),
        )
        app_config = dict(
            config.app,
            video_source="local",
            local_video_library_directory=str(root),
            local_bgm_library_directory="",
        )
        ui_config = dict(config.ui, language="en", bgm_type="")
        with (
            patch.object(config, "app", app_config),
            patch.object(config, "ui", ui_config),
            patch.object(config, "try_save_config", return_value=True),
            patch.object(
                asset_library,
                "scan_library",
                return_value=asset_library.ScanSummary(1, 0, 0, 1, 0, 0, ()),
            ),
            patch.object(asset_matching, "match_storyboard", return_value=match),
            patch.object(asset_library, "resolve_asset_path", return_value=video_path),
        ):
            app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=60)
            app.session_state["ui_language"] = "en"
            app.session_state["video_script"] = "人物正在工作。"
            app.run()

        assert [str(item.value) for item in app.exception] == []
        assert app.session_state["local_storyboard_selected_ids"] == {
            "1": "video-worker"
        }
        assert any(item.value == "video-worker" for item in app.selectbox)


def test_local_source_renders_smart_bgm_candidate():
    """智能本地音乐模式应使用分镜匹配结果并展示可试听候选。"""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        video_path = root / "worker.mp4"
        bgm_path = root / "calm.mp3"
        video_path.write_bytes(b"video")
        bgm_path.write_bytes(b"music")
        video_asset = asset_library.LibraryAsset(
            "video-worker",
            "video",
            str(root),
            "worker.mp4",
            "uncategorized",
            4.0,
            1080,
            1920,
            5,
            "video-hash",
            "人物工作场景",
            ("人物", "工作"),
            "ready",
            "",
            0,
            None,
        )
        bgm_asset = asset_library.LibraryAsset(
            "bgm-calm",
            "bgm",
            str(root),
            "calm.mp3",
            "轻快",
            10.0,
            None,
            None,
            5,
            "bgm-hash",
            "",
            (),
            "ready",
            "",
            0,
            None,
        )
        match = asset_matching.StoryboardMatch(
            shots=(
                asset_matching.StoryboardShot(
                    1,
                    "人物正在工作。",
                    0.0,
                    3.0,
                    (video_asset,),
                ),
            ),
            bgm_candidates=(bgm_asset,),
        )
        app_config = dict(
            config.app,
            video_source="local",
            local_video_library_directory=str(root),
            local_bgm_library_directory=str(root),
        )
        ui_config = dict(config.ui, language="en", bgm_type="smart")
        with (
            patch.object(config, "app", app_config),
            patch.object(config, "ui", ui_config),
            patch.object(config, "try_save_config", return_value=True),
            patch.object(
                asset_library,
                "scan_library",
                return_value=asset_library.ScanSummary(2, 0, 0, 1, 0, 0, ()),
            ),
            patch.object(asset_matching, "match_storyboard", return_value=match),
            patch.object(
                asset_library,
                "resolve_asset_path",
                side_effect=lambda asset: (
                    video_path if asset.kind == "video" else bgm_path
                ),
            ),
        ):
            app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=60)
            app.session_state["ui_language"] = "en"
            app.session_state["video_script"] = "人物正在工作。"
            app.run()

        assert [str(item.value) for item in app.exception] == []
        assert app.session_state["local_library_bgm_id"] == "bgm-calm"
        assert any(item.value == "bgm-calm" for item in app.selectbox)
