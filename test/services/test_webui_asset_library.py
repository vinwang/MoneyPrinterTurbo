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


def test_gap_shot_offers_a_manual_material_picker():
    """缺口镜必须给出人工替换入口，否则生成时只会抛「未覆盖全部分镜」。"""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        video_path = root / "人物" / "worker.mp4"
        video_path.parent.mkdir()
        video_path.write_bytes(b"video")
        spare_path = root / "机房" / "server.mp4"
        spare_path.parent.mkdir()
        spare_path.write_bytes(b"video")
        matched = asset_library.LibraryAsset(
            "video-worker",
            "video",
            str(root),
            "人物/worker.mp4",
            "人物",
            4.0,
            1080,
            1920,
            5,
            "hash-worker",
            "人物工作场景",
            ("人物",),
            "ready",
            "",
            0,
            None,
        )
        spare = asset_library.LibraryAsset(
            "video-server",
            "video",
            str(root),
            "机房/server.mp4",
            "机房",
            9.0,
            1080,
            1920,
            5,
            "hash-server",
            "机房服务器巡检",
            ("机房",),
            "ready",
            "",
            0,
            None,
        )
        match = asset_matching.StoryboardMatch(
            shots=(
                asset_matching.StoryboardShot(
                    index=1,
                    text="人物正在工作。",
                    start_seconds=0.0,
                    duration_seconds=3.0,
                    candidates=(matched,),
                ),
                asset_matching.StoryboardShot(
                    index=2,
                    text="服务器巡检。",
                    start_seconds=3.0,
                    duration_seconds=3.0,
                    candidates=(),
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
                return_value=asset_library.ScanSummary(2, 0, 0, 2, 0, 0, ()),
            ),
            patch.object(asset_matching, "match_storyboard", return_value=match),
            patch.object(
                asset_library,
                "list_assets",
                side_effect=lambda **kwargs: (matched, spare)
                if kwargs.get("kind") == "video"
                else (),
            ),
            patch.object(asset_library, "resolve_asset_path", return_value=video_path),
        ):
            app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=60)
            app.session_state["ui_language"] = "en"
            app.session_state["video_script"] = "人物正在工作。服务器巡检。"
            app.run()

            assert [str(item.value) for item in app.exception] == []
            # 缺口镜默认不自动填补：匹配器故意不硬凑无关素材，UI 不能替用户决定。
            assert app.session_state["local_storyboard_selected_ids"] == {
                "1": "video-worker"
            }
            gap_picker = app.selectbox(key="local_storyboard_gap_2")
            assert gap_picker.value == ""
            # 库里全部已索引素材都应可选，供人工替换；展示用不含绝对路径的标签。
            assert set(gap_picker.options) == {
                "Not selected — edit the script or add materials",
                "人物/worker.mp4",
                "机房/server.mp4",
            }
        # 人工选择写回快照后，冻结计划必须能覆盖两镜——这是修复前会抛
        # 「storyboard selections do not cover all shots」的那条路径。
        # AppTest 无法驱动这里的二次交互：本页已有带 format_func 的
        # selectbox，get_widget_states 会按格式化标签反查原始值并抛
        # ValueError，因此选择后的行为在服务层验证。
        filled = asset_matching.with_manual_candidates(match, {"2": spare})
        plan = asset_matching.build_storyboard_plan(
            filled, {"1": "video-worker", "2": "video-server"}
        )
        assert [item["asset_id"] for item in plan] == ["video-worker", "video-server"]

def test_asset_annotation_editor_saves_manual_description_and_tags():
    """标签编辑入口应回填已有人工标注，并把修改写入素材库。"""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        video_path = root / "人物" / "worker.mp4"
        video_path.parent.mkdir()
        video_path.write_bytes(b"video")
        asset = asset_library.LibraryAsset(
            "video-worker",
            "video",
            str(root),
            "人物/worker.mp4",
            "人物",
            4.0,
            1080,
            1920,
            5,
            "hash-worker",
            "自动识别的人物场景",
            ("自动",),
            "ready",
            "",
            0,
            None,
        )
        annotations = asset_library.AssetAnnotations(
            asset_id="video-worker",
            manual_description="工人在车间作业",
            manual_tags=("车间", "工人"),
            auto_description="自动识别的人物场景",
            auto_tags=("自动",),
        )
        app_config = dict(
            config.app,
            video_source="local",
            local_video_library_directory=str(root),
            local_bgm_library_directory="",
        )
        ui_config = dict(config.ui, language="en", bgm_type="")
        saved = []
        with (
            patch.object(config, "app", app_config),
            patch.object(config, "ui", ui_config),
            patch.object(config, "try_save_config", return_value=True),
            patch.object(
                asset_library,
                "scan_library",
                return_value=asset_library.ScanSummary(1, 0, 0, 1, 0, 0, ()),
            ),
            patch.object(
                asset_library,
                "list_assets",
                side_effect=lambda **kwargs: (asset,)
                if kwargs.get("kind") == "video"
                else (),
            ),
            patch.object(
                asset_library, "get_asset_annotations", return_value=annotations
            ),
            patch.object(
                asset_library,
                "update_asset_annotations",
                side_effect=lambda asset_id, **kwargs: saved.append(
                    (asset_id, kwargs)
                ),
            ),
            patch.object(asset_library, "resolve_asset_path", return_value=video_path),
        ):
            app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=60)
            app.session_state["ui_language"] = "en"
            # 标注入口在设置对话框的素材来源页，必须先打开对话框才会渲染。
            app.session_state["settings_dialog_open"] = True
            app.session_state["settings_dialog_target_tab"] = "material"
            app.run()

            assert [str(item.value) for item in app.exception] == []
            # 已有人工标注必须回填，否则保存会把用户之前填的内容清空。
            description_box = app.text_area(key="library_annotation_description")
            assert description_box.value == "工人在车间作业"
            tags_box = app.text_input(key="library_annotation_tags")
            assert tags_box.value == "车间, 工人"

            app.button(key="save_library_annotation_button").click().run()

            assert [str(item.value) for item in app.exception] == []
            assert saved == [
                (
                    "video-worker",
                    {
                        "description": "工人在车间作业",
                        "tags": ["车间", "工人"],
                    },
                )
            ]
