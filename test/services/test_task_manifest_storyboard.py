import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[2]))

from app.models.schema import MaterialInfo, VideoParams  # noqa: E402
from app.services import task  # noqa: E402


class TestTaskManifestStoryboard(unittest.TestCase):
    """Ensure manifest-backed storyboard plans use the direct snapshot path."""

    def test_get_video_materials_uses_manifest_snapshotter(self):
        plan = [
            {
                "shot_index": 1,
                "text": "画面",
                "visual_query": "画面",
                "asset_id": "video-1",
                "asset_sha256": "a" * 64,
                "source_path": "/managed/template-video-1.mp4",
                "source_kind": "manifest",
                "source_start_seconds": 0.0,
                "source_end_seconds": 1.0,
                "target_start_seconds": 0.0,
                "target_end_seconds": 1.0,
            }
        ]
        materialized = [
            {
                **plan[0],
                "source_path": "/task/storyboard_sources/manifest-video-1.mp4",
            }
        ]
        params = VideoParams(
            video_subject="测试",
            video_source="local",
            local_storyboard_plan=plan,
        )
        prepared = MaterialInfo(
            provider="local",
            url=materialized[0]["source_path"],
            duration=1,
        )
        with patch.object(
            task.asset_library_runtime,
            "materialize_manifest_storyboard_sources",
            return_value=tuple(materialized),
        ) as snapshotter, patch.object(
            task.video,
            "preprocess_video",
            return_value=(prepared,),
        ):
            result = task.get_video_materials(
                "manifest-task",
                params,
                video_terms=(),
                audio_duration=1.0,
            )

        snapshotter.assert_called_once_with("manifest-task", plan)
        self.assertEqual(result, (prepared.url,))
        self.assertEqual(
            params.local_storyboard_plan[0]["source_path"],
            prepared.url,
        )


if __name__ == "__main__":
    unittest.main()
