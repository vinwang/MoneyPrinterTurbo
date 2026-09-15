import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[2]))

from app.services import asset_library_runtime  # noqa: E402


class TestManifestStoryboardRuntime(unittest.TestCase):
    """Validate task-owned snapshots for selector manifest storyboard plans."""

    def test_manifest_storyboard_sources_are_snapshotted_without_library_db(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            managed_root = root / "storage" / "local_videos"
            managed_root.mkdir(parents=True)
            source = managed_root / "template-video-1.mp4"
            source.write_bytes(b"manifest-video")
            source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            plan = [
                {
                    "asset_id": "video-1",
                    "asset_sha256": source_hash,
                    "source_path": str(source),
                    "source_kind": "manifest",
                    "source_start_seconds": 0.0,
                    "source_end_seconds": 1.0,
                    "target_start_seconds": 0.0,
                    "target_end_seconds": 1.0,
                    "shot_index": 1,
                    "text": "画面",
                }
            ]
            with patch(
                "app.services.asset_library_runtime.utils.storage_dir",
                return_value=str(managed_root),
            ), patch(
                "app.services.asset_library_runtime.utils.task_dir",
                return_value=str(root / "storage" / "tasks" / "task-1"),
            ):
                materialized = asset_library_runtime.materialize_manifest_storyboard_sources(
                    "task-1",
                    plan,
                )

            snapshot = Path(materialized[0]["source_path"])
            self.assertTrue(snapshot.is_file())
            self.assertEqual(snapshot.read_bytes(), b"manifest-video")
            self.assertEqual(materialized[0]["source_sha256"], source_hash)


if __name__ == "__main__":
    unittest.main()
