import json
import sqlite3
import unittest

from app.services import asset_library_segments as segments


def _connection() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.executescript(
        """
        CREATE TABLE assets (
            asset_id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            root_path TEXT NOT NULL DEFAULT '/library',
            relative_path TEXT NOT NULL DEFAULT 'a.mp4',
            category TEXT NOT NULL DEFAULT 'uncategorized',
            duration REAL NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            tags_json TEXT NOT NULL DEFAULT '[]',
            analysis_status TEXT NOT NULL,
            analysis_error TEXT NOT NULL DEFAULT '',
            analysis_model TEXT NOT NULL DEFAULT '',
            segments_json TEXT NOT NULL DEFAULT '[]'
        );
        CREATE TABLE segments (
            segment_id TEXT PRIMARY KEY,
            asset_id TEXT NOT NULL,
            source_start_seconds REAL NOT NULL,
            source_end_seconds REAL NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            tags_json TEXT NOT NULL DEFAULT '[]',
            analysis_status TEXT NOT NULL,
            analysis_error TEXT NOT NULL DEFAULT '',
            analysis_model TEXT NOT NULL DEFAULT '',
            UNIQUE(asset_id, source_start_seconds, source_end_seconds)
        );
        """
    )
    return connection


def _record(**overrides):
    record = {
        "asset_id": "video-1",
        "kind": "video",
        "duration": 12.0,
        "description": "整条视频的汇总描述",
        "tags_json": json.dumps(["玉米"], ensure_ascii=False),
        "analysis_status": "ready",
        "analysis_error": "",
        "analysis_model": "test-model",
        "segments_json": "[]",
    }
    record.update(overrides)
    return record


class TestSyncSegmentsWithShotWindows(unittest.TestCase):
    """Per-window analysis must reach the segment rows, not the file-level summary."""

    def test_each_window_keeps_its_own_description_and_range(self):
        connection = _connection()
        record = _record(
            segments_json=json.dumps(
                [
                    {
                        "source_start_seconds": 0.0,
                        "source_end_seconds": 4.0,
                        "description": "进门放下包",
                        "tags": ["居家"],
                    },
                    {
                        "source_start_seconds": 4.0,
                        "source_end_seconds": 12.0,
                        "description": "坐沙发放松",
                        "tags": ["放松", "居家"],
                    },
                ],
                ensure_ascii=False,
            )
        )

        segments.sync_segments(connection, record)

        rows = connection.execute(
            "SELECT * FROM segments ORDER BY source_start_seconds"
        ).fetchall()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["description"], "进门放下包")
        self.assertEqual(rows[1]["description"], "坐沙发放松")
        self.assertEqual(rows[1]["source_start_seconds"], 4.0)
        self.assertEqual(rows[1]["source_end_seconds"], 12.0)
        self.assertEqual(json.loads(rows[1]["tags_json"]), ["放松", "居家"])

    def test_segment_ids_stay_stable_across_reruns(self):
        connection = _connection()
        record = _record(
            duration=6.0,
            segments_json=json.dumps(
                [
                    {
                        "source_start_seconds": 0.0,
                        "source_end_seconds": 6.0,
                        "description": "第一段",
                        "tags": ["玉米"],
                    }
                ],
                ensure_ascii=False,
            ),
        )

        segments.sync_segments(connection, record)
        first = connection.execute("SELECT segment_id FROM segments").fetchone()[0]
        segments.sync_segments(connection, record)
        second = connection.execute("SELECT segment_id FROM segments").fetchone()[0]

        self.assertEqual(first, second)

    def test_analysis_model_records_the_segment_index_version(self):
        connection = _connection()
        record = _record(
            duration=6.0,
            segments_json=json.dumps(
                [
                    {
                        "source_start_seconds": 0.0,
                        "source_end_seconds": 6.0,
                        "description": "第一段",
                        "tags": ["玉米"],
                    }
                ],
                ensure_ascii=False,
            ),
        )

        segments.sync_segments(connection, record)

        model = connection.execute("SELECT analysis_model FROM segments").fetchone()[0]
        self.assertIn("test-model", model)
        self.assertIn(segments._SEGMENT_INDEX_VERSION, model)

    def test_asset_without_per_window_data_falls_back_to_even_windows(self):
        connection = _connection()

        segments.sync_segments(connection, _record(segments_json="[]"))

        rows = connection.execute(
            "SELECT * FROM segments ORDER BY source_start_seconds"
        ).fetchall()
        self.assertEqual(len(rows), 3)
        # 没有逐窗口分析时沿用整条描述，行为与改造前一致。
        self.assertEqual(rows[0]["description"], "整条视频的汇总描述")
        self.assertEqual(rows[2]["source_end_seconds"], 12.0)

    def test_windows_that_do_not_cover_the_duration_are_rejected(self):
        connection = _connection()
        record = _record(
            segments_json=json.dumps(
                [
                    {
                        "source_start_seconds": 0.0,
                        "source_end_seconds": 5.0,
                        "description": "第一段",
                        "tags": ["玉米"],
                    }
                ],
                ensure_ascii=False,
            )
        )

        with self.assertRaisesRegex(segments.SegmentIndexError, "cover"):
            segments.sync_segments(connection, record)

    def test_overlapping_windows_are_rejected(self):
        connection = _connection()
        record = _record(
            segments_json=json.dumps(
                [
                    {
                        "source_start_seconds": 0.0,
                        "source_end_seconds": 7.0,
                        "description": "第一段",
                        "tags": ["玉米"],
                    },
                    {
                        "source_start_seconds": 5.0,
                        "source_end_seconds": 12.0,
                        "description": "第二段",
                        "tags": ["玉米"],
                    },
                ],
                ensure_ascii=False,
            )
        )

        with self.assertRaisesRegex(segments.SegmentIndexError, "contiguous"):
            segments.sync_segments(connection, record)

    def test_segment_without_description_is_rejected(self):
        connection = _connection()
        record = _record(
            segments_json=json.dumps(
                [
                    {
                        "source_start_seconds": 0.0,
                        "source_end_seconds": 12.0,
                        "description": "   ",
                        "tags": ["玉米"],
                    }
                ],
                ensure_ascii=False,
            )
        )

        with self.assertRaisesRegex(segments.SegmentIndexError, "description"):
            segments.sync_segments(connection, record)

    def test_failed_asset_keeps_its_windows_marked_failed(self):
        connection = _connection()
        record = _record(
            analysis_status="failed",
            analysis_error="vision unavailable",
            segments_json="[]",
        )

        segments.sync_segments(connection, record)

        rows = connection.execute("SELECT * FROM segments").fetchall()
        self.assertTrue(rows)
        self.assertTrue(all(row["analysis_status"] == "failed" for row in rows))
        self.assertTrue(all(row["analysis_error"] == "vision unavailable" for row in rows))


class TestBackfillWithShotWindows(unittest.TestCase):
    """Legacy rows must still be backfilled without per-window data."""

    def test_legacy_asset_is_backfilled_from_stored_columns(self):
        connection = _connection()
        connection.execute(
            """
            INSERT INTO assets(
                asset_id, kind, duration, description, tags_json,
                analysis_status, analysis_model, segments_json
            ) VALUES('video-legacy', 'video', 12.0, '旧版描述', '["玉米"]',
                     'ready', 'old-model', '[]')
            """
        )

        segments.backfill_segments(connection)

        rows = connection.execute(
            "SELECT * FROM segments ORDER BY source_start_seconds"
        ).fetchall()
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["description"], "旧版描述")


if __name__ == "__main__":
    unittest.main()
