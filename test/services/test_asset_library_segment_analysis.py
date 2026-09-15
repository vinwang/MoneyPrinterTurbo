import json
import unittest
from pathlib import Path
from unittest.mock import patch

from app.services import asset_library_shots as shots


class TestParseWindowAnalysis(unittest.TestCase):
    """Verify per-window analysis is only accepted when it aligns with the windows."""

    def _payload(self, count: int) -> str:
        return json.dumps(
            {
                "windows": [
                    {
                        "index": index + 1,
                        "description": f"第{index + 1}个镜头的画面",
                        "tags": [f"标签{index + 1}"],
                        "mood": "中性",
                    }
                    for index in range(count)
                ]
            },
            ensure_ascii=False,
        )

    def test_one_entry_per_window_is_parsed_in_order(self):
        parsed = shots.parse_window_analysis(self._payload(3), window_count=3)

        self.assertEqual(len(parsed), 3)
        self.assertEqual(parsed[0].description, "第1个镜头的画面")
        self.assertEqual(parsed[2].description, "第3个镜头的画面")
        self.assertEqual(parsed[0].tags, ("标签1", "中性"))

    def test_a_markdown_fenced_response_is_accepted(self):
        # 非 OpenAI 系 Provider 常把 JSON 包在 ```json 围栏里；这曾让真实
        # 重新索引里的一条素材整条失败。
        fenced = "```json\n" + self._payload(2) + "\n```"

        parsed = shots.parse_window_analysis(fenced, window_count=2)

        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0].description, "第1个镜头的画面")

    def test_too_few_entries_is_rejected_rather_than_padded(self):
        with self.assertRaisesRegex(ValueError, "one entry per window"):
            shots.parse_window_analysis(self._payload(2), window_count=3)

    def test_too_many_entries_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "one entry per window"):
            shots.parse_window_analysis(self._payload(4), window_count=3)

    def test_out_of_order_or_duplicate_indexes_are_rejected(self):
        payload = json.dumps(
            {
                "windows": [
                    {"index": 1, "description": "a", "tags": ["x"], "mood": "中性"},
                    {"index": 1, "description": "b", "tags": ["y"], "mood": "中性"},
                ]
            },
            ensure_ascii=False,
        )

        with self.assertRaisesRegex(ValueError, "index"):
            shots.parse_window_analysis(payload, window_count=2)

    def test_empty_description_is_rejected(self):
        payload = json.dumps(
            {
                "windows": [
                    {"index": 1, "description": "  ", "tags": ["x"], "mood": "中性"}
                ]
            },
            ensure_ascii=False,
        )

        with self.assertRaisesRegex(ValueError, "invalid fields"):
            shots.parse_window_analysis(payload, window_count=1)

    def test_unknown_window_fields_are_rejected(self):
        payload = json.dumps(
            {
                "windows": [
                    {
                        "index": 1,
                        "description": "a",
                        "tags": ["x"],
                        "mood": "中性",
                        "asset_id": "injected",
                    }
                ]
            },
            ensure_ascii=False,
        )

        with self.assertRaisesRegex(ValueError, "exactly"):
            shots.parse_window_analysis(payload, window_count=1)

    def test_provider_error_text_is_surfaced(self):
        with self.assertRaisesRegex(ValueError, "quota exceeded"):
            shots.parse_window_analysis("Error: quota exceeded", window_count=1)

    def test_non_json_text_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "JSON"):
            shots.parse_window_analysis("not json", window_count=1)


class TestAnalyzeWindows(unittest.TestCase):
    """Verify batching keeps request size bounded and reports real call volume."""

    def _frames(self, count: int) -> tuple[bytes, ...]:
        return tuple(f"frame-{index}".encode("utf-8") for index in range(count))

    def _responder(self, calls: list[int]):
        def vision_fn(prompt: str, frames, app_config=None) -> str:
            calls.append(len(frames))
            return json.dumps(
                {
                    "windows": [
                        {
                            "index": index + 1,
                            "description": f"镜头{index + 1}",
                            "tags": ["玉米"],
                            "mood": "中性",
                        }
                        for index in range(len(frames))
                    ]
                },
                ensure_ascii=False,
            )

        return vision_fn

    def test_windows_within_the_batch_size_use_one_call(self):
        calls: list[int] = []
        windows = ((0.0, 4.0), (4.0, 8.0), (8.0, 12.0))

        with patch.object(shots, "extract_frames_at", return_value=self._frames(3)):
            analyses, call_count = shots.analyze_windows(
                Path("/library/a.mp4"),
                windows,
                category="玉米",
                vision_fn=self._responder(calls),
                app_config={},
            )

        self.assertEqual(call_count, 1)
        self.assertEqual(calls, [3])
        self.assertEqual(len(analyses), 3)
        self.assertEqual(analyses[1].description, "镜头2")

    def test_many_windows_are_split_into_bounded_batches(self):
        calls: list[int] = []
        windows = tuple((float(i) * 4, float(i) * 4 + 4) for i in range(14))

        with patch.object(
            shots,
            "extract_frames_at",
            side_effect=lambda path, positions: self._frames(len(positions)),
        ):
            analyses, call_count = shots.analyze_windows(
                Path("/library/a.mp4"),
                windows,
                category="玉米",
                vision_fn=self._responder(calls),
                app_config={},
            )

        self.assertEqual(len(analyses), 14)
        self.assertEqual(sum(calls), 14)
        self.assertEqual(call_count, len(calls))
        for size in calls:
            self.assertLessEqual(size, shots._MAX_FRAMES_PER_CALL)

    def test_a_window_whose_frame_cannot_be_extracted_fails_the_asset(self):
        windows = ((0.0, 4.0), (4.0, 8.0))

        with patch.object(shots, "extract_frames_at", return_value=self._frames(1)):
            with self.assertRaisesRegex(ValueError, "one frame per window"):
                shots.analyze_windows(
                    Path("/library/a.mp4"),
                    windows,
                    category="玉米",
                    vision_fn=self._responder([]),
                    app_config={},
                )

    def test_a_broken_batch_is_split_and_retried_instead_of_dropping_the_asset(self):
        calls: list[int] = []
        good = self._responder([])

        def flaky(prompt: str, frames, app_config=None) -> str:
            calls.append(len(frames))
            # 首批 4 帧返回损坏 JSON；拆成 2+2 后都能正常解析。
            if len(frames) == 4 and calls.count(4) == 1:
                return "{broken"
            return good(prompt, frames, app_config)

        windows = tuple((float(i) * 4, float(i) * 4 + 4) for i in range(4))

        with patch.object(
            shots,
            "extract_frames_at",
            side_effect=lambda path, positions: self._frames(len(positions)),
        ):
            analyses, call_count = shots.analyze_windows(
                Path("/library/a.mp4"),
                windows,
                category="玉米",
                vision_fn=flaky,
                app_config={},
            )

        self.assertEqual(len(analyses), 4)
        self.assertEqual(calls, [4, 2, 2])
        self.assertEqual(call_count, 3)

    def test_a_single_frame_that_never_parses_raises_window_analysis_error(self):
        windows = ((0.0, 4.0), (4.0, 8.0))
        calls: list[int] = []

        def always_broken(prompt: str, frames, app_config=None) -> str:
            calls.append(len(frames))
            return "{broken"

        with patch.object(
            shots,
            "extract_frames_at",
            side_effect=lambda path, positions: self._frames(len(positions)),
        ):
            with self.assertRaises(shots.WindowAnalysisError):
                shots.analyze_windows(
                    Path("/library/a.mp4"),
                    windows,
                    category="玉米",
                    vision_fn=always_broken,
                    app_config={},
                )

        # 拆到 1 帧为止：2 -> 1+1，且首个单帧就抛错，第二个不再请求。
        self.assertEqual(calls, [2, 1])

    def test_no_window_means_no_call(self):
        calls: list[int] = []

        analyses, call_count = shots.analyze_windows(
            Path("/library/a.mp4"),
            (),
            category="玉米",
            vision_fn=self._responder(calls),
            app_config={},
        )

        self.assertEqual(analyses, ())
        self.assertEqual(call_count, 0)
        self.assertEqual(calls, [])


class TestAggregateWindowAnalysis(unittest.TestCase):
    """Verify the asset-level summary stays derived from the per-window results."""

    def test_asset_description_joins_windows_in_order(self):
        analyses = (
            shots.WindowAnalysis(description="进门", tags=("居家",)),
            shots.WindowAnalysis(description="坐沙发", tags=("放松", "居家")),
        )

        description, tags = shots.aggregate_window_analysis(analyses)

        self.assertEqual(description, "进门 坐沙发")
        self.assertEqual(tags, ("居家", "放松"))

    def test_aggregating_nothing_is_rejected(self):
        with self.assertRaises(ValueError):
            shots.aggregate_window_analysis(())

    def test_aggregate_description_is_truncated_to_the_stored_limit(self):
        analyses = tuple(
            shots.WindowAnalysis(description="镜头描述" * 40, tags=("玉米",))
            for _ in range(10)
        )

        description, _tags = shots.aggregate_window_analysis(analyses)

        self.assertLessEqual(len(description), shots._MAX_DESCRIPTION_CHARS)


if __name__ == "__main__":
    unittest.main()
