import unittest
from pathlib import Path
from unittest.mock import patch

from app.services import asset_library_shots as shots


class TestParseSceneBoundaries(unittest.TestCase):
    """Verify scene-score output is parsed without trusting the model of the day."""

    def test_boundaries_are_read_from_metadata_output(self):
        output = (
            "frame:0    pts:58368   pts_time:3.8\n"
            "lavfi.scene_score=0.455984\n"
            "frame:1    pts:108032  pts_time:7.033333\n"
            "lavfi.scene_score=0.493113\n"
        )

        self.assertEqual(
            shots._parse_scene_boundaries(output, duration=12.0),
            (3.8, 7.033333),
        )

    def test_boundaries_outside_the_duration_are_dropped(self):
        output = (
            "frame:0    pts_time:-1.0\n"
            "frame:1    pts_time:5.0\n"
            "frame:2    pts_time:99.0\n"
        )

        self.assertEqual(
            shots._parse_scene_boundaries(output, duration=12.0),
            (5.0,),
        )

    def test_duplicate_and_unsorted_boundaries_are_normalized(self):
        output = "pts_time:7.0\npts_time:3.0\npts_time:7.0\n"

        self.assertEqual(
            shots._parse_scene_boundaries(output, duration=12.0),
            (3.0, 7.0),
        )

    def test_output_without_any_timestamp_yields_no_boundary(self):
        self.assertEqual(
            shots._parse_scene_boundaries("no timestamps here", duration=12.0),
            (),
        )


class TestBuildShotWindows(unittest.TestCase):
    """Verify detected cuts become usable windows inside the 3-5s target."""

    def test_a_shot_inside_the_target_range_is_kept_as_one_window(self):
        self.assertEqual(
            shots.build_shot_windows(12.0, (4.0, 8.0)),
            ((0.0, 4.0), (4.0, 8.0), (8.0, 12.0)),
        )

    def test_a_short_shot_is_merged_forward_instead_of_kept_as_a_stub(self):
        # 1 秒的镜头裁不出可用画面，必须并入相邻镜头；合并后超过 5 秒上限
        # 的部分再均分，所以不会留下 1 秒的废候选。
        windows = shots.build_shot_windows(12.0, (1.0, 6.0))

        self.assertNotIn(1.0, [end for _start, end in windows])
        for start, end in windows:
            self.assertGreaterEqual(end - start, shots._MIN_SHOT_SECONDS)
            self.assertLessEqual(end - start, shots._MAX_SHOT_SECONDS)

    def test_a_trailing_short_shot_is_merged_backward(self):
        windows = shots.build_shot_windows(12.0, (5.0, 11.0))

        # 末尾 1 秒并入前一镜头，不作为独立窗口出现。
        self.assertEqual(windows[0], (0.0, 5.0))
        self.assertNotIn((11.0, 12.0), windows)
        for start, end in windows:
            self.assertGreaterEqual(end - start, shots._MIN_SHOT_SECONDS)

    def test_a_long_shot_is_subdivided_into_even_windows(self):
        windows = shots.build_shot_windows(20.0, ())

        self.assertEqual(len(windows), 4)
        for start, end in windows:
            self.assertAlmostEqual(end - start, 5.0)
        self.assertEqual(windows[0][0], 0.0)
        self.assertEqual(windows[-1][1], 20.0)

    def test_windows_are_contiguous_and_cover_the_whole_duration(self):
        windows = shots.build_shot_windows(37.5, (4.2, 9.9, 10.4, 28.0))

        self.assertEqual(windows[0][0], 0.0)
        self.assertAlmostEqual(windows[-1][1], 37.5)
        for previous, current in zip(windows, windows[1:]):
            self.assertEqual(previous[1], current[0])

    def test_a_span_just_over_the_maximum_stays_whole_rather_than_split_into_stubs(self):
        # 5–6 秒的镜头拆成两半会得到两个 2.5–3 秒以下的废窗口，宁可留一个
        # 略超上限的可用窗口。与 asset_library_segments 的既有取舍一致。
        windows = shots.build_shot_windows(5.2, ())

        self.assertEqual(windows, ((0.0, 5.2),))

    def test_a_video_shorter_than_the_minimum_stays_one_window(self):
        self.assertEqual(shots.build_shot_windows(2.0, ()), ((0.0, 2.0),))

    def test_invalid_duration_yields_no_window(self):
        self.assertEqual(shots.build_shot_windows(0.0, ()), ())
        self.assertEqual(shots.build_shot_windows(float("nan"), ()), ())

    def test_boundaries_are_ignored_when_they_fall_outside_the_duration(self):
        self.assertEqual(
            shots.build_shot_windows(8.0, (-1.0, 4.0, 20.0)),
            ((0.0, 4.0), (4.0, 8.0)),
        )


class TestDetectShotWindows(unittest.TestCase):
    """Verify detection degrades to time slicing instead of failing the scan."""

    def test_detected_cuts_drive_the_windows(self):
        with patch.object(
            shots,
            "_run_scene_detection",
            return_value="pts_time:4.0\npts_time:8.0\n",
        ):
            windows = shots.detect_shot_windows(Path("/library/a.mp4"), 12.0)

        self.assertEqual(windows, ((0.0, 4.0), (4.0, 8.0), (8.0, 12.0)))

    def test_detection_failure_falls_back_to_even_windows(self):
        with patch.object(
            shots,
            "_run_scene_detection",
            side_effect=shots.ShotDetectionError("ffmpeg unavailable"),
        ):
            windows = shots.detect_shot_windows(Path("/library/a.mp4"), 12.0)

        # 检测不可用时退回均分，不能让整次扫描失败。
        self.assertEqual(windows, shots.build_shot_windows(12.0, ()))
        self.assertEqual(windows[0][0], 0.0)
        self.assertEqual(windows[-1][1], 12.0)

    def test_detection_is_skipped_for_clips_under_the_minimum(self):
        with patch.object(shots, "_run_scene_detection") as detector:
            windows = shots.detect_shot_windows(Path("/library/a.mp4"), 2.5)

        detector.assert_not_called()
        self.assertEqual(windows, ((0.0, 2.5),))


class TestFrameSampling(unittest.TestCase):
    """Verify each window gets its own sampling position."""

    def test_每个窗口取中点作为代表帧位置(self):
        positions = shots.window_frame_positions(((0.0, 4.0), (4.0, 10.0)))

        self.assertEqual(positions, (2.0, 7.0))

    def test_position_stays_inside_the_window_for_a_very_short_window(self):
        positions = shots.window_frame_positions(((0.0, 0.2),))

        self.assertEqual(len(positions), 1)
        self.assertGreater(positions[0], 0.0)
        self.assertLess(positions[0], 0.2)


if __name__ == "__main__":
    unittest.main()
