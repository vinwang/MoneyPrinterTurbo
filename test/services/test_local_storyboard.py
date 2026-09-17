import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.services import (
    asset_library,
    asset_library_runtime,
    asset_library_segments,
    asset_library_shots,
    asset_matching,
    asset_shot_planner,
    video,
)


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
                    "description": description,
                    "tags": tags,
                    "mood": "平静",
                }
                for index in range(frame_count)
            ]
        },
        ensure_ascii=False,
    )


def _asset(
    asset_id: str,
    *,
    relative_path: str,
    description: str,
    tags: tuple[str, ...],
    duration: float = 6.0,
) -> asset_library.LibraryAsset:
    return asset_library.LibraryAsset(
        asset_id=asset_id,
        kind="video",
        root_path="/library",
        relative_path=relative_path,
        category="uncategorized",
        duration=duration,
        width=1080,
        height=1920,
        size_bytes=1,
        sha256=f"hash-{asset_id}",
        description=description,
        tags=tags,
        analysis_status="ready",
        analysis_error="",
        use_count=0,
        last_used_at=None,
    )


def _segment(asset: asset_library.LibraryAsset) -> asset_library.LibrarySegment:
    return asset_library.LibrarySegment(
        segment_id=f"segment-{asset.asset_id}",
        asset_id=asset.asset_id,
        source_start_seconds=0.0,
        source_end_seconds=5.0,
        description=asset.description,
        tags=asset.tags,
        analysis_status="ready",
        analysis_error="",
        analysis_model="test",
    )


class TestLocalStoryboard(unittest.TestCase):
    def test_relevance_filter_ignores_filename_and_does_not_reuse_excluded_assets(self):
        unrelated = _asset(
            "unrelated",
            relative_path="海边.mp4",
            description="办公室会议",
            tags=("会议",),
        )
        relevant = _asset(
            "relevant",
            relative_path="office.mp4",
            description="海边散步",
            tags=("海边",),
        )
        ranked = asset_matching._rank_segments(
            "海边",
            (_segment(unrelated), _segment(relevant)),
            {unrelated.asset_id: unrelated, relevant.asset_id: relevant},
            count=3,
        )

        self.assertEqual(tuple(item[0].asset_id for item in ranked), ("relevant",))
        self.assertEqual(
            asset_matching._rank_segments(
                "海边",
                (_segment(relevant),),
                {relevant.asset_id: relevant},
                excluded_ids={"relevant"},
            ),
            (),
        )

    def test_segments_of_one_video_are_scored_on_their_own_description(self):
        # 逐窗口分析后，资产级描述是所有窗口的汇总。若继续把它计入片段分数，
        # 同一视频的每个片段都会拿到整条视频的语义，逐窗口索引就白做了。
        asset = _asset(
            "multi-shot",
            relative_path="home.mp4",
            description="进门放下包 坐沙发放松 产品特写",
            tags=("居家", "放松", "产品"),
            duration=12.0,
        )
        entrance = asset_library.LibrarySegment(
            segment_id="segment-entrance",
            asset_id=asset.asset_id,
            source_start_seconds=0.0,
            source_end_seconds=4.0,
            description="进门放下包",
            tags=("居家",),
            analysis_status="ready",
            analysis_error="",
            analysis_model="test",
        )
        sofa = asset_library.LibrarySegment(
            segment_id="segment-sofa",
            asset_id=asset.asset_id,
            source_start_seconds=4.0,
            source_end_seconds=8.0,
            description="坐沙发放松",
            tags=("放松",),
            analysis_status="ready",
            analysis_error="",
            analysis_model="test",
        )

        entrance_score = asset_matching._segment_score("坐沙发放松", entrance, asset)
        sofa_score = asset_matching._segment_score("坐沙发放松", sofa, asset)

        self.assertGreater(sofa_score, entrance_score)

    def test_bgm_still_scores_on_asset_level_description(self):
        # BGM 没有片段索引，资产级描述是唯一语义来源，不能一并摘掉。
        track = asset_library.LibraryAsset(
            asset_id="bgm-1",
            kind="bgm",
            root_path="/library",
            relative_path="warm.mp3",
            category="轻快",
            duration=120.0,
            width=None,
            height=None,
            size_bytes=1,
            sha256="hash-bgm",
            description="温暖治愈的原声吉他",
            tags=("治愈",),
            analysis_status="ready",
            analysis_error="",
            use_count=0,
            last_used_at=None,
        )
        segment = asset_library.LibrarySegment(
            segment_id="bgm-segment",
            asset_id=track.asset_id,
            source_start_seconds=0.0,
            source_end_seconds=120.0,
            description="",
            tags=(),
            analysis_status="ready",
            analysis_error="",
            analysis_model="bgm-metadata",
        )

        self.assertGreater(
            asset_matching._segment_score("温暖治愈", segment, track),
            0.0,
        )

    def test_must_not_appear_excludes_a_candidate_that_would_otherwise_win(self):
        # 方案里的原例：「下班回家，终于可以放松一下」不应因为「工作」相关词
        # 就选办公会议。模型给出的 must_not_appear 必须真的挡住这类候选。
        office = _asset(
            "office",
            relative_path="a.mp4",
            description="办公室会议桌前讨论工作",
            tags=("办公室", "会议"),
        )
        sofa = _asset(
            "sofa",
            relative_path="b.mp4",
            description="下班回家坐在沙发上休息",
            tags=("居家",),
        )
        assets = {office.asset_id: office, sofa.asset_id: sofa}
        segments = (_segment(office), _segment(sofa))

        unconstrained = asset_matching._rank_segments(
            "办公室 放松",
            segments,
            assets,
            count=3,
        )
        constrained = asset_matching._rank_segments(
            "办公室 放松",
            segments,
            assets,
            count=3,
            must_not_appear=("办公室", "会议"),
        )

        self.assertIn("office", [item[0].asset_id for item in unconstrained])
        self.assertNotIn("office", [item[0].asset_id for item in constrained])

    def test_must_match_ranks_the_matching_candidate_first_without_gapping_others(self):
        # must_match 是加分而非硬过滤：模型给的是它想象中的画面元素，多半不会
        # 逐字出现在素材描述里。实测按字面硬过滤会把 17 个分镜里的 13 个变成
        # 缺口，连本该命中的素材也被挡掉，所以这里只要求「命中的排前面」。
        packaged = _asset(
            "packaged",
            relative_path="a.mp4",
            description="真空包装玉米整齐排列",
            tags=("真空包装",),
        )
        plain = _asset(
            "plain",
            relative_path="b.mp4",
            description="金黄玉米特写颗粒饱满",
            tags=("玉米",),
        )
        assets = {packaged.asset_id: packaged, plain.asset_id: plain}
        segments = (_segment(packaged), _segment(plain))

        ranked = asset_matching._rank_segments(
            "玉米",
            segments,
            assets,
            count=3,
            must_match=("真空包装",),
        )

        self.assertEqual(ranked[0][0].asset_id, "packaged")
        self.assertEqual(len(ranked), 2)

    def test_an_unmatchable_must_match_does_not_empty_the_candidate_list(self):
        plain = _asset(
            "plain",
            relative_path="b.mp4",
            description="金黄玉米特写",
            tags=("玉米",),
        )

        ranked = asset_matching._rank_segments(
            "玉米",
            (_segment(plain),),
            {plain.asset_id: plain},
            count=3,
            must_match=("数据中心",),
        )

        self.assertEqual([item[0].asset_id for item in ranked], ["plain"])

    def test_planned_direct_shot_does_not_get_the_topic_prepended(self):
        # 评测集暴露的问题：query_context 被拼进每一镜，把「现在下单立减」
        # 这类短 CTA 句的语义冲掉。模型判为 direct 的镜不应再补主题词。
        asset = _asset(
            "asset-1",
            relative_path="corn.mp4",
            description="促销价格与购物车截图",
            tags=("促销",),
        )
        planned = (
            asset_shot_planner.PlannedShot(
                index=1,
                text="现在下单立减。",
                script_start=0,
                script_end=7,
                visual_query="促销价格页面",
                must_match=(),
                must_not_appear=(),
                expression="direct",
            ),
        )
        with (
            patch.object(
                asset_matching.library,
                "list_assets",
                side_effect=lambda **kwargs: (asset,)
                if kwargs.get("kind") == "video"
                else (),
            ),
            patch.object(
                asset_matching.library,
                "list_segments",
                return_value=(_segment(asset),),
            ),
        ):
            result = asset_matching.match_storyboard(
                "现在下单立减。",
                clip_duration=3,
                query_context="东北糯玉米",
                planned_shots=planned,
            )

        self.assertEqual(result.shots[0].visual_query, "促销价格页面")

    def test_planned_ambience_shot_still_receives_the_topic(self):
        asset = _asset(
            "asset-1",
            relative_path="corn.mp4",
            description="居家餐桌摆盘",
            tags=("居家",),
        )
        planned = (
            asset_shot_planner.PlannedShot(
                index=1,
                text="忙碌了一整天。",
                script_start=0,
                script_end=7,
                visual_query="居家放松氛围",
                must_match=(),
                must_not_appear=(),
                expression="ambience",
            ),
        )
        with (
            patch.object(
                asset_matching.library,
                "list_assets",
                side_effect=lambda **kwargs: (asset,)
                if kwargs.get("kind") == "video"
                else (),
            ),
            patch.object(
                asset_matching.library,
                "list_segments",
                return_value=(_segment(asset),),
            ),
        ):
            result = asset_matching.match_storyboard(
                "忙碌了一整天。",
                clip_duration=3,
                query_context="东北糯玉米",
                planned_shots=planned,
            )

        self.assertEqual(result.shots[0].visual_query, "东北糯玉米 居家放松氛围")

    def test_planned_shots_cover_the_script_verbatim(self):
        asset = _asset(
            "asset-1",
            relative_path="corn.mp4",
            description="玉米特写",
            tags=("玉米",),
        )
        script = "姐妹们快看。这个玉米真甜。"
        planned = (
            asset_shot_planner.PlannedShot(
                index=1,
                text="姐妹们快看。",
                script_start=0,
                script_end=6,
                visual_query="玉米特写",
                must_match=(),
                must_not_appear=(),
                expression="direct",
            ),
            asset_shot_planner.PlannedShot(
                index=2,
                text="这个玉米真甜。",
                script_start=6,
                script_end=13,
                visual_query="掰开玉米",
                must_match=(),
                must_not_appear=(),
                expression="direct",
            ),
        )
        with (
            patch.object(
                asset_matching.library,
                "list_assets",
                side_effect=lambda **kwargs: (asset,)
                if kwargs.get("kind") == "video"
                else (),
            ),
            patch.object(
                asset_matching.library,
                "list_segments",
                return_value=(_segment(asset),),
            ),
        ):
            result = asset_matching.match_storyboard(
                script,
                clip_duration=3,
                planned_shots=planned,
            )

        self.assertEqual("".join(shot.text for shot in result.shots), script)
        self.assertEqual(result.shots[0].script_start, 0)
        self.assertEqual(result.shots[1].script_end, 13)

    def test_long_clause_is_not_cut_at_arbitrary_character_boundary(self):
        shots = asset_matching._make_shots(
            "下班回家终于可以坐在沙发上放松一下然后好好休息。",
            clip_duration=3,
        )

        self.assertEqual(len(shots), 1)
        self.assertEqual(shots[0].text, "下班回家终于可以坐在沙发上放松一下然后好好休息。")

    def test_segment_windows_keep_usable_lengths_without_short_tail(self):
        self.assertEqual(
            asset_library_segments._segment_windows(6.0),
            ((0.0, 3.0), (3.0, 6.0)),
        )
        self.assertEqual(
            asset_library_segments._segment_windows(12.0),
            ((0.0, 4.0), (4.0, 8.0), (8.0, 12.0)),
        )

    def test_match_storyboard_keeps_a_missing_shot_as_an_explicit_gap(self):
        asset = _asset(
            "asset-1",
            relative_path="relax.mp4",
            description="沙发休息",
            tags=("放松",),
        )
        with (
            patch.object(
                asset_matching.library,
                "list_assets",
                side_effect=lambda **kwargs: (asset,)
                if kwargs.get("kind") == "video"
                else (),
            ),
            patch.object(
                asset_matching.library,
                "list_segments",
                return_value=(_segment(asset),),
            ),
        ):
            result = asset_matching.match_storyboard(
                "放松一下。会议开始。",
                clip_duration=1,
            )

        self.assertEqual(result.shots[0].candidates[0].asset_id, "asset-1")
        self.assertEqual(result.shots[1].candidates, ())
        with self.assertRaisesRegex(
            asset_library.AssetLibraryError,
            "do not cover",
        ):
            asset_matching.build_storyboard_plan(result, {"1": "asset-1"})

    def test_uncovered_shot_error_names_the_shots_that_need_a_material(self):
        # 缺口镜的报错必须指出是哪一镜缺素材，否则用户在 WebUI 只看到
        # 「selections do not cover all shots」，无从判断该改文案还是补素材。
        asset = _asset(
            "asset-1",
            relative_path="relax.mp4",
            description="沙发休息",
            tags=("放松",),
        )
        match = asset_matching.StoryboardMatch(
            shots=(
                asset_matching.StoryboardShot(
                    index=1,
                    text="放松一下。",
                    start_seconds=0.0,
                    duration_seconds=2.0,
                    candidates=(asset,),
                ),
                asset_matching.StoryboardShot(
                    index=2,
                    text="会议开始。",
                    start_seconds=2.0,
                    duration_seconds=2.0,
                    candidates=(),
                ),
                asset_matching.StoryboardShot(
                    index=3,
                    text="服务器巡检。",
                    start_seconds=4.0,
                    duration_seconds=2.0,
                    candidates=(),
                ),
            ),
            bgm_candidates=(),
        )

        with self.assertRaises(asset_library.AssetLibraryError) as caught:
            asset_matching.build_storyboard_plan(match, {"1": "asset-1"})

        message = str(caught.exception)
        self.assertIn("do not cover", message)
        self.assertIn("2", message)
        self.assertIn("3", message)

    def test_manual_candidate_fills_a_gap_shot_so_the_plan_can_be_built(self):
        # 方案要求缺口镜「留缺口让人工替换」。人工选择的素材写回候选快照后，
        # 冻结计划的既有校验（候选必须在快照里）无需放宽即可通过。
        matched = _asset(
            "asset-1",
            relative_path="relax.mp4",
            description="沙发休息",
            tags=("放松",),
        )
        manual = _asset(
            "asset-2",
            relative_path="server.mp4",
            description="机房服务器",
            tags=("机房",),
            duration=9.0,
        )
        match = asset_matching.StoryboardMatch(
            shots=(
                asset_matching.StoryboardShot(
                    index=1,
                    text="放松一下。",
                    start_seconds=0.0,
                    duration_seconds=2.0,
                    candidates=(matched,),
                    candidate_segments=(_segment(matched),),
                ),
                asset_matching.StoryboardShot(
                    index=2,
                    text="服务器巡检。",
                    start_seconds=2.0,
                    duration_seconds=2.0,
                    candidates=(),
                ),
            ),
            bgm_candidates=(),
        )

        filled = asset_matching.with_manual_candidates(match, {"2": manual})
        plan = asset_matching.build_storyboard_plan(
            filled, {"1": "asset-1", "2": "asset-2"}
        )

        self.assertEqual([item["asset_id"] for item in plan], ["asset-1", "asset-2"])
        manual_entry = plan[1]
        self.assertEqual(manual_entry["source_start_seconds"], 0.0)
        # 人工素材没有片段分析，整条可用，裁切范围就是整个素材时长。
        self.assertEqual(manual_entry["source_end_seconds"], 9.0)
        self.assertIn("manual", manual_entry["selection_reason"].lower())
        # 人工选择没有相关性分数，不能伪造一个数字冒充匹配结果。
        self.assertIsNone(manual_entry["relevance_score"])

    def test_manual_candidate_refuses_to_overwrite_a_matched_shot(self):
        # 人工替换只用于填补缺口；覆盖已有候选会让快照与用户看到的不一致。
        asset = _asset(
            "asset-1",
            relative_path="relax.mp4",
            description="沙发休息",
            tags=("放松",),
        )
        match = asset_matching.StoryboardMatch(
            shots=(
                asset_matching.StoryboardShot(
                    index=1,
                    text="放松一下。",
                    start_seconds=0.0,
                    duration_seconds=2.0,
                    candidates=(asset,),
                ),
            ),
            bgm_candidates=(),
        )

        with self.assertRaisesRegex(asset_library.AssetLibraryError, "already has"):
            asset_matching.with_manual_candidates(match, {"1": asset})

    def test_manual_candidate_rejects_an_unknown_shot_index(self):
        asset = _asset(
            "asset-1",
            relative_path="relax.mp4",
            description="沙发休息",
            tags=("放松",),
        )
        match = asset_matching.StoryboardMatch(
            shots=(
                asset_matching.StoryboardShot(
                    index=1,
                    text="放松一下。",
                    start_seconds=0.0,
                    duration_seconds=2.0,
                    candidates=(),
                ),
            ),
            bgm_candidates=(),
        )

        with self.assertRaisesRegex(asset_library.AssetLibraryError, "shot"):
            asset_matching.with_manual_candidates(match, {"7": asset})

    def test_match_storyboard_uses_query_context_for_plain_narration(self):
        asset = _asset(
            "asset-1",
            relative_path="corn.mp4",
            description="东北糯玉米颗粒饱满",
            tags=("玉米", "糯玉米"),
        )
        with (
            patch.object(
                asset_matching.library,
                "list_assets",
                side_effect=lambda **kwargs: (asset,)
                if kwargs.get("kind") == "video"
                else (),
            ),
            patch.object(
                asset_matching.library,
                "list_segments",
                return_value=(_segment(asset),),
            ),
        ):
            result = asset_matching.match_storyboard(
                "姐妹们快看",
                clip_duration=1,
                query_context="东北糯玉米",
            )

        self.assertEqual(result.shots[0].visual_query, "东北糯玉米 姐妹们快看")
        self.assertEqual(result.shots[0].candidates[0].asset_id, "asset-1")

    def test_storyboard_plan_aligns_to_subtitles_and_rejects_mismatch(self):
        asset = _asset(
            "asset-1",
            relative_path="relax.mp4",
            description="沙发休息",
            tags=("放松",),
        )
        shot = asset_matching.StoryboardShot(
            index=1,
            text="下班回家，放松一下。",
            start_seconds=0.0,
            duration_seconds=4.0,
            candidates=(asset,),
            candidate_segments=(_segment(asset),),
            candidate_scores=(0.8,),
            candidate_reasons=("semantic description/tag match score=0.800",),
        )
        match = asset_matching.StoryboardMatch((shot,), ())
        plan = asset_matching.build_storyboard_plan(match, {"1": "asset-1"})

        with tempfile.TemporaryDirectory() as directory:
            subtitle_path = Path(directory) / "subtitle.srt"
            subtitle_path.write_text(
                "1\n00:00:00,000 --> 00:00:03,000\n下班回家，放松一下。\n",
                encoding="utf-8",
            )
            aligned = asset_matching.align_storyboard_plan(
                plan,
                audio_duration=3.0,
                subtitle_path=subtitle_path,
            )

        self.assertEqual(aligned[0]["target_end_seconds"], 3.0)
        self.assertEqual(aligned[0]["source_end_seconds"], 3.0)
        with tempfile.TemporaryDirectory() as directory:
            subtitle_path = Path(directory) / "subtitle.srt"
            subtitle_path.write_text(
                "1\n00:00:00,000 --> 00:00:03,000\n不一致的旁白。\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                asset_library.AssetLibraryError,
                "does not cover",
            ):
                asset_matching.align_storyboard_plan(
                    plan,
                    audio_duration=3.0,
                    subtitle_path=subtitle_path,
                )

    def test_shot_boundary_follows_subtitle_timing_not_character_count(self):
        # 第一句念得慢（0–6s），第二句念得快（6.5–8s）。按字数比例分配会让画面在
        # 3.6s 就切走，而那时第一句还没念完——这正是「文字对不上口播」。
        plan = (
            {
                "shot_index": 1,
                "text": "姐妹们快看。",
                "source_start_seconds": 0.0,
                "source_end_seconds": 20.0,
            },
            {
                "shot_index": 2,
                "text": "这个玉米真甜。",
                "source_start_seconds": 0.0,
                "source_end_seconds": 20.0,
            },
        )

        with tempfile.TemporaryDirectory() as directory:
            subtitle_path = Path(directory) / "subtitle.srt"
            subtitle_path.write_text(
                "1\n00:00:00,000 --> 00:00:06,000\n姐妹们快看。\n\n"
                "2\n00:00:06,500 --> 00:00:08,000\n这个玉米真甜。\n",
                encoding="utf-8",
            )
            aligned = asset_matching.align_storyboard_plan(
                plan,
                audio_duration=8.0,
                subtitle_path=subtitle_path,
            )

        # 切点落在两句之间的停顿里（6.0–6.5），而不是字数比例的 3.64。
        self.assertGreaterEqual(aligned[0]["target_end_seconds"], 6.0)
        self.assertLessEqual(aligned[0]["target_end_seconds"], 6.5)
        self.assertEqual(aligned[1]["target_end_seconds"], 8.0)

    def test_shot_boundary_inside_one_subtitle_cue_is_interpolated(self):
        plan = (
            {
                "shot_index": 1,
                "text": "姐妹们快看。",
                "source_start_seconds": 0.0,
                "source_end_seconds": 20.0,
            },
            {
                "shot_index": 2,
                "text": "这个玉米真甜。",
                "source_start_seconds": 0.0,
                "source_end_seconds": 20.0,
            },
        )

        with tempfile.TemporaryDirectory() as directory:
            subtitle_path = Path(directory) / "subtitle.srt"
            subtitle_path.write_text(
                "1\n00:00:00,000 --> 00:00:11,000\n姐妹们快看。这个玉米真甜。\n",
                encoding="utf-8",
            )
            aligned = asset_matching.align_storyboard_plan(
                plan,
                audio_duration=11.0,
                subtitle_path=subtitle_path,
            )

        # 一条字幕跨两个分镜时按字符位置插值：5/11 * 11 = 5.0。
        self.assertAlmostEqual(aligned[0]["target_end_seconds"], 5.0, places=3)

    def test_leading_and_trailing_silence_stay_inside_the_edge_shots(self):
        plan = (
            {
                "shot_index": 1,
                "text": "姐妹们快看。",
                "source_start_seconds": 0.0,
                "source_end_seconds": 30.0,
            },
            {
                "shot_index": 2,
                "text": "这个玉米真甜。",
                "source_start_seconds": 0.0,
                "source_end_seconds": 30.0,
            },
        )

        with tempfile.TemporaryDirectory() as directory:
            subtitle_path = Path(directory) / "subtitle.srt"
            subtitle_path.write_text(
                "1\n00:00:02,000 --> 00:00:05,000\n姐妹们快看。\n\n"
                "2\n00:00:05,000 --> 00:00:09,000\n这个玉米真甜。\n",
                encoding="utf-8",
            )
            aligned = asset_matching.align_storyboard_plan(
                plan,
                audio_duration=12.0,
                subtitle_path=subtitle_path,
            )

        # 片头静音归第一个分镜，片尾静音归最后一个分镜，时间轴不留空洞。
        self.assertEqual(aligned[0]["target_start_seconds"], 0.0)
        self.assertAlmostEqual(aligned[0]["target_end_seconds"], 5.0, places=3)
        self.assertEqual(aligned[-1]["target_end_seconds"], 12.0)

    def test_subtitle_alignment_rejects_a_shot_without_alignable_text(self):
        plan = (
            {
                "shot_index": 1,
                "text": "，，，",
                "source_start_seconds": 0.0,
                "source_end_seconds": 20.0,
            },
            {
                "shot_index": 2,
                "text": "这个玉米真甜。",
                "source_start_seconds": 0.0,
                "source_end_seconds": 20.0,
            },
        )

        with tempfile.TemporaryDirectory() as directory:
            subtitle_path = Path(directory) / "subtitle.srt"
            subtitle_path.write_text(
                "1\n00:00:00,000 --> 00:00:08,000\n，，，这个玉米真甜。\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                asset_library.AssetLibraryError,
                "alignable text",
            ):
                asset_matching.align_storyboard_plan(
                    plan,
                    audio_duration=8.0,
                    subtitle_path=subtitle_path,
                )

    def test_source_window_follows_the_subtitle_aligned_duration(self):
        # 源片段按对齐后的真实时长裁切；第一镜 6.25s 就要 6.25s 的源画面。
        plan = (
            {
                "shot_index": 1,
                "text": "姐妹们快看。",
                "source_start_seconds": 1.0,
                "source_end_seconds": 20.0,
            },
            {
                "shot_index": 2,
                "text": "这个玉米真甜。",
                "source_start_seconds": 0.0,
                "source_end_seconds": 20.0,
            },
        )

        with tempfile.TemporaryDirectory() as directory:
            subtitle_path = Path(directory) / "subtitle.srt"
            subtitle_path.write_text(
                "1\n00:00:00,000 --> 00:00:06,000\n姐妹们快看。\n\n"
                "2\n00:00:06,500 --> 00:00:08,000\n这个玉米真甜。\n",
                encoding="utf-8",
            )
            aligned = asset_matching.align_storyboard_plan(
                plan,
                audio_duration=8.0,
                subtitle_path=subtitle_path,
            )

        first = aligned[0]
        self.assertAlmostEqual(
            first["source_end_seconds"] - first["source_start_seconds"],
            first["target_end_seconds"] - first["target_start_seconds"],
            places=6,
        )

    def test_a_source_segment_too_short_for_its_aligned_shot_is_rejected(self):
        plan = (
            {
                "shot_index": 1,
                "text": "姐妹们快看。",
                "source_start_seconds": 0.0,
                "source_end_seconds": 2.0,
            },
            {
                "shot_index": 2,
                "text": "这个玉米真甜。",
                "source_start_seconds": 0.0,
                "source_end_seconds": 20.0,
            },
        )

        with tempfile.TemporaryDirectory() as directory:
            subtitle_path = Path(directory) / "subtitle.srt"
            subtitle_path.write_text(
                "1\n00:00:00,000 --> 00:00:06,000\n姐妹们快看。\n\n"
                "2\n00:00:06,500 --> 00:00:08,000\n这个玉米真甜。\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                asset_library.AssetLibraryError,
                "too short",
            ):
                asset_matching.align_storyboard_plan(
                    plan,
                    audio_duration=8.0,
                    subtitle_path=subtitle_path,
                )

    def test_storyboard_plan_preserves_explicit_scene_duration_weights(self):
        plan = (
            {
                "shot_index": 1,
                "text": "短镜头",
                "target_duration_seconds": 1.0,
                "source_start_seconds": 0.0,
                "source_end_seconds": 2.0,
            },
            {
                "shot_index": 2,
                "text": "长镜头",
                "target_duration_seconds": 3.0,
                "source_start_seconds": 0.0,
                "source_end_seconds": 6.0,
            },
        )

        aligned = asset_matching.align_storyboard_plan(
            plan,
            audio_duration=8.0,
            subtitle_path=None,
        )

        self.assertEqual(aligned[0]["target_start_seconds"], 0.0)
        self.assertEqual(aligned[0]["target_end_seconds"], 2.0)
        self.assertEqual(aligned[1]["target_start_seconds"], 2.0)
        self.assertEqual(aligned[1]["target_end_seconds"], 8.0)

    def test_storyboard_snapshot_detects_external_file_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "relax.mp4"
            source.write_bytes(b"original")
            db_path = root / "library.sqlite3"

            def probe(_path):
                return {"duration": 5.0, "width": 1080, "height": 1920}

            with patch.object(
                asset_library_shots,
                "extract_frames_at",
                side_effect=_frames_per_window,
            ):
                asset_library.scan_library(
                    root,
                    None,
                    db_path=db_path,
                    probe_fn=probe,
                    vision_fn=lambda _path, frames: _window_analysis(
                        "沙发休息", ["放松"], len(frames)
                    ),
                )
            indexed = asset_library.list_assets(
                kind="video", analysis_status="ready", db_path=db_path
            )[0]
            plan = (
                {
                    "shot_index": 1,
                    "asset_id": indexed.asset_id,
                    "asset_sha256": indexed.sha256,
                    "source_start_seconds": 0.0,
                    "source_end_seconds": 3.0,
                    "target_start_seconds": 0.0,
                    "target_end_seconds": 3.0,
                },
            )
            source.write_bytes(b"changed")
            with patch.object(
                asset_library_runtime.utils,
                "task_dir",
                return_value=str(root / "task"),
            ):
                with self.assertRaisesRegex(
                    asset_library.AssetLibraryError,
                    "changed since indexing",
                ):
                    asset_library_runtime.materialize_storyboard_sources(
                        "task-1", plan, db_path=db_path
                    )

    def test_storyboard_snapshot_is_reused_after_successful_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "relax.mp4"
            source.write_bytes(b"original")
            db_path = root / "library.sqlite3"

            with patch.object(
                asset_library_shots,
                "extract_frames_at",
                side_effect=_frames_per_window,
            ):
                asset_library.scan_library(
                    root,
                    None,
                    db_path=db_path,
                    probe_fn=lambda _path: {
                        "duration": 5.0,
                        "width": 1080,
                        "height": 1920,
                    },
                    vision_fn=lambda _path, frames: _window_analysis(
                        "沙发休息", ["放松"], len(frames)
                    ),
                )
            indexed = asset_library.list_assets(
                kind="video", analysis_status="ready", db_path=db_path
            )[0]
            plan = (
                {
                    "shot_index": 1,
                    "asset_id": indexed.asset_id,
                    "asset_sha256": indexed.sha256,
                    "source_start_seconds": 0.0,
                    "source_end_seconds": 3.0,
                    "target_start_seconds": 0.0,
                    "target_end_seconds": 3.0,
                },
            )
            task_root = root / "task"
            with patch.object(
                asset_library_runtime.utils,
                "task_dir",
                return_value=str(task_root),
            ):
                first = asset_library_runtime.materialize_storyboard_sources(
                    "task-1", plan, db_path=db_path
                )
                self.assertEqual(Path(first[0]["source_path"]).read_bytes(), b"original")
                second = asset_library_runtime.materialize_storyboard_sources(
                    "task-1", plan, db_path=db_path
                )

        self.assertEqual(first, second)

    def test_manual_annotations_survive_rescan_and_are_used_for_matching(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video_root = root / "videos"
            video_root.mkdir()
            video_path = video_root / "clip.mp4"
            video_path.write_bytes(b"video")
            db_path = root / "library.sqlite3"

            with patch.object(
                asset_library_shots,
                "extract_frames_at",
                side_effect=_frames_per_window,
            ):
                asset_library.scan_library(
                    video_root,
                    None,
                    db_path=db_path,
                    probe_fn=lambda _path: {
                        "duration": 5.0,
                        "width": 1080,
                        "height": 1920,
                    },
                    vision_fn=lambda _path, frames: _window_analysis(
                        "普通场景", ["普通"], len(frames)
                    ),
                )
            indexed = asset_library.list_assets(
                kind="video", db_path=db_path
            )[0]
            asset_library.update_asset_annotations(
                indexed.asset_id,
                description="咖啡店里喝咖啡",
                tags=("咖啡", "休息"),
                db_path=db_path,
            )
            with patch.object(
                asset_library_shots,
                "extract_frames_at",
                side_effect=_frames_per_window,
            ):
                asset_library.scan_library(
                    video_root,
                    None,
                    db_path=db_path,
                    probe_fn=lambda _path: {
                        "duration": 5.0,
                        "width": 1080,
                        "height": 1920,
                    },
                    vision_fn=lambda _path, frames: _window_analysis(
                        "普通场景", ["普通"], len(frames)
                    ),
                    force_reanalyze=True,
                )
            rescanned = asset_library.get_asset(indexed.asset_id, db_path=db_path)

        self.assertEqual(rescanned.description, "咖啡店里喝咖啡")
        self.assertEqual(rescanned.tags[-2:], ("咖啡", "休息"))
        ranked = asset_matching._rank_assets("咖啡", (rescanned,))
        self.assertEqual(tuple(asset.asset_id for asset in ranked), (rescanned.asset_id,))

    def test_legacy_library_schema_migrates_persistently(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.sqlite3"
            connection = sqlite3.connect(path)
            connection.row_factory = sqlite3.Row
            connection.executescript(
                """
                CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                INSERT INTO metadata VALUES ('schema_version', '1');
                CREATE TABLE assets(
                    asset_id TEXT PRIMARY KEY, kind TEXT NOT NULL,
                    root_path TEXT NOT NULL, relative_path TEXT NOT NULL,
                    category TEXT NOT NULL, duration REAL NOT NULL,
                    width INTEGER, height INTEGER, size_bytes INTEGER NOT NULL,
                    mtime_ns INTEGER NOT NULL, sha256 TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    tags_json TEXT NOT NULL DEFAULT '[]',
                    analysis_status TEXT NOT NULL,
                    analysis_error TEXT NOT NULL DEFAULT '',
                    analysis_model TEXT NOT NULL DEFAULT '',
                    use_count INTEGER NOT NULL DEFAULT 0, last_used_at TEXT
                );
                """
            )
            connection.execute(
                "INSERT INTO assets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "video-legacy",
                    "video",
                    "/library",
                    "clip.mp4",
                    "人物",
                    6.0,
                    1080,
                    1920,
                    1,
                    1,
                    "hash",
                    "人物在走路",
                    '["人物"]',
                    "ready",
                    "",
                    "model-v1",
                    2,
                    None,
                ),
            )
            connection.commit()
            connection.close()

            connection = sqlite3.connect(path)
            connection.row_factory = sqlite3.Row
            asset_library._ensure_schema(connection)
            connection.close()

            connection = sqlite3.connect(path)
            self.assertEqual(
                connection.execute(
                    "SELECT value FROM metadata WHERE key = 'schema_version'"
                ).fetchone()[0],
                "2",
            )
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM segments").fetchone()[0],
                2,
            )
            self.assertEqual(
                {
                    row[1]
                    for row in connection.execute("PRAGMA table_info(assets)")
                    if row[1].startswith("manual_")
                },
                {"manual_description", "manual_tags_json"},
            )
            connection.close()

    def test_timeline_validation_rejects_gaps_and_accepts_contiguous_ranges(self):
        timeline = (
            {
                "source_path": "one.mp4",
                "source_start_seconds": 1.0,
                "source_end_seconds": 3.0,
                "target_start_seconds": 0.0,
                "target_end_seconds": 2.0,
            },
            {
                "source_path": "two.mp4",
                "source_start_seconds": 0.0,
                "source_end_seconds": 2.0,
                "target_start_seconds": 2.0,
                "target_end_seconds": 4.0,
            },
        )
        self.assertEqual(len(video._validate_storyboard_timeline(timeline, 4.0)), 2)
        with self.assertRaisesRegex(ValueError, "gap"):
            video._validate_storyboard_timeline(
                (*timeline[:1], {**timeline[1], "target_start_seconds": 2.5}),
                4.5,
            )
