import unittest
from unittest.mock import patch

from app.services import asset_library, asset_matching


def _asset(asset_id: str, *, description: str, tags=()) -> asset_library.LibraryAsset:
    return asset_library.LibraryAsset(
        asset_id=asset_id,
        kind="video",
        root_path="/library",
        relative_path=f"{asset_id}.mp4",
        category="uncategorized",
        duration=12.0,
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


def _segment(asset, index=0, description=None):
    return asset_library.LibrarySegment(
        segment_id=f"segment-{asset.asset_id}-{index}",
        asset_id=asset.asset_id,
        source_start_seconds=float(index) * 4,
        source_end_seconds=float(index) * 4 + 4,
        description=description if description is not None else asset.description,
        tags=asset.tags,
        analysis_status="ready",
        analysis_error="",
        analysis_model="test",
    )


def _corpus(asset_count=400, per_asset=3):
    """构造一个可复现的中等规模库，描述各不相同。"""
    topics = (
        "金黄玉米在田间随风摆动",
        "真空包装玉米整齐堆放",
        "掰开玉米展示晶莹颗粒",
        "水煮玉米搭配鸡蛋黄瓜摆盘",
        "促销价格与购物车截图",
        "口红小样与护肤水开箱",
    )
    assets = {}
    segments = []
    for index in range(asset_count):
        asset = _asset(
            f"video-{index:05d}",
            description=topics[index % len(topics)],
            tags=(f"标签{index % 17}",),
        )
        assets[asset.asset_id] = asset
        for window in range(per_asset):
            segments.append(
                _segment(
                    asset,
                    window,
                    description=f"{topics[(index + window) % len(topics)]}第{window}段",
                )
            )
    return assets, tuple(segments)


class TestRecallIndex(unittest.TestCase):
    """Stage-1 recall must narrow the pool without changing the result."""

    def test_recall_returns_the_same_ranking_as_a_full_scan(self):
        assets, segments = _corpus(asset_count=120)
        index = asset_matching.build_recall_index(segments, assets)

        for query in (
            "真空包装玉米",
            "掰开玉米颗粒",
            "促销 购物车",
            "水煮玉米 鸡蛋",
        ):
            with self.subTest(query=query):
                full = asset_matching._rank_segments(query, segments, assets, count=3)
                narrowed = asset_matching._rank_segments(
                    query, segments, assets, count=3, recall_index=index
                )
                self.assertEqual(
                    [(item[0].segment_id, round(item[1], 9)) for item in full],
                    [(item[0].segment_id, round(item[1], 9)) for item in narrowed],
                )

    def test_recall_is_exact_because_zero_overlap_cannot_pass_the_threshold(self):
        # 无 token 交集的片段语义分恒为 0，低于最低相关性阈值，因此按倒排
        # 索引只打分「至少共享一个 token」的片段与全表扫描等价。
        assets, segments = _corpus(asset_count=40)
        index = asset_matching.build_recall_index(segments, assets)

        ranked = asset_matching._rank_segments(
            "完全不相关的查询词汇",
            segments,
            assets,
            count=3,
            recall_index=index,
        )

        self.assertEqual(ranked, ())

    def test_exclusions_still_apply_after_recall(self):
        office = _asset("office", description="办公室会议桌前讨论")
        sofa = _asset("sofa", description="居家沙发休息放松")
        assets = {office.asset_id: office, sofa.asset_id: sofa}
        segments = (_segment(office), _segment(sofa))
        index = asset_matching.build_recall_index(segments, assets)

        ranked = asset_matching._rank_segments(
            "办公室 放松",
            segments,
            assets,
            count=3,
            recall_index=index,
            must_not_appear=("办公室",),
        )

        self.assertNotIn("office", [item[0].asset_id for item in ranked])

    def test_excluded_ids_still_apply_after_recall(self):
        assets, segments = _corpus(asset_count=30)
        index = asset_matching.build_recall_index(segments, assets)
        first = asset_matching._rank_segments(
            "真空包装玉米", segments, assets, count=1, recall_index=index
        )
        excluded = first[0][0].asset_id

        ranked = asset_matching._rank_segments(
            "真空包装玉米",
            segments,
            assets,
            count=3,
            recall_index=index,
            excluded_ids={excluded},
        )

        self.assertNotIn(excluded, [item[0].asset_id for item in ranked])

    def test_index_is_built_once_per_match_not_once_per_shot(self):
        # 一条文案通常拆 2-4 镜，逐镜重建索引会把建索引成本乘上分镜数，
        # 反而比全表打分更慢。这里断言调用次数而不是墙钟耗时：
        # 耗时断言在整套测试的负载下会随机翻转，而调用次数是确定的。
        assets, segments = _corpus(asset_count=40)
        script = "真空包装玉米整齐堆放。掰开玉米展示晶莹颗粒。水煮玉米搭配鸡蛋。"
        calls = []
        real_build = asset_matching.build_recall_index

        def counting_build(*args, **kwargs):
            calls.append(1)
            return real_build(*args, **kwargs)

        with (
            patch.object(
                asset_matching.library,
                "list_assets",
                side_effect=lambda **kwargs: tuple(assets.values())
                if kwargs.get("kind") == "video"
                else (),
            ),
            patch.object(
                asset_matching.library, "list_segments", return_value=segments
            ),
            patch.object(
                asset_matching, "build_recall_index", side_effect=counting_build
            ),
        ):
            match = asset_matching.match_storyboard(script, clip_duration=3)

        self.assertGreater(len(match.shots), 1)
        self.assertEqual(len(calls), 1)

    def test_recall_narrows_the_pool_it_has_to_score(self):
        # 收窄候选池是加速的来源；池子必须真的比全库小，且结果与全表一致。
        assets, segments = _corpus(asset_count=200)
        index = asset_matching.build_recall_index(segments, assets)
        query = "真空包装玉米整齐堆放"

        recalled = asset_matching._recalled_segments(
            index, frozenset(asset_matching._tokens(query))
        )

        self.assertLess(len(recalled), len(segments))
        self.assertEqual(
            [
                (item[0].segment_id, round(item[1], 9))
                for item in asset_matching._rank_segments(
                    query, segments, assets, count=3, recall_index=index
                )
            ],
            [
                (item[0].segment_id, round(item[1], 9))
                for item in asset_matching._rank_segments(
                    query, segments, assets, count=3
                )
            ],
        )


if __name__ == "__main__":
    unittest.main()
