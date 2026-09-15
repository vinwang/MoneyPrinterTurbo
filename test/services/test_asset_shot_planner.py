import json
import unittest

from app.services import asset_shot_planner as planner


def _plan_payload(shots):
    return json.dumps({"shots": shots}, ensure_ascii=False)


class TestParseShotPlan(unittest.TestCase):
    """A plan is only usable when its spans reproduce the script exactly."""

    def test_contiguous_spans_are_parsed_in_order(self):
        script = "下班回家，放松一下。这个玉米真甜。"
        payload = _plan_payload(
            [
                {
                    "index": 1,
                    "script_start": 0,
                    "script_end": 10,
                    "visual_query": "居家沙发休息",
                    "must_match": ["居家"],
                    "must_not_appear": ["办公室"],
                    "expression": "ambience",
                },
                {
                    "index": 2,
                    "script_start": 10,
                    "script_end": 17,
                    "visual_query": "玉米特写",
                    "must_match": [],
                    "must_not_appear": [],
                    "expression": "direct",
                },
            ]
        )

        parsed = planner.parse_shot_plan(payload, script)

        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0].text, "下班回家，放松一下。")
        self.assertEqual(parsed[0].visual_query, "居家沙发休息")
        self.assertEqual(parsed[0].must_not_appear, ("办公室",))
        self.assertEqual(parsed[0].expression, "ambience")
        self.assertEqual(parsed[1].text, "这个玉米真甜。")

    def test_a_markdown_fenced_response_is_accepted(self):
        script = "文案一。文案二。"
        payload = _plan_payload(
            [
                {
                    "index": 1,
                    "script_start": 0,
                    "script_end": len(script),
                    "visual_query": "画面",
                    "must_match": [],
                    "must_not_appear": [],
                    "expression": "direct",
                }
            ]
        )

        parsed = planner.parse_shot_plan("```json\n" + payload + "\n```", script)

        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0].text, script)

    def test_a_gap_between_spans_is_rejected(self):
        script = "第一句。第二句。"
        payload = _plan_payload(
            [
                {
                    "index": 1,
                    "script_start": 0,
                    "script_end": 3,
                    "visual_query": "画面一",
                    "must_match": [],
                    "must_not_appear": [],
                    "expression": "direct",
                },
                {
                    "index": 2,
                    "script_start": 4,
                    "script_end": 8,
                    "visual_query": "画面二",
                    "must_match": [],
                    "must_not_appear": [],
                    "expression": "direct",
                },
            ]
        )

        with self.assertRaisesRegex(planner.ShotPlanError, "contiguous"):
            planner.parse_shot_plan(payload, script)

    def test_overlapping_spans_are_rejected(self):
        script = "第一句。第二句。"
        payload = _plan_payload(
            [
                {
                    "index": 1,
                    "script_start": 0,
                    "script_end": 5,
                    "visual_query": "画面一",
                    "must_match": [],
                    "must_not_appear": [],
                    "expression": "direct",
                },
                {
                    "index": 2,
                    "script_start": 3,
                    "script_end": 8,
                    "visual_query": "画面二",
                    "must_match": [],
                    "must_not_appear": [],
                    "expression": "direct",
                },
            ]
        )

        with self.assertRaisesRegex(planner.ShotPlanError, "contiguous"):
            planner.parse_shot_plan(payload, script)

    def test_a_plan_that_does_not_reach_the_end_is_rejected(self):
        script = "第一句。第二句。"
        payload = _plan_payload(
            [
                {
                    "index": 1,
                    "script_start": 0,
                    "script_end": 4,
                    "visual_query": "画面一",
                    "must_match": [],
                    "must_not_appear": [],
                    "expression": "direct",
                }
            ]
        )

        with self.assertRaisesRegex(planner.ShotPlanError, "cover"):
            planner.parse_shot_plan(payload, script)

    def test_out_of_range_span_is_rejected(self):
        script = "短文案。"
        payload = _plan_payload(
            [
                {
                    "index": 1,
                    "script_start": 0,
                    "script_end": 99,
                    "visual_query": "画面",
                    "must_match": [],
                    "must_not_appear": [],
                    "expression": "direct",
                }
            ]
        )

        with self.assertRaisesRegex(planner.ShotPlanError, "range"):
            planner.parse_shot_plan(payload, script)

    def test_unknown_expression_is_rejected(self):
        script = "文案。"
        payload = _plan_payload(
            [
                {
                    "index": 1,
                    "script_start": 0,
                    "script_end": 3,
                    "visual_query": "画面",
                    "must_match": [],
                    "must_not_appear": [],
                    "expression": "cinematic",
                }
            ]
        )

        with self.assertRaisesRegex(planner.ShotPlanError, "expression"):
            planner.parse_shot_plan(payload, script)

    def test_unknown_shot_field_is_rejected(self):
        script = "文案。"
        payload = _plan_payload(
            [
                {
                    "index": 1,
                    "script_start": 0,
                    "script_end": 3,
                    "visual_query": "画面",
                    "must_match": [],
                    "must_not_appear": [],
                    "expression": "direct",
                    "asset_id": "injected",
                }
            ]
        )

        with self.assertRaisesRegex(planner.ShotPlanError, "exactly"):
            planner.parse_shot_plan(payload, script)

    def test_empty_visual_query_is_rejected(self):
        script = "文案。"
        payload = _plan_payload(
            [
                {
                    "index": 1,
                    "script_start": 0,
                    "script_end": 3,
                    "visual_query": "   ",
                    "must_match": [],
                    "must_not_appear": [],
                    "expression": "direct",
                }
            ]
        )

        with self.assertRaisesRegex(planner.ShotPlanError, "visual_query"):
            planner.parse_shot_plan(payload, script)

    def test_provider_error_text_is_surfaced(self):
        with self.assertRaisesRegex(planner.ShotPlanError, "quota"):
            planner.parse_shot_plan("Error: quota exceeded", "文案。")

    def test_non_json_response_is_rejected(self):
        with self.assertRaisesRegex(planner.ShotPlanError, "JSON"):
            planner.parse_shot_plan("not json at all", "文案。")

    def test_a_plan_whose_spans_rewrite_the_script_cannot_exist(self):
        # 分镜正文由原文切片得到，模型无法改写旁白——这是设计上的保证，
        # 此处固定住：拼接所有分镜正文必须逐字等于原文。
        script = "姐妹们快看。这个玉米真甜。"
        payload = _plan_payload(
            [
                {
                    "index": 1,
                    "script_start": 0,
                    "script_end": 6,
                    "visual_query": "画面一",
                    "must_match": [],
                    "must_not_appear": [],
                    "expression": "direct",
                },
                {
                    "index": 2,
                    "script_start": 6,
                    "script_end": 13,
                    "visual_query": "画面二",
                    "must_match": [],
                    "must_not_appear": [],
                    "expression": "direct",
                },
            ]
        )

        parsed = planner.parse_shot_plan(payload, script)

        self.assertEqual("".join(shot.text for shot in parsed), script)


class TestPlanShots(unittest.TestCase):
    """Planning degrades to punctuation splitting instead of failing generation."""

    def _payload(self, script):
        return _plan_payload(
            [
                {
                    "index": 1,
                    "script_start": 0,
                    "script_end": len(script),
                    "visual_query": "整段画面",
                    "must_match": [],
                    "must_not_appear": [],
                    "expression": "direct",
                }
            ]
        )

    def test_model_plan_is_used_when_valid(self):
        script = "下班回家，放松一下。"
        calls = []

        def fake_llm(prompt, app_config=None):
            calls.append(prompt)
            return self._payload(script)

        shots = planner.plan_shots(
            script,
            clip_duration=3.0,
            llm_fn=fake_llm,
            app_config={},
        )

        self.assertEqual(len(shots), 1)
        self.assertEqual(shots[0].text, script)
        self.assertEqual(len(calls), 1)

    def test_invalid_plan_falls_back_to_none(self):
        def broken_llm(prompt, app_config=None):
            return "{not json"

        self.assertIsNone(
            planner.plan_shots(
                "文案。",
                clip_duration=3.0,
                llm_fn=broken_llm,
                app_config={},
            )
        )

    def test_empty_script_is_rejected_before_calling_the_model(self):
        called = []

        def fake_llm(prompt, app_config=None):
            called.append(prompt)
            return "{}"

        with self.assertRaises(planner.ShotPlanError):
            planner.plan_shots(
                "   ",
                clip_duration=3.0,
                llm_fn=fake_llm,
                app_config={},
            )
        self.assertEqual(called, [])


if __name__ == "__main__":
    unittest.main()
