# MoneyPrinterTurbo Test Directory

This directory contains unit tests for the **MoneyPrinterTurbo** project.

## Directory Structure

- `services/`: Domain-focused unit and controller tests
  - `test_task.py`: Task pipeline tests
  - `test_task_manager.py`: In-memory and Redis queue tests
  - `test_controller_*.py`: API controller tests split by controller domain
  - `test_video.py`, `test_voice.py`: Media service tests
- `test_main.py`: Application entry-point test

## Running Tests

The CI suite uses pytest, which also runs the existing `unittest.TestCase`
tests:

```bash
# Run all tests
uv run python -X utf8 -m pytest -q test

# Run a specific test file
uv run python -X utf8 -m pytest -q test/services/test_video.py

# Run a specific test class
uv run python -X utf8 -m pytest -q test/services/test_video.py::TestVideoService

# Run a specific test method
uv run python -X utf8 -m pytest -q test/services/test_video.py::TestVideoService::test_preprocess_video
```

To run the same branch coverage check used by CI:

```bash
uv run python -X utf8 -m coverage run -m pytest -q test
uv run python -m coverage report
```

Live provider tests are skipped by default. To run tests that may call external
TTS or LLM services, set `MPT_RUN_INTEGRATION_TESTS=1` and provide the required
provider credentials.

## Adding New Tests

To add tests for other components, follow these guidelines:

1. Name files `test_<domain>.py` and keep each file focused on one domain.
2. Split broad controller suites into files such as `test_controller_video.py`.
3. Use either pytest functions or `unittest.TestCase`; pytest collects both.
4. Name test functions and methods with the `test_` prefix.

## Test Resources

Place any resource files required for testing in the `test/resources` directory.

## Local Matching Eval Set and Scale Baseline

Two scripts measure local storyboard matching. They are not part of the pytest
suite, because both need a populated asset index; their own logic is unit-tested
in `test/scripts/test_matching_eval.py` and `test/scripts/test_matching_scale.py`.

`scripts/matching_eval.py` scores matching against human labels in
`test/resources/matching_eval_set.json`. Metrics come from the labels, never from
the matcher's own scores.

```bash
# Score the current matcher against the labelled eval set
uv run python -m scripts.matching_eval --eval-set test/resources/matching_eval_set.json

# Compare against the recorded baseline; exits 1 when a tracked metric regresses
uv run python -m scripts.matching_eval \
  --eval-set test/resources/matching_eval_set.json \
  --baseline test/resources/matching_eval_baseline.json
```

Labelled `asset_id` values contain a content hash, so re-encoding an asset
changes its ID. The runner then fails with `missing from the index` instead of
silently reporting a zero hit rate — re-label the affected cases.

Both scripts are run against the human-labelled set and the synthetic corpus
respectively; neither calls a real vision model, so neither can show a quality
improvement on its own. See `docs/MPT广告素材生产系统-改动方案.md` §11.7 for the
acceptance rules these numbers feed.

`scripts/matching_scale.py` measures index and query cost on a synthetic library.
The vision model is replaced by a deterministic stand-in, so `vision_calls` is the
call volume a real run would spend, and the timings exclude real model latency and
cost.

```bash
uv run python -m scripts.matching_scale \
  --library-root /tmp/mpt_scale/videos \
  --db-path /tmp/mpt_scale/index.sqlite3 \
  --asset-count 10000
```

Recorded baselines live in `test/resources/matching_eval_baseline.json` and
`test/resources/matching_scale_baseline.json`. Re-record them only together with
the change that moved the numbers, and state which machine produced them.
