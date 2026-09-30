from __future__ import annotations

import pytest

from arbiter.schemas import JOB_TYPE_PARAMS, JOB_TYPE_TO_MODEL, JobType, NimbleScoreParams


def test_nimble_score_schema_is_registered_for_the_exact_development_job():
    assert JobType.NIMBLE_SCORE.value == "nimble-score"
    assert JOB_TYPE_TO_MODEL["nimble-score"] == "nimble-scorer"
    assert JOB_TYPE_PARAMS["nimble-score"] is NimbleScoreParams


def test_nimble_score_schema_rejects_non_development_suites():
    for suite in ("decision-v4/test", "transfer-v4/test", "unknown"):
        with pytest.raises(ValueError):
            NimbleScoreParams(suite=suite, dataset_file="/not/read")
