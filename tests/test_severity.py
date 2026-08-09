"""Severity scoring is a pure function, so it gets the most thorough tests in the repo.

Every downstream number - map marker colours, the video summary, the priority ordering -
is derived from this module. A silent error here corrupts everything without ever
raising.
"""

import pytest

from src.core.severity import (
    area_factor,
    classify,
    severity_level,
    severity_score,
)
from src.utils.constants import AREA_SATURATION, SEVERITY_BY_NAME

POTHOLE = 3
ALLIGATOR = 2
TRANSVERSE = 1
LONGITUDINAL = 0


# --------------------------------------------------------------------------------------
# area_factor
# --------------------------------------------------------------------------------------


def test_area_factor_is_zero_for_zero_area():
    assert area_factor(0.0) == 0.0


def test_area_factor_saturates_at_one():
    assert area_factor(AREA_SATURATION) == pytest.approx(1.0)


def test_area_factor_does_not_exceed_one_beyond_saturation():
    # A damage covering the whole frame must not push the score above its ceiling.
    assert area_factor(1.0) == 1.0


def test_area_factor_increases_with_area():
    assert area_factor(0.01) < area_factor(0.05) < area_factor(0.10)


def test_area_factor_uses_sqrt_scaling_not_linear():
    # Half the saturation area should score well above 0.5 under sqrt scaling.
    # Under linear scaling this would be exactly 0.5. This test is what pins the
    # curve choice; it fails if someone "simplifies" it back to linear.
    assert area_factor(AREA_SATURATION / 2) > 0.65


def test_area_factor_clamps_negative_input():
    assert area_factor(-0.5) == 0.0


# --------------------------------------------------------------------------------------
# severity_score
# --------------------------------------------------------------------------------------


def test_score_is_one_when_every_factor_is_maximal():
    assert severity_score(AREA_SATURATION, POTHOLE, 1.0) == pytest.approx(1.0)


def test_score_stays_within_unit_interval():
    for area in (0.0, 0.001, 0.05, AREA_SATURATION, 1.0):
        for cid in (LONGITUDINAL, TRANSVERSE, ALLIGATOR, POTHOLE):
            for conf in (0.0, 0.25, 0.9, 1.0):
                assert 0.0 <= severity_score(area, cid, conf) <= 1.0


def test_pothole_outranks_longitudinal_crack_all_else_equal():
    area, conf = 0.03, 0.8
    assert severity_score(area, POTHOLE, conf) > severity_score(area, LONGITUDINAL, conf)


def test_damage_types_are_ordered_by_seriousness():
    area, conf = 0.03, 0.8
    scores = [severity_score(area, c, conf)
              for c in (LONGITUDINAL, TRANSVERSE, ALLIGATOR, POTHOLE)]
    assert scores == sorted(scores)


def test_bigger_damage_scores_higher():
    assert severity_score(0.01, POTHOLE, 0.8) < severity_score(0.08, POTHOLE, 0.8)


def test_higher_confidence_scores_higher():
    assert severity_score(0.03, POTHOLE, 0.3) < severity_score(0.03, POTHOLE, 0.95)


def test_unknown_class_id_is_rejected():
    # Better to fail loudly than to silently score an out-of-range class as harmless.
    with pytest.raises(KeyError):
        severity_score(0.03, 99, 0.8)


# --------------------------------------------------------------------------------------
# severity_level
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "score,expected",
    [
        (0.00, "Low"),
        (0.30, "Low"),        # inclusive upper bound
        (0.3001, "Medium"),
        (0.55, "Medium"),
        (0.5501, "High"),
        (0.75, "High"),
        (0.7501, "Critical"),
        (1.00, "Critical"),
    ],
)
def test_score_maps_to_expected_band(score, expected):
    assert severity_level(score).name == expected


def test_level_carries_a_colour_and_an_action():
    level = severity_level(0.9)
    assert level.name == "Critical"
    assert level.hex.startswith("#")
    assert level.action


def test_every_band_is_reachable():
    produced = {severity_level(s).name for s in (0.1, 0.4, 0.6, 0.9)}
    assert produced == set(SEVERITY_BY_NAME)


# --------------------------------------------------------------------------------------
# classify
# --------------------------------------------------------------------------------------


def test_classify_returns_score_and_level_name():
    score, level = classify(AREA_SATURATION, POTHOLE, 1.0)
    assert score == pytest.approx(1.0)
    assert level == "Critical"


def test_classify_agrees_with_its_parts():
    area, cid, conf = 0.04, ALLIGATOR, 0.7
    score, level = classify(area, cid, conf)
    assert score == pytest.approx(severity_score(area, cid, conf))
    assert level == severity_level(score).name


def test_a_faint_hairline_crack_is_low_severity():
    # Anchors the heuristic to a real-world case: a tiny, low-confidence crack must
    # not surface as an urgent repair.
    score, level = classify(0.002, LONGITUDINAL, 0.3)
    assert level == "Low"


def test_a_large_confident_pothole_is_critical():
    score, level = classify(0.12, POTHOLE, 0.92)
    assert level == "Critical"
