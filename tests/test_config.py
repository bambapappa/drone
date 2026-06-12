"""Settings validation, in particular the analysis ROI."""

import pytest
from pydantic import ValidationError

from app.core.config import Settings


def test_roi_empty_is_none():
    s = Settings(analysis_roi="")
    assert s.roi_tuple() is None


def test_roi_valid():
    s = Settings(analysis_roi="0,0,0.5,1")
    assert s.roi_tuple() == (0.0, 0.0, 0.5, 1.0)


def test_roi_right_half():
    s = Settings(analysis_roi="0.5,0,0.5,1")
    assert s.roi_tuple() == (0.5, 0.0, 0.5, 1.0)


@pytest.mark.parametrize(
    "bad",
    [
        "0.5,0.5",  # too few values
        "0,0,1.5,1",  # w out of range
        "0.8,0,0.5,1",  # sticks outside the frame
        "0,0,0.01,1",  # sliver
        "a,b,c,d",  # garbage
    ],
)
def test_roi_invalid_rejected(bad):
    with pytest.raises(ValidationError):
        Settings(analysis_roi=bad)


def test_class_sets_parse():
    s = Settings(human_classes="person, Pedestrian ,people", threat_classes="")
    assert s.human_class_set() == {"person", "pedestrian", "people"}
    assert s.threat_class_set() == set()
