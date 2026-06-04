"""Options validation / error-path tests.

These cover the validator and consistency-check branches that the demo integration
tests never hit (they only ever build valid options).
"""

import pytest
import pydantic

import thuner.option.track as track_option
import thuner.option.data as data_option
import thuner.option.grid as grid_option
from thuner.utils import Retrieval, BaseDatasetOptions
from thuner.detect.preprocess import cross_section


def _ds(name, **kwargs):
    """A minimal, data-free dataset options object."""
    base = dict(
        name=name,
        start="2005-11-13T14:00:00",
        end="2005-11-13T15:00:00",
        fields=["reflectivity"],
        parent_local="/tmp",
    )
    base.update(kwargs)
    return BaseDatasetOptions(**base)


def test_unknown_field_rejected():
    with pytest.raises(pydantic.ValidationError):
        grid_option.GridOptions(not_a_field=1)


def test_typo_field_rejected():
    with pytest.raises(pydantic.ValidationError):
        _ds("x", reuse_regridderr=True)  # typo: trailing r


def test_duplicate_dataset_names_rejected():
    with pytest.raises(pydantic.ValidationError):
        data_option.DataOptions(datasets=[_ds("cpol"), _ds("cpol")])


def test_regridder_from_missing_target():
    borrower = _ds("b", reuse_regridder=True, regridder_from="ghost")
    with pytest.raises(pydantic.ValidationError):
        data_option.DataOptions(datasets=[borrower])


def test_regridder_from_self_reference():
    borrower = _ds("b", reuse_regridder=True, regridder_from="b")
    with pytest.raises(pydantic.ValidationError):
        data_option.DataOptions(datasets=[borrower])


def test_regridder_from_cycle():
    a = _ds("a", reuse_regridder=True, regridder_from="b")
    b = _ds("b", reuse_regridder=True, regridder_from="a")
    with pytest.raises(pydantic.ValidationError):
        data_option.DataOptions(datasets=[a, b])


def test_regridder_from_borrower_not_reusing():
    src = _ds("src", reuse_regridder=True)
    borrower = _ds("b", reuse_regridder=False, regridder_from="src")
    with pytest.raises(pydantic.ValidationError):
        data_option.DataOptions(datasets=[borrower, src])


def test_regridder_from_source_not_reusing():
    src = _ds("src", reuse_regridder=False)
    borrower = _ds("b", reuse_regridder=True, regridder_from="src")
    with pytest.raises(pydantic.ValidationError):
        data_option.DataOptions(datasets=[borrower, src])


def test_regridder_from_valid():
    src = _ds("src", reuse_regridder=True)
    borrower = _ds("b", reuse_regridder=True, regridder_from="src")
    options = data_option.DataOptions(datasets=[src, borrower])
    assert options.dataset_by_name("b").regridder_from == "src"


def test_flatten_method_requires_altitudes():
    # vertical_max is the default flatten_method; altitudes defaults to None.
    with pytest.raises(pydantic.ValidationError):
        track_option.DetectionOptions(method="steiner")


def test_vertical_max_requires_range():
    with pytest.raises(pydantic.ValidationError):
        track_option.DetectionOptions(method="steiner", altitudes=3000.0)


def test_cross_section_requires_single_altitude():
    with pytest.raises(pydantic.ValidationError):
        track_option.DetectionOptions(
            method="steiner",
            altitudes=(500.0, 3000.0),
            flatten_method=Retrieval(function=cross_section),
        )


def test_valid_detection_options():
    # vertical_max + range, cross_section + single altitude, and 2D (no flatten).
    track_option.DetectionOptions(method="steiner", altitudes=(500.0, 3000.0))
    track_option.DetectionOptions(
        method="steiner",
        altitudes=3000.0,
        flatten_method=Retrieval(function=cross_section),
    )
    track_option.DetectionOptions(method="steiner", flatten_method=None)
