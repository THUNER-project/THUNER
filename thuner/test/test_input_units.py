"""Tests for the InputUnit filepath abstraction and the unit-aware loader."""

import tempfile
from pathlib import Path

import numpy as np
import xarray as xr

import thuner.utils as utils
import thuner.option as option
import thuner.track.utils as track_utils


def _make_cartesian_dataset(time, domain_mask=False):
    """Tiny converted-style cartesian dataset with a single time."""
    y = np.arange(0, 5) * 1000.0
    x = np.arange(0, 5) * 1000.0
    refl = np.random.rand(1, len(y), len(x)).astype("float32")
    data_vars = {"reflectivity": (("time", "y", "x"), refl)}
    if domain_mask:
        mask = np.ones((len(y), len(x)), dtype=bool)
        data_vars["domain_mask"] = (("y", "x"), mask)
    ds = xr.Dataset(
        data_vars,
        coords={"time": [np.datetime64(time)], "y": y, "x": x},
    )
    return ds


class TestNormalization:
    def test_strings_become_single_source_units(self):
        units = utils.normalize_input_units(["a.nc", "b.nc"])
        assert all(isinstance(u, utils.InputUnit) for u in units)
        assert units[0].sources == ["a.nc"]
        assert units[1].sources == ["b.nc"]

    def test_archive_member_becomes_one_source(self):
        units = utils.normalize_input_units([("arch.zip", "member.h5")])
        assert len(units) == 1
        # A two element reference is a single (archive, member) source.
        assert units[0].sources == [["arch.zip", "member.h5"]]

    def test_dict_and_none_pass_through(self):
        d = {"u": ["u1.nc"], "v": ["v1.nc"]}
        assert utils.normalize_input_units(d) is d
        assert utils.normalize_input_units(None) is None

    def test_existing_units_pass_through(self):
        unit = utils.InputUnit(sources=["a.nc"])
        out = utils.normalize_input_units([unit])
        assert out[0] is unit


class TestRoundTrip:
    def test_unit_json_round_trip_equal(self):
        unit = utils.InputUnit(
            sources=[["arch.zip", "m.h5"]],
            start_time="2021-10-14T16:00:00",
            end_time="2021-10-14T16:00:00",
        )
        reloaded = utils.InputUnit.model_validate_json(unit.model_dump_json())
        assert reloaded == unit

    def test_dataset_options_filepaths_round_trip(self):
        opts = utils.BaseDatasetOptions(
            name="generic",
            start="2021-10-14T16:00:00",
            end="2021-10-14T17:00:00",
            fields=["reflectivity"],
            filepaths=["/data/a.nc", "/data/b.nc"],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "opts.json"
            opts.to_json(path)
            reloaded = utils.BaseDatasetOptions.from_json(path)
        assert reloaded == opts
        assert isinstance(reloaded.filepaths[0], utils.InputUnit)


class TestRecordStr:
    """``record_str`` must round-trip through ``from_record_str`` so visualization can
    re-derive the grid from the persisted provenance record."""

    def test_plain_single_source_round_trip(self):
        unit = utils.InputUnit(sources=["/data/a.nc"])
        # A bare path stays human-readable (unwrapped) in the record.
        assert unit.record_str() == "/data/a.nc"
        assert utils.InputUnit.from_record_str(unit.record_str()).sources == [
            "/data/a.nc"
        ]

    def test_multi_plain_source_round_trip(self):
        # e.g. ERA5 groups one file per field in a single unit.
        unit = utils.InputUnit(sources=["/data/u.nc", "/data/v.nc"])
        rebuilt = utils.InputUnit.from_record_str(unit.record_str())
        assert rebuilt.sources == ["/data/u.nc", "/data/v.nc"]

    def test_archive_member_round_trip(self):
        # e.g. a single operational radar (member inside a zip archive).
        unit = utils.InputUnit(sources=[["/raw/3.zip", "3_160000.pvol.h5"]])
        rebuilt = utils.InputUnit.from_record_str(unit.record_str())
        assert rebuilt.sources == [["/raw/3.zip", "3_160000.pvol.h5"]]

    def test_ensemble_archive_member_round_trip(self):
        # e.g. an operational radar ensemble: several [archive, member] sources.
        sources = [
            ["/raw/3.zip", "3_160000.pvol.h5"],
            ["/raw/4.zip", "4_160000.pvol.h5"],
        ]
        unit = utils.InputUnit(sources=sources)
        rebuilt = utils.InputUnit.from_record_str(unit.record_str())
        assert rebuilt.sources == sources


class TestConvertedFilepath:
    def test_plain_file_mirror(self):
        opts = utils.BaseDatasetOptions(
            name="generic",
            start="2021-10-14T16:00:00",
            end="2021-10-14T17:00:00",
            fields=["reflectivity"],
            parent_local="/data/raw",
            filepaths=["/data/raw/sub/a.nc"],
            converted_options={"parent_converted": "/data/converted"},
        )
        unit = opts.filepaths[0]
        assert opts.converted_filepath(unit) == "/data/converted/sub/a.nc"

    def test_archive_member_named_by_member(self):
        opts = utils.BaseDatasetOptions(
            name="generic",
            start="2021-10-14T16:00:00",
            end="2021-10-14T17:00:00",
            fields=["reflectivity"],
            parent_local="/data/raw",
            filepaths=[
                ("/data/raw/vol/3_20211014.pvol.zip", "3_20211014_160000.pvol.h5")
            ],
            converted_options={"parent_converted": "/data/converted"},
        )
        unit = opts.filepaths[0]
        out = opts.converted_filepath(unit)
        assert out == "/data/converted/vol/3_20211014_160000.pvol.nc"

    def test_operational_ensemble_directory_named_by_radars(self):
        """The converted directory should use the ensemble's radar string (matching the
        converted filenames), not the raw single-radar directory."""
        import thuner.data.aura as aura

        opts = aura.OperationalEnsembleOptions(
            start="2021-10-14T16:00:00",
            end="2021-10-14T16:00:00",
            radars=[3, 4, 40],
            parent_local="/data/raw",
            converted_options={"parent_converted": "/data/converted"},
        )
        unit = opts.filepaths[0]
        out = opts.converted_filepath(unit)
        # Both the directory and the filename are named by the radar ensemble; the raw
        # tree's single-radar '3' directory is replaced by '3_4_40'.
        assert out == "/data/converted/rq0/3_4_40/2021/vol/3_4_40_20211014_160000.nc"


class TestLoaderRoundTrip:
    """Drive the unit-aware update_input_record over synthetic converted files."""

    def test_subset_generate_and_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            times = [f"2021-10-14T16:{m:02}:00" for m in (0, 10, 20)]
            paths = []
            for t in times:
                p = Path(tmp) / f"grid_{t.replace(':', '')}.nc"
                _make_cartesian_dataset(t).to_netcdf(p)
                paths.append(str(p))

            opts = utils.BaseDatasetOptions(
                name="generic",
                start="2021-10-14T16:00:00",
                end="2021-10-14T16:10:00",  # excludes the 16:20 file
                fields=["reflectivity"],
                filepaths=paths,
            )

            # get_filepaths subsets units by their (inferred) time span.
            subset = opts.get_filepaths()
            assert len(subset) == 2
            assert all(isinstance(u, utils.InputUnit) for u in subset)

            # generate_times expands the (single time) files in order.
            gen_times = list(utils.generate_times(opts.filepaths))
            assert gen_times == [np.datetime64(t) for t in times]

            # The base loader opens unit.sources[0] for the first file.
            grid_options = option.grid.GridOptions(name="cartesian", regrid=False)
            record = track_utils.TrackInputRecord(
                name="generic", filepaths=opts.filepaths
            )
            opts.update_input_record(
                np.datetime64(times[0]), record, None, grid_options
            )
            assert record._current_file_index == 0
            assert np.datetime64(times[0]) in record.dataset.time.values

    def test_save_writes_converted_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            raw.mkdir()
            t = "2021-10-14T16:00:00"
            src = raw / "a.nc"
            _make_cartesian_dataset(t).to_netcdf(src)

            opts = utils.BaseDatasetOptions(
                name="generic",
                start=t,
                end=t,
                fields=["reflectivity"],
                parent_local=str(raw),
                filepaths=[str(src)],
                converted_options={
                    "save": True,
                    "parent_converted": str(Path(tmp) / "converted"),
                },
            )
            grid_options = option.grid.GridOptions(name="cartesian", regrid=False)
            record = track_utils.TrackInputRecord(
                name="generic", filepaths=opts.filepaths
            )
            opts.update_input_record(np.datetime64(t), record, None, grid_options)
            expected = Path(tmp) / "converted" / "a.nc"
            assert expected.exists()
