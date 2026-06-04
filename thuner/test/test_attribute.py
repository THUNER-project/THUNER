"""Unit tests for attribute reading, including the DataTree fast path."""

import numpy as np
import xarray as xr

from thuner.attribute.utils import read_attribute_zarr, _attribute_df_from_dataset


def _write_core(store):
    times = np.array(["2005-11-13T14:00:00", "2005-11-13T14:10:00"], dtype="datetime64[ns]")
    area = xr.DataArray(
        np.arange(4.0).reshape(2, 2),
        dims=("time", "universal_id"),
        coords={"time": times, "universal_id": [1, 2]},
    )
    ds = xr.Dataset({"area": area})
    ds.attrs["index_columns"] = ["time", "universal_id"]
    ds.to_zarr(store, group="attributes/mcs/core", mode="w")


def test_read_attribute_roundtrip_sets_index(tmp_path):
    store = tmp_path / "output.zarr"
    _write_core(store)
    df = read_attribute_zarr(store, "attributes/mcs/core")
    assert list(df.index.names) == ["time", "universal_id"]
    assert "area" in df.columns
    assert df.shape == (4, 1)


def test_read_attribute_tree_matches_reopen(tmp_path):
    """The tree= fast path must produce an identical DataFrame to re-opening."""
    store = tmp_path / "output.zarr"
    _write_core(store)
    df_reopen = read_attribute_zarr(store, "attributes/mcs/core")
    tree = xr.open_datatree(store, engine="zarr")
    df_tree = read_attribute_zarr(store, "attributes/mcs/core", tree=tree)
    assert df_reopen.equals(df_tree)


def test_placeholder_strings_fold_to_nan():
    """Empty/"nan" strings (how missing ids round-trip from zarr) become NaN."""
    ds = xr.Dataset(
        {"parents": ("universal_id", np.array(["", "nan", "3 5"], dtype=object))},
        coords={"universal_id": [1, 2, 3]},
    )
    ds.attrs["index_columns"] = ["universal_id"]
    df = _attribute_df_from_dataset(ds)
    assert df["parents"].isna().tolist() == [True, True, False]
    assert df.loc[3, "parents"] == "3 5"
