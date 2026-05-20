from pathlib import Path
import shutil
import numpy as np
import thuner.data as data
import thuner.default as default
import thuner.track.track as track
import thuner.option as option
import thuner.data.synthetic as synthetic
from pathlib import Path
import os
import xarray as xr
from thuner.attribute.utils import read_attribute_zarr
from thuner.config import get_zarr_store_name


def test_synthetic():
    # # Testing: Synthetic Data
    # The synthetic module is a work in progress. The idea is to allow synthetic meteorological
    # datasets to be readily created for testing purposes. While an entire synthetic dataset
    # could be created first, then fed into THUNER in the usual way (see previous tutorials/demos)
    # with this module we instead generate the synthetic data as we go. The approach avoids the
    # need for storing large datasets.
    """Synthetic data demo/test."""
    # Set a flag for whether or not to remove existing output directories
    remove_existing_outputs = True
    # Parent directory for saving outputs
    base_local = Path.home() / "THUNER_output"
    start = "2005-11-13T00:00:00"
    end = "2005-11-13T02:00:00"
    output_parent = base_local / "runs/synthetic/geographic"
    if output_parent.exists() and remove_existing_outputs:
        shutil.rmtree(output_parent)
    options_directory = output_parent / "options"
    options_directory.mkdir(parents=True, exist_ok=True)
    # Create a grid
    lat = np.arange(-14, -6 + 0.025, 0.025).tolist()
    lon = np.arange(128, 136 + 0.025, 0.025).tolist()
    grid_options = option.grid.GridOptions(
        name="geographic", latitude=lat, longitude=lon
    )
    grid_options.to_yaml(options_directory / "grid.yml")
    # Initialize synthetic objects
    starting_objects = []
    for i in range(5):
        obj = synthetic.create_object(
            time=start,
            center_latitude=np.mean(lat),
            center_longitude=lon[(i + 1) * len(lon) // 6],
            direction=-np.pi / 4 + i * np.pi / 6,
            speed=30 - 4 * i,
            horizontal_radius=5 + 4 * i,
        )
        starting_objects.append(obj)
    # Create data options dictionary
    synthetic_options = data.synthetic.SyntheticOptions(
        starting_objects=starting_objects
    )
    data_options = option.data.DataOptions(datasets=[synthetic_options])
    data_options.to_yaml(options_directory / "data.yml")
    track_options = default.synthetic_track()
    track_options.to_yaml(options_directory / "track.yml")
    # Create the display_options dictionary
    visualize_options = default.synthetic_runtime(options_directory / "visualize.yml")
    visualize_options.to_yaml(options_directory / "visualize.yml")
    times = np.arange(
        np.datetime64(start),
        np.datetime64(end) + np.timedelta64(10, "m"),
        np.timedelta64(10, "m"),
    )
    args = [times, data_options, grid_options, track_options, visualize_options]
    track.track(*args, output_directory=output_parent)
    # ![THUNER applied to synthetic data.](https://raw.githubusercontent.com/THUNER-project/THUNER/refs/heads/main/gallery/synthetic.gif)
    central_latitude = -10
    central_longitude = 132
    y = np.arange(-400e3, 400e3 + 2.5e3, 2.5e3).tolist()
    x = np.arange(-400e3, 400e3 + 2.5e3, 2.5e3).tolist()
    grid_options = option.grid.GridOptions(
        name="cartesian",
        x=x,
        y=y,
        central_latitude=central_latitude,
        central_longitude=central_longitude,
    )
    grid_options.to_yaml(options_directory / "grid.yml")
    output_parent = base_local / "runs/synthetic/cartesian"
    if output_parent.exists() & remove_existing_outputs:
        shutil.rmtree(output_parent)
    times = np.arange(
        np.datetime64(start),
        np.datetime64(end) + np.timedelta64(10, "m"),
        +np.timedelta64(10, "m"),
    )
    args = [times, data_options, grid_options, track_options, visualize_options]
    track.track(*args, output_directory=output_parent)
    # ## ZARR output layout
    #
    # THUNER consolidates all outputs into a single unified zarr store at the run root (`output.zarr` by default; configurable via `thuner.config.get_zarr_store_name`), with hierarchical groups for masks, attributes and filepath records. Attribute metadata is distributed across each group's dataset and variable `.attrs`. Below we verify the layout produced by the most recent (cartesian) run.
    store_path = output_parent / get_zarr_store_name()
    assert store_path.exists(), "Unified zarr store was not created."
    # Find one attribute group and confirm it round-trips.
    leaf_groups = []
    for root, dirs, files in os.walk(store_path / "attributes"):
        if any((Path(root) / d / ".zarray").exists() for d in dirs):
            rel = Path(root).relative_to(store_path).as_posix()
            leaf_groups.append(rel)
    assert leaf_groups, "No attribute groups written to zarr store."
    print(f"Found {len(leaf_groups)} attribute groups, e.g. {leaf_groups[0]}")
    df = read_attribute_zarr(store_path, leaf_groups[0])
    assert len(df) > 0, "Round-tripped attribute DataFrame is empty."
    print(df.head())
    # Confirm a mask group is non-empty.
    mask_groups = [d for d in (store_path / "masks").iterdir() if d.is_dir()]
    assert mask_groups, "No mask groups written to zarr store."
    ds = xr.open_zarr(store_path, group=f"masks/{mask_groups[0].name}")
    assert ds.sizes.get("time", 0) > 0, "Mask zarr group has no time dimension."
    print(f"masks/{mask_groups[0].name}:", dict(ds.sizes))
    # Confirm per-variable metadata is distributed (no giant JSON at root).
    sample = xr.open_zarr(store_path, group=leaf_groups[0])
    assert (
        "attribute_type" not in sample.attrs
    ), "Legacy giant-JSON metadata blob should no longer be present."
    has_var_attrs = any(sample[v].attrs.get("data_type") for v in sample.variables)
    assert has_var_attrs, "Expected distributed per-variable metadata."


if __name__ == "__main__":
    test_synthetic()
