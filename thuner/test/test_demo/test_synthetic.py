import xarray as xr
from pathlib import Path
import shutil
import numpy as np
import thuner.data as data
import thuner.default as default
import thuner.track.track as track
import thuner.option as option
import thuner.analyze as analyze
import thuner.data.synthetic as synthetic


def test_synthetic():
    # # Testing: Synthetic Data
    # The synthetic module is a work in progress. The idea is to allow synthetic meteorological datasets to be readily created for testing purposes. While an entire synthetic dataset could be created first, then fed into THUNER in the usual way (see previous tutorials/demos) with this module we instead generate the synthetic data as we go. The approach allows us to create large synthetic datasets for testing, but avoid storing them!
    """Synthetic data demo/test."""
    # Set a flag for whether or not to remove existing output directories
    remove_existing_outputs = True
    # Parent directory for saving outputs
    base_local = Path.home() / "THUNER_output"
    start = "2005-11-13T00:00:00"
    end = "2005-11-13T01:00:00"
    output_parent = base_local / "runs/synthetic/geographic"
    if output_parent.exists() and remove_existing_outputs:
        shutil.rmtree(output_parent)
    options_directory = output_parent / "options"
    options_directory.mkdir(parents=True, exist_ok=True)
    # Create a grid
    lat = np.arange(-14, -6 + 0.025, 0.025).tolist()
    lon = np.arange(128, 136 + 0.025, 0.025).tolist()
    grid_options = option.grid.GridOptions(name="geographic", latitude=lat, longitude=lon)
    grid_options.to_json(options_directory / "grid.json")
    # Initialize synthetic objects
    starting_objects = []
    for i in range(5):
        obj = synthetic.EllipsoidObject(
            time=start,
            center_latitude=np.mean(lat),
            center_longitude=lon[(i + 1) * len(lon) // 6],
            direction=-np.pi / 4 + i * np.pi / 8,
            speed=30 - 4 * i,
            horizontal_radius=7 + 4 * i,
            orientation=0.25 * np.pi + i * np.pi / 8,
        )
        starting_objects.append(obj)
    # Create data options dictionary
    synthetic_options = data.synthetic.SyntheticOptions(objects=starting_objects)
    data_options = option.data.DataOptions(datasets=[synthetic_options])
    data_options.to_json(options_directory / "data.json")
    track_options = default.track.synthetic_track()
    track_options.to_json(options_directory / "track.json")
    # Create the display_options dictionary
    visualize_options = default.visualize.synthetic_runtime(
        options_directory / "visualize.json"
    )
    visualize_options.to_json(options_directory / "visualize.json")
    visualize_options.model_dump()
    times = np.arange(
        np.datetime64(start),
        np.datetime64(end) + np.timedelta64(10, "m"),
        np.timedelta64(10, "m"),
    )
    track.track(
        times=times,
        data_options=data_options,
        grid_options=grid_options,
        track_options=track_options,
        visualize_options=visualize_options,
        output_directory=output_parent,
    )
    # ![THUNER applied to synthetic data.](https://raw.githubusercontent.com/THUNER-project/THUNER/refs/heads/main/gallery/synthetic.gif)
    analyze.synthetic.write_ground_truth(output_parent, data_options=data_options, times=times)
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
    grid_options.to_json(options_directory / "grid.json")
    output_parent = base_local / "runs/synthetic/cartesian"
    if output_parent.exists() & remove_existing_outputs:
        shutil.rmtree(output_parent)
    times = np.arange(
        np.datetime64(start),
        np.datetime64(end) + np.timedelta64(10, "m"),
        +np.timedelta64(10, "m"),
    )
    track.track(
        times=times,
        data_options=data_options,
        grid_options=grid_options,
        track_options=track_options,
        visualize_options=visualize_options,
        output_directory=output_parent,
    )


if __name__ == '__main__':
    test_synthetic()
