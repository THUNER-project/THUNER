"""Test synthetic data generation and tracking."""

from pathlib import Path
import shutil
import numpy as np
import thor.data as data
import thor.data.dispatch as dispatch
import thor.grid as grid
import thor.track as track
import thor.option as option
import thor.visualize as visualize
import thor.data.synthetic as synthetic


def test_synthetic():
    # Parent directory for saving outputs
    base_local = Path.home() / "THOR_output"
    start = "2005-11-13T00:00:00"
    end = "2005-11-13T01:00:00"

    # Create a grid
    lat = np.arange(-14, -6 + 0.025, 0.025).tolist()
    lon = np.arange(128, 136 + 0.025, 0.025).tolist()
    grid_options = grid.create_options(name="geographic", latitude=lat, longitude=lon)
    grid.check_options(grid_options)
    grid.save_grid_options(grid_options, filename="synth_geographic")

    # Initialize synthetic objects
    synthetic_object = synthetic.create_object(
        time=start,
        center_latitude=np.mean(lat),
        center_longitude=np.mean(lon),
        direction=np.pi / 2,
        speed=10,
    )
    starting_objects = [synthetic_object]
    # Create data options dictionary
    synthetic_options = synthetic.synthetic_data_options(
        starting_objects=starting_objects
    )
    data_options = option.consolidate_options([synthetic_options])
    dispatch.check_data_options(data_options)
    data.option.save_data_options(data_options, filename="synthetic")

    # Create the track_options dictionary
    track_options = option.cell(
        dataset="synthetic",
        tracking_method="mint",
        global_flow_margin=70,
        unique_global_flow=False,
    )
    option.save_track_options(track_options, filename="synthetic_cell")

    # Create the display_options dictionary
    visualize_options = {
        "cell": visualize.option.runtime_options(
            "cell", save=True, style="presentation"
        )
    }
    visualize.option.save_display_options(
        visualize_options, filename="runtime_synthetic"
    )

    output_directory = base_local / "runs/synthetic_demo_geographic"
    if output_directory.exists():
        shutil.rmtree(output_directory)
    times = np.arange(
        np.datetime64(start),
        np.datetime64(end) + np.timedelta64(10, "m"),
        +np.timedelta64(10, "m"),
    )
    track.simultaneous_track(
        times,
        data_options,
        grid_options,
        track_options,
        visualize_options,
        output_directory=output_directory,
    )
