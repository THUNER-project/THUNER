"""Test synthetic data generation and tracking."""

from pathlib import Path
import shutil
import os
import numpy as np
import thuner.data as data
import thuner.data.dispatch as dispatch
import thuner.grid as grid
import thuner.track as track
import thuner.option as option
import thuner.visualize as visualize
import thuner.data.synthetic as synthetic
import thuner.analyze as analyze

# Suppress the "wayland" plugin warning
os.environ["QT_QPA_PLATFORM"] = "offscreen"


def test_synthetic():
    # Parent directory for saving outputs
    base_local = Path.home() / "THUNER_output"
    start = "2005-11-13T00:00:00"
    end = "2005-11-13T01:00:00"

    output_directory = base_local / "runs/synthetic_demo_geographic"
    if output_directory.exists():
        shutil.rmtree(output_directory)
    options_directory = output_directory / "options"

    # Create a grid
    lat = np.arange(-14.0, -10.0 + 0.025 / 2, 0.025).round(3).tolist()
    lon = np.arange(129.0, 133.0 + 0.025 / 2, 0.025).round(3).tolist()
    grid_options = grid.create_options(name="geographic", latitude=lat, longitude=lon)
    grid.check_options(grid_options)
    grid.save_grid_options(grid_options, options_directory)

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

    # Restrict the ERA5 data to a smaller region containing the CPOL radar
    era5_pl_options = data.era5.data_options(
        start=start,
        end=end,
        latitude_range=[lat[0], lat[-1]],
        longitude_range=[lon[0], lon[-1]],
    )
    era5_sl_options = data.era5.data_options(
        start=start,
        end=end,
        data_format="single-levels",
        latitude_range=[lat[0], lat[-1]],
        longitude_range=[lon[0], lon[-1]],
    )

    data_options = option.consolidate_options(
        [synthetic_options, era5_pl_options, era5_sl_options]
    )
    dispatch.check_data_options(data_options)
    data.option.save_data_options(data_options, options_directory)

    # Create the track_options dictionary
    track_options = option.synthetic_track_options()
    track_options.to_yaml(options_directory / "track.yml")

    # Create the display_options dictionary
    visualize_options = {
        "cell": visualize.option.runtime_options(
            "cell", save=True, style="presentation"
        )
    }
    visualize.option.save_display_options(visualize_options, options_directory)

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

    # Test cartesian grid tracking
    output_directory = base_local / "runs/synthetic_demo_cartesian"
    if output_directory.exists():
        shutil.rmtree(output_directory)
    options_directory = output_directory / "options"

    central_latitude = -10
    central_longitude = 132

    y = np.arange(-400e3, 400e3 + 2.5e3, 2.5e3).tolist()
    x = np.arange(-400e3, 400e3 + 2.5e3, 2.5e3).tolist()

    grid_options = grid.create_options(
        name="cartesian",
        x=x,
        y=y,
        central_latitude=central_latitude,
        central_longitude=central_longitude,
    )
    grid.check_options(grid_options)
    grid.save_grid_options(grid_options, options_directory)
    track_options = option.synthetic_track_options()
    track_options.to_yaml(options_directory / "track.yml")
    data.option.save_data_options(data_options, options_directory)

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


if __name__ == "__main__":
    test_synthetic()
