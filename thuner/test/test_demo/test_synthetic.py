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
from thuner.utils import format_time
from thuner.log import setup_logger


def test_synthetic():
    # # Testing: Synthetic Data
    # The `synthetic` module allows us to create artificial datasets with known velocities, extents etc. which we can then compare with those estimated by THUNER.
    """Synthetic data demo/test."""
    logger = setup_logger(__name__)
    # ## Geographic Coordinates
    # Parent directory for saving outputs
    base_local = Path.home() / "THUNER_output"
    start = "2005-11-13T00:00:00"
    end = "2005-11-13T03:00:00"
    # Set a flag for whether or not to remove existing output directories
    remove_existing_outputs = True
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
    grid_options.to_json(options_directory / "grid.json")
    # Initialize synthetic objects. Each is given a finite lifetime (30-120 min) and linear
    # fade-in/out, so objects appear, intensify, weaken and disappear over the run.
    starting_objects = []
    for i in range(5):
        major = 3 * (7 + 4 * i)  # full axis length in km
        obj = synthetic.EllipsoidObject(
            time=start,
            center_latitude=np.mean(lat),
            center_longitude=lon[(i + 1) * len(lon) // 6],
            direction=-np.pi / 4 + i * np.pi / 8,
            speed=30 - 4 * i,
            major=major,
            minor=0.4 * major,
            orientation=0.25 * np.pi + i * np.pi / 8,
            life_time=120 + i * 30,
            fade_in_time=60,
            fade_out_time=60,
        )
        starting_objects.append(obj)
    # Create data options dictionary. The objects are owned by a generator; FixedGenerator
    # simply replays this fixed list (procedural generators are a future extension).
    generator = synthetic.FixedGenerator(objects=starting_objects)
    synthetic_options = data.synthetic.SyntheticOptions(generator=generator)
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
    # ![THUNER applied to synthetic data.](https://raw.githubusercontent.com/THUNER-project/THUNER/refs/heads/main/gallery/synthetic_convective_20051113.gif)
    gallery_directory = Path(track.__file__).parent.parent.parent / "gallery"
    if gallery_directory.exists():
        gif_filename = f"convective_{format_time(start, day_only=True)}.gif"
        logger.info(f"Copying {gif_filename} to gallery.")
        gif_filepath = output_parent / f"visualize/match/{gif_filename}"
        shutil.copy(gif_filepath, gallery_directory / f"synthetic_{gif_filename}")
    else:
        logger.warning("Gallery missing. Skipping GIF copy.")
    # ## Cartesian Coordinates
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
        visualize_options=None,
        output_directory=output_parent,
    )
    # ## Procedural scenes
    #
    # Instead of placing objects by hand, a `RandomEllipseGenerator` spawns random cells over time: `initial_count` cells at the start, then new ones as a Poisson process at `spawn_rate` per hour, each with random geometry, motion and lifetime drawn from the configured ranges. It is deterministic given its `seed`, so re-running reproduces the same scene and the ground truth still matches the rendered data exactly. Procedural generators like this are the eventual aim of the synthetic module — building richer, more realistic scenes for testing.
    # A procedural scene over two hours, on the same geographic grid.
    output_parent = base_local / "runs/synthetic/random"
    if output_parent.exists() and remove_existing_outputs:
        shutil.rmtree(output_parent)
    options_directory = output_parent / "options"
    options_directory.mkdir(parents=True, exist_ok=True)
    lat = np.arange(-14, -6 + 0.025, 0.025).tolist()
    lon = np.arange(128, 136 + 0.025, 0.025).tolist()
    grid_options = option.grid.GridOptions(
        name="geographic", latitude=lat, longitude=lon
    )
    grid_options.to_json(options_directory / "grid.json")
    generator = synthetic.RandomEllipseGenerator(
        seed=42,
        spawn_rate=8,  # ~8 new cells per hour
        initial_count=3,
        major_range=(30, 60),  # full major axis, km
        speed_range=(5, 25),  # m/s
        life_time_range=(30, 120),  # minutes
    )
    synthetic_options = data.synthetic.SyntheticOptions(generator=generator)
    data_options = option.data.DataOptions(datasets=[synthetic_options])
    data_options.to_json(options_directory / "data.json")
    track_options = default.track.synthetic_track()
    track_options.to_json(options_directory / "track.json")
    visualize_options = default.visualize.synthetic_runtime(
        options_directory / "visualize.json"
    )
    visualize_options.to_json(options_directory / "visualize.json")
    times = np.arange(
        np.datetime64(start),
        np.datetime64(start) + np.timedelta64(2, "h") + np.timedelta64(10, "m"),
        np.timedelta64(10, "m"),
    )
    track.track(
        times=times,
        data_options=data_options,
        grid_options=grid_options,
        track_options=track_options,
        visualize_options=None,
        output_directory=output_parent,
    )
    ground_truth = analyze.synthetic.write_ground_truth(
        output_parent, data_options=data_options, times=times, grid_options=grid_options
    )
    print(ground_truth["synthetic"].head(10).to_string())


if __name__ == "__main__":
    test_synthetic()
