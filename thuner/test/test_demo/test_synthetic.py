import xarray as xr
from pathlib import Path
import shutil
import matplotlib.pyplot as plt
import numpy as np
import thuner.data as data
import thuner.default as default
import thuner.track.track as track
import thuner.option as option
import thuner.analyze as analyze
import thuner.data.synthetic as synthetic
import thuner.attribute as attribute
import thuner.visualize as visualize
from thuner.utils import format_time, copy_to_gallery
from thuner.log import setup_logger
import matplotlib.pyplot as plt
from matplotlib import colors


def test_synthetic():
    # # Testing: Synthetic Data
    # The `synthetic` module allows us to create artificial datasets with known velocities, extents etc. which we can then compare with those estimated by THUNER.
    """Synthetic data demo/test."""
    logger = setup_logger(__name__)
    # ## Geographic Coordinates
    # Parent directory for saving outputs
    base_local = Path.home() / "THUNER_output"
    start = "2005-11-13T00:00:00"
    end = "2005-11-13T02:00:00"
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
    # target_objects tells analyze.synthetic.write_ground_truth which tracked object's
    # masks to match the synthetic truth objects against (by truth-mask overlap).
    synthetic_options = data.synthetic.SyntheticOptions(
        generator=generator, target_objects=["convective"]
    )
    data_options = option.data.DataOptions(datasets=[synthetic_options])
    data_options.to_json(options_directory / "data.json")
    track_options = default.track.synthetic_track()
    track_options.to_json(options_directory / "track.json")
    # Create the display_options dictionary
    visualize_options = default.visualize.synthetic_runtime(
        options_directory / "visualize.json"
    )
    visualize_options.to_json(options_directory / "visualize.json")
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
    gif_filename = f"convective_{format_time(start, day_only=True)}.gif"
    gif_filepath = output_parent / f"visualize/match/{gif_filename}"
    copy_to_gallery(gif_filepath, gallery_name=f"synthetic_{gif_filename}")
    # ![THUNER applied to synthetic data.](https://raw.githubusercontent.com/THUNER-project/THUNER/refs/heads/main/gallery/synthetic_convective_20051113.gif)
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
    start = "2005-11-13T00:00:00"
    end = "2005-11-13T04:00:00"
    lat = np.arange(-14, -6 + 0.025, 0.025).tolist()
    lon = np.arange(128, 136 + 0.025, 0.025).tolist()
    grid_options = option.grid.GridOptions(
        name="geographic", latitude=lat, longitude=lon
    )
    grid_options.to_json(options_directory / "grid.json")
    generator = synthetic.RandomEllipseGenerator(
        seed=42,
        spawn_rate=8,  # ~8 new cells per hour
        initial_count=8,
        major_range=(50, 100),  # full major axis, km
        speed_range=(5, 45),  # m/s
        life_time_range=(30, 240),  # minutes
    )
    # target_objects tells analyze.synthetic.write_ground_truth which tracked object's
    # masks to match the synthetic truth objects against (by truth-mask overlap).
    synthetic_options = data.synthetic.SyntheticOptions(
        generator=generator,
        target_objects=["convective"],
        # Save each generated grid so attribute.series can reload it when plotting.
        converted_options={"save": True},
    )
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
        np.datetime64(end) + np.timedelta64(10, "m"),
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
    analysis_options = analyze.mcs.AnalysisOptions()
    analysis_options.to_json(options_directory / "analysis.json")
    analyze.utils.smooth_flow_velocities("convective", output_parent)
    analyze.utils.quality_control("convective", output_parent, analysis_options)
    style = "presentation"
    attribute_handlers = default.visualize.detected_attribute_handlers(
        output_parent, style
    )
    figure_options = option.visualize.HorizontalAttributeOptions(
        name="synthetic_convective",
        object_name="convective",
        style=style,
        attribute_handlers=attribute_handlers,
    )
    visualize.attribute.series(
        output_directory=output_parent,
        start_time=start,
        end_time=end,
        figure_options=figure_options,
        dataset_name="synthetic",
        parallel_figure=True,
        by_date=False,
        num_processes=8,
    )
    match_tables = analyze.synthetic.write_ground_truth(output_parent, times=times)
    matched = match_tables["synthetic"].reset_index()
    matched = matched[
        matched["convective_universal_id"] != 0
    ]  # drop unmatched truth objects
    matched["n_sharing"] = matched.groupby(["time", "convective_universal_id"])[
        "id"
    ].transform("size")
    matched["overlapped"] = matched["n_sharing"] > 1
    velocities = attribute.utils.read_attribute(output_parent, "analysis", "velocities")
    ellipse = attribute.utils.read_attribute(
        output_parent, "attributes", "convective", "ellipse"
    )
    dt = xr.open_datatree(output_parent / "output.zarr")
    quality = attribute.utils.read_attribute(output_parent, "analysis", "quality")
    truth = match_tables["synthetic"].reset_index()
    truth = truth[truth["convective_universal_id"] != 0]
    # flag truth objects merged into a shared detected object (see the overlap demo above)
    truth["overlapped"] = (
        truth.groupby(["time", "convective_universal_id"])["id"].transform("size") > 1
    )
    truth = truth.rename(columns={"u": "u_true", "v": "v_true"})
    detected = velocities.reset_index()[["time", "universal_id", "u", "v"]]
    detected = detected.rename(columns={"u": "u_detected", "v": "v_detected"})
    quality = quality.reset_index()[["time", "universal_id", "contained"]]
    comparison = truth.merge(
        detected,
        left_on=["time", "convective_universal_id"],
        right_on=["time", "universal_id"],
        how="inner",
    )
    comparison = comparison.merge(
        quality,
        left_on=["time", "convective_universal_id"],
        right_on=["time", "universal_id"],
        how="inner",
    )
    comparison
    cond = comparison["contained"] & ~(comparison["overlapped"])
    comparison = (
        comparison.where(cond).dropna().drop(columns=["contained", "overlapped"])
    )
    comparison["u_error"] = comparison["u_detected"] - comparison["u_true"]
    comparison["v_error"] = comparison["v_detected"] - comparison["v_true"]
    comparison["u_rmse"] = np.sqrt(np.mean(comparison["u_error"] ** 2))
    comparison["v_rmse"] = np.sqrt(np.mean(comparison["v_error"] ** 2))
    style = "dark_background"
    with plt.style.context(style):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, component in zip(axes, ["u", "v"]):
            ax.text(
                0.025,
                0.975,
                f"RMSE: {comparison[f'{component}_rmse'].iloc[0]:.2f} m/s",
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                color="white",
                backgroundcolor="dimgrey",
            )
            true_velocity = comparison[f"{component}_true"]
            detected_velocity = comparison[f"{component}_detected"]
            boundaries = np.arange(0, 11)
            n_colors = len(boundaries) - 1
            # 3. Set up the discrete colormap and normalization
            cmap = plt.get_cmap("Reds", n_colors)  # Choose your base colormap
            norm = colors.BoundaryNorm(boundaries, cmap.N, clip=True)
            hb = ax.hexbin(
                true_velocity,
                detected_velocity,
                gridsize=15,
                cmap=cmap,
                norm=norm,
                bins=10,
                mincnt=1,
                extent=(-40, 40, -40, 40),
                linewidth=0.5,
                edgecolor="black",
            )
            ax.set_facecolor("dimgrey")
            lims = [-40, 40]
            ax.plot(lims, lims, "white", lw=1)
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_xticks(np.arange(-40, 41, 20))
            ax.set_yticks(np.arange(-40, 41, 20))
            ax.set_xlabel(f"true {component} [m/s]")
            ax.set_ylabel(f"detected {component} [m/s]")
            ax.set_title(f"{component} velocity: true vs detected")
            ax.set_aspect("equal")
            ax.grid()
            ax.set_axisbelow(True)
            cbar = plt.colorbar(hb, ax=ax)
            cbar.set_label("Count [-]")
        (output_parent / "visualize").mkdir(parents=True, exist_ok=True)
        plt.savefig(
            output_parent / "visualize" / "true_vs_detected.png", bbox_inches="tight"
        )
    plt.show()
    comparison["u_error"] = comparison["u_detected"] - comparison["u_true"]
    comparison["v_error"] = comparison["v_detected"] - comparison["v_true"]
    u_rmse = np.sqrt(np.mean(comparison["u_error"] ** 2))
    v_rmse = np.sqrt(np.mean(comparison["v_error"] ** 2))
    print(f"u velocity RMSE: {u_rmse:.2f} [m/s]")
    print(f"v velocity RMSE: {v_rmse:.2f} [m/s]")


if __name__ == "__main__":
    test_synthetic()
