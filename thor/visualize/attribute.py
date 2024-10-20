"""Functions for visualizing object attributes and classifications."""

import gc
import os
import multiprocessing
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import thor.visualize.horizontal as horizontal
from thor.visualize.visualize import figure_colors, styles, animate_object
from thor.attribute.utils import read_attribute_csv
from thor.analyze.utils import read_options
import thor.data.dispatch as dispatch
import thor.detect.detect as detect
from thor.utils import format_time
import thor.parallel as parallel
import thor.visualize.utils as utils
from thor.log import setup_logger, logging_listener

logger = setup_logger(__name__)
proj = ccrs.PlateCarree()


mcs_legend_options = {"loc": "lower center", "bbox_to_anchor": (1.15, -0.375)}
mcs_legend_options.update({"ncol": 3, "fancybox": True, "shadow": True})


def get_altitude_labels(track_options, mcs_name="mcs", mcs_level=1):
    """Get altitude labels for convective and stratiform objects."""
    mcs_options = track_options.levels[mcs_level].options_by_name(mcs_name)
    convective = mcs_options.grouping.member_objects[0]
    convective_level = mcs_options.grouping.member_levels[0]
    stratiform = mcs_options.grouping.member_objects[-1]
    stratiform_level = mcs_options.grouping.member_levels[-1]
    convective_options = track_options.levels[convective_level]
    convective_options = convective_options.options_by_name(convective)
    stratiform_options = track_options.levels[stratiform_level]
    stratiform_options = stratiform_options.options_by_name(stratiform)
    convective_altitudes = np.array(convective_options.detection.altitudes)
    stratiform_altitudes = np.array(stratiform_options.detection.altitudes)
    convective_altitudes = np.round(convective_altitudes / 1e3, 1)
    stratiform_altitudes = np.round(stratiform_altitudes / 1e3, 1)
    convective_label = f"{convective_altitudes[0]:g} to {convective_altitudes[1]:g} km"
    stratiform_label = f"{stratiform_altitudes[0]:g} to {stratiform_altitudes[1]:g} km"
    return convective_label + " Altitude", stratiform_label + " Altitude"


def mcs_series(
    output_directory,
    start_time,
    end_time,
    figure_options,
    convective_label="convective",
    dataset_name=None,
    animate=True,
    parallel_figure=False,
    dt=3600,
    by_date=True,
):
    """Visualize mcs attributes at specified times."""
    plt.close("all")
    # Switch to non-interactive backend
    original_backend = matplotlib.get_backend()
    matplotlib.use("Agg")

    start_time = np.datetime64(start_time)
    end_time = np.datetime64(end_time)
    options = read_options(output_directory)
    track_options = options["track"]
    if dataset_name is None:
        try:
            object_options = track_options.levels[0].options_by_name(convective_label)
            dataset_name = object_options.dataset
        except KeyError:
            message = "Could not infer dataset used for detection. Provide manually."
            raise KeyError(message)

    masks_filepath = output_directory / "masks/mcs.nc"
    masks = xr.open_dataset(masks_filepath)
    times = masks.time.values
    times = times[(times >= start_time) & (times <= end_time)]
    record_filepath = output_directory / f"records/filepaths/{dataset_name}.csv"
    filepaths = read_attribute_csv(record_filepath)
    time = times[0]
    args = [time, filepaths, masks, output_directory, figure_options]
    args += [options, track_options, dataset_name, dt]
    visualize_mcs(*args)
    if len(times) == 1:
        # Switch back to original backend
        plt.close("all")
        matplotlib.use(original_backend)
        return
    if parallel_figure:
        num_processes = int(0.75 * os.cpu_count())
        num_processes = 4
        with logging_listener(), multiprocessing.get_context("spawn").Pool(
            initializer=parallel.initialize_process, processes=num_processes
        ) as pool:
            results = []
            for time in times[1:]:
                args = [time, filepaths, masks, output_directory, figure_options]
                args += [options, track_options, dataset_name, dt]
                args = tuple(args)
                results.append(pool.apply_async(visualize_mcs, args))
            pool.close()
            pool.join()
            parallel.check_results(results)
    else:
        for time in times[1:]:
            args = [time, filepaths, masks, output_directory, figure_options]
            args += [options, track_options, dataset_name, dt]
            visualize_mcs(*args)
    if animate:
        figure_name = figure_options["name"]
        save_directory = output_directory / f"visualize"
        figure_directory = output_directory / f"visualize/{figure_name}"
        args = [figure_name, "mcs", output_directory, save_directory]
        args += [figure_directory, figure_name]
        animate_object(*args, by_date=by_date)
    # Switch back to original backend
    plt.close("all")
    matplotlib.use(original_backend)


def visualize_mcs(
    time,
    filepaths,
    masks,
    output_directory,
    figure_options,
    options,
    track_options,
    dataset_name,
    dt,
):
    """Wrapper for mcs_horizontal."""
    filepath = filepaths[dataset_name].loc[time]
    dataset_options = options["data"][dataset_name]
    convert = dispatch.convert_dataset_dispatcher.get(dataset_name)
    if convert is None:
        message = f"Dataset {dataset_name} not found in dispatch."
        logger.debug(f"Getting grid from dataset at time {time}.")
        raise KeyError(message)
    convert_args_dispatcher = {
        "cpol": [time, filepath, dataset_options, options["grid"]],
        "gridrad": [time, filepath, track_options, dataset_options, options["grid"]],
    }
    args = convert_args_dispatcher[dataset_name]
    ds, boundary_coords = convert(*args)
    logger.debug(f"Getting grid from dataset at time {time}.")
    get_grid = dispatch.grid_from_dataset_dispatcher.get(dataset_name)
    if get_grid is None:
        message = f"Dataset {dataset_name} not found in grid from dataset "
        message += "dispatcher."
        raise KeyError(message)
    grid = get_grid(ds, "reflectivity", time)
    logger.debug(f"Rebuilding processed grid for time {time}.")
    processed_grid = detect.rebuild_processed_grid(grid, track_options, "mcs", 1)
    mask = masks.sel(time=time)
    args = [output_directory, processed_grid, mask, boundary_coords]
    args += [figure_options, options["grid"]]
    figure_name = figure_options["name"]
    with plt.style.context(styles[figure_options["style"]]):
        fig, ax = mcs_horizontal(*args, dt=dt)
        filename = f"{format_time(time)}.png"
        filepath = output_directory / f"visualize/{figure_name}/{filename}"
        lock = multiprocessing.Lock()
        with lock:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Saving {figure_name} figure for {time}.")
            fig.savefig(filepath, bbox_inches="tight")
            utils.reduce_color_depth(filepath)
            plt.close(fig)
    gc.collect()


def mcs_horizontal(
    output_directory,
    grid,
    mask,
    boundary_coordinates,
    figure_options,
    grid_options,
    convective_label="convective",
    anvil_label="anvil",
    dt=3600,
):
    """Create a horizontal cross section plot."""
    member_objects = [convective_label, anvil_label]
    options = read_options(output_directory)
    track_options = options["track"]

    args = [grid, mask, grid_options, figure_options, member_objects]
    args += [boundary_coordinates]
    time = grid.time.values
    logger.debug(f"Creating grouped mask figure at time {time}.")
    fig, axes = horizontal.grouped_mask(*args)

    time = grid.time.values

    try:
        filepath = output_directory / "attributes/mcs/group.csv"
        group = read_attribute_csv(filepath, times=[time]).loc[time]
        filepath = output_directory / "analysis/velocities.csv"
        velocities = read_attribute_csv(filepath, times=[time]).loc[time]
        filepath = output_directory / "analysis/classification.csv"
        classification = read_attribute_csv(filepath, times=[time]).loc[time]
        filepath = output_directory / f"attributes/mcs/{convective_label}/ellipse.csv"
        ellipse = read_attribute_csv(filepath, times=[time]).loc[time]
        new_names = {"latitude": "ellipse_latitude", "longitude": "ellipse_longitude"}
        ellipse = ellipse.rename(columns=new_names)
        filepath = output_directory / "analysis/quality.csv"
        quality = read_attribute_csv(filepath, times=[time]).loc[time]
        attributes = pd.concat(
            [ellipse, group, velocities, classification, quality], axis=1
        )
        objs = group.reset_index()["universal_id"].values

    except KeyError:
        # If no attributes, return early
        objs = []

    # Display velocity attributes
    for obj_id in objs:
        obj_attr = attributes.loc[obj_id]
        args = [axes, figure_options, obj_attr]
        velocity_attributes_horizontal(*args, dt=dt)
        displacement_attributes_horizontal(*args)
        ellipse_attributes(*args)

    style = figure_options["style"]
    key_color = figure_colors[style]["key"]
    horizontal.vector_key(axes[0], color=key_color, dt=dt)
    args_dict = {"mcs_name": "mcs", "mcs_level": 1}
    convective_label, stratiform_label = get_altitude_labels(track_options, **args_dict)

    axes[0].set_title(convective_label)
    axes[1].set_title(stratiform_label)

    # Get legend proxy artists
    legend_handles = []
    handle = horizontal.domain_boundary_legend_artist()
    legend_handles += [handle]
    handle = horizontal.ellipse_legend_artist("Major Axis", figure_options["style"])
    legend_handles += [handle]
    attribute_names = figure_options["attributes"]
    for name in attribute_names:
        color = colors_dispatcher[name]
        label = label_dispatcher[name]
        handle = horizontal.displacement_legend_artist(color, label)
        legend_handles.append(handle)

    legend_color = figure_colors[figure_options["style"]]["legend"]
    legend = axes[0].legend(handles=legend_handles[::-1], **mcs_legend_options)
    legend.get_frame().set_alpha(None)
    legend.get_frame().set_facecolor(legend_color)

    return fig, axes


names_dispatcher = {
    "velocity": ["u", "v"],
    "relative_velocity": ["u_relative", "v_relative"],
    "shear": ["u_shear", "v_shear"],
    "ambient": ["u_ambient", "v_ambient"],
    "offset": ["x_offset", "y_offset"],
}
colors_dispatcher = {
    "velocity": "tab:purple",
    "relative_velocity": "darkgreen",
    "shear": "tab:purple",
    "ambient": "tab:red",
    "offset": "tab:blue",
}
label_dispatcher = {
    "velocity": "System Velocity",
    "relative_velocity": "Relative System Velocity",
    "shear": "Ambient Shear",
    "ambient": "Ambient Wind",
    "offset": "Stratiform-Offset",
}
system_contained = ["convective_contained", "anvil_contained"]
quality_dispatcher = {
    "ambient": system_contained,
    "velocity": system_contained + ["velocity"],
    "shear": system_contained + ["shear"],
    "relative_velocity": system_contained + ["relative_velocity"],
    "offset": system_contained + ["offset"],
    "major": ["convective_contained", "axis_ratio"],
    "minor": ["convective_contained", "axis_ratio"],
}


def velocity_attributes_horizontal(axes, figure_options, object_attributes, dt=3600):
    """
    Add velocity attributes. Assumes the attribtes dataframe has already
    been subset to the desired time and object, so is effectively a dictionary.
    """

    velocity_attributes = ["ambient", "relative_velocity", "velocity", "shear"]
    attribute_names = figure_options["attributes"]
    velocity_attributes = [v for v in attribute_names if v in velocity_attributes]
    latitude = object_attributes["latitude"]
    longitude = object_attributes["longitude"]
    legend_handles = []

    for attribute in velocity_attributes:
        [u_name, v_name] = names_dispatcher[attribute]
        u, v = object_attributes[u_name], object_attributes[v_name]
        quality_names = quality_dispatcher.get(attribute)
        quality = get_quality(quality_names, object_attributes)
        color = colors_dispatcher[attribute]
        label = label_dispatcher[attribute]
        args = [axes[0], latitude, longitude, u, v, color, label]
        axes[0] = horizontal.cartesian_velocity(*args, quality=quality, dt=dt)

    return legend_handles


def get_quality(quality_names, object_attributes):
    if quality_names is None:
        quality = True
    else:
        qualities = object_attributes[quality_names]
        quality = qualities.all()
    return quality


def ellipse_attributes(axes, figure_options, object_attributes):
    """Add ellipse axis attributes."""

    quality_names = quality_dispatcher.get("major")
    quality = get_quality(quality_names, object_attributes)
    latitude = object_attributes["ellipse_latitude"]
    longitude = object_attributes["ellipse_longitude"]
    major, orientation = object_attributes["major"], object_attributes["orientation"]
    style = figure_options["style"]
    args = [axes[0], latitude, longitude, major, orientation, "Major Axis", style]
    args += [quality]
    legend_handles = []
    legend_handle = horizontal.ellipse_axis(*args)
    legend_handles.append(legend_handle)

    return legend_handles


def displacement_attributes_horizontal(axes, figure_options, object_attributes):
    """Add displacement attributes."""

    displacement_attributes = ["offset"]
    attribute_names = figure_options["attributes"]
    displacement_attributes = [
        v for v in attribute_names if v in displacement_attributes
    ]
    latitude = object_attributes["latitude"]
    longitude = object_attributes["longitude"]
    legend_handles = []

    for attribute in displacement_attributes:
        [dx_name, dy_name] = names_dispatcher[attribute]
        # Convert displacements from km to metres
        if object_attributes is not None:
            dx, dy = object_attributes[dx_name] * 1e3, object_attributes[dy_name] * 1e3
            color = colors_dispatcher[attribute]
            label = label_dispatcher[attribute]
            quality_names = quality_dispatcher.get(attribute)
            quality = get_quality(quality_names, object_attributes)
            args = [axes[0], latitude, longitude, dx, dy, color, label]
            args_dict = {"quality": quality}
            axes[0] = horizontal.cartesian_displacement(*args, **args_dict, arrow=False)
            args[0] = axes[1]
            axes[1] = horizontal.cartesian_displacement(*args, **args_dict, arrow=False)
        legend_artist = horizontal.displacement_legend_artist(color, label)
        legend_handles.append(legend_artist)

    return legend_handles
