"""Functions for visualizing object attributes and classifications."""

import gc
from time import sleep
import multiprocessing
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import thuner.visualize.horizontal as horizontal

from thuner.attribute.utils import read_attribute_csv
from thuner.analyze.utils import read_options
import thuner.data.dispatch as dispatch
import thuner.detect.detect as detect
from thuner.utils import format_time, new_angle, circular_mean
import thuner.parallel as parallel
import thuner.visualize.utils as utils
import thuner.visualize.visualize as visualize
from thuner.log import setup_logger, logging_listener

logger = setup_logger(__name__)
proj = ccrs.PlateCarree()


mcs_legend_options = {"ncol": 3, "loc": "lower center"}


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
    num_processes=6,
):
    """Visualize mcs attributes at specified times."""
    plt.close("all")
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

    masks_filepath = output_directory / "masks/mcs.zarr"
    masks = xr.open_dataset(masks_filepath, engine="zarr")
    times = masks.time.values
    times = times[(times >= start_time) & (times <= end_time)]

    # Get colors
    color_angle_df = get_color_angle_df(output_directory)

    record_filepath = output_directory / f"records/filepaths/{dataset_name}.csv"
    filepaths = read_attribute_csv(record_filepath)
    time = times[0]
    args = [time, filepaths, masks, output_directory, figure_options]
    args += [options, track_options, dataset_name, dt, color_angle_df]
    visualize_mcs(*args)
    if len(times) == 1:
        # Switch back to original backend
        plt.close("all")
        matplotlib.use(original_backend)
        return
    if parallel_figure:
        with logging_listener(), multiprocessing.get_context("spawn").Pool(
            initializer=parallel.initialize_process, processes=num_processes
        ) as pool:
            results = []
            for time in times[1:]:
                sleep(2)
                args = [time, filepaths, masks, output_directory, figure_options]
                args += [options, track_options, dataset_name, dt, color_angle_df]
                args = tuple(args)
                results.append(pool.apply_async(visualize_mcs, args))
            pool.close()
            pool.join()
            parallel.check_results(results)
    else:
        for time in times[1:]:
            args = [time, filepaths, masks, output_directory, figure_options]
            args += [options, track_options, dataset_name, dt, color_angle_df]
            visualize_mcs(*args)
    if animate:
        figure_name = figure_options["name"]
        save_directory = output_directory / f"visualize"
        figure_directory = output_directory / f"visualize/{figure_name}"
        args = [figure_name, "mcs", output_directory, save_directory]
        args += [figure_directory, figure_name]
        visualize.animate_object(*args, by_date=by_date)
    # Switch back to original backend
    plt.close("all")
    matplotlib.use(original_backend)


# # @profile
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
    color_angle_df,
):
    """Wrapper for mcs_horizontal."""
    logger.info(f"Visualizing MCS at time {time}.")

    # Get object colors
    keys = color_angle_df.loc[color_angle_df["time"] == time]["universal_id"].values
    values = color_angle_df.loc[color_angle_df["time"] == time]["color_angle"].values
    values = [visualize.mask_colormap(v / (2 * np.pi)) for v in values]
    object_colors = dict(zip(keys, values))

    filepath = filepaths[dataset_name].loc[time]
    dataset_options = options["data"].dataset_by_name(dataset_name)
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
    ds, boundary_coords, simple_boundary_coords = convert(*args)
    del boundary_coords
    logger.debug(f"Getting grid from dataset at time {time}.")
    get_grid = dispatch.grid_from_dataset_dispatcher.get(dataset_name)
    if get_grid is None:
        message = f"Dataset {dataset_name} not found in grid from dataset "
        message += "dispatcher."
        raise KeyError(message)
    grid = get_grid(ds, "reflectivity", time)
    del ds
    logger.debug(f"Rebuilding processed grid for time {time}.")
    processed_grid = detect.rebuild_processed_grid(grid, track_options, "mcs", 1)
    del grid
    mask = masks.sel(time=time)
    mask = mask.load()
    args = [output_directory, processed_grid, mask, simple_boundary_coords]
    args += [figure_options, options["grid"]]
    figure_name = figure_options["name"]
    style = figure_options["style"]
    with plt.style.context(visualize.styles[style]), visualize.set_style(style):
        fig, ax = mcs_horizontal(*args, dt=dt, object_colors=object_colors)
        # Remove mask and processed_grid from memory after generating the figure
        del mask, processed_grid
        filename = f"{format_time(time)}.png"
        filepath = output_directory / f"visualize/{figure_name}/{filename}"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving {figure_name} figure for {time}.")
        fig.savefig(filepath, bbox_inches="tight")
        utils.reduce_color_depth(filepath)
        plt.clf()
        plt.close()
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
    object_colors=None,
):
    """Create a horizontal cross section plot."""
    member_objects = [convective_label, anvil_label]
    options = read_options(output_directory)
    track_options = options["track"]

    time = grid.time.values
    logger.debug(f"Creating grouped mask figure at time {time}.")
    try:
        filepath = output_directory / "analysis/quality.csv"
        kwargs = {"times": [time], "columns": ["duration", "parents"]}
        object_quality = read_attribute_csv(filepath, **kwargs).loc[time]
        object_quality = object_quality.any(axis=1).to_dict()
    except (FileNotFoundError, KeyError):
        object_quality = None

    args = [grid, mask, grid_options, figure_options, member_objects]
    args += [boundary_coordinates]
    kwargs = {"object_colors": object_colors, "mask_quality": object_quality}
    fig, axes, colorbar_axes, legend_axes = horizontal.grouped_mask(*args, **kwargs)

    try:
        filepath = output_directory / "attributes/mcs/group.csv"
        group = read_attribute_csv(filepath, times=[time]).loc[time]
        filepath = output_directory / "analysis/velocities.csv"
        velocities = read_attribute_csv(filepath, times=[time]).loc[time]
        # filepath = output_directory / "analysis/classification.csv"
        # classification = read_attribute_csv(filepath, times=[time]).loc[time]
        filepath = output_directory / f"attributes/mcs/{convective_label}/ellipse.csv"
        ellipse = read_attribute_csv(filepath, times=[time]).loc[time]
        new_names = {"latitude": "ellipse_latitude", "longitude": "ellipse_longitude"}
        ellipse = ellipse.rename(columns=new_names)
        filepath = output_directory / "analysis/quality.csv"
        quality = read_attribute_csv(filepath, times=[time]).loc[time]
        attributes = pd.concat([ellipse, group, velocities, quality], axis=1)
        objs = group.reset_index()["universal_id"].values
    except KeyError:
        # If no attributes, set objs=[]
        objs = []

    # Display velocity attributes
    for obj_id in objs:
        obj_attr = attributes.loc[obj_id]
        args = [axes, figure_options, obj_attr]
        velocity_attributes_horizontal(*args, dt=dt)
        displacement_attributes_horizontal(*args)
        ellipse_attributes(*args)
        if object_quality[obj_id]:
            text_attributes_horizontal(*args, object_quality=object_quality)

    style = figure_options["style"]
    scale = utils.get_extent(grid_options)[1]

    key_color = visualize.figure_colors[style]["key"]
    horizontal.vector_key(axes[0], color=key_color, dt=dt, scale=scale)
    kwargs = {"mcs_name": "mcs", "mcs_level": 1}
    convective_label, stratiform_label = get_altitude_labels(track_options, **kwargs)

    axes[0].set_title(convective_label)
    axes[1].set_title(stratiform_label)

    # Get legend proxy artists
    handles = []
    labels = []
    handle = horizontal.domain_boundary_legend_artist()
    handles += [handle]
    labels += ["Domain Boundary"]
    handle = horizontal.ellipse_legend_artist("Major Axis", figure_options["style"])
    handles += [handle]
    labels += ["Major Axis"]
    attribute_names = figure_options["attributes"]
    for name in [attr for attr in attribute_names if attr != "id"]:
        color = colors_dispatcher[name]
        label = label_dispatcher[name]
        handle = horizontal.displacement_legend_artist(color, label)
        handles.append(handle)
        labels.append(label)

    handle, handler = horizontal.mask_legend_artist()
    handles += [handle]
    labels += ["Object Masks"]
    legend_color = visualize.figure_colors[figure_options["style"]]["legend"]
    handles, labels = handles[::-1], labels[::-1]

    args = [handles, labels]
    leg_ax = legend_axes[0]
    if scale == 1:
        legend = leg_ax.legend(*args, **mcs_legend_options, handler_map=handler)
    elif scale == 2:
        mcs_legend_options["loc"] = "lower left"
        mcs_legend_options["bbox_to_anchor"] = (-0.0, -0.425)
        legend = leg_ax.legend(*args, **mcs_legend_options, handler_map=handler)
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
    "ambient": system_contained + ["duration"],
    "velocity": system_contained + ["velocity", "duration"],
    "shear": system_contained + ["shear", "duration"],
    "relative_velocity": system_contained + ["relative_velocity", "duration"],
    "offset": system_contained + ["offset", "duration"],
    "major": ["convective_contained", "anvil_contained", "axis_ratio", "duration"],
    "minor": ["convective_contained", "anvil_contained", "axis_ratio", "duration"],
    "mask": ["parents", "duration"],
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
        args = [axes[0], latitude, longitude, u, v, color]
        axes[0] = horizontal.cartesian_velocity(*args, quality=quality, dt=dt)

    return legend_handles


def text_attributes_horizontal(
    axes, figure_options, object_attributes, object_quality=None
):
    """Add object ID attributes."""

    if "id" in figure_options["attributes"]:
        latitude = object_attributes["latitude"]
        longitude = object_attributes["longitude"]
        args = [axes[0], str(object_attributes.name), longitude, latitude]
        horizontal.embossed_text(*args)
        args[0] = axes[1]
        horizontal.embossed_text(*args)
    return


def get_quality(quality_names, object_attributes, method="all"):
    if quality_names is None:
        quality = True
    else:
        qualities = object_attributes[quality_names]
        if method == "all":
            quality = qualities.all()
        elif method == "any":
            quality = qualities.any()
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
            args = [axes[0], latitude, longitude, dx, dy, color]
            kwargs = {"quality": quality}
            axes[0] = horizontal.cartesian_displacement(*args, **kwargs, arrow=False)
            args[0] = axes[1]
            axes[1] = horizontal.cartesian_displacement(*args, **kwargs, arrow=False)
        legend_artist = horizontal.displacement_legend_artist(color, label)
        legend_handles.append(legend_artist)

    return legend_handles


def convert_parents(parents):
    """Convert a parents string to a list of integers."""
    if str(parents) == "nan":
        return []
    parents_list = parents.split(" ")
    return [int(parent) for parent in parents_list]


def get_parent_angles(df, row, color_dict, previous_time):
    """Get the parent angles for the object in row."""
    obj_parents = convert_parents(row["parents"])
    parent_angles = []
    areas = []
    for parent in obj_parents:
        dict_universal_ids = np.array(color_dict["universal_id"])
        times = np.array(color_dict["time"])
        cond = (dict_universal_ids == parent) & (times == previous_time)
        parent_angle = np.array(color_dict["color_angle"])[cond][0]
        parent_angles.append(parent_angle)
        parent_universal_id = dict_universal_ids[cond][0]
        area = df.loc[previous_time, parent_universal_id]["area"]
        areas.append(area)
    return parent_angles, areas


def new_color_angle(df, row, color_dict, previous_time, angle_list):
    """Get a new color for the new object in row."""
    # Object not yet in color_dict
    if str(row["parents"]) == "nan":
        # If object has no parents, get a new color angle as different as possible
        # from existing color angles
        angles = color_dict["color_angle"]
        return new_angle(angles + angle_list)
    else:
        # If object has parents, get the average color angle of the parents,
        # weighting the average by object area
        args = [df, row, color_dict, previous_time]
        parent_angles, areas = get_parent_angles(*args)
        return circular_mean(parent_angles, areas)


def update_color_angle(df, row, color_dict, previous_time, universal_id):
    # If object is already in color_dict, get its color angle
    dict_universal_ids = np.array(color_dict["universal_id"])
    times = np.array(color_dict["time"])
    cond = (dict_universal_ids == universal_id) & (times == previous_time)
    previous_angle = np.array(color_dict["color_angle"])[cond][0]
    previous_area = df.loc[previous_time, universal_id]["area"]
    if str(row["parents"]) == "nan":
        # If the object has no new parents, i.e. no mergers have occured,
        # retain the same color
        return previous_angle
    else:
        # If the object has new parents, get the average color angle of the
        # parents and the current object
        args = [df, row, color_dict, previous_time]
        parent_angles, areas = get_parent_angles(*args)
        args = [parent_angles + [previous_angle], areas + [previous_area]]
        return circular_mean(*args)


def get_color_angle_df(output_parent):
    """
    Get a dictionary containing color angles, i.e. indices, for displaying masks.
    The color angle is calculated to reflect object splits/merges.
    """
    filepath = output_parent / "attributes/mcs/core.csv"
    df = read_attribute_csv(filepath, columns=["parents", "area"])
    color_dict = {"time": [], "universal_id": [], "color_angle": []}
    times = sorted(np.unique(df.reset_index().time))
    previous_time = None
    for i, time in enumerate(times):
        df_time = df.xs(time, level="time")
        universal_ids = sorted(np.unique(df_time.reset_index().universal_id))
        time_list, universal_id_list, angle_list = [], [], []
        if i > 0:
            previous_time = times[i - 1]
        for j, universal_id in enumerate(universal_ids):
            row = df_time.loc[universal_id]
            if universal_id not in color_dict["universal_id"]:
                # Object not yet in color_dict
                angle = new_color_angle(df, row, color_dict, previous_time, angle_list)
            else:
                # If object is already in color_dict, get its color angle
                args = [df, row, color_dict, previous_time, universal_id]
                angle = update_color_angle(*args)
            time_list.append(time)
            universal_id_list.append(universal_id)
            angle_list.append(angle)
        color_dict["time"] += time_list
        color_dict["universal_id"] += universal_id_list
        color_dict["color_angle"] += angle_list
    return pd.DataFrame(color_dict)
