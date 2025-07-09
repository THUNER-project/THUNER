"""Functions for visualizing object attributes and classifications."""

import gc
from pathlib import Path
from pydantic import Field, model_validator
import tempfile
from typing import Any, Dict
from time import sleep
import xesmf as xe
import multiprocessing as mp
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import thuner.visualize.horizontal as horizontal
from thuner.utils import initialize_process, check_results
from thuner.utils import format_time, new_angle, circular_mean
from thuner.utils import BaseHandler, AttributeHandler
from thuner.attribute.utils import read_attribute_csv
from thuner.analyze.utils import read_options
import thuner.detect.detect as detect
import thuner.visualize.utils as utils
import thuner.visualize.visualize as visualize
from thuner.option.visualize import FigureOptions, GroupedHorizontalAttributeOptions
from thuner.option.visualize import HorizontalAttributeOptions
from thuner.log import setup_logger, logging_listener
from thuner.config import get_outputs_directory


__all__ = ["series", "grouped_horizontal"]

logger = setup_logger(__name__)
proj = ccrs.PlateCarree()


mcs_legend_options = {"ncol": 3, "loc": "lower center"}


def get_altitude_labels(
    track_options,
    object_name="mcs",
    object_level=1,
    member_objects=None,
    member_levels=None,
):
    """Get altitude labels for convective and stratiform objects."""
    object_options = track_options.levels[object_level].object_by_name(object_name)
    if member_objects is None:
        member_objects = object_options.grouping.member_objects
    if member_levels is None:
        all_member_objects = object_options.grouping.member_objects
        all_member_levels = object_options.grouping.member_levels
        member_levels = []
        for i, level in enumerate(all_member_levels):
            if all_member_objects[i] in member_objects:
                member_levels.append(level)
    labels = []
    for i, obj in enumerate(member_objects):
        level = member_levels[i]
        options = track_options.levels[level].object_by_name(obj)
        altitudes = np.array(options.detection.altitudes)
        altitudes = np.round(altitudes / 1e3, 1)
        labels.append(f"{altitudes[0]:g} to {altitudes[1]:g} km")
    return labels


def series(
    output_directory: str | Path,
    start_time,
    end_time,
    figure_options,
    dataset_name,
    animate=True,
    parallel_figure=False,
    by_date=True,
    num_processes=4,
):
    """Visualize attributes at specified times."""

    # Setup plt backend
    plt.close("all")
    original_backend = matplotlib.get_backend()
    matplotlib.use("Agg")

    # Setup times and masks
    start_time = np.datetime64(start_time)
    end_time = np.datetime64(end_time)
    options = read_options(output_directory)
    object_name = figure_options.object_name
    masks_filepath = output_directory / f"masks/{object_name}.zarr"
    masks = xr.open_dataset(masks_filepath, engine="zarr")
    times = masks.time.values
    times = times[(times >= start_time) & (times <= end_time)]

    figure_function = figure_options.method.function

    # Initialize the paths to save xesmf regridder weights
    dataset_options = options["data"].dataset_by_name(dataset_name)
    if dataset_options.reuse_regridder:
        if dataset_options.weights_filepath is None:
            filepath = output_directory / "records/regridder_weights"
            filepath = filepath / f"{dataset_options.name}.nc"
            dataset_options.weights_filepath = filepath

    # Start with first time
    args = [times[0], masks, output_directory, figure_options.model_dump()]
    args += [options, dataset_name]
    figure_function(*args)

    if len(times) == 1:
        # Switch back to original backend
        plt.close("all"), matplotlib.use(original_backend)
        return

    if parallel_figure:
        kwargs = {"initializer": initialize_process, "processes": num_processes}
        with logging_listener(), mp.get_context("spawn").Pool(**kwargs) as pool:
            results = []
            for time in times[1:]:
                sleep(2)
                # Note need to define a new args for each iteration! Can't simply
                # change the first element, or we break parallization! Note also it's
                # bad practice to pass dataframes to mp workers.
                args = [time, masks, output_directory]
                args += [figure_options.model_dump()]
                args += [options, dataset_name]
                args = tuple(args)
                results.append(pool.apply_async(figure_function, args))
            pool.close()
            pool.join()
            check_results(results)
    else:
        for time in times[1:]:
            args[0] = time
            figure_function(*args)

    if animate:
        figure_name = figure_options.name
        save_directory = output_directory / f"visualize"
        figure_directory = output_directory / f"visualize/{figure_name}"
        args = [figure_name, "mcs", output_directory, save_directory]
        args += [figure_directory, figure_name]
        visualize.animate_object(*args, by_date=by_date)
    # Close all figures to clear memory
    plt.close("all")
    # Switch back to original backend
    matplotlib.use(original_backend)


def get_mask_grid_boundary(
    object_name, time, filepaths_df, masks, dataset_name, options
):
    """Get the mask and grid for a given time."""

    filepath = filepaths_df[dataset_name].loc[time]
    dataset_options = options["data"].dataset_by_name(dataset_name)
    object_level = options["track"].object_by_name(object_name).hierarchy_level

    message = f"Converting {dataset_name}."
    logger.debug(message)
    args = [time, filepath, options["track"], options["grid"]]
    outs = dataset_options.convert_dataset(*args)
    ds, boundary_coords, simple_boundary_coords = outs
    del boundary_coords
    logger.debug(f"Getting grid from dataset at time {time}.")

    if len(dataset_options.fields) > 1:
        raise ValueError("Non-unique dataset field.")

    grid = dataset_options.grid_from_dataset(ds, dataset_options.fields[0], time)
    del ds
    logger.debug(f"Rebuilding processed grid for time {time}.")
    args = [grid, options["track"], object_name, object_level]
    processed_grid = detect.rebuild_processed_grid(*args)
    del grid
    mask = masks.sel(time=time).load()

    grid_time = processed_grid.time.values
    mask_time = mask.time.values
    if grid_time != time or mask_time != time:
        message = f"Grid or mask time {grid_time} does not match requested time {time}."
        raise ValueError(message)

    return mask, processed_grid, simple_boundary_coords


def get_object_colors(time, color_angle_df):
    """Get the object colors for a given time."""
    keys = color_angle_df.loc[color_angle_df["time"] == time]["universal_id"].values
    values = color_angle_df.loc[color_angle_df["time"] == time]["color_angle"].values
    values = [visualize.mask_colormap(v / (2 * np.pi)) for v in values]
    return dict(zip(keys, values))


def detected_horizontal(
    time,
    masks,
    output_directory,
    figure_options_dict,
    options,
    dataset_name,
):
    """Create a horizontal cross section plot."""
    logger.info(f"Visualizing attributes at time {time}.")

    # Rebuild the figure options
    figure_options = HorizontalAttributeOptions(**figure_options_dict)
    object_name = figure_options.object_name

    # Get filepaths dataframe
    record_filepath = output_directory / f"records/filepaths/{dataset_name}.csv"
    filepaths_df = read_attribute_csv(record_filepath, columns=[dataset_name])

    # Setup colors
    color_angle_df = get_color_angle_df(object_name, output_directory)

    grid_options = options["grid"]
    obj_name = figure_options.object_name

    args = [obj_name, time, filepaths_df, masks, dataset_name, options]
    mask, grid, boundary_coords = get_mask_grid_boundary(*args)
    mask = mask[obj_name + "_mask"]
    grid = grid[obj_name + "_grid"]
    object_colors = get_object_colors(time, color_angle_df)

    time = grid.time.values
    style = figure_options.style

    attribute_handlers = figure_options.attribute_handlers
    args = [grid, mask, grid_options, figure_options, boundary_coords]
    kwargs = {"object_colors": object_colors}

    with plt.style.context(visualize.styles[style]), visualize.set_style(style):
        figure_features = horizontal.detected_mask(*args, **kwargs)
        fig, subplot_axes, colorbar_axes, legend_axes = figure_features

    # Create the grouped object figure instance
    kwargs = {"object_name": object_name, "time": time, "grid": grid, "mask": mask}
    kwargs.update({"boundary_coordinates": boundary_coords})
    kwargs.update({"attribute_handlers": attribute_handlers})
    kwargs.update({"figure": fig, "subplot_axes": subplot_axes})
    kwargs.update({"colorbar_axes": colorbar_axes, "legend_axes": legend_axes})
    core_filepath = output_directory / f"attributes/{obj_name}/core.csv"
    kwargs["core_filepath"] = str(core_filepath)
    detected_figure = BaseFigure(**kwargs)
    # Remove duplicate mask and grid from memory after generating the figure
    del mask, grid, boundary_coords
    add_attributes(time, detected_figure)
    create_legend(detected_figure, grid_options, figure_options)

    filename = f"{format_time(time)}.png"
    filepath = output_directory / f"visualize/{figure_options.name}/{filename}"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving {figure_options.name} figure for {time}.")
    with plt.style.context(visualize.styles[style]), visualize.set_style(style):
        detected_figure.figure.savefig(filepath, bbox_inches="tight")
    del detected_figure
    utils.reduce_color_depth(filepath)
    plt.clf(), plt.close(), gc.collect()


def grouped_horizontal(
    time,
    masks,
    output_directory,
    figure_options_dict,
    options,
    dataset_name,
):
    """Create a horizontal cross section plot."""
    logger.info(f"Visualizing attributes at time {time}.")

    # Rebuild the figure options
    figure_options = GroupedHorizontalAttributeOptions(**figure_options_dict)

    # Get filepaths dataframe
    record_filepath = output_directory / f"records/filepaths/{dataset_name}.csv"
    filepaths_df = read_attribute_csv(record_filepath, columns=[dataset_name])
    obj_name = figure_options.object_name

    # Setup colors
    color_angle_df = get_color_angle_df(obj_name, output_directory)

    grid_options = options["grid"]
    args = [obj_name, time, filepaths_df, masks, dataset_name, options]
    mask, grid, boundary_coords = get_mask_grid_boundary(*args)
    object_colors = get_object_colors(time, color_angle_df)

    time = grid.time.values
    style = figure_options.style

    member_objects = figure_options.member_objects
    attribute_handlers = figure_options.attribute_handlers
    args = [grid, mask, grid_options, figure_options, member_objects]
    args += [boundary_coords]
    kwargs = {"object_colors": object_colors}

    with plt.style.context(visualize.styles[style]), visualize.set_style(style):
        figure_features = horizontal.grouped_mask(*args, **kwargs)
        fig, subplot_axes, colorbar_axes, legend_axes = figure_features

    # Set the subplot figure titles to altitudes if specified
    if figure_options.altitude_titles:
        # Get altitude labels for the member objects
        track_options = options["track"]
        args = [track_options, obj_name, 1, member_objects]
        altitude_labels = get_altitude_labels(*args)
        for i, label in enumerate(altitude_labels):
            subplot_axes[i].set_title(label)

    # Create the grouped object figure instance
    kwargs = {"object_name": obj_name, "time": time, "grid": grid, "mask": mask}
    kwargs.update({"boundary_coordinates": boundary_coords})
    kwargs.update({"attribute_handlers": attribute_handlers})
    kwargs.update({"member_objects": member_objects})
    kwargs.update({"figure": fig, "subplot_axes": subplot_axes})
    kwargs.update({"colorbar_axes": colorbar_axes, "legend_axes": legend_axes})
    core_filepath = output_directory / f"attributes/{obj_name}/core.csv"
    kwargs["core_filepath"] = str(core_filepath)
    base_directory = output_directory / f"attributes/{obj_name}/"
    filepaths_list = [str(base_directory / f"{obj}/core.csv") for obj in member_objects]
    kwargs["member_core_filepaths"] = dict(zip(member_objects, filepaths_list))
    grouped_figure = GroupedObjectFigure(**kwargs)
    # Remove duplicate mask and grid from memory after generating the figure
    del mask, grid, boundary_coords
    add_attributes(time, grouped_figure)
    create_legend(grouped_figure, grid_options, figure_options)

    filename = f"{format_time(time)}.png"
    filepath = output_directory / f"visualize/{figure_options.name}/{filename}"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving {figure_options.name} figure for {time}.")
    with plt.style.context(visualize.styles[style]), visualize.set_style(style):
        grouped_figure.figure.savefig(filepath, bbox_inches="tight")
    del grouped_figure
    utils.reduce_color_depth(filepath)
    plt.clf(), plt.close(), gc.collect()


def add_attribute(
    ax, object_name, handler, attribute_artists, legend_artists, time, figure
):
    """Add an attribute to the figure for a given object."""
    # Get attribute df
    attribute_artists[object_name][handler.name] = {}
    attribute_df = read_attribute_csv(handler.filepath, times=[time])
    kwargs = {"times": [time], "columns": handler.quality_variables}
    quality_df = read_attribute_csv(handler.quality_filepath, **kwargs)
    if handler.quality_method == "all":
        quality_df = quality_df.all(axis=1)
    elif handler.quality_method == "any":
        quality_df = quality_df.any(axis=1)

    try:
        id_type = "universal_id"
        object_ids = attribute_df.reset_index()[id_type].values
    except KeyError:
        id_type = "id"
        object_ids = attribute_df.reset_index()[id_type].values

    # Will also need to load in core attributes
    if hasattr(figure, "member_core_filepaths"):
        core_filepath = figure.member_core_filepaths[object_name]
    elif hasattr(figure, "core_filepath"):
        core_filepath = figure.core_filepath
    core_df = read_attribute_csv(core_filepath, times=[time])
    # Join the core attributes with the attribute df
    # Prepend column names with handler name if necessary
    for col in core_df.columns:
        if col in attribute_df.columns:
            core_df.rename(columns={col: f"{handler.name}_{col}"}, inplace=True)
    attribute_df = attribute_df.join(core_df)

    leg_method = handler.legend_method
    if leg_method is not None and handler.name not in legend_artists.keys():
        # Create the legend artist
        func = leg_method.function
        keyword_arguments = leg_method.keyword_arguments
        legend_artist = func(**keyword_arguments)
        legend_artists[handler.label] = legend_artist

    for obj_id in object_ids:
        # Add the attribute for the given object to the figure
        object_df = attribute_df.xs(obj_id, level=id_type, drop_level=False)
        obj_quality_df = quality_df.xs(obj_id, level=id_type, drop_level=False)
        attributes = handler.attributes
        func = handler.method.function
        kwargs = handler.method.keyword_arguments
        artist = func(ax, attributes, object_df, obj_quality_df, **kwargs)
        attribute_artists[object_name][handler.name][obj_id] = artist


def add_attributes(time, figure):
    """Add all the requisite attributes to the figure."""
    legend_artists = {}
    attribute_artists = {}
    for i, obj in enumerate(figure.attribute_handlers.keys()):
        attribute_artists[obj] = {}
        for handler in figure.attribute_handlers[obj]:
            ax = figure.subplot_axes[i]
            args = [ax, obj, handler, attribute_artists, legend_artists, time]
            args += [figure]
            add_attribute(*args)
    figure.legend_artists = legend_artists
    figure.attribute_artists = attribute_artists


def create_legend(figure, grid_options, figure_options):
    """Create a legend for the figure."""
    legend_options = {"ncol": 3, "loc": "lower center"}

    scale = visualize.utils.get_extent(grid_options)[1]
    handles, labels = [], []
    handle, handler = horizontal.mask_legend_artist()
    handles += [handle]
    labels += ["Object Masks"]
    handle = horizontal.domain_boundary_legend_artist()
    handles += [handle]
    labels += ["Domain Boundary"]
    handles += list(figure.legend_artists.values())
    labels += list(figure.legend_artists.keys())
    legend_color = visualize.figure_colors[figure_options.style]["legend"]
    args = [handles, labels]
    style = figure_options.style
    leg_ax = figure.legend_axes[0]

    with plt.style.context(visualize.styles[style]), visualize.set_style(style):
        if scale == 1:
            legend = leg_ax.legend(*args, **mcs_legend_options, handler_map=handler)
        elif scale == 2:
            legend_options["loc"] = "lower left"
            legend_options["bbox_to_anchor"] = (-0.0, -0.425)
            legend = leg_ax.legend(*args, **mcs_legend_options, handler_map=handler)
    legend.get_frame().set_alpha(None)
    legend.get_frame().set_facecolor(legend_color)


class BaseFigure(BaseHandler):
    """Base class for a figure visualizing a field, objects, and object attributes."""

    object_name: str = Field(..., description="The name of the object.")
    time: np.datetime64 = Field(..., description="The time of the figure.")
    _desc = "A dictionary with a list of attribute handlers for the given object."
    attribute_handlers: dict[str, list[AttributeHandler]] = Field([], description=_desc)
    _desc = "The artist used to visualize the domain boundary."
    boundary_artists: list[Any] = Field([], description=_desc)
    _desc = "The artists used to visualize the field, e.g. reflectivity."
    field_artists: list[Any] = Field([], description=_desc)
    _desc = "The artists used to visualize object masks."
    mask_artists: list[Any] = Field([], description=_desc)
    _desc = "The artists used to visualize object attributes."
    attribute_artists: Dict[str, Any] = Field({}, description=_desc)
    _desc = "The proxy artists used for creating legends."
    legend_artists: Dict[str, Any] = Field({}, description=_desc)
    _desc = "Layout class instance for the figure."
    layout: utils.BaseLayout = Field(None, description=_desc)
    _desc = "Options for the figure."
    options: FigureOptions | None = Field(None, description=_desc)
    figure: Any = Field(None, description="The Matplotlib figure object.")
    _desc = "The Matplotlib axes containing subplots."
    subplot_axes: list[Any] = Field([], description=_desc)
    _desc = "The Matplotlib axes containing legends."
    legend_axes: list[Any] = Field([], description=_desc)
    _desc = "The Matplotlib axes containing colorbars."
    colorbar_axes: list[Any] = Field([], description=_desc)
    _desc = "Filepath to csv of core attributes."
    core_filepath: str | None = Field(None, description=_desc)


class GroupedObjectFigure(BaseFigure):
    """Class for visualizing grouped objects."""

    member_objects: list[str] = Field([], description="Member object names.")
    _desc = "Filepaths to core attributes for each member object."
    member_core_filepaths: dict[str, str] = Field({}, description=_desc)

    @model_validator(mode="after")
    def _check_number_subplots(cls, values):
        """
        Check the number of subplots matches the number of member objects and number
        of member_core_filepaths keys.
        """
        lengths = [len(values.member_objects), len(values.subplot_axes)]
        lengths += [len(values.member_core_filepaths)]
        if len(set(lengths)) != 1:
            message = "Number of member objects, subplot axes, and member core "
            message += "filepaths must agree."
            raise ValueError(message)


def velocity_horizontal(
    ax, attributes, object_df, quality_df, color="tab:red", dt=3600, reverse=False
):
    """
    Add velocity attributes. Assumes the attribtes dataframe has already
    been subset to the desired time and object, so is effectively a dictionary.
    """
    latitude = object_df["latitude"].values[0]
    longitude = object_df["longitude"].values[0]
    u, v = object_df[attributes[0]].values[0], object_df[attributes[1]].values[0]
    args = [ax, latitude, longitude, u, v, color]
    kwargs = {"quality": quality_df.values, "dt": dt, "reverse": reverse}
    return horizontal.cartesian_velocity(*args, **kwargs)


def text_horizontal(
    ax,
    attributes,
    object_df,
    quality_df,
    formatter=None,
    labelled_attribute="universal_id",
):
    """Add object ID attributes."""
    latitude = object_df["latitude"].values[0]
    longitude = object_df["longitude"].values[0]
    label = object_df.reset_index()[labelled_attribute].values[0]
    if formatter is not None:
        label = formatter(label)
    args = [ax, label, longitude, latitude]
    if quality_df.values[0]:
        return horizontal.embossed_text(*args)
    else:
        return None


def orientation_horizontal(ax, attributes, object_df, quality_df=None):
    """Add orientation attributes to axes."""
    latitude = object_df["orientation_latitude"].values[0]
    longitude = object_df["orientation_longitude"].values[0]
    if "major" in attributes:
        length = object_df["major"].values[0]
    elif "minor" in attributes:
        length = object_df["minor"].values[0]
    else:
        raise ValueError("No major or minor attribute in object_df.")
    if "orientation" in attributes:
        orientation = object_df["orientation"].values[0]
    else:
        raise ValueError("No orientation attribute in object_df.")
    args = [ax, latitude, longitude, length, orientation, quality_df.values]
    return horizontal.ellipse_axis(*args)


def displacement_horizontal(
    ax, attributes, object_df, quality_df, color="tab:blue", reverse=False
):
    """Add displacement attributes."""
    # Convert displacements from km to metres
    dx = object_df[attributes[0]].values[0] * 1e3
    dy = object_df[attributes[1]].values[0] * 1e3
    if reverse:
        dx, dy = -dx, -dy
    latitude = object_df["latitude"].values[0]
    longitude = object_df["longitude"].values[0]
    args = [ax, latitude, longitude, dx, dy, color, quality_df.values]
    return horizontal.cartesian_displacement(*args, arrow=True, reverse=reverse)


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


def get_color_angle_df(object_name, output_parent, filepath=None):
    """
    Get a dictionary containing color angles, i.e. indices, for displaying masks.
    The color angle is calculated to reflect object splits/merges.
    """
    if filepath is None:
        filepath = output_parent / f"attributes/{object_name}/core.csv"
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
