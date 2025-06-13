"""Functions for visualizing object attributes and classifications."""

import gc
from pathlib import Path
from pydantic import Field, BaseModel, model_validator, ConfigDict
from typing import Any, Callable, Dict, Literal
from time import sleep
import multiprocessing as mp
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import thuner.visualize.horizontal as horizontal
from thuner.utils import initialize_process, check_results, DataObject
from thuner.attribute.utils import read_attribute_csv
from thuner.analyze.utils import read_options
import thuner.detect.detect as detect
from thuner.utils import format_time, new_angle, circular_mean
import thuner.visualize.utils as utils
import thuner.visualize.visualize as visualize
from thuner.option.visualize import FigureOptions
from thuner.log import setup_logger, logging_listener


__all__ = ["mcs_series", "mcs_horizontal"]

logger = setup_logger(__name__)
proj = ccrs.PlateCarree()


mcs_legend_options = {"ncol": 3, "loc": "lower center"}


def get_altitude_labels(track_options, mcs_name="mcs", mcs_level=1):
    """Get altitude labels for convective and stratiform objects."""
    mcs_options = track_options.levels[mcs_level].object_by_name(mcs_name)
    convective = mcs_options.grouping.member_objects[0]
    convective_level = mcs_options.grouping.member_levels[0]
    stratiform = mcs_options.grouping.member_objects[-1]
    stratiform_level = mcs_options.grouping.member_levels[-1]
    convective_options = track_options.levels[convective_level]
    convective_options = convective_options.object_by_name(convective)
    stratiform_options = track_options.levels[stratiform_level]
    stratiform_options = stratiform_options.object_by_name(stratiform)
    convective_altitudes = np.array(convective_options.detection.altitudes)
    stratiform_altitudes = np.array(stratiform_options.detection.altitudes)
    convective_altitudes = np.round(convective_altitudes / 1e3, 1)
    stratiform_altitudes = np.round(stratiform_altitudes / 1e3, 1)
    convective_label = f"{convective_altitudes[0]:g} to {convective_altitudes[1]:g} km"
    stratiform_label = f"{stratiform_altitudes[0]:g} to {stratiform_altitudes[1]:g} km"
    return convective_label + " Altitude", stratiform_label + " Altitude"


def series(
    output_directory: str | Path,
    start_time,
    end_time,
    figure_options,
    dataset_name,
    animate=True,
    parallel_figure=False,
    dt=3600,
    by_date=True,
    num_processes=4,
):
    """Visualize mcs attributes at specified times."""
    plt.close("all")
    original_backend = matplotlib.get_backend()
    matplotlib.use("Agg")

    start_time = np.datetime64(start_time)
    end_time = np.datetime64(end_time)
    options = read_options(output_directory)
    track_options = options["track"]

    masks_filepath = output_directory / "masks/mcs.zarr"
    masks = xr.open_dataset(masks_filepath, engine="zarr")
    times = masks.time.values
    times = times[(times >= start_time) & (times <= end_time)]

    # Get colors
    color_angle_df = get_color_angle_df(output_directory)

    record_filepath = output_directory / f"records/filepaths/{dataset_name}.csv"
    filepaths = read_attribute_csv(record_filepath, columns=[dataset_name])
    # Start with first time
    time = times[0]
    args = [time, filepaths, masks, output_directory, figure_options]
    args += [options, dataset_name, dt, color_angle_df]
    create_figure(*args)
    if len(times) == 1:
        # Switch back to original backend
        plt.close("all")
        matplotlib.use(original_backend)
        return
    if parallel_figure:
        kwargs = {"initializer": initialize_process, "processes": num_processes}
        with logging_listener(), mp.get_context("spawn").Pool(**kwargs) as pool:
            results = []
            for time in times[1:]:
                sleep(2)
                args = [time, filepaths, masks, output_directory, figure_options]
                args += [options, dataset_name, dt, color_angle_df]
                args = tuple(args)
                results.append(pool.apply_async(create_figure, args))
            pool.close()
            pool.join()
            check_results(results)
    else:
        for time in times[1:]:
            args = [time, filepaths, masks, output_directory, figure_options]
            args += [options, dataset_name, dt, color_angle_df]
            create_figure(*args)
    if animate:
        figure_name = figure_options.name
        save_directory = output_directory / f"visualize"
        figure_directory = output_directory / f"visualize/{figure_name}"
        args = [figure_name, "mcs", output_directory, save_directory]
        args += [figure_directory, figure_name]
        visualize.animate_object(*args, by_date=by_date)
    # Switch back to original backend
    plt.close("all")
    matplotlib.use(original_backend)


def get_mask_grid_boundary(
    object_name, time, color_angle_df, filepaths, masks, dataset_name, options
):
    """Get the mask and grid for a given time."""
    logger.info(f"Visualizing attribtues at time {time}.")

    filepath = filepaths[dataset_name].loc[time]
    dataset_options = options["data"].dataset_by_name(dataset_name)

    args = [time, filepath, options["track"], options["grid"]]
    ds, boundary_coords, simple_boundary_coords = dataset_options.convert_dataset(*args)
    del boundary_coords
    logger.debug(f"Getting grid from dataset at time {time}.")
    grid = dataset_options.grid_from_dataset(ds, "reflectivity", time)
    del ds
    logger.debug(f"Rebuilding processed grid for time {time}.")
    args = [grid, options["track"], object_name, 1]
    processed_grid = detect.rebuild_processed_grid(*args)
    del grid
    mask = masks.sel(time=time).load()
    return mask, processed_grid, simple_boundary_coords


def get_object_colors(time, color_angle_df):
    """Get the object colors for a given time."""
    keys = color_angle_df.loc[color_angle_df["time"] == time]["universal_id"].values
    values = color_angle_df.loc[color_angle_df["time"] == time]["color_angle"].values
    values = [visualize.mask_colormap(v / (2 * np.pi)) for v in values]
    return dict(zip(keys, values))


def grouped_horizontal(
    object_name,
    time,
    filepaths,
    masks,
    output_directory,
    figure_options,
    options,
    dataset_name,
    dt,
    color_angle_df,
):
    """Create a horizontal cross section plot."""
    logger.info(f"Visualizing attribtues at time {time}.")
    args = [object_name, time, color_angle_df, filepaths, masks, dataset_name, options]
    mask, grid, boundary_coords = get_mask_grid_boundary(*args)
    object_colors = get_object_colors(time, color_angle_df)

    grid_options = options["grid"]
    track_options = options["track"]
    obj_name = figure_options.object_name

    grid = processed_grid
    time = grid.time.values
    style = figure_options.style

    args = [grid, mask, grid_options, figure_options, member_objects]
    args += [boundary_coords]
    kwargs = {"object_colors": object_colors}

    with plt.style.context(
        visualize.visualize.styles[style]
    ), visualize.visualize.set_style(style):
        figure_features = visualize.horizontal.grouped_mask(*args, **kwargs)
        fig, subplot_axes, colorbar_axes, legend_axes = figure_features

    kwargs = {"object_name": "mcs", "time": time, "grid": grid, "mask": mask}
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
    grouped_figure = visualize.attribute.GroupedObjectFigure(**kwargs)

    # Remove mask and processed_grid from memory after generating the figure
    del mask, processed_grid
    filename = f"{format_time(time)}.png"
    filepath = output_directory / f"visualize/{figure_options.figure_name}/{filename}"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving {figure_options.figure_name} figure for {time}.")
    fig.savefig(filepath, bbox_inches="tight")
    utils.reduce_color_depth(filepath)
    plt.clf()
    plt.close()
    gc.collect()


def create_figure(
    object_name,
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
    """Create an attributes figure for a given time."""
    logger.info(f"Visualizing attribtues at time {time}.")

    # Get object colors
    keys = color_angle_df.loc[color_angle_df["time"] == time]["universal_id"].values
    values = color_angle_df.loc[color_angle_df["time"] == time]["color_angle"].values
    values = [visualize.mask_colormap(v / (2 * np.pi)) for v in values]
    object_colors = dict(zip(keys, values))

    filepath = filepaths[dataset_name].loc[time]
    dataset_options = options["data"].dataset_by_name(dataset_name)

    args = [time, filepath, track_options, options["grid"]]
    ds, boundary_coords, simple_boundary_coords = dataset_options.convert_dataset(*args)
    del boundary_coords
    logger.debug(f"Getting grid from dataset at time {time}.")
    grid = dataset_options.grid_from_dataset(ds, "reflectivity", time)
    del ds
    logger.debug(f"Rebuilding processed grid for time {time}.")
    processed_grid = detect.rebuild_processed_grid(grid, track_options, object_name, 1)
    del grid
    mask = masks.sel(time=time).load()
    args = [output_directory, processed_grid, mask, simple_boundary_coords]
    args += [figure_options, options["grid"]]
    figure_name, style = figure_options.name, figure_options.style

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


class BaseHandler(BaseModel):
    """Base class for figure handlers defined in this module."""

    # Allow arbitrary types in the input record classes.
    model_config = ConfigDict(arbitrary_types_allowed=True)


class AttributeVisualizeMethod(BaseHandler):
    """
    Class for handling the visualization of a group of attributes. Includes a reference
    to a function, and requisite keyword arguments.
    """

    _desc = "The function used to visualize the attributes."
    function: Callable | str | None = Field(None, description=_desc)
    _desc = "Keyword arguments for the visualization function."
    keyword_arguments: dict = Field({}, description=_desc)


class LegendArtistMethod(BaseHandler):
    """
    Class for handling the visualization of a legend artist. Includes a reference
    to a function, and requisite keyword arguments.
    """

    _desc = "The function used to create the legend artist."
    function: Callable | str | None = Field(None, description=_desc)
    _desc = "Keyword arguments for the legend artist function."
    keyword_arguments: dict = Field({}, description=_desc)


class AttributeHandler(BaseHandler):
    """
    Class for handling the visualization of attributes, e.g. orientation, or groups of
    attributes visualized together, e.g. u, v.
    """

    _desc = "The name of the attribute or attributes being handled, e.g. velocity."
    name: str = Field(..., description=_desc)
    _desc = "The axes in which the attributes are to be visualized."
    axes: list[Any] = Field([], description=_desc)
    _desc = "The label to appear in legends etc for this attribute."
    label: str = Field(..., description=_desc)
    _desc = "The names of the attributes to be visualized."
    attributes: list[str] = Field(..., description=_desc)
    _desc = "The filepath to the attribute file, i.e. an attribute type csv file."
    filepath: str = Field(..., description=_desc)
    _desc = "The method used to visualize the attributes."
    method: AttributeVisualizeMethod = Field(..., description=_desc)
    _desc = "The method used to create the legend artist for this attribute."
    legend_method: LegendArtistMethod | None = Field(None, description=_desc)
    _desc = "The filepath of the quality control file."
    quality_filepath: str | None = Field(None, description=_desc)
    _desc = "The quality control variables for this attribute."
    quality_variables: list[str] = Field([], description=_desc)
    _desc = "The logic used to determine if an object is of sufficient quality."
    quality_method: Literal["any", "all"] = Field("all", description=_desc)


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


# def mcs_horizontal(
#     output_directory,
#     grid,
#     mask,
#     boundary_coordinates,
#     figure_options,
#     grid_options,
#     convective_label="convective",
#     anvil_label="anvil",
#     dt=3600,
#     object_colors=None,
# ):
#     """Create a horizontal cross section plot."""
#     member_objects = [convective_label, anvil_label]
#     options = read_options(output_directory)
#     track_options = options["track"]

#     time = grid.time.values
#     logger.debug(f"Creating grouped mask figure at time {time}.")
#     try:
#         filepath = output_directory / "analysis/quality.csv"
#         kwargs = {"times": [time], "columns": ["duration", "parents"]}
#         object_quality = read_attribute_csv(filepath, **kwargs).loc[time]
#         object_quality = object_quality.any(axis=1).to_dict()
#     except (FileNotFoundError, KeyError):
#         object_quality = None

#     args = [grid, mask, grid_options, figure_options, member_objects]
#     args += [boundary_coordinates]
#     kwargs = {"object_colors": object_colors, "mask_quality": object_quality}
#     fig, axes, colorbar_axes, legend_axes = horizontal.grouped_mask(*args, **kwargs)

#     try:
#         filepath = output_directory / "attributes/mcs/core.csv"
#         columns = ["latitude", "longitude"]
#         core = read_attribute_csv(filepath, times=[time], columns=columns).loc[time]
#         filepath = output_directory / "attributes/mcs/group.csv"
#         group = read_attribute_csv(filepath, times=[time]).loc[time]
#         filepath = output_directory / "analysis/velocities.csv"
#         velocities = read_attribute_csv(filepath, times=[time]).loc[time]
#         # filepath = output_directory / "analysis/classification.csv"
#         # classification = read_attribute_csv(filepath, times=[time]).loc[time]
#         filepath = output_directory / f"attributes/mcs/{convective_label}/ellipse.csv"
#         ellipse = read_attribute_csv(filepath, times=[time]).loc[time]
#         new_names = {"latitude": "ellipse_latitude", "longitude": "ellipse_longitude"}
#         ellipse = ellipse.rename(columns=new_names)
#         filepath = output_directory / "analysis/quality.csv"
#         quality = read_attribute_csv(filepath, times=[time]).loc[time]
#         attributes = pd.concat([core, ellipse, group, velocities, quality], axis=1)
#         objs = group.reset_index()["universal_id"].values
#     except KeyError:
#         # If no attributes, set objs=[]
#         objs = []

#     for obj_id in objs:
#         obj_attr = attributes.loc[obj_id]
#         args = [axes, figure_options, obj_attr]
#         velocity_attributes_horizontal(*args, dt=dt)
#         displacement_attributes_horizontal(*args)
#         ellipse_attributes(*args)
#         if object_quality[obj_id]:
#             text_attributes_horizontal(*args, object_quality=object_quality)

#     style = figure_options.style
#     scale = utils.get_extent(grid_options)[1]

#     key_color = visualize.figure_colors[style]["key"]
#     horizontal.vector_key(axes[0], color=key_color, dt=dt, scale=scale)
#     kwargs = {"mcs_name": "mcs", "mcs_level": 1}
#     convective_label, stratiform_label = get_altitude_labels(track_options, **kwargs)

#     axes[0].set_title(convective_label)
#     axes[1].set_title(stratiform_label)

#     # Get legend proxy artists
#     handles = []
#     labels = []
#     handle = horizontal.domain_boundary_legend_artist()
#     handles += [handle]
#     labels += ["Domain Boundary"]
#     handle = horizontal.ellipse_legend_artist("Major Axis", figure_options.style)
#     handles += [handle]
#     labels += ["Major Axis"]
#     attribute_names = figure_options.attributes
#     for name in [attr for attr in attribute_names if attr != "id"]:
#         color = colors_dispatcher[name]
#         label = label_dispatcher[name]
#         handle = horizontal.displacement_legend_artist(color, label)
#         handles.append(handle)
#         labels.append(label)

#     handle, handler = horizontal.mask_legend_artist()
#     handles += [handle]
#     labels += ["Object Masks"]
#     legend_color = visualize.figure_colors[figure_options.style]["legend"]
#     handles, labels = handles[::-1], labels[::-1]

#     args = [handles, labels]
#     leg_ax = legend_axes[0]
#     if scale == 1:
#         legend = leg_ax.legend(*args, **mcs_legend_options, handler_map=handler)
#     elif scale == 2:
#         mcs_legend_options["loc"] = "lower left"
#         mcs_legend_options["bbox_to_anchor"] = (-0.0, -0.425)
#         legend = leg_ax.legend(*args, **mcs_legend_options, handler_map=handler)
#     legend.get_frame().set_alpha(None)
#     legend.get_frame().set_facecolor(legend_color)

#     return fig, axes


def velocity_horizontal(
    ax, attributes, object_df, quality_df, color="tab:red", dt=3600
):
    """
    Add velocity attributes. Assumes the attribtes dataframe has already
    been subset to the desired time and object, so is effectively a dictionary.
    """
    latitude = object_df["latitude"].values[0]
    longitude = object_df["longitude"].values[0]
    u, v = object_df[attributes[0]].values[0], object_df[attributes[1]].values[0]
    args = [ax, latitude, longitude, u, v, color]
    return horizontal.cartesian_velocity(*args, quality=quality_df.values, dt=dt)


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
    if quality_df.values:
        return horizontal.embossed_text(*args)
    else:
        return None


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


def get_color_angle_df(output_parent, filepath=None):
    """
    Get a dictionary containing color angles, i.e. indices, for displaying masks.
    The color angle is calculated to reflect object splits/merges.
    """
    if filepath is None:
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
