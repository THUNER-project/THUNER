"""Functions for visualizing object attributes and classifications."""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import thor.visualize.horizontal as horizontal
from thor.visualize.visualize import map_colors, styles, animate_object
from thor.log import setup_logger
from thor.attribute.utils import read_attribute_csv
from thor.analyze.utils import read_options
import thor.data.dispatch as dispatch
import thor.detect.detect as detect
from thor.utils import format_time

logger = setup_logger(__name__)
proj = ccrs.PlateCarree()


mcs_legend_options = {"loc": "lower center", "bbox_to_anchor": (1.15, -0.3), "ncol": 4}
mcs_legend_options.update({"fancybox": True, "shadow": True})


def mcs_series(
    output_directory,
    start_time,
    end_time,
    figure_options,
    convective_label="cell",
    dataset_name=None,
    animate=True,
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
            dataset_name = track_options[0][convective_label]["dataset"]
        except KeyError:
            message = "Could not infer dataset used for detection. Provide manually."
            raise KeyError(message)

    filepath = masks_filepath = output_directory / "masks/mcs.nc"
    masks = xr.open_dataset(masks_filepath)
    times = masks.time.values
    times = times[(times >= start_time) & (times <= end_time)]
    record_filepath = output_directory / f"records/filepaths/{dataset_name}.csv"
    filepaths = read_attribute_csv(record_filepath)
    convert = dispatch.convert_dataset_dispatcher.get(dataset_name)
    if convert is None:
        message = f"Dataset {dataset_name} not found in dispatch."
        raise KeyError(message)
    for time in times:
        filepath = filepaths[dataset_name].loc[time]
        args = [time, filepath, options["data"][dataset_name], options["grid"]]
        ds, boundary_coords = convert(*args)
        get_grid = dispatch.grid_from_dataset_dispatcher.get(dataset_name)
        if get_grid is None:
            message = f"Dataset {dataset_name} not found in grid from dataset "
            message += "dispatcher."
            raise KeyError(message)
        grid = get_grid(ds, "reflectivity", time)
        processed_grid = detect.rebuild_processed_grid(grid, track_options, "mcs", 1)
        mask = masks.sel(time=time)
        args = [output_directory, processed_grid, mask, boundary_coords]
        args += [figure_options, options["grid"]]
        figure_name = figure_options["name"]
        with plt.style.context(styles[figure_options["style"]]):
            fig, ax = mcs_horizontal(*args)
            filename = f"{format_time(time)}.png"
            filepath = output_directory / f"visualize/{figure_name}/{filename}"
            filepath.parent.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Saving {figure_name} figure for {time}.")
            fig.savefig(filepath, bbox_inches="tight")
    if animate:
        save_directory = output_directory / f"visualize"
        figure_directory = output_directory / f"visualize/{figure_name}"
        args = [figure_name, "mcs", output_directory, save_directory]
        args += [figure_directory, figure_name]
        animate_object(*args)
    # Switch back to original backend
    matplotlib.use(original_backend)


def mcs_horizontal(
    output_directory,
    grid,
    mask,
    boundary_coordinates,
    figure_options,
    grid_options,
    convective_label="cell",
    anvil_label="anvil",
):
    """Create a horizontal cross section plot."""
    member_objects = [convective_label, anvil_label]

    args = [grid, mask, grid_options, figure_options, member_objects]
    args += [boundary_coordinates]
    fig, axes = horizontal.grouped_mask(*args)

    time = grid.time.values

    try:
        filepath = output_directory / "attributes/mcs/group.csv"
        group = read_attribute_csv(filepath, times=[time]).loc[time]
        filepath = output_directory / "analysis/velocities.csv"
        velocities = read_attribute_csv(filepath, times=[time]).loc[time]
        filepath = output_directory / "analysis/classification.csv"
        classification = read_attribute_csv(filepath, times=[time]).loc[time]
        filepath = output_directory / "analysis/quality.csv"
        quality = read_attribute_csv(filepath, times=[time]).loc[time]
        attributes = pd.concat([group, velocities, classification, quality], axis=1)
        objs = group.reset_index()["universal_id"].values
    except KeyError:
        # If no attributes, return early
        objs = []

    # Display velocity attributes
    for obj_id in objs:
        obj_attr = attributes.loc[obj_id]
        args = [axes, figure_options, obj_attr]
        velocity_attributes_horizontal(*args)
        displacement_attributes_horizontal(*args)

    # Get legend proxy artists
    legend_handles = []
    attribute_names = figure_options["attributes"]
    for name in attribute_names:
        color = colors_dispatcher[name]
        label = label_dispatcher[name]
        handle = horizontal.displacement_legend_artist(color, label)
        legend_handles.append(handle)

    legend_color = map_colors[figure_options["style"]]["legend_color"]
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
    "velocity": system_contained + ["velocity"],
    "shear": ["shear"],
    "relative_velocity": system_contained + ["relative_velocity"],
    "offset": system_contained + ["offset"],
}


def velocity_attributes_horizontal(axes, figure_options, object_attributes):
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
        axes[0] = horizontal.cartesian_velocity(*args, quality=quality)

    return legend_handles


def get_quality(quality_names, object_attributes):
    if quality_names is None:
        quality = True
    else:
        qualities = object_attributes[quality_names]
        quality = qualities.all()
    return quality


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
            axes[0] = horizontal.cartesian_displacement(*args, **args_dict)
            args[0] = axes[1]
            axes[1] = horizontal.cartesian_displacement(*args, **args_dict)
        legend_artist = horizontal.displacement_legend_artist(color, label)
        legend_handles.append(legend_artist)

    return legend_handles
