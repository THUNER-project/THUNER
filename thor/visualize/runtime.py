"""
Plotting functions to be called during algorithm runtime for debugging 
and visualization purposes.
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import cartopy.crs as ccrs
import thor.visualize.horizontal as horizontal
from thor.visualize.visualize import styles
from thor.utils import format_time
from thor.match.utils import get_grids, get_masks
from thor.log import setup_logger
from thor.visualize.utils import make_subplot_labels


logger = setup_logger(__name__)


def get_extent(grid):
    """Get the cartopy extent."""
    return (
        grid.longitude.values.min(),
        grid.longitude.values.max(),
        grid.latitude.values.min(),
        grid.latitude.values.max(),
    )


def detected_mask_template(grid, input_record, figure_options, extent):
    """Create a template figure for masks."""
    fig = plt.figure(figsize=(6, 3.5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    horizontal.add_cartographic_features(
        ax, style=figure_options["style"], scale="10m", extent=extent
    )
    if "radar" in grid.attrs["instrument"]:
        radar_longitude = float(grid.attrs["origin_longitude"])
        radar_latitude = float(grid.attrs["origin_latitude"])
        horizontal.add_radar_features(
            ax, radar_longitude, radar_latitude, extent, input_record
        )
    return fig, ax


def detected_mask(
    input_record, tracks, level_index, obj, track_options, grid_options, figure_options
):
    """Plot masks for a detected object."""

    object_tracks = tracks[level_index][obj]
    object_options = track_options[level_index][obj]
    grid = object_tracks["current_grid"]
    extent = get_extent(grid)

    if "figure_template" not in figure_options.keys():
        fig, ax = detected_mask_template(grid, input_record, figure_options, extent)
        figure_options["figure_template"] = fig

    fig = copy.deepcopy(figure_options["figure_template"])
    ax = fig.axes[0]

    pcm = horizontal.grid(grid, ax, add_colorbar=False)

    if object_options["tracking"]["method"] is None:
        mask = object_tracks["current_mask"]
    else:
        mask = object_tracks["current_matched_mask"]
    if mask is not None:
        horizontal.mask(mask, ax)

    cbar_label = grid.name.title() + f" [{grid.units}]"
    fig.colorbar(pcm, label=cbar_label)
    ax.set_title(f"{grid.time.values.astype('datetime64[s]')} UTC")

    return fig, ax


def grouped_mask_template(
    grid, input_record, figure_options, extent, figsize, member_objects
):
    """Create a template figure for grouped masks."""
    fig = plt.figure(figsize=figsize)
    nrows = 1
    ncols = len(member_objects) + 1
    width_ratios = [1] * len(member_objects) + [0.05]
    gs = gridspec.GridSpec(nrows, ncols, width_ratios=width_ratios)
    axes = []
    for i in range(len(member_objects)):
        ax = fig.add_subplot(gs[0, i], projection=ccrs.PlateCarree())
        axes.append(ax)
        ax = horizontal.add_cartographic_features(
            ax,
            extent=extent,
            style=figure_options["style"],
            scale="10m",
            left_labels=(i == 0),
        )[0]
        grid_i = grid[f"{member_objects[i]}_grid"]
        if "radar" in grid_i.attrs["instrument"]:
            radar_longitude = float(grid_i.attrs["origin_longitude"])
            radar_latitude = float(grid_i.attrs["origin_latitude"])
            horizontal.add_radar_features(
                ax, radar_longitude, radar_latitude, extent, input_record
            )
        ax.set_title(member_objects[i].replace("_", " ").title())
    cbar_ax = fig.add_subplot(gs[0, -1])
    make_subplot_labels(axes, x_shift=-0.12, y_shift=0.06)
    return fig, axes, cbar_ax


def grouped_mask(
    input_record,
    tracks,
    level_index,
    obj,
    track_options,
    grid_options,
    figure_options,
):
    """Plot masks for a grouped object."""
    object_options = track_options[level_index][obj]
    object_tracks = tracks[level_index][obj]
    if object_options["tracking"]["method"] is None:
        mask = object_tracks["current_mask"]
    else:
        mask = object_tracks["current_matched_mask"]

    try:
        member_objects = figure_options["member_objects"]
    except KeyError:
        member_objects = object_options["grouping"]["member_objects"]

    grid = object_tracks["current_grid"]
    extent = get_extent(grid)

    try:
        figsize = figure_options["figsize"]
    except KeyError:
        figsize = (len(member_objects) * 4, 3.5)

    if "figure_template" not in figure_options.keys():
        fig, axes, cbar_ax = grouped_mask_template(
            grid, input_record, figure_options, extent, figsize, member_objects
        )
        figure_options["figure_template"] = fig

    fig = copy.deepcopy(figure_options["figure_template"])
    axes = fig.axes[:-1]
    cbar_ax = fig.axes[-1]

    for i in range(len(member_objects)):
        ax = axes[i]
        grid_i = grid[f"{member_objects[i]}_grid"]
        mask_i = mask[f"{member_objects[i]}_mask"]
        pcm = horizontal.grid(grid_i, ax, add_colorbar=False)
        if mask_i is not None:
            horizontal.mask(mask_i, ax)

    cbar_label = grid_i.attrs["long_name"].title() + f" [{grid_i.attrs['units']}]"
    fig.colorbar(pcm, cax=cbar_ax, label=cbar_label)
    fig.suptitle(f"{grid.time.values.astype('datetime64[s]')} UTC", y=1.05)

    return fig, ax


def get_box_coords(box, latitudes, longitudes):
    """Get the coordinates of a box."""
    box_latitudes = [
        latitudes[box["row_min"]],
        latitudes[box["row_min"]],
        latitudes[box["row_max"]],
        latitudes[box["row_max"]],
        latitudes[box["row_min"]],
    ]
    box_longitudes = [
        longitudes[box["col_min"]],
        longitudes[box["col_max"]],
        longitudes[box["col_max"]],
        longitudes[box["col_min"]],
        longitudes[box["col_min"]],
    ]
    return box_latitudes, box_longitudes


def get_box_center_coords(box, latitudes, longitudes):
    """Get the coordinates of the center of a box."""
    center_row = int(np.ceil((box["row_min"] + box["row_max"]) / 2))
    center_col = int(np.ceil((box["col_min"] + box["col_max"]) / 2))
    center_lat = latitudes[center_row]
    center_lon = longitudes[center_col]
    return center_lat, center_lon, center_row, center_col


arrow_options = {"arrowstyle": "->", "linewidth": 1.5, "mutation_scale": 7, "zorder": 3}
arrow_origin_options = {"marker": "o", "zorder": 3, "markersize": 2}


def plot_flow(ax, start_lat, start_lon, flow, grid_options, color="w"):

    # Exaggerate the flow vector for visualization
    geographic_flow = np.array(flow) * np.array(grid_options["geographic_spacing"])
    geographic_flow = np.array(geographic_flow) * 5
    start_coords = [start_lon, start_lat]
    end_coords = np.array(start_coords) + geographic_flow[::-1]
    ax.plot(start_lon, start_lat, color=color, **arrow_origin_options)
    arrow = FancyArrowPatch(start_coords, end_coords, color=color, **arrow_options)
    ax.add_patch(arrow)


def plot_displacement(ax, previous_centers, current_centers, color="w"):

    # Exaggerate the displacement for visualization
    geographic_displacement = (current_centers - previous_centers) * 5
    start_lat, start_lon = previous_centers
    start_coords = [start_lon, start_lat]
    end_coords = np.array(start_coords) + geographic_displacement[::-1]
    ax.plot(start_lon, start_lat, alpha=0.6, color=color, **arrow_origin_options)
    arrow = FancyArrowPatch(
        start_coords, end_coords, alpha=0.6, color=color, **arrow_options
    )
    ax.add_patch(arrow)


def plot_box(ax, box, grid, color, linestyle="--"):
    latitudes = grid.latitude.values
    longitudes = grid.longitude.values
    box_latitudes, box_longitudes = get_box_coords(box, latitudes, longitudes)
    ax.plot(
        box_longitudes, box_latitudes, color=color, linewidth=1.5, linestyle=linestyle
    )


def match_template(reference_grid, input_record, figure_options, extent):
    """Create a template for match figures."""
    fig = plt.figure(figsize=(7, 3))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05])
    axes = []
    subtitles = ["Previous Objects", "Current Objects"]
    for i in range(2):
        ax = fig.add_subplot(gs[0, i], projection=ccrs.PlateCarree())
        axes.append(ax)
        ax = horizontal.add_cartographic_features(
            ax,
            extent=extent,
            style=figure_options["style"],
            scale="10m",
            left_labels=(i == 0),
        )[0]
        if "radar" in reference_grid.attrs["instrument"]:
            radar_longitude = float(reference_grid.attrs["origin_longitude"])
            radar_latitude = float(reference_grid.attrs["origin_latitude"])
            horizontal.add_radar_features(
                ax, radar_longitude, radar_latitude, extent, input_record
            )
        ax.set_title(subtitles[i])
    cbar_ax = fig.add_subplot(gs[0, -1])
    make_subplot_labels(axes, x_shift=-0.12, y_shift=0.06)
    return fig, axes, cbar_ax


def match_features(grid, object_record, axes, colors, grid_options):
    latitudes = grid.latitude.values
    longitudes = grid.longitude.values
    if "radar" in grid.attrs["instrument"]:
        center_lon = float(grid.attrs["origin_longitude"])
        center_lat = float(grid.attrs["origin_latitude"])
    else:
        center_lon = longitudes[len(longitudes) // 2]
        center_lat = latitudes[len(longitudes) // 2]
    if object_record["global_flow"] is not None:
        global_flow = object_record["global_flow"]
        plot_flow(
            axes[0], center_lat, center_lon, global_flow, grid_options, color="tab:red"
        )
    for i in range(len(object_record["matched_current_ids"])):
        if object_record["matched_current_ids"][i] == 0:
            continue
        id = object_record["universal_ids"][i]
        color_index = (id - 1) % len(colors)
        color = colors[color_index]
        flow_box = object_record["flow_boxes"][i]
        flow = object_record["flows"][i]
        search_box = object_record["search_boxes"][i]
        previous_centers = object_record["previous_centers"][i]
        current_centers = object_record["matched_current_centers"][i]
        plot_box(axes[0], flow_box, grid, color)
        plot_box(axes[1], search_box, grid, color)
        row, col = get_box_center_coords(flow_box, latitudes, longitudes)[2:]
        lat = latitudes[row]
        lon = longitudes[col]
        plot_flow(axes[0], lat, lon, flow, grid_options)
        plot_displacement(axes[0], previous_centers, current_centers)


def visualize_match(
    input_record,
    tracks,
    level_index,
    obj,
    track_options,
    grid_options,
    figure_options,
):
    """Visualize the matching process."""

    object_tracks = tracks[level_index][obj]
    object_record = object_tracks["object_record"]
    object_options = track_options[level_index][obj]
    current_grid, previous_grid = get_grids(object_tracks, object_options)
    current_mask, previous_mask = get_masks(object_tracks, object_options, matched=True)

    extent = get_extent(current_grid)

    if "figure_template" not in figure_options.keys():
        fig, ax, cbar_ax = match_template(
            current_grid, input_record, figure_options, extent
        )
        figure_options["figure_template"] = fig

    fig = copy.deepcopy(figure_options["figure_template"])
    axes = fig.axes[:-1]
    cbar_ax = fig.axes[-1]

    if previous_mask is not None:
        pcm = horizontal.grid(previous_grid, axes[0], add_colorbar=False)
        colors = horizontal.mask(previous_mask, axes[0])
    if current_mask is not None:
        pcm = horizontal.grid(current_grid, axes[1], add_colorbar=False)
        colors = horizontal.mask(current_mask, axes[1])
        match_features(current_grid, object_record, axes, colors, grid_options)

    cbar_label = (
        current_grid.attrs["long_name"].title() + f" [{current_grid.attrs['units']}]"
    )
    fig.colorbar(pcm, cax=cbar_ax, label=cbar_label)
    fig.suptitle(f"{current_grid.time.values.astype('datetime64[s]')} UTC", y=1.05)

    return fig, axes


create_mask_figure_dispatcher = {"detect": detected_mask, "group": grouped_mask}


def visualize_mask(
    input_record, tracks, level_index, obj, track_options, grid_options, figure_options
):
    """Plot masks for an object."""
    object_options = track_options[level_index][obj]
    create_figure = create_mask_figure_dispatcher.get(object_options["method"])
    if not create_figure:
        message = "create_mask_figure function for object track option "
        message += f"{object_options['method']} not found."
        raise KeyError(message)

    fig, ax = create_figure(
        input_record,
        tracks,
        level_index,
        obj,
        track_options,
        grid_options,
        figure_options,
    )
    return fig, ax


create_figure_dispatcher = {"mask": visualize_mask, "match": visualize_match}


def visualize(
    track_input_records,
    tracks,
    level_index,
    obj,
    track_options,
    grid_options,
    visualize_options,
    output_directory,
):
    # Close all current figures
    plt.close("all")

    object_options = track_options[level_index][obj]

    if not visualize_options or not visualize_options.get(object_options["name"]):
        return
    input_record = track_input_records[object_options["dataset"]]
    object_visualize_options = visualize_options.get(object_options["name"])
    for figure in object_visualize_options["figures"].keys():
        create_figure = create_figure_dispatcher.get(figure)
        if not create_figure:
            message = "create_figure function for figure type "
            message += f"{figure} not found."
            raise KeyError(message)

        figure_options = object_visualize_options["figures"][figure]
        with plt.style.context(styles[figure_options["style"]]):

            fig, ax = create_figure(
                input_record,
                tracks,
                level_index,
                obj,
                track_options,
                grid_options,
                figure_options,
            )
            if object_visualize_options["save"]:
                grid_time = input_record["current_grid"].time.values
                filename = f"{format_time(grid_time)}.png"
                filepath = (
                    output_directory
                    / "visualize/runtime"
                    / figure
                    / object_visualize_options["name"]
                    / filename
                )
                filepath.parent.mkdir(parents=True, exist_ok=True)
                logger.debug(
                    f"Saving {figure} figure for {object_visualize_options['name']}."
                )
                fig.savefig(filepath, bbox_inches="tight")
