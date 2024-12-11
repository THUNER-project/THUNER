"""
Plotting functions to be called during algorithm runtime for debugging 
and visualization purposes.
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import thuner.visualize.horizontal as horizontal
from thuner.visualize.visualize import styles
from thuner.utils import format_time
from thuner.match.utils import get_grids, get_masks
from thuner.log import setup_logger
from thuner.visualize.utils import make_subplot_labels, get_extent
from thuner.visualize.visualize import mask_colors, set_style
from thuner.object.box import get_box_center_coords
import thuner.grid as thuner_grid
import thuner.visualize.utils as utils


logger = setup_logger(__name__)
proj = ccrs.PlateCarree()


def get_boundaries(input_record, num_previous=1):
    """Get the appropriate current and previous masks for matching."""
    current_boundaries = input_record["current_boundary_coordinates"]
    previous_boundaries = input_record["previous_boundary_coordinates"]
    previous_boundaries = [previous_boundaries[-i] for i in range(1, num_previous + 1)]
    boundaries = [current_boundaries] + previous_boundaries
    return boundaries


def detected_mask(
    input_record, tracks, level_index, obj, track_options, grid_options, figure_options
):
    """Plot masks for a detected object."""

    object_tracks = tracks[level_index][obj]
    object_options = track_options.levels[level_index].options_by_name(obj)
    grid = object_tracks["current_grid"]

    if "tracking" not in object_options.model_fields:
        mask = object_tracks["current_mask"]
    else:
        mask = object_tracks["current_matched_mask"]

    boundary_coordinates = input_record["current_boundary_coordinates"]
    args = [grid, mask, grid_options, figure_options, boundary_coordinates]
    fig, ax = horizontal.detected_mask(*args)

    return fig, ax


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
    object_options = track_options.levels[level_index].options_by_name(obj)
    object_tracks = tracks[level_index][obj]
    if "tracking" in object_options.model_fields:
        mask = object_tracks["current_mask"]
    else:
        mask = object_tracks["current_matched_mask"]

    try:
        member_objects = figure_options["member_objects"]
    except KeyError:
        member_objects = object_options.grouping.member_objects

    grid = object_tracks["current_grid"]
    extent, scale = get_extent(grid_options)

    boundary_coordinates = input_record["current_boundary_coordinates"]
    args = [grid, mask, grid_options, figure_options, member_objects]
    args += [boundary_coordinates]
    fig, subplot_axes = horizontal.grouped_mask(*args)[:2]

    return fig, subplot_axes


def match_template(reference_grid, figure_options, extent):
    """Create a template for match figures."""
    fig = plt.figure(figsize=(12, 3))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05])
    axes = []
    for i in range(3):
        ax = fig.add_subplot(gs[0, i], projection=proj)
        ax.set_rasterized(True)
        kwargs = {"extent": extent, "scale": "10m"}
        kwargs.update({"left_labels": (i == 0)})
        ax = horizontal.cartographic_features(ax, **kwargs)[0]
        if (
            "instrument" in reference_grid.attrs.keys()
            and "radar" in reference_grid.attrs["instrument"]
        ):
            radar_longitude = float(reference_grid.attrs["origin_longitude"])
            radar_latitude = float(reference_grid.attrs["origin_latitude"])
            horizontal.radar_features(ax, radar_longitude, radar_latitude, extent)
        axes.append(ax)
    cbar_ax = fig.add_subplot(gs[0, -1])
    make_subplot_labels(axes, x_shift=-0.12, y_shift=0.06)
    return fig, axes, cbar_ax


def match_features(grid, object_record, axes, grid_options, unique_global_flow=True):
    colors = mask_colors
    if unique_global_flow and len(object_record["global_flows"]) > 0:
        global_flow = object_record["global_flows"][0]
        if "instrument" in grid.attrs.keys() and "radar" in grid.attrs["instrument"]:
            lon = float(grid.attrs["origin_longitude"])
            lat = float(grid.attrs["origin_latitude"])
        else:
            lon, lat = None, None
        [row, col] = np.ceil(np.array(grid_options.shape) / 2).astype(int)
        vector_options = {"start_lat": lat, "start_lon": lon, "color": "tab:red"}
        horizontal.pixel_vector(
            axes[1], row, col, global_flow, grid_options, **vector_options
        )
    for i in range(len(object_record["previous_ids"])):
        # Get the flows, displacements and boxes.
        id = object_record["universal_ids"][i]
        color_index = (id - 1) % len(colors)
        color = colors[color_index]
        flow_box = object_record["flow_boxes"][i]
        flow = object_record["flows"][i]
        corrected_flow = object_record["corrected_flows"][i]
        search_box = object_record["search_boxes"][i]
        center = object_record["previous_centers"][i]
        displacement = object_record["previous_displacements"][i]
        row, col = get_box_center_coords(flow_box, grid_options)[2:]
        if not unique_global_flow:
            # If global flow not unique, plot for current object
            global_flow = object_record["global_flows"][i]
            global_flow_box = object_record["global_flow_boxes"][i]
            # horizontal.plot_box(axes[1], global_flow_box, grid_options, alpha=0.8)
            horizontal.pixel_vector(
                axes[1], row, col, global_flow, grid_options, color="tab:red"
            )
        # Plot the local flow box, and the local and corrected flow vectors
        horizontal.plot_box(axes[1], flow_box, grid_options, color=color)
        horizontal.pixel_vector(axes[1], row, col, flow, grid_options, color="silver")
        horizontal.pixel_vector(
            axes[1], row, col, corrected_flow, grid_options, linestyle=":"
        )
        # Plot the search box
        horizontal.plot_box(axes[2], search_box, grid_options, color=color)

        if np.all(np.logical_not(np.isnan(displacement))):
            # Subtract displacement from previous center to get the origin
            origin = center - displacement.astype(int)
            args = [axes[0], origin[0], origin[1], displacement, grid_options]
            horizontal.pixel_vector(*args, color="silver")
        # Label object with corrected flow case and cost
        case = object_record["cases"][i]
        lat = np.array(grid_options.latitude)
        lat_shift = 0.01 * (lat.max() - lat.min())  # Shift text up slightly
        row, col = flow_box["row_max"], flow_box["col_min"]
        text_lat, text_lon = thuner_grid.get_pixels_geographic(row, col, grid_options)
        text_lat = text_lat + lat_shift
        text_properties = {"fontsize": 6, "zorder": 4, "color": color}
        text_properties.update({"weight": "bold", "transform": proj})
        if object_record["matched_current_ids"][i] != 0:
            distance = int(np.round(object_record["distances"][i]))
            area_difference = int(np.round(object_record["area_differences"][i]))
            area_overlap = int(np.round(object_record["overlap_areas"][i]))
            object_text = f"{case}, {distance}+{area_difference}-{area_overlap}"
        else:
            object_text = f"{case}, No Match"
        axes[1].text(text_lon, text_lat, object_text, **text_properties)


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
    object_options = track_options.levels[level_index].options_by_name(obj)
    grids = get_grids(object_tracks, object_options, num_previous=2)
    masks = get_masks(object_tracks, object_options, matched=True, num_previous=2)
    all_boundaries = get_boundaries(input_record, num_previous=2)

    extent, scale = get_extent(grid_options)

    if figure_options["template"] is None:
        fig, ax, cbar_ax = match_template(grids[0], figure_options, extent)
        figure_options["template"] = fig

    fig = copy.deepcopy(figure_options["template"])
    axes = fig.axes[:-1]
    cbar_ax = fig.axes[-1]

    for i in range(3):
        j = 2 - i
        if grids[j] is not None:
            axes[i].set_title(grids[j].time.values.astype("datetime64[s]"))
            args = [grids[j], axes[i], grid_options, False]
            pcm = horizontal.show_grid(*args)
            if masks[j] is not None:
                horizontal.show_mask(masks[j], axes[i], grid_options)
            if input_record["current_boundary_coordinates"] is not None:
                horizontal.domain_boundary(axes[i], all_boundaries[j], grid_options)
        axes[i].set_extent(extent)
    unique_global_flow = object_options.tracking.unique_global_flow
    match_features(grids[0], object_record, axes, grid_options, unique_global_flow)
    cbar_label = grids[0].attrs["long_name"].title() + f" [{grids[0].attrs['units']}]"
    fig.colorbar(pcm, cax=cbar_ax, label=cbar_label)

    return fig, axes


def create_mask_figure_dispatcher(object_options):
    """Dispatch the mask figure creation based on the method."""
    if "detection" in object_options.model_fields:
        return detected_mask
    elif "grouping" in object_options.model_fields:
        return grouped_mask
    else:
        return None


def visualize_mask(
    input_record, tracks, level_index, obj, track_options, grid_options, figure_options
):
    """Plot masks for an object."""
    object_options = track_options.levels[level_index].options_by_name(obj)
    create_figure = create_mask_figure_dispatcher(object_options)
    if not create_figure:
        message = "create_mask_figure function for object detection option not found."
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

    object_options = track_options.levels[level_index].options_by_name(obj)

    if not visualize_options or not visualize_options.get(object_options.name):
        return
    input_record = track_input_records[object_options.dataset]
    object_visualize_options = visualize_options.get(object_options.name)
    logger.info("Generating runtime visualizations.")
    for figure in object_visualize_options["figures"].keys():
        create_figure = create_figure_dispatcher.get(figure)
        if not create_figure:
            message = "create_figure function for figure type "
            message += f"{figure} not found."
            raise KeyError(message)

        figure_options = object_visualize_options["figures"][figure]
        style = figure_options["style"]
        with plt.style.context(styles[style]), set_style(style):
            fig, ax = create_figure(
                input_record,
                tracks,
                level_index,
                obj,
                track_options,
                grid_options,
                figure_options,
            )
            if not object_visualize_options["save"]:
                return
            grid_time = input_record["current_grid"].time.values
            filename = f"{format_time(grid_time)}.png"
            obj_name = object_visualize_options["name"]
            filepath = output_directory / "visualize" / figure / obj_name / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)
            logger.debug(
                f"Saving {figure} figure for {object_visualize_options['name']}."
            )
            fig.savefig(filepath, bbox_inches="tight")
            utils.reduce_color_depth(filepath)
            plt.close(fig)
