"""
Plotting functions to be called during algorithm runtime for debugging
and visualization purposes.
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import thuner.visualize.horizontal as horizontal
from thuner.visualize.visualize import styles
from thuner.utils import format_time
from thuner.match.utils import get_grids, get_masks
from thuner.log import setup_logger
from thuner.visualize.utils import get_extent
from thuner.visualize.visualize import runtime_colors, set_style
from thuner.match.box import get_box_center_coords
import thuner.grid as thuner_grid
import thuner.visualize.utils as utils


logger = setup_logger(__name__)
proj = ccrs.PlateCarree()


def get_boundaries(input_record, num_previous=1):
    """Get the appropriate current and next boundaries for matching."""
    next_boundaries = input_record.next_boundary_coordinates
    current_boundaries = input_record.boundary_coodinates
    current_boundaries = [current_boundaries[-i] for i in range(1, num_previous + 1)]
    boundaries = [next_boundaries] + current_boundaries
    return boundaries


def detected_mask(
    input_record, tracks, level_index, obj, track_options, grid_options, figure_options
):
    """Plot masks for a detected object."""

    object_tracks = tracks.levels[level_index].objects[obj]
    object_options = track_options.levels[level_index].object_by_name(obj)
    grid = object_tracks.next_grid

    if object_options.tracking is None:
        mask = object_tracks.next_mask
    else:
        mask = object_tracks.next_matched_mask

    boundary_coordinates = input_record.next_boundary_coordinates
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
    object_options = track_options.levels[level_index].object_by_name(obj)
    object_tracks = tracks.levels[level_index].objects[obj]
    if object_options.tracking is None:
        mask = object_tracks.next_mask
    else:
        mask = object_tracks.next_matched_mask

    member_objects = object_options.grouping.member_objects
    grid = object_tracks.next_grid

    boundary_coordinates = input_record.next_boundary_coordinates
    args = [grid, mask, grid_options, figure_options, member_objects]
    args += [boundary_coordinates]
    fig, subplot_axes = horizontal.grouped_mask(*args)[:2]

    return fig, subplot_axes


def match_template(reference_grid, extent, scale):
    """Create a template for match figures."""
    if scale == 1:
        rows = 1
        columns = 3
        subplot_width = 4
    elif scale == 2:
        rows = 3
        columns = 1
        subplot_width = 8
    else:
        raise ValueError("Only scales of 1 or 2 implemented so far.")
    kwargs = {"extent": extent, "subplot_width": subplot_width, "rows": rows}
    kwargs.update({"columns": columns, "colorbar": True, "legend_rows": 2})
    kwargs.update({"shared_legends": "all"})
    layout = horizontal.PanelledUniformMaps(**kwargs)
    fig, subplot_axes, colorbar_axes, legend_axes = layout.initialize_layout()
    for i in range(len(subplot_axes)):
        ax = subplot_axes[i]
        ax.set_extent(extent)
        keys = reference_grid.attrs.keys()
        if "instrument" in keys and "radar" in reference_grid.attrs["instrument"]:
            radar_longitude = float(reference_grid.attrs["origin_longitude"])
            radar_latitude = float(reference_grid.attrs["origin_latitude"])
            horizontal.radar_features(ax, radar_longitude, radar_latitude, extent)

    return fig, subplot_axes, colorbar_axes, legend_axes, layout


def match_features(grid, match_record, axes, grid_options, unique_global_flow=True):
    """Visualizing the matching process for TINT/MINT matching."""
    colors = runtime_colors

    if unique_global_flow and len(match_record["global_flows"]) > 0:
        global_flow = match_record["global_flows"][0]
        if "instrument" in grid.attrs.keys() and "radar" in grid.attrs["instrument"]:
            lon = float(grid.attrs["origin_longitude"])
            lat = float(grid.attrs["origin_latitude"])
        else:
            lon, lat = None, None
        [row, col] = np.ceil(np.array(grid_options.shape) / 2).astype(int)
        args = [axes[1], row, col, global_flow, grid_options]
        vector_options = {"start_lat": lat, "start_lon": lon, "color": "tab:red"}
        horizontal.pixel_displacement(*args, **vector_options)
    for i in range(len(match_record["ids"])):
        # Get the flows, displacements and boxes.
        obj_id = match_record["universal_ids"][i]
        color_index = (obj_id - 1) % len(colors)
        color = colors[color_index]
        flow_box = match_record["flow_boxes"][i]
        flow = match_record["flows"][i]
        corrected_flow = match_record["corrected_flows"][i]
        search_box = match_record["search_boxes"][i]

        # Plot object universal ID in current grid.
        center = match_record["centers"][i]
        coords = thuner_grid.get_coordinates(grid_options, center[0], center[1])
        horizontal.embossed_text(axes[1], str(obj_id), coords[1], coords[0])
        # Plot object universal IDs in next grid.
        next_center = match_record["next_centers"][i]
        if np.all(np.logical_not(np.isnan(next_center))):
            args = [grid_options, next_center[0], next_center[1]]
            next_coords = thuner_grid.get_coordinates(*args)
            args = [axes[2], str(obj_id), next_coords[1], next_coords[0]]
            horizontal.embossed_text(*args)

        displacement = match_record["displacements"][i]
        row, col = get_box_center_coords(flow_box, grid_options)[2:]
        if not unique_global_flow:
            # If global flow not unique, plot for current object
            global_flow = match_record["global_flows"][i]
            global_flow_box = match_record["global_flow_boxes"][i]
            horizontal.plot_box(axes[1], global_flow_box, grid_options, alpha=0.8)
            args = [axes[1], row, col, global_flow, grid_options]
            horizontal.pixel_displacement(*args, color="tab:red")
        # Plot the local flow box, and the local and corrected flow vectors
        horizontal.plot_box(axes[1], flow_box, grid_options, color=color)
        args = [axes[1], row, col, flow, grid_options]
        horizontal.pixel_displacement(*args, color="tab:blue")
        args = [axes[1], row, col, corrected_flow, grid_options]
        horizontal.pixel_displacement(*args, color="tab:purple")
        # Plot the search box
        kwargs = {"color": color, "linestyle": "dashdot"}
        horizontal.plot_box(axes[2], search_box, grid_options, **kwargs)

        if np.all(np.logical_not(np.isnan(displacement))):
            # Subtract displacement from center to get the origin
            origin = center - displacement.astype(int)
            args = [axes[0], origin[0], origin[1], displacement, grid_options]
            horizontal.pixel_displacement(*args, color="tab:green")
        # Label object with corrected flow case and cost
        case = match_record["cases"][i]
        lat = np.array(grid_options.latitude)
        lat_shift = 0.01 * (lat.max() - lat.min())  # Shift text up slightly
        row, col = flow_box["row_max"], flow_box["col_min"]
        text_lat, text_lon = thuner_grid.get_pixels_geographic(row, col, grid_options)
        text_lat = text_lat + lat_shift
        text_properties = {"fontsize": 6, "zorder": 4, "color": color}
        text_properties.update({"weight": "bold", "transform": proj})
        if match_record["next_ids"][i] != 0:
            distance = int(np.round(match_record["distances"][i]))
            area_difference = int(np.round(match_record["area_differences"][i]))
            area_overlap = int(np.round(match_record["overlap_areas"][i]))
            object_text = f"{case}, {distance}+{area_difference}-{area_overlap}"
        else:
            object_text = f"{case}, No Match"
        axes[1].text(text_lon, text_lat, object_text, **text_properties)


def visualize_tint_match(
    input_record,
    tracks,
    level_index,
    obj,
    track_options,
    grid_options,
    figure_options,
):
    """Visualize the matching process."""

    object_tracks = tracks.levels[level_index].objects[obj]
    match_record = object_tracks.match_record
    object_options = track_options.levels[level_index].object_by_name(obj)
    grids = get_grids(object_tracks, object_options, num_previous=2)
    if object_options.tracking is not None:
        matched = True
    else:
        matched = False
    masks = get_masks(object_tracks, object_options, matched=matched, num_previous=2)
    all_boundaries = get_boundaries(input_record, num_previous=2)

    extent, scale = get_extent(grid_options)

    if figure_options.template is None:
        figure_options.template = match_template(grids[0], extent, scale)

    template = copy.deepcopy(figure_options.template)
    fig, subplot_axes, colorbar_axes, legend_axes, layout = template

    for i in range(3):
        j = 2 - i
        ax = subplot_axes[i]
        if grids[j] is not None:
            ax.set_title(grids[j].time.values.astype("datetime64[s]"))
            args = [grids[j], ax, grid_options, False]
            pcm = horizontal.show_grid(*args)
            if masks[j] is not None:
                horizontal.show_mask(masks[j], ax, grid_options)
            if input_record.next_boundary_coordinates is not None:
                horizontal.domain_boundary(ax, all_boundaries[j], grid_options)
        ax.set_extent(extent)
    unique_global_flow = object_options.tracking.unique_global_flow
    args = [grids[0], match_record, subplot_axes, grid_options, unique_global_flow]
    match_features(*args)
    cbar_label = grids[0].attrs["long_name"].title() + f" [{grids[0].attrs['units']}]"
    fig.colorbar(pcm, cax=colorbar_axes[0], label=cbar_label)

    # Create legend patches and labels
    handles = []
    labels = ["Displacement", "Global Flow", "Local Flow", "Corrected Flow"]
    colors = ["tab:green", "tab:red", "tab:blue", "tab:purple"]
    for i in range(len(labels)):
        handles.append(horizontal.displacement_legend_artist(colors[i], labels[i]))

    handle, handler = horizontal.mask_legend_artist()
    handles = [handle] + handles
    labels = ["Object Masks"] + labels

    if not unique_global_flow:
        kwargs = {"color": "tab:red", "single_color": True}
        handle, handler = horizontal.box_legend_artist(**kwargs)
        handles = handles + [handle]
        labels = labels + ["Global Flow Boxes"]

    handle, handler = horizontal.box_legend_artist()
    handles = handles + [handle]
    labels = labels + ["Local Flow Boxes"]

    handle, handler = horizontal.box_legend_artist(linestyle="dashdot")
    handles = handles + [handle]
    labels = labels + ["Search Boxes"]

    legend_options = {"ncol": 4, "loc": "lower center", "handler_map": handler}
    legend_axes[0].legend(handles, labels, **legend_options)

    return fig, subplot_axes


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
    object_options = track_options.levels[level_index].object_by_name(obj)
    create_figure = create_mask_figure_dispatcher(object_options)
    if not create_figure:
        message = "create_mask_figure function for object detection option not found."
        raise KeyError(message)

    args = [input_record, tracks, level_index, obj, track_options]
    args += [grid_options, figure_options]
    fig, ax = create_figure(*args)
    return fig, ax


def visualize(
    track_input_records,
    tracks,
    level_index,
    obj,
    track_options,
    grid_options,
    runtime_options,
    output_directory,
):
    # Close all current figures
    plt.close("all")

    if not runtime_options or not runtime_options.objects.get(obj):
        return
    object_options = track_options.levels[level_index].object_by_name(obj)
    object_runtime_options = runtime_options.objects.get(obj)
    input_record = track_input_records[object_options.dataset]
    logger.info("Creating runtime visualization figures.")
    for figure_options in object_runtime_options.figures:
        if not figure_options.function:
            message = f"{object_options.name} {figure_options.name} figure "
            message += f"function undefined."
            raise KeyError(message)
        style = figure_options.style
        with plt.style.context(styles[style]), set_style(style):
            args = [input_record, tracks, level_index, obj, track_options]
            args += [grid_options, figure_options]
            fig, ax = figure_options.function(*args)

            grid_time = input_record.next_grid.time.values
            filename = f"{format_time(grid_time)}.png"
            figure_name = figure_options.name
            filepath = output_directory / "visualize" / figure_name
            filepath = filepath / obj / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Saving {figure_name} figure for {obj}.")
            fig.savefig(filepath, bbox_inches="tight")
            utils.reduce_color_depth(filepath)
            plt.close(fig)
