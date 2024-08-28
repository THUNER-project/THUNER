"""
Plotting functions to be called during algorithm runtime for debugging 
and visualization purposes.
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import thor.visualize.horizontal as horizontal
from thor.visualize.visualize import styles
from thor.utils import format_time
from thor.match.utils import get_grids, get_masks
from thor.log import setup_logger
from thor.visualize.utils import make_subplot_labels
from thor.visualize.visualize import mask_colors
from thor.object.box import get_box_center_coords
import thor.grid as thor_grid


logger = setup_logger(__name__)
proj = ccrs.PlateCarree()


def get_extent(grid_options):
    """Get the cartopy extent."""
    lon = np.array(grid_options["longitude"])
    lat = np.array(grid_options["latitude"])
    return (lon.min(), lon.max(), lat.min(), lat.max())


def get_boundaries(input_record, num_previous=1):
    """Get the appropriate current and previous masks for matching."""
    current_boundaries = input_record["current_boundary_coordinates"]
    previous_boundaries = input_record["previous_boundary_coordinates"]
    previous_boundaries = [previous_boundaries[-i] for i in range(1, num_previous + 1)]
    boundaries = [current_boundaries] + previous_boundaries
    return boundaries


def detected_mask_template(grid, input_record, figure_options, extent):
    """Create a template figure for masks."""
    fig = plt.figure(figsize=(6, 3.5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    horizontal.add_cartographic_features(
        ax, style=figure_options["style"], scale="10m", extent=extent
    )
    if "instrument" in grid.attrs.keys() and "radar" in grid.attrs["instrument"]:
        radar_longitude = float(grid.attrs["origin_longitude"])
        radar_latitude = float(grid.attrs["origin_latitude"])
        horizontal.add_radar_features(ax, radar_longitude, radar_latitude, extent)
    return fig, ax


def detected_mask(
    input_record, tracks, level_index, obj, track_options, grid_options, figure_options
):
    """Plot masks for a detected object."""

    object_tracks = tracks[level_index][obj]
    object_options = track_options[level_index][obj]
    grid = object_tracks["current_grid"]
    extent = get_extent(grid_options)

    if "figure_template" not in figure_options.keys():
        fig, ax = detected_mask_template(grid, input_record, figure_options, extent)
        figure_options["figure_template"] = fig

    fig = copy.deepcopy(figure_options["figure_template"])
    ax = fig.axes[0]

    pcm = horizontal.grid(grid, ax, grid_options, add_colorbar=False)

    if object_options["tracking"]["method"] is None:
        mask = object_tracks["current_mask"]
    else:
        mask = object_tracks["current_matched_mask"]
    if mask is not None:
        horizontal.mask(mask, ax, grid_options)

    if input_record["current_boundary_coordinates"] is not None:
        boundaries = input_record["current_boundary_coordinates"]
        horizontal.add_domain_boundary(ax, boundaries)

    cbar_label = grid.name.title() + f" [{grid.units}]"
    fig.colorbar(pcm, label=cbar_label)
    ax.set_title(f"{grid.time.values.astype('datetime64[s]')} UTC")

    return fig, ax


def grouped_mask_template(
    grid, input_record, figure_options, extent, figsize, member_objects
):
    """Create a template figure for grouped masks."""
    fig = plt.figure(figsize=figsize)
    style = figure_options["style"]
    nrows = 1
    ncols = len(member_objects) + 1
    width_ratios = [1] * len(member_objects) + [0.05]
    gs = gridspec.GridSpec(nrows, ncols, width_ratios=width_ratios)
    axes = []
    for i in range(len(member_objects)):
        ax = fig.add_subplot(gs[0, i], projection=proj)
        axes.append(ax)
        ax = horizontal.add_cartographic_features(
            ax, extent=extent, style=style, scale="10m", left_labels=(i == 0)
        )[0]
        grid_i = grid[f"{member_objects[i]}_grid"]
        if (
            "instrument" in grid_i.attrs.keys()
            and "radar" in grid_i.attrs["instrument"]
        ):
            radar_longitude = float(grid_i.attrs["origin_longitude"])
            radar_latitude = float(grid_i.attrs["origin_latitude"])
            horizontal.add_radar_features(ax, radar_longitude, radar_latitude, extent)
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
    extent = get_extent(grid_options)

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
        pcm = horizontal.grid(grid_i, ax, grid_options, add_colorbar=False)
        if mask_i is not None:
            horizontal.mask(mask_i, ax, grid_options)
        if "current_boundary_coordinates" in input_record.keys():
            boundaries = input_record["current_boundary_coordinates"]
            horizontal.add_domain_boundary(ax, boundaries)

    cbar_label = grid_i.attrs["long_name"].title() + f" [{grid_i.attrs['units']}]"
    fig.colorbar(pcm, cax=cbar_ax, label=cbar_label)
    fig.suptitle(f"{grid.time.values.astype('datetime64[s]')} UTC", y=1.05)

    return fig, ax


def match_template(reference_grid, input_record, figure_options, extent):
    """Create a template for match figures."""
    fig = plt.figure(figsize=(12, 3))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05])
    axes = []
    for i in range(3):
        ax = fig.add_subplot(gs[0, i], projection=proj)
        axes.append(ax)
        ax = horizontal.add_cartographic_features(
            ax,
            extent=extent,
            style=figure_options["style"],
            scale="10m",
            left_labels=(i == 0),
        )[0]
        if (
            "instrument" in reference_grid.attrs.keys()
            and "radar" in reference_grid.attrs["instrument"]
        ):
            radar_longitude = float(reference_grid.attrs["origin_longitude"])
            radar_latitude = float(reference_grid.attrs["origin_latitude"])
            horizontal.add_radar_features(ax, radar_longitude, radar_latitude, extent)
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
        [row, col] = np.ceil(np.array(grid_options["shape"]) / 2).astype(int)
        vector_options = {"start_lat": lat, "start_lon": lon, "color": "tab:red"}
        horizontal.plot_vector(
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
            horizontal.plot_box(axes[1], global_flow_box, grid_options, alpha=0.8)
            horizontal.plot_vector(
                axes[1], row, col, global_flow, grid_options, color="tab:red"
            )
        # Plot the local flow box, and the local and corrected flow vectors
        horizontal.plot_box(axes[1], flow_box, grid_options, color=color)
        horizontal.plot_vector(axes[1], row, col, flow, grid_options, color="silver")
        horizontal.plot_vector(
            axes[1], row, col, corrected_flow, grid_options, linestyle=":"
        )
        # Plot the search box
        horizontal.plot_box(axes[2], search_box, grid_options, color=color)

        if np.all(np.logical_not(np.isnan(displacement))):
            # Subtract displacement from previous center to get the origin
            origin = center - displacement.astype(int)
            horizontal.plot_vector(
                axes[0],
                origin[0],
                origin[1],
                displacement,
                grid_options,
                color="silver",
            )
        # Label object with corrected flow case and cost
        case = object_record["cases"][i]
        lat = np.array(grid_options["latitude"])
        lat_shift = 0.01 * (lat.max() - lat.min())  # Shift text up slightly
        row, col = flow_box["row_max"], flow_box["col_min"]
        text_lat, text_lon = thor_grid.get_pixels_geographic(row, col, grid_options)
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
    object_options = track_options[level_index][obj]
    grids = get_grids(object_tracks, object_options, num_previous=2)
    masks = get_masks(object_tracks, object_options, matched=True, num_previous=2)
    all_boundaries = get_boundaries(input_record, num_previous=2)

    extent = get_extent(grid_options)

    if "figure_template" not in figure_options.keys():
        fig, ax, cbar_ax = match_template(
            grids[0], input_record, figure_options, extent
        )
        figure_options["figure_template"] = fig

    fig = copy.deepcopy(figure_options["figure_template"])
    axes = fig.axes[:-1]
    cbar_ax = fig.axes[-1]

    for i in range(3):
        j = 2 - i
        if grids[j] is not None:
            axes[i].set_title(grids[j].time.values.astype("datetime64[s]"))
            pcm = horizontal.grid(grids[j], axes[i], grid_options, add_colorbar=False)
            if masks[j] is not None:
                horizontal.mask(masks[j], axes[i], grid_options)
            if input_record["current_boundary_coordinates"] is not None:
                horizontal.add_domain_boundary(axes[i], all_boundaries[j])
    unique_global_flow = object_options["tracking"]["options"]["unique_global_flow"]
    match_features(grids[0], object_record, axes, grid_options, unique_global_flow)
    cbar_label = grids[0].attrs["long_name"].title() + f" [{grids[0].attrs['units']}]"
    fig.colorbar(pcm, cax=cbar_ax, label=cbar_label)

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

    logger.info("Generating runtime visualizations.")

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
