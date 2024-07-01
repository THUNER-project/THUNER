"""
Plotting functions to be called during algorithm runtime for debugging 
and visualization purposes.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import thor.visualize.horizontal as horizontal
from thor.visualize.visualize import styles
from thor.utils import format_time
from thor.log import setup_logger
from thor.visualize.utils import make_subplot_labels


logger = setup_logger(__name__)


def detected_mask(
    input_record, tracks, level_index, obj, track_options, figure_options
):
    """Plot masks for a detected object."""

    object_tracks = tracks[level_index][obj]
    grid = object_tracks["processed_grid"]

    extent = (
        grid.longitude.values.min(),
        grid.longitude.values.max(),
        grid.latitude.values.min(),
        grid.latitude.values.max(),
    )

    fig = plt.figure(figsize=(6, 3))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    horizontal.add_cartographic_features(
        ax, style=figure_options["style"], scale="10m", extent=extent
    )
    if "radar" in grid.attrs["instrument"]:
        horizontal.add_radar_features(
            ax,
            float(grid.attrs["origin_longitude"]),
            float(grid.attrs["origin_latitude"]),
            extent,
            input_record,
        )
    horizontal.grid(grid, ax)
    mask = object_tracks["current_mask"]
    if mask is not None:
        horizontal.mask(mask, ax)

    ax.set_title(f"{grid.time.values.astype('datetime64[s]')} UTC")

    return fig, ax


def grouped_mask(input_record, tracks, level_index, obj, track_options, figure_options):
    """Plot masks for a grouped object."""
    object_options = track_options[level_index][obj]
    object_tracks = tracks[level_index][obj]
    mask = object_tracks["current_mask"]

    try:
        member_objects = figure_options["member_objects"]
        member_levels = figure_options["member_levels"]
    except KeyError:
        member_objects = object_options["grouping"]["member_objects"]
        member_levels = object_options["grouping"]["member_levels"]

    # Initialise figures using extent of processed grid for first member object
    grid = tracks[member_levels[0]][member_objects[0]]["processed_grid"]

    extent = (
        grid.longitude.min(),
        grid.longitude.max(),
        grid.latitude.min(),
        grid.latitude.max(),
    )

    try:
        figsize = figure_options["figsize"]
    except KeyError:
        figsize = (len(member_objects) * 4, 4)

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
        if "radar" in grid.attrs["instrument"]:
            horizontal.add_radar_features(
                ax,
                float(grid.attrs["origin_longitude"]),
                float(grid.attrs["origin_latitude"]),
                extent,
                input_record,
            )
        grid = tracks[member_levels[i]][member_objects[i]]["processed_grid"]
        pcm = horizontal.grid(grid, ax, add_colorbar=False)
        if mask[f"{member_objects[i]}_mask"] is not None:
            horizontal.mask(mask[f"{member_objects[i]}_mask"], ax)
        ax.set_title(member_objects[i].replace("_", " ").title())

    cbar_ax = fig.add_subplot(gs[0, -1])
    cbar_label = grid.name.title() + f" [{grid.units}]"
    fig.colorbar(pcm, cax=cbar_ax, label=cbar_label)

    fig.suptitle(f"{grid.time.values.astype('datetime64[s]')} UTC", y=1.05)

    make_subplot_labels(axes, x_shift=-0.12, y_shift=0.06)

    return fig, ax


def match(grid, mask):
    """Plot masks."""

    return


create_mask_figure_dispatcher = {"detect": detected_mask, "group": grouped_mask}


def mask(input_record, tracks, level_index, obj, track_options, figure_options):
    """Plot masks for an object."""
    object_options = track_options[level_index][obj]
    create_figure = create_mask_figure_dispatcher.get(object_options["method"])
    if not create_figure:
        message = "create_mask_figure function for object track option "
        message += f"{object_options['method']} not found."
        raise KeyError(message)

    fig, ax = create_figure(
        input_record, tracks, level_index, obj, track_options, figure_options
    )
    return fig, ax


create_figure_dispatcher = {"mask": mask, "match": match}


def visualize(
    track_input_records,
    tracks,
    level_index,
    obj,
    track_options,
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
                input_record, tracks, level_index, obj, track_options, figure_options
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
                fig.savefig(filepath, bbox_inches="tight")
