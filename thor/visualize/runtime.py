"""
Plotting functions to be called during algorithm runtime for debugging 
and visualization purposes.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import thor.visualize.horizontal as horizontal
import thor.detect as detect
from thor.visualize.visualize import styles
from thor.utils import format_time


def mask(input_record, object_tracks, object_options, figure_options):
    """Plot masks."""

    grid = input_record["current_grid"]
    if object_options["detection"]["flatten_method"] is not None:
        flattener = detect.detect.flattener_dispatcher.get(
            object_options["detection"]["flatten_method"]
        )
        flat_grid = flattener(grid, object_options)
    else:
        flat_grid = detect.preprocess.vertical_max(grid, object_options)

    extent = (
        grid.longitude.min(),
        grid.longitude.max(),
        grid.latitude.min(),
        grid.latitude.max(),
    )

    fig, ax = horizontal.initialize_cartographic_figure(
        style=figure_options["style"], scale="10m", extent=extent
    )[:2]

    horizontal.grid(flat_grid, fig, ax)

    mask = object_tracks["current_mask"]
    if mask is not None:
        horizontal.mask(mask.where(mask > 0), fig, ax)

    ax.set_title(f"{grid.time.values.astype('datetime64[s]')} UTC")

    return fig, ax


def match(grid, mask):
    """Plot masks."""

    return


create_figure_dispatcher = {"mask": mask, "match": match}


def visualize(track_input_records, object_tracks, object_options, visualize_options):
    # Close all current figures
    plt.close("all")

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
                input_record, object_tracks, object_options, figure_options
            )
            if object_visualize_options["save"]:
                grid_time = input_record["current_grid"].time.values
                filename = f"{format_time(grid_time)}.png"
                filepath = (
                    Path(object_visualize_options["parent_local"])
                    / "runtime"
                    / figure
                    / object_visualize_options["name"]
                    / filename
                )
                filepath.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(filepath, bbox_inches="tight")
