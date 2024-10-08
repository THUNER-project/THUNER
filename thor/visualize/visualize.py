"""General display functions."""

from PIL import Image
import imageio
from pathlib import Path
import glob
import numpy as np
from matplotlib.colors import BoundaryNorm
import pyart.graph.cm_colorblind as pcm
import thor.visualize.utils as utils
from thor.log import setup_logger

logger = setup_logger(__name__)

reflectivity_levels = np.arange(-10, 60 + 5, 5)
reflectivity_norm = BoundaryNorm(
    reflectivity_levels, ncolors=pcm.HomeyerRainbow.N, clip=True
)

pcolormesh_style = {
    "reflectivity": {
        "cmap": pcm.HomeyerRainbow,
        "shading": "nearest",
        "norm": reflectivity_norm,
    },
}

mask_colors = ["cyan", "magenta", "purple", "teal", "saddlebrown"]
mask_colors += ["hotpink", "darkslategrey", "rosybrown", "plum"]
mask_colors += ["indigo", "darksalmon", "tan"]

figure_colors = {
    "paper": {
        "land": tuple(np.array([249.0, 246.0, 216.0]) / (256)),
        "sea": tuple(np.array([240.0, 240.0, 256.0]) / (256)),
        "coast": "black",
        "legend": "w",
        "key": "k",
        "ellipse_axis": "w",
        "ellipse_axis_shadow": "grey",
    },
    "presentation": {
        "land": tuple(np.array([249.0, 246.0, 216.0]) / (256 * 3.5)),
        "sea": tuple(np.array([245.0, 245.0, 256.0]) / (256 * 3.5)),
        "coast": "white",
        "legend": tuple(np.array([249.0, 246.0, 216.0]) / (256 * 3.5)),
        "key": "tab:purple",
        "ellipse_axis": "w",
        "ellipse_axis_shadow": "k",
    },
}

base_styles = {"paper": "default", "presentation": "dark_background"}
custom_styles_dir = Path(__file__).parent / "styles"

styles = {
    style: [base_styles[style], custom_styles_dir / f"{style}.mplstyle"]
    for style in base_styles.keys()
}


def get_filepaths_dates(directory):
    filepaths = np.array(sorted(glob.glob(str(directory / "*.png"))))
    dates = []
    for filepath in filepaths:
        date = Path(filepath).stem
        date = f"{date[:8]}"
        dates.append(date)
    dates = np.array(dates)
    return filepaths, dates


def animate_all(visualize_options, output_directory):
    if visualize_options is None:
        return
    for obj in visualize_options.keys():
        for fig_type in visualize_options[obj]["figures"].keys():
            if visualize_options[obj]["figures"][fig_type]["animate"]:
                animate_object(fig_type, obj, output_directory)


def animate_object(
    fig_type,
    obj,
    output_directory,
    save_directory=None,
    figure_directory=None,
    animation_name=None,
    by_date=True,
):
    """
    Animate object figures.
    """
    if save_directory is None:
        save_directory = output_directory / "visualize" / fig_type
    if figure_directory is None:
        figure_directory = output_directory / "visualize" / fig_type / obj
    if animation_name is None:
        animation_name = obj

    logger.info(f"Animating {fig_type} figures for {obj} objects.")

    filepaths, dates = get_filepaths_dates(figure_directory)
    if by_date:
        for date in np.unique(dates):
            filepaths_date = filepaths[dates == date]
            output_filepath = save_directory / f"{animation_name}_{date}.gif"
            logger.info(f"Saving animation to {output_filepath}.")
            images = [Image.open(f) for f in filepaths_date]
            imageio.mimsave(output_filepath, images, fps=5, loop=0)
    else:
        output_filepath = save_directory / f"{animation_name}.gif"
        logger.info(f"Saving animation to {output_filepath}.")
        images = [Image.open(f) for f in filepaths]
        imageio.mimsave(output_filepath, images, fps=5, loop=0)


def get_grid(time, filename, field, data_options, grid_options):
    """
    Get the grid from a file.
    """
    grid = utils.load_grid(filename)
    return grid[field]
