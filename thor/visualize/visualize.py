"""General display methods."""

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

mask_colors = [
    "cyan",
    "magenta",
    "purple",
    "teal",
    "saddlebrown",
    "hotpink",
]

map_colors = {
    "paper": {
        "land_color": tuple(np.array([249.0, 246.0, 216.0]) / (256)),
        "sea_color": tuple(np.array([240.0, 240.0, 256.0]) / (256)),
        "coast_color": "black",
        "leg_color": "w",
    },
    "presentation": {
        "land_color": tuple(np.array([249.0, 246.0, 216.0]) / (256 * 3.5)),
        "sea_color": tuple(np.array([245.0, 245.0, 256.0]) / (256 * 3.5)),
        "coast_color": "white",
        "leg_color": tuple(np.array([249.0, 246.0, 216.0]) / (256 * 3.5)),
    },
}

base_styles = {"paper": "default", "presentation": "dark_background"}
custom_styles_dir = Path(__file__).parent / "styles"

styles = {
    style: [base_styles[style], custom_styles_dir / f"{style}.mplstyle"]
    for style in base_styles.keys()
}


def get_filepaths_dates(fig_type, obj, output_directory):
    filepaths = glob.glob(
        str(output_directory / "visualize" / fig_type / obj / "*.png")
    )
    filepaths = np.array(sorted(filepaths))
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


def animate_object(fig_type, obj, output_directory):
    """
    Animate object figures.
    """
    filepaths, dates = get_filepaths_dates(fig_type, obj, output_directory)
    for date in np.unique(dates):
        filepaths_date = filepaths[dates == date]
        output_filepath = (
            output_directory / "visualize" / fig_type / f"{obj}_{date}.gif"
        )
        logger.info(
            f"Animating {fig_type} figures for {obj} objects on "
            f"{date[:4]}-{date[4:6]}-{date[6:8]}."
        )
        images = [Image.open(f) for f in filepaths_date]
        imageio.mimsave(output_filepath, images, fps=5, loop=0)


def get_grid(time, filename, field, data_options, grid_options):
    """
    Get the grid from a file.
    """
    grid = utils.load_grid(filename)
    return grid[field]
