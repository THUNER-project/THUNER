"""General display functions."""

from PIL import Image
import imageio
import colorsys
from pathlib import Path
import glob
import numpy as np
import contextlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pyart.graph.cm_colorblind as pcm
import thuner.visualize.utils as utils
from thuner.log import setup_logger

logger = setup_logger(__name__)


style = "presentation"


def discrete_cmap_norm(
    levels, cmap_name="Reds", pad_start=0, pad_end=0, extend="neither"
):
    """Create a discrete colormap."""
    number_levels = len(levels)

    extend_above = 1 if extend in ["both", "max"] else 0
    extend_below = 1 if extend in ["both", "min"] else 0
    number_colors = pad_start + extend_below + number_levels + extend_above + pad_end
    cmap = plt.get_cmap(cmap_name, number_colors)
    colors = list(cmap(np.arange(0, number_colors)))
    if extend in ["both", "max"]:
        end = -pad_end - extend_above if pad_end != 0 else -extend_above
    else:
        end = -pad_end if pad_end != 0 else None
    start = pad_start + extend_below
    cmap = mcolors.ListedColormap(colors[start:end], f"{cmap_name}_discrete")
    norm = mcolors.BoundaryNorm(levels, ncolors=number_levels, clip=False)

    if extend in ["both", "min"]:
        cmap.set_under(colors[start - extend_below])
    if extend in ["both", "max"]:
        cmap.set_over(colors[end])

    return cmap, norm


def desaturate_colormap(cmap, factor=0.15):
    """Desaturate a colormap by a given factor."""
    colors = cmap(np.linspace(0, 1, cmap.N))
    hls_colors = [colorsys.rgb_to_hls(*color[:3]) for color in colors]
    desaturated_hls_colors = [(h, l, s * factor) for h, l, s in hls_colors]
    desaturated_rgb_colors = [
        colorsys.hls_to_rgb(h, l, s) for h, l, s in desaturated_hls_colors
    ]
    desaturated_cmap = mcolors.ListedColormap(
        desaturated_rgb_colors, name=f"{cmap.name}_desaturated"
    )
    return desaturated_cmap


def hls_colormap(N=1, lightness=0.9, saturation=1):
    """Create a hls colormap."""
    hls_colors = [(i / N, lightness, saturation) for i in range(N)]
    rgb_colors = [colorsys.hls_to_rgb(h, l, s) for h, l, s in hls_colors]
    hls_colormap = mcolors.ListedColormap(rgb_colors, name=f"hls_{lightness}_{N}")
    return hls_colormap


mask_colors = ["cyan", "magenta", "gold", "cyan"]
mask_colormap = mcolors.LinearSegmentedColormap.from_list("mask", mask_colors, N=64)


@contextlib.contextmanager
def set_style(new_style):
    """Set the style for the visualization."""
    global style
    original_style = style
    style = new_style
    try:
        yield
    finally:
        style = original_style


# Desaturate the HomeyerRainbow colormap
desaturated_homeyer_rainbow = desaturate_colormap(pcm.HomeyerRainbow, factor=0.35)

reflectivity_levels = np.arange(-10, 60 + 5, 5)
reflectivity_norm = mcolors.BoundaryNorm(
    reflectivity_levels, ncolors=desaturated_homeyer_rainbow.N, clip=True
)

pcolormesh_style = {
    "reflectivity": {
        "cmap": desaturated_homeyer_rainbow,
        "shading": "nearest",
        "norm": reflectivity_norm,
    },
}


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
figure_colors["gadi"] = figure_colors["presentation"]

base_styles = {
    "paper": "default",
    "presentation": "dark_background",
    "gadi": "dark_background",
}
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
            images = [Image.open(f).convert("RGBA") for f in filepaths_date]
            kwargs = {"duration": 200, "loop": 0}
            imageio.mimsave(output_filepath, images, **kwargs)
    else:
        output_filepath = save_directory / f"{animation_name}.gif"
        logger.info(f"Saving animation to {output_filepath}.")
        images = [Image.open(f).convert("RGBA") for f in filepaths]
        kwargs = {"duration": 200, "loop": 0}
        imageio.mimsave(output_filepath, images, **kwargs)


def get_grid(time, filename, field, data_options, grid_options):
    """
    Get the grid from a file.
    """
    grid = utils.load_grid(filename)
    return grid[field]
