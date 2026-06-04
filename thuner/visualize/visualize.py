"""General display functions."""

import random
import copy
from PIL import Image
import imageio
import colorsys
from pathlib import Path
import glob
import numpy as np
import contextlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import os

# Set the environment variable to turn off the pyart welcome message
os.environ["PYART_QUIET"] = "True"
from cmweather import cm_colorblind as pcm
import thuner.visualize.utils as utils
from thuner.log import setup_logger

logger = setup_logger(__name__)


style = "presentation"

__all__ = ["animate_object", "set_style"]


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


MASK_COLORS = ["cyan", "magenta", "gold", "cyan"]
MASK_COLORMAP = mcolors.LinearSegmentedColormap.from_list("mask", MASK_COLORS, N=64)
RUNTIME_COLORMAP = mcolors.LinearSegmentedColormap.from_list("mask", MASK_COLORS, N=12)
RUNTIME_COLORS = [RUNTIME_COLORMAP(i) for i in range(12)]
RUNTIME_COLORS = [mcolors.to_hex(color) for color in RUNTIME_COLORS]
random.seed(4189)
random.shuffle(RUNTIME_COLORS)


@contextlib.contextmanager
def set_style(new_style):
    """Custom style manager for matplotlib."""
    global style
    original_style = style
    style = new_style
    try:
        yield
    finally:
        style = original_style


# Desaturate the HomeyerRainbow colormap
DESATURATED_HOMEYER_RAINBOW = desaturate_colormap(pcm.HomeyerRainbow, factor=0.35)

REFLECTIVITY_LEVELS = np.arange(-10, 60 + 5, 5)
REFLECTIVITY_NORM = mcolors.BoundaryNorm(
    REFLECTIVITY_LEVELS, ncolors=DESATURATED_HOMEYER_RAINBOW.N, clip=True
)

BRIGHTNESS_RAINBOW = desaturate_colormap(pcm.HomeyerRainbow, factor=0.35)
BRIGHTNESS_RAINBOW.set_over((0, 0, 0, 0))  # Set over color to transparent
BRIGHTNESS_RAINBOW.set_under((0, 0, 0, 0))
BRIGHTNESS_LEVELS = np.arange(180, 250 + 10, 10)
BRIGHTNESS_NORM = mcolors.BoundaryNorm(
    BRIGHTNESS_LEVELS, ncolors=BRIGHTNESS_RAINBOW.N, clip=False
)

PCOLORMESH_STYLE = {
    "reflectivity": {
        "cmap": DESATURATED_HOMEYER_RAINBOW,
        "shading": "nearest",
        "norm": REFLECTIVITY_NORM,
    },
    "brightness_temperature": {
        "cmap": BRIGHTNESS_RAINBOW,
        "shading": "nearest",
        "norm": BRIGHTNESS_NORM,
    },
}


FIGURE_COLORS = {
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
FIGURE_COLORS["gadi"] = FIGURE_COLORS["presentation"]

BASE_STYLES = {
    "paper": "default",
    "presentation": "dark_background",
    "gadi": "dark_background",
}
CUSTOM_STYLES_DIR = Path(__file__).parent / "styles"

STYLES = {
    style: [BASE_STYLES[style], CUSTOM_STYLES_DIR / f"{style}.mplstyle"]
    for style in BASE_STYLES.keys()
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
    for obj_options in visualize_options.objects.values():
        for fig_options in obj_options.figures:
            if fig_options.animate:
                animate_object(fig_options.name, obj_options.name, output_directory)


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
            imageio.mimsave(
                output_filepath,
                images,
                duration=200,
                loop=0,
            )
    else:
        output_filepath = save_directory / f"{animation_name}.gif"
        logger.info(f"Saving animation to {output_filepath}.")
        images = [Image.open(f).convert("RGBA") for f in filepaths]
        imageio.mimsave(
            output_filepath,
            images,
            duration=200,
            loop=0,
        )
