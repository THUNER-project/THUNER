"""Horizontal cross-section features."""

import numpy as np

import cv2
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
import cartopy.feature as cfeature
import thor.visualize.visualize as visualize
from thor.log import setup_logger

logger = setup_logger(__name__)


def grid(grid, ax, add_colorbar=True):
    """Plot a grid cross section."""

    pcm = grid.plot.pcolormesh(
        ax=ax,
        shading="nearest",
        zorder=1,
        **visualize.grid_formats[grid.name],
        add_colorbar=add_colorbar,
        extend="both",
    )

    return pcm


def mask(mask, ax):
    """Plot masks."""

    colors = ["purple", "teal", "cyan", "magenta", "saddlebrown"]

    try:
        row_vals = mask.latitude.values
        col_vals = mask.longitude.values
    except AttributeError:
        row_vals = mask.y.values
        col_vals = mask.x.values

    cmap = LinearSegmentedColormap.from_list("custom", colors, N=len(colors))

    object_labels = np.unique(mask.where(mask > 0).values)
    object_labels = object_labels[~np.isnan(object_labels)]
    for i in object_labels:
        binary_mask = mask.where(mask == i, 0).astype(bool)
        color_index = (i - 1) % len(colors)
        pcm_mask = (color_index * binary_mask).where(binary_mask)
        pcm = pcm_mask.plot.pcolormesh(
            ax=ax,
            shading="nearest",
            alpha=0.4,
            zorder=2,
            add_colorbar=False,
            cmap=cmap,
            levels=np.arange(0, len(colors) + 1),
        )
        contours = cv2.findContours(
            binary_mask.values.astype(np.uint8),
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE,
        )[0]
        for contour in contours:
            contour = np.append(contour, [contour[0]], axis=0)
            row_coords = row_vals[contour[:, :, 1]]
            col_coords = col_vals[contour[:, :, 0]]
            ax.plot(
                col_coords,
                row_coords,
                color=colors[int(color_index)],
                linewidth=1.5,
                zorder=3,
            )


def add_radar_features(ax, radar_lon, radar_lat, extent, input_record):
    """Add radar features to an ax."""
    ax.plot(
        [radar_lon, radar_lon],
        [extent[2], extent[3]],
        color="tab:red",
        linewidth=1,
        alpha=0.8,
        linestyle="--",
        zorder=1,
    )
    ax.plot(
        [extent[0], extent[1]],
        [radar_lat, radar_lat],
        color="tab:red",
        linewidth=1,
        alpha=0.8,
        linestyle="--",
        zorder=1,
    )
    try:
        ax.plot(
            input_record["range_longitudes"],
            input_record["range_latitudes"],
            color="tab:red",
            linewidth=1,
            alpha=0.8,
            linestyle="--",
            zorder=1,
        )
    except:
        logger.debug("No range rings to plot.")

    return


def add_cartographic_features(
    ax,
    style="paper",
    scale="110m",
    extent=(-180, 180, -90, 90),
    left_labels=True,
    bottom_labels=True,
):
    """
    Initialize a figure.

    Parameters
    ----------
    nrows : int, optional
        Number of rows in the figure.
    ncols : int, optional
        Number of columns in the figure.
    style : str, optional
        Style of the figure.
    figsize : tuple, optional
        Size of the figure.
    scale : str, optional
        Scale of the features.
    gridline_spacing : int, optional
        Spacing of the grid lines.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Axes object.
    """

    colors = visualize.map_colors[style]
    ocean = cfeature.NaturalEarthFeature(
        "physical", "ocean", scale, edgecolor="face", facecolor=colors["sea_color"]
    )
    land = cfeature.NaturalEarthFeature(
        "physical", "land", scale, edgecolor="face", facecolor=colors["land_color"]
    )
    states_provinces = cfeature.NaturalEarthFeature(
        category="cultural",
        name="admin_1_states_provinces_lines",
        scale=scale,
        facecolor="none",
        edgecolor="gray",
    )
    ax.add_feature(land, zorder=0)
    ax.add_feature(ocean, zorder=0)
    ax.add_feature(states_provinces, zorder=0)
    ax.coastlines(resolution=scale, zorder=1, color=colors["coast_color"])
    gridlines = initialize_gridlines(
        ax, extent=extent, left_labels=left_labels, bottom_labels=bottom_labels
    )

    return ax, colors, gridlines


def initialize_gridlines(ax, extent, left_labels=True, bottom_labels=True):
    """
    Initialize gridlines.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Axes object.
    gridline_spacing : int, optional
        Spacing of the grid lines.

    Returns
    -------
    gridlines : cartopy.mpl.gridliner.Gridliner
        Gridliner object.
    """

    if plt.rcParams["font.family"][0] == "sans-serif":
        font = plt.rcParams["font.sans-serif"][0]
    elif plt.rcParams["font.family"][0] == "serif":
        font = plt.rcParams["font.serif"][0]

    gridlines = ax.gridlines(
        draw_labels=True,
        x_inline=False,
        y_inline=False,
        linewidth=1,
        color="gray",
        alpha=0.4,
        linestyle="--",
    )

    gridlines.right_labels = False
    gridlines.top_labels = False
    gridlines.left_labels = left_labels
    gridlines.bottom_labels = bottom_labels

    gridlines.xlabel_style = {
        "rotation": 0,
        "ha": "center",
        "font": font,
        "color": plt.rcParams["text.color"],
    }
    gridlines.ylabel_style = {
        "rotation": 0,
        "font": font,
        "color": plt.rcParams["text.color"],
    }

    delta_grid = np.max([extent[1] - extent[0], extent[3] - extent[2]])
    grid_spacing = (10 ** np.floor(np.log10(delta_grid))) / 2
    lon_start = np.floor(extent[0] / grid_spacing) * grid_spacing
    lon_end = np.ceil(extent[1] / grid_spacing) * grid_spacing
    lat_start = np.floor(extent[2] / grid_spacing) * grid_spacing
    lat_end = np.ceil(extent[3] / grid_spacing) * grid_spacing

    gridlines.xlocator = mticker.FixedLocator(
        np.arange(lon_start, lon_end + grid_spacing, grid_spacing)
    )
    gridlines.ylocator = mticker.FixedLocator(
        np.arange(lat_start, lat_end + grid_spacing, grid_spacing)
    )

    return gridlines
