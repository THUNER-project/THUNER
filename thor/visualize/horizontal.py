"""Horizontal cross-section features."""

import numpy as np

import cv2
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch
import cartopy.feature as cfeature
from cartopy import crs as ccrs
import thor.visualize.visualize as visualize
from thor.log import setup_logger
from thor.object.box import get_box_coords

logger = setup_logger(__name__)


def grid(grid, ax, add_colorbar=True):
    """Plot a grid cross section."""

    title = ax.get_title()
    pcm = grid.plot.pcolormesh(
        ax=ax,
        shading="nearest",
        zorder=1,
        **visualize.grid_formats[grid.attrs["long_name"].lower()],
        add_colorbar=add_colorbar,
        extend="both",
        transform=ccrs.PlateCarree(),
    )
    ax.set_title(title)

    return pcm


def mask(mask, ax):
    """Plot masks."""

    title = ax.get_title()
    colors = visualize.mask_colors

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
            transform=ccrs.PlateCarree(),
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
                linewidth=1,
                zorder=3,
                transform=ccrs.PlateCarree(),
            )
    ax.set_title(title)
    return colors


def add_radar_features(ax, radar_lon, radar_lat, extent, input_record):
    """Add radar features to an ax."""
    alpha = 0.8
    ax.plot(
        [radar_lon, radar_lon],
        [extent[2], extent[3]],
        color="tab:red",
        linewidth=1,
        alpha=alpha,
        linestyle="--",
        zorder=1,
        transform=ccrs.PlateCarree(),
    )
    ax.plot(
        [extent[0], extent[1]],
        [radar_lat, radar_lat],
        color="tab:red",
        linewidth=1,
        alpha=alpha,
        linestyle="--",
        zorder=1,
        transform=ccrs.PlateCarree(),
    )
    try:
        ax.plot(
            input_record["range_longitudes"],
            input_record["range_latitudes"],
            color="tab:red",
            linewidth=1,
            alpha=alpha,
            linestyle="--",
            zorder=1,
            transform=ccrs.PlateCarree(),
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
    gridlines.xlocator = mticker.FixedLocator(
        np.arange(-180, 180 + grid_spacing, grid_spacing)
    )
    gridlines.ylocator = mticker.FixedLocator(
        np.arange(-90, 90 + grid_spacing, grid_spacing)
    )
    ax.set_extent(extent)

    return gridlines


def get_domain_center(grid):
    if "instrument" in grid.attrs.keys() and "radar" in grid.attrs["instrument"]:
        center_lon = float(grid.attrs["origin_longitude"])
        center_lat = float(grid.attrs["origin_latitude"])
    else:
        latitudes = grid.latitude.values
        longitudes = grid.longitude.values
        center_lon = longitudes[len(longitudes) // 2]
        center_lat = latitudes[len(longitudes) // 2]
    return center_lat, center_lon


arrow_options = {"arrowstyle": "->", "linewidth": 1, "mutation_scale": 7, "zorder": 3}
arrow_origin_options = {"marker": "o", "zorder": 3, "markersize": 2}


def get_flow_scale(lats, lons):
    """
    Scale flow vectors so that a geographic flow of 1 gridcell corresponds to 1/50 of the
    domain.
    """
    return 0.01 * (lats[-1] - lats[0]) / (lats[1] - lats[0])


def plot_flow(
    ax, start_lat, start_lon, flow, grid_options, color="w", alpha=1, linestyle="-"
):
    """Plot a flow type vector."""
    flow_scale = get_flow_scale(grid_options["latitude"], grid_options["longitude"])
    geographic_flow = np.array(flow) * np.array(grid_options["geographic_spacing"])
    geographic_flow = np.array(geographic_flow) * flow_scale
    start_coords = [start_lon, start_lat]
    end_coords = np.array(start_coords) + geographic_flow[::-1]
    ax.plot(
        start_lon,
        start_lat,
        color=color,
        alpha=alpha,
        **arrow_origin_options,
        transform=ccrs.PlateCarree(),
    )
    arrow = FancyArrowPatch(
        start_coords,
        end_coords,
        color=color,
        alpha=alpha,
        **arrow_options,
        linestyle=linestyle,
        transform=ccrs.PlateCarree(),
    )
    ax.add_patch(arrow)


def plot_displacement(displacement, center, ax, latitudes, longitudes, grid_options):
    if np.all(np.isnan(displacement)):
        return
    lat = latitudes[center[0]]
    lon = longitudes[center[1]]
    plot_flow(ax, lat, lon, displacement, grid_options, color="silver")


def plot_box(ax, box, grid, color, linestyle="--", alpha=1):
    latitudes = grid.latitude.values
    longitudes = grid.longitude.values
    box_lats, box_lons = get_box_coords(box, latitudes, longitudes)
    ax.plot(
        box_lons,
        box_lats,
        color=color,
        linewidth=1,
        linestyle=linestyle,
        alpha=alpha,
        transform=ccrs.PlateCarree(),
    )
