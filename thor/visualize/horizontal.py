"""Horizontal cross-section features."""

import numpy as np

import cv2
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import BoundaryNorm
import cartopy.feature as cfeature
from cartopy import crs as ccrs
import thor.visualize.visualize as visualize
from thor.log import setup_logger
from thor.object.box import get_geographic_box_coords
import thor.grid as thor_grid

logger = setup_logger(__name__)

proj = ccrs.PlateCarree()
domain_plot_style = {"color": "tab:red", "linewidth": 1, "alpha": 0.8}
domain_plot_style.update({"zorder": 1, "transform": proj, "linestyle": "--"})


def grid(grid, ax, grid_options, add_colorbar=True):
    """Plot a grid cross section."""

    if grid_options["name"] == "geographic":
        LON, LAT = np.meshgrid(grid_options["longitude"], grid_options["latitude"])
    elif grid_options["name"] == "cartesian":
        LON, LAT = grid_options["longitude"], grid_options["latitude"]

    title = ax.get_title()
    mesh_style = visualize.pcolormesh_style[grid.attrs["long_name"].lower()]
    mesh_style["transform"] = proj
    pcm = ax.pcolormesh(LON, LAT, grid.values, zorder=1, **mesh_style)
    ax.set_title(title)
    if add_colorbar:
        cbar = plt.colorbar(pcm, ax=ax, orientation="vertical")
    return pcm


contour_options = {"mode": cv2.RETR_LIST, "method": cv2.CHAIN_APPROX_SIMPLE}


def mask(mask, ax, grid_options, single_color=False):
    """Plot masks."""

    title = ax.get_title()
    colors = visualize.mask_colors
    if single_color:
        colors = [colors[0]]
    cmap = LinearSegmentedColormap.from_list("custom", colors, N=len(colors))
    object_labels = np.unique(mask.where(mask > 0).values)
    object_labels = object_labels[~np.isnan(object_labels)]

    if grid_options["name"] == "geographic":
        LON, LAT = np.meshgrid(grid_options["longitude"], grid_options["latitude"])
    elif grid_options["name"] == "cartesian":
        LON, LAT = grid_options["longitude"], grid_options["latitude"]

    levels = np.arange(0, len(colors) + 1)
    norm = BoundaryNorm(levels, ncolors=len(colors), clip=True)
    mesh_style = {"shading": "nearest", "transform": proj, "cmap": cmap, "alpha": 0.4}
    mesh_style.update({"zorder": 2, "norm": norm})
    for i in object_labels:
        binary_mask = mask.where(mask == i, 0).astype(bool)
        color_index = (i - 1) % len(colors)
        pcm_mask = (color_index * binary_mask).where(binary_mask)
        pcm = ax.pcolormesh(LON, LAT, pcm_mask, **mesh_style)
        binary_array = binary_mask.values.astype(np.uint8)
        contours = cv2.findContours(binary_array, **contour_options)[0]
        for contour in contours:
            contour = np.append(contour, [contour[0]], axis=0)
            cols = contour[:, :, 0].flatten()
            rows = contour[:, :, 1].flatten()
            lats, lons = thor_grid.get_pixels_geographic(rows, cols, grid_options)
            color = colors[int(color_index)]
            ax.plot(lons, lats, color=color, linewidth=1, zorder=3, transform=proj)
    ax.set_title(title)
    return colors


def add_radar_features(ax, radar_lon, radar_lat, extent):
    """Add radar features to an ax."""
    ax.plot([radar_lon, radar_lon], [extent[2], extent[3]], **domain_plot_style)
    ax.plot([extent[0], extent[1]], [radar_lat, radar_lat], **domain_plot_style)
    return ax


def add_domain_boundary(ax, boundaries):
    """Add domain boundary to an ax."""
    logger.debug("Plotting boundary.")
    for boundary in boundaries:
        lons = boundary["longitude"]
        lats = boundary["latitude"]
        ax.plot(lons, lats, **domain_plot_style)
    return ax


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

    grid_style = {"draw_labels": True, "linewidth": 1, "color": "gray", "alpha": 0.4}
    grid_style.update({"linestyle": "--", "x_inline": False, "y_inline": False})
    gridlines = ax.gridlines(**grid_style)

    gridlines.right_labels = False
    gridlines.top_labels = False
    gridlines.left_labels = left_labels
    gridlines.bottom_labels = bottom_labels

    text_color = plt.rcParams["text.color"]
    label_style = {"rotation": 0, "font": font, "color": text_color}
    gridlines.xlabel_style = {"ha": "center", **label_style}
    gridlines.ylabel_style = {**label_style}

    delta_grid = np.max([extent[1] - extent[0], extent[3] - extent[2]])
    spacing = (10 ** np.floor(np.log10(delta_grid))) / 2
    if spacing < 1:
        spacing = 1
    gridlines.xlocator = mticker.FixedLocator(np.arange(-180, 180 + spacing, spacing))
    gridlines.ylocator = mticker.FixedLocator(np.arange(-90, 90 + spacing, spacing))
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


arrow_options = {"arrowstyle": "->", "linewidth": 1, "mutation_scale": 7}
arrow_options.update({"zorder": 3, "transform": proj})
arrow_origin_options = {"marker": "o", "zorder": 3, "markersize": 1, "transform": proj}


def get_geographic_vector_scale(grid_options):
    """
    Scale vectors so that a vector of 1 gridcell corresponds to 1/50 of the domain.
    """
    lats, lons = grid_options["latitude"], grid_options["longitude"]
    if grid_options["name"] == "cartesian":
        lats = lats[:, 0]
        lons = lons[0, :]
    row_scale = 0.02 * (lats[-1] - lats[0]) / (lats[1] - lats[0])
    col_scale = 0.02 * (lons[-1] - lons[0]) / (lons[1] - lons[0])
    vector_scale = np.min([row_scale, col_scale])
    return vector_scale


def plot_vector(
    ax,
    row,
    col,
    vector,
    grid_options,
    start_lat=None,
    start_lon=None,
    color="w",
    alpha=1,
    linestyle="-",
):
    """Plot a vector given in gridcell, i.e. "pixel", coordinates."""
    latitudes = grid_options["latitude"]
    longitudes = grid_options["longitude"]
    if grid_options["name"] == "cartesian":
        if start_lat is None or start_lon is None:
            start_lon = longitudes[row, col]
            start_lat = latitudes[row, col]
        # Convert a vector [row, col] to azimuth direction. First get direction
        # counter-clockwise from east.
        vector_direction = np.rad2deg(np.arctan2(vector[0], vector[1]))
        # Now convert to azimuth direction, i.e. clockwise from north.
        azimuth = (90 - vector_direction) % 360
        spacing = np.array(grid_options["cartesian_spacing"])
        cartesian_vector = np.array(vector) * spacing
        distance = np.sqrt(np.sum(cartesian_vector**2))
        end_lon, end_lat = thor_grid.geodesic_forward(
            start_lon, start_lat, azimuth, distance
        )[:2]
        geographic_vector = [end_lat - start_lat, end_lon - start_lon]
    elif grid_options["name"] == "geographic":
        if start_lat is None or start_lon is None:
            start_lat = latitudes[row]
            start_lon = longitudes[col]
        geographic_vector = np.array(vector) * np.array(
            grid_options["geographic_spacing"]
        )
    else:
        raise ValueError(f"Grid name must be 'cartesian' or 'geographic'.")
    scale = get_geographic_vector_scale(grid_options)
    geographic_vector = np.array(geographic_vector) * scale
    start_coords = [start_lon, start_lat]
    end_coords = np.array(start_coords) + geographic_vector[::-1]
    ax.plot(start_lon, start_lat, color=color, alpha=alpha, **arrow_origin_options)
    vector_style = {"color": color, "alpha": alpha, "linestyle": linestyle}
    arrow = FancyArrowPatch(start_coords, end_coords, **vector_style, **arrow_options)
    ax.add_patch(arrow)


def plot_box(ax, box, grid_options, linestyle="--", alpha=1, color="tab:red"):
    lats, lons = get_geographic_box_coords(box, grid_options)
    box_style = {"color": color, "linewidth": 1, "linestyle": linestyle}
    box_style.update({"alpha": alpha, "transform": proj})
    ax.plot(lons, lats, **box_style)
