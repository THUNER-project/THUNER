"""Horizontal cross-section features."""

import copy
import numpy as np
import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patheffects as patheffects
import matplotlib.lines as mlines
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import BoundaryNorm
import cartopy.feature as cfeature
from cartopy import crs as ccrs
import thor.visualize.visualize as visualize
from thor.visualize.utils import get_extent, make_subplot_labels
from thor.log import setup_logger
from thor.object.box import get_geographic_box_coords
import thor.grid as thor_grid

logger = setup_logger(__name__)
# Set the number of cv2 threads to 0 to avoid crashes.
# See https://github.com/opencv/opencv/issues/5150#issuecomment-675019390
cv2.setNumThreads(0)

proj = ccrs.PlateCarree()
domain_plot_style = {"color": "tab:red", "linewidth": 1, "alpha": 0.8}
domain_plot_style.update({"zorder": 1, "transform": proj, "linestyle": "--"})


def show_grid(grid, ax, grid_options, add_colorbar=True):
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


def show_mask(mask, ax, grid_options, single_color=False):
    """Plot masks."""

    title = ax.get_title()
    colors = visualize.mask_colors
    if single_color:
        colors = [colors[0]] * len(colors)
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
        ax.pcolormesh(LON, LAT, pcm_mask, **mesh_style)
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


def domain_boundary_legend_artist():
    """Create a legend artist for a domain boundary."""
    legend_artist = mlines.Line2D([], [], **domain_plot_style)
    legend_artist.set_label("Domain Boundary")
    return legend_artist


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

    colors = visualize.figure_colors[style]
    ocean = cfeature.NaturalEarthFeature(
        "physical", "ocean", scale, edgecolor="face", facecolor=colors["sea"]
    )
    land = cfeature.NaturalEarthFeature(
        "physical", "land", scale, edgecolor="face", facecolor=colors["land"]
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
    ax.coastlines(resolution=scale, zorder=1, color=colors["coast"])
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


displacement_linewidth = 3
head_width = 0.01  # Specified in percent of x limits
head_length = 0.0125  # Specified in percent of x limits
ellipse_axis_linewidth = 1


def percent_to_data(ax, percent):
    """Get the percentage of x limits."""
    x_min, x_max = ax.get_xlim()
    data = percent * (x_max - x_min)
    return data


vector_options = {"color": "w", "zorder": 5, "head_width": head_width}
vector_options.update({"head_length": head_length})
vector_options.update({"linewidth": displacement_linewidth / 3})
vector_options.update({"length_includes_head": True, "transform": proj})


def displacement_legend_artist(color, label):
    """Create a legend artist for a displacement provided in cartesian coordinates."""
    linewidth = displacement_linewidth
    path_effects = [
        patheffects.Stroke(linewidth=linewidth, foreground=color),
        patheffects.Normal(),
    ]
    args_dict = {"color": "w", "linewidth": linewidth / 3, "linestyle": "-"}
    args_dict.update({"zorder": 4, "transform": proj, "path_effects": path_effects})
    legend_artist = mlines.Line2D([], [], **args_dict)
    legend_artist.set_label(label)
    legend_artist.set_path_effects(path_effects)
    return legend_artist


def vector_key(ax, u=-10, v=0, color="k", dt=3600):
    """Add a vector key to the plot."""
    fig = ax.get_figure()
    start_point = fig.transFigure.transform((0.875, 1))
    [longitude, latitude] = ax.transData.inverted().transform(start_point)
    longitude = longitude % 360
    args = [ax, latitude, longitude, u, v, color, None]
    cartesian_velocity(*args, quality=True, dt=dt)
    start_point = fig.transFigure.transform((0.89, 1))
    [longitude, latitude] = ax.transData.inverted().transform(start_point)
    ax.text(longitude, latitude, f"{np.abs(u)} m/s", ha="left", va="center")


def ellipse_axis(
    ax, latitude, longitude, axis_length, orientation, label, style, quality=True
):
    """Display an ellipse axis."""
    azimuth = (90 - np.rad2deg(orientation)) % 360
    args = [longitude, latitude, azimuth, axis_length * 1e3 / 2]
    lon_1, lat_1 = thor_grid.geodesic_forward(*args)[:2]
    args[2] = (azimuth + 180) % 360
    lon_2, lat_2 = thor_grid.geodesic_forward(*args)[:2]

    colors = visualize.figure_colors[style]
    axis_color = colors["ellipse_axis"]
    shadow_color = colors["ellipse_axis_shadow"]
    args_dict = {"shadow_color": shadow_color, "alpha": 0.9}
    args_dict.update({"offset": (ellipse_axis_linewidth, -ellipse_axis_linewidth)})
    path_effects = [patheffects.SimpleLineShadow(**args_dict), patheffects.Normal()]
    args_dict = {"color": axis_color, "linewidth": ellipse_axis_linewidth}
    args_dict.update({"zorder": 3, "path_effects": path_effects, "transform": proj})
    args_dict.update({"linestyle": "--"})
    if quality:
        ax.plot([lon_1, lon_2], [lat_1, lat_2], **args_dict)
    args_dict = {"color": axis_color, "linewidth": ellipse_axis_linewidth}
    args_dict.update({"zorder": 5, "path_effects": path_effects, "linestyle": "--"})
    args_dict.update({"label": label})
    legend_handle = mlines.Line2D([], [], **args_dict)
    return legend_handle


def ellipse_legend_artist(label, style):
    """Create a legend artist for an ellipse axis."""
    colors = visualize.figure_colors[style]
    axis_color = colors["ellipse_axis"]
    shadow_color = colors["ellipse_axis_shadow"]
    args_dict = {"shadow_color": shadow_color, "alpha": 0.9}
    path_effects = [patheffects.SimpleLineShadow(**args_dict), patheffects.Normal()]
    args_dict = {"color": axis_color, "linewidth": ellipse_axis_linewidth}
    args_dict.update({"zorder": 3, "path_effects": path_effects, "linestyle": "--"})
    args_dict.update({"label": label})
    legend_handle = mlines.Line2D([], [], **args_dict)
    return legend_handle


def cartesian_displacement(
    ax, start_latitude, start_longitude, dx, dy, color, label, quality=True, arrow=True
):
    """Plot a displacement provided in cartesian coordinates."""
    linewidth = displacement_linewidth
    distance = np.sqrt(dx**2 + dy**2)
    vector_direction = np.rad2deg(np.arctan2(dy, dx))
    # Now convert to azimuth direction, i.e. clockwise from north.
    azimuth = (90 - vector_direction) % 360
    args = [start_longitude, start_latitude, azimuth, distance]
    end_longitude, end_latitude = thor_grid.geodesic_forward(*args)[:2]
    # Ensure that the end longitude is within the range [0, 360).
    end_longitude = end_longitude % 360
    dlon = end_longitude - start_longitude
    dlat = end_latitude - start_latitude

    args = [start_longitude, start_latitude, dlon, dlat]
    path_effects = [
        patheffects.Stroke(linewidth=linewidth, foreground=color),
        patheffects.Normal(),
    ]
    args_dict = {"path_effects": path_effects}
    tmp_vector_options = copy.deepcopy(vector_options)
    if not arrow:
        tmp_vector_options.update({"head_width": 0, "head_length": 0})
    else:
        width = tmp_vector_options["head_width"]
        length = tmp_vector_options["head_length"]
        new_width = percent_to_data(ax, width)
        new_length = percent_to_data(ax, length)
        tmp_vector_options.update({"head_width": new_width, "head_length": new_length})
    if quality:
        ax.arrow(*args, **tmp_vector_options, **args_dict, clip_on=False)

    return ax


def cartesian_velocity(
    ax, start_latitude, start_longitude, u, v, color, label, dt=3600, quality=True
):
    """Plot a velocity provided in cartesian coordinates."""

    # Scale velocities so they represent the displacement after dt seconds
    dx, dy = u * dt, v * dt
    args = [ax, start_latitude, start_longitude, dx, dy, color, label, quality]
    return cartesian_displacement(*args)


def pixel_vector(
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


def detected_mask_template(grid, figure_options, extent):
    """Create a template figure for masks."""
    fig = plt.figure(figsize=(6, 3.5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    add_cartographic_features(
        ax, style=figure_options["style"], scale="10m", extent=extent
    )
    if "instrument" in grid.attrs.keys() and "radar" in grid.attrs["instrument"]:
        radar_longitude = float(grid.attrs["origin_longitude"])
        radar_latitude = float(grid.attrs["origin_latitude"])
        add_radar_features(ax, radar_longitude, radar_latitude, extent)
    return fig, ax


def detected_mask(grid, mask, grid_options, figure_options, boundary_coordinates):
    """Plot masks for a detected object."""

    extent = get_extent(grid_options)
    single_color = figure_options["single_color"]
    if figure_options["template"] is None:
        fig, ax = detected_mask_template(grid, figure_options, extent)
        figure_options["template"] = fig
    fig = copy.deepcopy(figure_options["template"])
    ax = fig.axes[0]
    if grid is not None:
        pcm = show_grid(grid, ax, grid_options, add_colorbar=False)
    if mask is not None:
        show_mask(mask, ax, grid_options, single_color)
    if boundary_coordinates is not None:
        add_domain_boundary(ax, boundary_coordinates)
    cbar_label = grid.name.title() + f" [{grid.units}]"
    fig.colorbar(pcm, label=cbar_label)
    ax.set_title(f"{grid.time.values.astype('datetime64[s]')} UTC")

    return fig, ax


def grouped_mask_template(grid, figure_options, extent, figsize, member_objects):
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
        args_dict = {"extent": extent, "style": style, "scale": "10m"}
        args_dict.update({"left_labels": (i == 0)})
        ax = add_cartographic_features(ax, **args_dict)[0]
        ax.set_title(member_objects[i].replace("_", " ").title())
        if grid is None:
            continue
        grid_i = grid[f"{member_objects[i]}_grid"]
        if (
            "instrument" in grid_i.attrs.keys()
            and "radar" in grid_i.attrs["instrument"]
        ):
            radar_longitude = float(grid_i.attrs["origin_longitude"])
            radar_latitude = float(grid_i.attrs["origin_latitude"])
            add_radar_features(ax, radar_longitude, radar_latitude, extent)
    cbar_ax = fig.add_subplot(gs[0, -1])
    make_subplot_labels(axes, x_shift=-0.12, y_shift=0.06)
    return fig, axes, cbar_ax


def grouped_mask(
    grid, mask, grid_options, figure_options, member_objects, boundary_coordinates
):
    """Plot masks for a grouped object."""

    extent = get_extent(grid_options)
    single_color = figure_options["single_color"]

    try:
        figsize = figure_options["figsize"]
    except KeyError:
        logger.info("No figsize provided. Using default.")
        figsize = (len(member_objects) * 4, 3.5)

    if figure_options["template"] is None:
        args = [grid, figure_options, extent, figsize, member_objects]
        fig, axes, cbar_ax = grouped_mask_template(*args)
        figure_options["template"] = fig

    fig = copy.deepcopy(figure_options["template"])
    axes = fig.axes[:-1]
    cbar_ax = fig.axes[-1]

    pcm = None
    for i in range(len(member_objects)):
        ax = axes[i]
        mask_i = mask[f"{member_objects[i]}_mask"]
        if mask_i is not None:
            show_mask(mask_i, ax, grid_options, single_color)
        if boundary_coordinates is not None:
            add_domain_boundary(ax, boundary_coordinates)
        if grid is None:
            continue
        grid_i = grid[f"{member_objects[i]}_grid"]
        if grid_i is not None:
            pcm = show_grid(grid_i, ax, grid_options, add_colorbar=False)
    if pcm is not None and grid is not None:
        cbar_label = grid_i.attrs["long_name"].title() + f" [{grid_i.attrs['units']}]"
        fig.colorbar(pcm, cax=cbar_ax, label=cbar_label)
    fig.suptitle(f"{mask.time.values.astype('datetime64[s]')} UTC", y=1.05)

    return fig, axes
