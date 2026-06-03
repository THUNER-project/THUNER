"""Horizontal cross-section features."""

from itertools import product
from typing import List, Any, Literal
import copy
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patheffects as patheffects
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.legend_handler import HandlerTuple
import cartopy.feature as cfeature
from cartopy import crs as ccrs
import thuner.visualize.visualize as visualize
from thuner.visualize.utils import get_extent, Panelled
from thuner.log import setup_logger
from thuner.match.box import get_geographic_box_coords
import thuner.grid as thuner_grid

logger = setup_logger(__name__)
# Set the number of cv2 threads to 0 to avoid crashes.
# See https://github.com/opencv/opencv/issues/5150#issuecomment-675019390
cv2.setNumThreads(0)

proj = ccrs.PlateCarree()
domain_plot_style = {"color": "tab:red", "linewidth": 1, "alpha": 0.6}
domain_plot_style.update({"zorder": 1, "transform": proj, "linestyle": "-"})


def show_grid(grid, ax, grid_options, add_colorbar=True):
    """Plot a grid cross section."""

    # pcolormesh accepts the 1-D coords of a geographic grid directly (it meshes them
    # internally) and the 2-D curvilinear coords of a cartesian grid as-is.
    longitude, latitude = grid_options.longitude, grid_options.latitude

    title = ax.get_title()
    mesh_style = visualize.pcolormesh_style[grid.attrs["field_name"].lower()]
    mesh_style["transform"] = proj
    pcm = ax.pcolormesh(longitude, latitude, grid.values, zorder=1, **mesh_style)
    ax.set_title(title)
    if add_colorbar:
        cbar = plt.colorbar(pcm, ax=ax, orientation="vertical")
    return pcm


contour_options = {"mode": cv2.RETR_LIST, "method": cv2.CHAIN_APPROX_SIMPLE}


def show_mask(
    mask, ax, grid_options, single_color=False, object_colors=None, mask_quality=None
):
    """
    Visualize masks. object_colors should be a dictionary mapping object labels to colors.
    If object_colors is None, default colors are used, with indexing based on
    the object id."""

    title = ax.get_title()
    colors = visualize.runtime_colors
    if single_color:
        colors = [colors[0]] * len(colors)

    mask_values = mask.values
    object_labels = np.unique(mask_values)
    object_labels = object_labels[object_labels > 0].astype(np.int32)

    longitude, latitude = grid_options.longitude, grid_options.latitude
    if mask_quality is None:
        mask_quality = {obj: True for obj in object_labels}

    # Fill all objects with a single QuadMesh rather than one (mostly transparent)
    # pcolormesh per object: each kept object gets a contiguous index into a colormap
    # of its colour. Outlines are still drawn per object.
    fill = np.full(mask_values.shape, np.nan)
    fill_colors = []
    for i in object_labels:
        if not mask_quality[i]:
            continue
        is_object = mask_values == i
        if object_colors is not None:
            color = object_colors[i]
        else:
            color = colors[int((i - 1) % len(colors))]
        fill[is_object] = len(fill_colors) + 1
        fill_colors.append(color)

        contours = cv2.findContours(is_object.astype(np.uint8), **contour_options)[0]
        for contour in contours:
            contour = np.append(contour, [contour[0]], axis=0)
            cols = contour[:, :, 0].flatten()
            rows = contour[:, :, 1].flatten()
            lats, lons = thuner_grid.get_pixels_geographic(rows, cols, grid_options)
            ax.plot(lons, lats, color=color, linewidth=1, zorder=3, transform=proj)

    if fill_colors:
        cmap = mcolors.ListedColormap(fill_colors)
        norm = mcolors.BoundaryNorm(np.arange(0.5, len(fill_colors) + 1.5), cmap.N)
        mesh_style = {"shading": "nearest", "transform": proj, "alpha": 0.4, "zorder": 2}
        ax.pcolormesh(longitude, latitude, fill, cmap=cmap, norm=norm, **mesh_style)
    ax.set_title(title)
    return colors


def mask_legend_artist(single_color=False):
    """Create a legend artist for masks."""
    colors = visualize.mask_colors
    single_color = False
    if single_color:
        colors = [colors[0]] * len(colors)

    patches = []
    for i in range(3):
        edge_color = mcolors.to_rgb(colors[i])
        fill_color = list(edge_color) + [0.4]  # Add alpha of .4
        patch = mpatches.Rectangle(
            (0, 0),
            1,
            1,
            edgecolor=edge_color,
            facecolor=fill_color,
        )
        patches.append(patch)

    handler_map = {tuple: HandlerTuple(ndivide=None)}
    return tuple(patches), handler_map


def box_legend_artist(single_color=False, color=None, linestyle="--"):
    """Create a legend artist for boxes."""
    colors = visualize.mask_colors
    if single_color and color is not None:
        colors = [color] * len(colors)

    patches = []
    for i in range(3):
        edge_color = mcolors.to_rgb(colors[i])
        fill_color = list(edge_color) + [0]  # Make transparent
        patch = mpatches.Rectangle(
            (0, 0),
            1,
            1,
            edgecolor=edge_color,
            linewidth=1,
            linestyle=linestyle,
            facecolor=fill_color,
        )
        patches.append(patch)

    handler_map = {tuple: HandlerTuple(ndivide=None)}
    return tuple(patches), handler_map


def radar_features(ax, radar_lon, radar_lat, extent):
    """Add radar features to an ax."""
    ax.plot([radar_lon, radar_lon], [extent[2], extent[3]], **domain_plot_style)
    ax.plot([extent[0], extent[1]], [radar_lat, radar_lat], **domain_plot_style)
    return ax


def embossed_text(ax, text, longitude, latitude):
    """Add embossed text to an ax."""
    extent = ax.get_extent(crs=proj)
    dlon = np.abs(extent[1] - extent[0])
    offset = dlon / 50
    path_effects = [patheffects.Stroke(linewidth=1.5, foreground="k")]
    path_effects += [patheffects.Normal()]
    fontsize = int(plt.rcParams["font.size"] / 2)
    artist = ax.text(
        longitude - offset,
        latitude + offset,
        text,
        transform=proj,
        zorder=5,
        fontweight="bold",
        color="w",
        path_effects=path_effects,
        fontsize=fontsize,
    )
    artist.set_clip_on(True)
    artist.set_clip_box(ax.bbox)
    return artist


def domain_boundary(ax, boundaries, grid_options):
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


def cartographic_features(
    ax,
    scale="110m",
    extent=(-180, 180, -90, 90),
    left_labels=True,
    bottom_labels=True,
    border_zorder=0,
    coastline_zorder=1,
    grid_zorder=1,
):
    """
    Add cartographic features to an ax. 
    """

    colors = visualize.figure_colors[visualize.style]
    ocean = cfeature.NaturalEarthFeature(
        "physical",
        "ocean",
        scale,
        edgecolor="face",
        facecolor=colors["sea"],
    )
    land = cfeature.NaturalEarthFeature(
        "physical", "land", scale, edgecolor="face", facecolor=colors["land"]
    )
    state_scale = min([int(scale.replace("m", "")), 50])
    state_scale = f"{state_scale}m"
    states_provinces = cfeature.NaturalEarthFeature(
        scale=state_scale,
        category="cultural",
        name="admin_1_states_provinces_lines",
        facecolor="none",
        edgecolor="gray",
    )
    national_borders = cfeature.BORDERS.with_scale(scale)

    ax.add_feature(land, zorder=0)
    ax.add_feature(ocean, zorder=0)
    ax.add_feature(states_provinces, zorder=border_zorder, alpha=0.6)
    ax.add_feature(national_borders, zorder=border_zorder, edgecolor=colors["coast"])
    ax.coastlines(resolution=scale, zorder=coastline_zorder, color=colors["coast"])
    gridlines = initialize_gridlines(
        ax,
        extent=extent,
        left_labels=left_labels,
        bottom_labels=bottom_labels,
        zorder=grid_zorder,
    )

    return ax, colors, gridlines


def initialize_gridlines(ax, extent, left_labels=True, bottom_labels=True, zorder=1):
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
    grid_style.update({"zorder": zorder})
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
    elif delta_grid / spacing > 10:
        spacing *= 2
    gridlines.xlocator = mticker.FixedLocator(np.arange(-180, 180 + spacing, spacing))
    gridlines.ylocator = mticker.FixedLocator(np.arange(-90, 90 + spacing, spacing))

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
    lats, lons = grid_options.latitude, grid_options.longitude
    if grid_options.name == "cartesian":
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
    legend_artist = mlines.Line2D(
        [],
        [],
        color="w",
        linewidth=linewidth / 3,
        linestyle="-",
        zorder=4,
        transform=proj,
        path_effects=path_effects,
    )
    legend_artist.set_label(label)
    legend_artist.set_path_effects(path_effects)
    return legend_artist


def vector_key(ax, u=-10, v=0, color="k", dt=3600, scale=1):
    """Add a vector key to the plot."""
    fig = ax.get_figure()
    if scale == 1:
        y_position = 0.95
        x_position = 0.92
    elif scale == 2:
        y_position = 0.925
        x_position = 0.875
    start_point = fig.transFigure.transform((x_position, y_position))
    [longitude, latitude] = ax.transData.inverted().transform(start_point)
    longitude = longitude % 360
    cartesian_velocity(
        ax=ax,
        start_latitude=latitude,
        start_longitude=longitude,
        u=u,
        v=v,
        color=color,
        quality=True,
        dt=dt,
        clip=False,
    )

    start_point = fig.transFigure.transform((x_position + 0.015, y_position))
    [longitude, latitude] = ax.transData.inverted().transform(start_point)
    ax.text(longitude, latitude, f"{np.abs(u)} m/s", ha="left", va="center")


def ellipse_axis(ax, latitude, longitude, axis_length, orientation, quality=True):
    """Display an ellipse axis."""
    azimuth = (90 - np.rad2deg(orientation)) % 360
    lon_1, lat_1 = thuner_grid.geodesic_forward(
        longitude, latitude, azimuth, axis_length * 1e3 / 2
    )[:2]
    lon_2, lat_2 = thuner_grid.geodesic_forward(
        longitude, latitude, (azimuth + 180) % 360, axis_length * 1e3 / 2
    )[:2]

    colors = visualize.figure_colors[visualize.style]
    axis_color = colors["ellipse_axis"]
    shadow_color = colors["ellipse_axis_shadow"]
    offset = (0.85 * ellipse_axis_linewidth / 2, -0.85 * ellipse_axis_linewidth)
    path_effects = [
        patheffects.SimpleLineShadow(
            shadow_color=shadow_color,
            alpha=0.9,
            offset=offset,
        ),
        patheffects.Normal(),
    ]
    if quality:
        artist = ax.plot(
            [lon_1, lon_2],
            [lat_1, lat_2],
            color=axis_color,
            linewidth=ellipse_axis_linewidth,
            zorder=3,
            path_effects=path_effects,
            transform=proj,
            linestyle="--",
        )
    else:
        artist = None
    return artist


def orientation_legend_artist(label, style):
    """Create a legend artist for an ellipse axis."""
    colors = visualize.figure_colors[style]
    axis_color = colors["ellipse_axis"]
    shadow_color = colors["ellipse_axis_shadow"]
    offset = (0.85 * ellipse_axis_linewidth, -0.85 * ellipse_axis_linewidth)
    path_effects = [
        patheffects.SimpleLineShadow(
            shadow_color=shadow_color,
            alpha=0.9,
            offset=offset,
        ),
        patheffects.Normal(),
    ]
    legend_handle = mlines.Line2D(
        [],
        [],
        color=axis_color,
        linewidth=ellipse_axis_linewidth,
        zorder=3,
        path_effects=path_effects,
        linestyle="--",
        label=label,
    )
    return legend_handle


def pixel_displacement(
    ax,
    row,
    col,
    vector,
    grid_options,
    start_lat=None,
    start_lon=None,
    color="w",
):
    """Plot a vector given in gridcell, i.e. "pixel", coordinates."""
    latitudes = grid_options.latitude
    longitudes = grid_options.longitude
    if grid_options.name == "cartesian":
        if start_lat is None or start_lon is None:
            start_lon = longitudes[row, col]
            start_lat = latitudes[row, col]
        dy, dx = vector * np.array(grid_options.cartesian_spacing)
    elif grid_options.name == "geographic":
        if start_lat is None or start_lon is None:
            start_lat = latitudes[row]
            start_lon = longitudes[col]
        dlat, dlon = vector * np.array(grid_options.geographic_spacing)
        end_lat, end_lon = start_lat + dlat, start_lon + dlon
        forward_dir, backward_dir, distance = thuner_grid.geodesic_inverse(
            start_lon, start_lat, end_lon, end_lat
        )
        # Convert azimuth to vector direction from east, measured counter-clockwise as normal
        # in cartesian coordinates.
        forward_dir = (90 - forward_dir) % 360
        dy = distance * np.sin(np.deg2rad(forward_dir))
        dx = distance * np.cos(np.deg2rad(forward_dir))
    else:
        raise ValueError(f"Grid name must be 'cartesian' or 'geographic'.")
    cartesian_displacement(
        ax,
        start_latitude=start_lat,
        start_longitude=start_lon,
        dx=dx,
        dy=dy,
        color=color,
        quality=True,
    )


def cartesian_displacement(
    ax,
    start_latitude,
    start_longitude,
    dx,
    dy,
    color,
    quality=True,
    arrow=True,
    reverse=False,
    clip=True,
):
    """Plot a displacement provided in cartesian coordinates."""
    linewidth = displacement_linewidth
    distance = np.sqrt(dx**2 + dy**2)
    vector_direction = np.rad2deg(np.arctan2(dy, dx))
    # Now convert to azimuth direction, i.e. clockwise from north.
    azimuth = (90 - vector_direction) % 360
    end_longitude, end_latitude = thuner_grid.geodesic_forward(
        start_longitude, start_latitude, azimuth, distance
    )[:2]
    # Ensure that the end longitude is within the range [0, 360).
    end_longitude = end_longitude % 360
    dlon = end_longitude - start_longitude
    dlat = end_latitude - start_latitude
    if reverse:
        kwargs = {
            "x": start_longitude + dlon,
            "y": start_latitude + dlat,
            "dx": -dlon,
            "dy": -dlat,
        }
    else:
        kwargs = {"x": start_longitude, "y": start_latitude, "dx": dlon, "dy": dlat}
    path_effects = [
        patheffects.Stroke(linewidth=linewidth, foreground=color),
        patheffects.Normal(),
    ]
    tmp_vector_options = copy.deepcopy(vector_options)
    if not arrow:
        tmp_vector_options.update({"head_width": 0, "head_length": 0})
    else:
        width = tmp_vector_options["head_width"]
        length = tmp_vector_options["head_length"]
        new_width = percent_to_data(ax, width)
        new_length = percent_to_data(ax, length)
        tmp_vector_options.update({"head_width": new_width, "head_length": new_length})
    if clip:
        clip_options = {"clip_on": True, "clip_box": ax.bbox}
    else:
        clip_options = {"clip_on": False}
    if quality:
        artist = ax.arrow(
            **kwargs,
            **tmp_vector_options,
            **clip_options,
            path_effects=path_effects,
        )
    else:
        artist = None

    return artist


def cartesian_velocity(
    ax,
    start_latitude,
    start_longitude,
    u,
    v,
    color,
    dt=3600,
    quality=True,
    clip=True,
    reverse=False,
):
    """Plot a velocity provided in cartesian coordinates."""

    # Scale velocities so they represent the displacement after dt seconds
    dx, dy = u * dt, v * dt
    return cartesian_displacement(
        ax=ax,
        start_latitude=start_latitude,
        start_longitude=start_longitude,
        dx=dx,
        dy=dy,
        color=color,
        quality=quality,
        clip=clip,
        reverse=reverse,
    )


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
    latitudes = grid_options.latitude
    longitudes = grid_options.longitude
    if grid_options.name == "cartesian":
        if start_lat is None or start_lon is None:
            start_lon = longitudes[row, col]
            start_lat = latitudes[row, col]
        # Convert a vector [row, col] to azimuth direction. First get direction
        # counter-clockwise from east.
        vector_direction = np.rad2deg(np.arctan2(vector[0], vector[1]))
        # Now convert to azimuth direction, i.e. clockwise from north.
        azimuth = (90 - vector_direction) % 360
        spacing = np.array(grid_options.cartesian_spacing)
        cartesian_vector = np.array(vector) * spacing
        distance = np.sqrt(np.sum(cartesian_vector**2))
        end_lon, end_lat = thuner_grid.geodesic_forward(
            start_lon, start_lat, azimuth, distance
        )[:2]
        geographic_vector = [end_lat - start_lat, end_lon - start_lon]
    elif grid_options.name == "geographic":
        if start_lat is None or start_lon is None:
            start_lat = latitudes[row]
            start_lon = longitudes[col]
        geographic_vector = np.array(vector) * np.array(grid_options.geographic_spacing)
    else:
        raise ValueError(f"Grid name must be 'cartesian' or 'geographic'.")
    scale = get_geographic_vector_scale(grid_options)
    geographic_vector = np.array(geographic_vector) * scale
    start_coords = [start_lon, start_lat]
    end_coords = np.array(start_coords) + geographic_vector[::-1]
    ax.plot(start_lon, start_lat, color=color, alpha=alpha, **arrow_origin_options)
    vector_style = {"color": color, "alpha": alpha, "linestyle": linestyle}
    kwargs = {**vector_style, **arrow_options}
    arrow = mpatches.FancyArrowPatch(start_coords, end_coords, **kwargs)
    ax.add_patch(arrow)


def plot_box(
    ax, box, grid_options, linestyle="--", alpha=1, color="tab:red", linewidth=1
):
    lats, lons = get_geographic_box_coords(box, grid_options)
    box_style = {"color": color, "linewidth": linewidth, "linestyle": linestyle}
    box_style.update({"alpha": alpha, "transform": proj})
    ax.plot(lons, lats, **box_style)


def detected_mask_template(grid, figure_options, extent, scale):
    """Create a template figure for masks."""
    rows, columns = 1, 1

    if scale == 1:
        subplot_width = 4  # Default subplot width in inches
    elif scale == 2:
        subplot_width = 8

    layout = PanelledUniformMaps(
        extent=extent,
        subplot_width=subplot_width,
        rows=rows,
        columns=columns,
        colorbar=True,
        legend_rows=2,
        shared_legends="all",
    )
    fig, subplot_axes, colorbar_axes, legend_axes = layout.initialize_layout()

    object_name = figure_options.object_name

    ax = subplot_axes[0]
    ax.set_title(object_name.replace("_", " ").title(), y=1)
    if grid is not None:
        keys = grid.attrs.keys()
        if "instrument" in keys and "radar" in grid.attrs["instrument"]:
            radar_longitude = float(grid.attrs["origin_longitude"])
            radar_latitude = float(grid.attrs["origin_latitude"])
            radar_features(ax, radar_longitude, radar_latitude, extent)
    return fig, subplot_axes, colorbar_axes, legend_axes, layout


def detected_mask(
    grid,
    mask,
    grid_options,
    figure_options,
    boundary_coordinates,
    object_colors=None,
    mask_quality=None,
):
    """Plot masks for a detected object."""

    extent, scale = get_extent(grid_options)
    single_color = figure_options.single_color
    if figure_options.template is None:
        figure_options.template = detected_mask_template(
            grid=grid,
            figure_options=figure_options,
            extent=extent,
            scale=scale,
        )
    template = copy.deepcopy(figure_options.template)
    [fig, subplot_axes, colorbar_axes, legend_axes, layout] = template
    ax = fig.axes[0]
    pcm, colorbar_label = None, None
    if grid is not None:
        pcm = show_grid(grid, ax, grid_options, add_colorbar=False)
        colorbar_label = grid.attrs["field_name"].replace("_", " ").title()
        colorbar_label += f" [{grid.attrs['units']}]"
    if mask is not None:
        show_mask(mask, ax, grid_options, single_color, object_colors, mask_quality)
    if boundary_coordinates is not None:
        domain_boundary(ax, boundary_coordinates, grid_options)
    if pcm is not None and colorbar_label is not None:
        fig.colorbar(pcm, cax=colorbar_axes[0], label=colorbar_label)

    ax.set_title(f"{grid.time.values.astype('datetime64[s]')} UTC")
    ax.set_extent(extent, crs=proj)

    return fig, subplot_axes, colorbar_axes, legend_axes


def grouped_mask_template(grid, extent, member_objects, scale):
    """Create a template figure for grouped masks."""
    if scale == 1:
        rows = 1
        columns = len(member_objects)
        subplot_width = 4  # Default subplot width in inches
    elif scale == 2:
        rows = len(member_objects)
        columns = 1
        subplot_width = 8
    else:
        raise ValueError("Only scales of 1 or 2 implemented so far.")
    layout = PanelledUniformMaps(
        extent=extent,
        subplot_width=subplot_width,
        rows=rows,
        columns=columns,
        colorbar=True,
        legend_rows=2,
        shared_legends="all",
    )
    fig, subplot_axes, colorbar_axes, legend_axes = layout.initialize_layout()
    for i in range(len(member_objects)):
        ax = subplot_axes[i]
        ax.set_title(member_objects[i].replace("_", " ").title(), y=1)
        if grid is None:
            continue
        grid_i = grid[f"{member_objects[i]}_grid"]
        keys = grid_i.attrs.keys()
        if "instrument" in keys and "radar" in grid_i.attrs["instrument"]:
            radar_longitude = float(grid_i.attrs["origin_longitude"])
            radar_latitude = float(grid_i.attrs["origin_latitude"])
            radar_features(ax, radar_longitude, radar_latitude, extent)
    return fig, subplot_axes, colorbar_axes, legend_axes, layout


def grouped_mask(
    grid,
    mask,
    grid_options,
    figure_options,
    member_objects,
    boundary_coordinates,
    object_colors=None,
    mask_quality=None,
):
    """Plot masks for a grouped object."""

    extent, scale = get_extent(grid_options)
    single_color = figure_options.single_color

    if figure_options.template is None:
        figure_options.template = grouped_mask_template(
            grid=grid,
            extent=extent,
            member_objects=member_objects,
            scale=scale,
        )

    template = copy.deepcopy(figure_options.template)
    [fig, subplot_axes, colorbar_axes, legend_axes, layout] = template
    pcm, colorbar_label = None, None
    for i in range(len(member_objects)):
        ax = subplot_axes[i]
        mask_i = mask[f"{member_objects[i]}_mask"]
        if mask_i is not None:
            show_mask(
                mask=mask_i,
                ax=ax,
                grid_options=grid_options,
                single_color=single_color,
                object_colors=object_colors,
                mask_quality=mask_quality,
            )
        if boundary_coordinates is not None:
            domain_boundary(ax, boundary_coordinates, grid_options)
        if grid is None:
            continue
        grid_i = grid[f"{member_objects[i]}_grid"]
        if grid_i is not None:
            pcm = show_grid(grid_i, ax, grid_options, add_colorbar=False)
            colorbar_label = grid_i.attrs["field_name"].replace("_", " ").title()
            colorbar_label += f" [{grid_i.attrs['units']}]"
        ax.set_extent(extent, crs=proj)
    if pcm is not None and colorbar_label is not None:
        for ax in colorbar_axes:
            fig.colorbar(pcm, cax=ax, label=colorbar_label)
    title = f"{mask.time.values.astype('datetime64[s]')} UTC"
    fig.suptitle(title, y=layout.suptitle_height)

    return fig, subplot_axes, colorbar_axes, legend_axes


class PanelledUniformMaps(Panelled):
    """Class to handle layout of cartopy maps with uniform extent."""

    def __init__(
        self,
        # Extent of the figure [lon_min, lon_max, lat_min, lat_max]
        extent: List[float],
        subplot_width: float = 5.75,  # Width of the subplots in inches
        rows: int = 1,  # Number of rows in the figure
        columns: int = 1,  # Number of columns in the figure
        # Spacing between subplots as fraction of subplot width
        horizontal_spacing: float = 0.3,  # Spacing between subplots in inches
        vertical_spacing: float = 0.6,  # Spacing between subplots in inches
        colorbar: bool = True,  # Add a colorbar to the figure
        legend_rows: int | None = 2,  # Number of rows in the legend
        shared_legends: Literal["columns", "all", None] = None,  # Share legends
        projections: Any | List[Any] | None = proj,  # Projections for each subplot
        label_offset_x: float = -0.12,
        label_offset_y: float = 0.06,
        border_zorder: int = 0,  # Political border zorder
        coastline_zorder: int = 1,  # Coastline zorder
        grid_zorder: int = 1,  # Grid zorder
    ):
        lat_range = extent[3] - extent[2]
        lon_range = extent[1] - extent[0]
        subplot_height = (lat_range / lon_range) * subplot_width
        super().__init__(
            subplot_width=subplot_width,
            subplot_height=subplot_height,
            rows=rows,
            columns=columns,
            horizontal_spacing=horizontal_spacing,
            vertical_spacing=vertical_spacing,
            colorbar=colorbar,
            legend_rows=legend_rows,
            shared_legends=shared_legends,
            projections=projections,
            label_offset_x=label_offset_x,
            label_offset_y=label_offset_y,
        )
        self.extent = extent
        self.colorbar = colorbar
        self.legend_rows = legend_rows
        self.suptitle_height = 1
        self.border_zorder = border_zorder
        self.coastline_zorder = coastline_zorder
        self.grid_zorder = grid_zorder

    # Redefine the initialize_layout method to handle cartopy projections
    def initialize_layout(self):
        """Initialize the figure."""
        self.initialize_gridspec()

        colorbar_axes = []
        subplot_axes = []
        legend_axes = []

        dlon = self.extent[1] - self.extent[0]
        # Choose cartopy scale based on dlon
        if dlon < 10:
            scale = "10m"
        elif dlon < 20:
            scale = "50m"
        else:
            scale = "110m"
        # Looping over self.rows and self.columns rather than rows, columns
        # ignores possible colorbar columns and possible legend row

        if self.shared_legends == "columns" or self.shared_legends == None:
            subplot_rows = range(0, 2 * self.rows, 2)
        else:
            subplot_rows = range(self.rows)

        # Initialize cartographic_features kwargs
        # Create maps
        for i, j in product(range(self.rows), range(self.columns)):
            subplot_row = subplot_rows[i]
            prj = self.projections[i, j]
            ax = self.fig.add_subplot(self.grid_spec[subplot_row, j], projection=prj)
            ax.set_rasterized(True)
            ax = cartographic_features(
                ax,
                border_zorder=self.border_zorder,
                grid_zorder=self.grid_zorder,
                coastline_zorder=self.coastline_zorder,
                scale=scale,
                extent=self.extent,
                left_labels=j == 0,
                bottom_labels=i == self.rows - 1,
            )[0]
            ax.set_extent(self.extent, crs=proj)
            subplot_axes.append(ax)

        colorbar_axes, legend_axes = self.initialize_legend(subplot_axes)

        return self.fig, subplot_axes, colorbar_axes, legend_axes
