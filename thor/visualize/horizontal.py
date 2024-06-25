"""Horizontal cross-section features."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
import thor.visualize.visualize as visualize


def grid(grid, fig, ax):
    """Plot a grid cross section."""

    pcm = grid.plot.pcolormesh(
        ax=ax,
        shading="nearest",
        zorder=0,
        **visualize.grid_formats[grid.name],
        extend="both",
    )

    return pcm


def mask(mask, fig, ax):
    """Plot masks."""

    cmap = plt.get_cmap("tab10")
    # ax.set_prop_cycle(color=colors)

    pcm = ((mask - 1) % 10 + 1).plot.pcolormesh(
        ax=ax,
        shading="nearest",
        alpha=1,
        zorder=1,
        add_colorbar=False,
        cmap=cmap,
        levels=np.arange(1, 11, 1),
    )

    return pcm


def initialize_cartographic_figure(
    nrows=1,
    ncols=1,
    style="paper",
    figsize=None,
    scale="110m",
    extent=(-180, 180, -90, 90),
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

    if figsize is None:
        figsize = (ncols * 6, nrows * 3)
    projection = ccrs.PlateCarree()

    style_path = Path(__file__).parent / "styles" / f"{style}.mplstyle"
    with plt.style.context([visualize.base_styles[style], style_path]):
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=figsize,
            subplot_kw={"projection": projection},
        )
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

        for ax in fig.get_axes():
            ax.add_feature(land, zorder=0)
            ax.add_feature(ocean, zorder=0)
            ax.add_feature(states_provinces, zorder=0)
            ax.coastlines(resolution=scale, zorder=1, color=colors["coast_color"])

            gridlines = initialize_gridlines(ax, extent=extent)

    return fig, axes, colors, gridlines


def initialize_gridlines(ax, extent):
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
