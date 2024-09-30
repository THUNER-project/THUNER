"""Methods for visualizing object attributes and classifications."""

import copy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import thor.visualize.horizontal as horizontal
from thor.visualize.utils import get_extent, make_subplot_labels

proj = ccrs.PlateCarree()


def mcs_horizontal_template(
    grid, figure_options, extent, figsize, convective_label="cell", anvil_label="anvil"
):
    """Create a template figure for mcs visualisation."""
    member_objects = [convective_label, anvil_label]
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
        ax = horizontal.add_cartographic_features(
            ax, extent=extent, style=style, scale="10m", left_labels=(i == 0)
        )[0]
        grid_i = grid[f"{member_objects[i]}_grid"]
        if (
            "instrument" in grid_i.attrs.keys()
            and "radar" in grid_i.attrs["instrument"]
        ):
            radar_longitude = float(grid_i.attrs["origin_longitude"])
            radar_latitude = float(grid_i.attrs["origin_latitude"])
            horizontal.add_radar_features(ax, radar_longitude, radar_latitude, extent)
        ax.set_title(member_objects[i].replace("_", " ").title())
    cbar_ax = fig.add_subplot(gs[0, -1])
    make_subplot_labels(axes, x_shift=-0.12, y_shift=0.06)
    return fig, axes, cbar_ax


def mcs_horizontal(
    grid,
    mask,
    boundaries,
    figure_options,
    grid_options,
    convective_label="cell",
    anvil_label="anvil",
):
    """Create a horizontal cross section plot."""
    extent = get_extent(grid_options)
    member_objects = [convective_label, anvil_label]

    if "template" not in figure_options.keys():
        fig, ax = mcs_horizontal_template(grid, figure_options, extent)
        figure_options["template"] = fig

    fig = copy.deepcopy(figure_options["template"])
    ax = fig.axes[0]

    pcm = horizontal.grid(grid, ax, grid_options, add_colorbar=False)
    horizontal.mask(mask, ax, grid_options)
    horizontal.add_domain_boundary(ax, boundaries)

    cbar_label = grid.name.title() + f" [{grid.units}]"
    fig.colorbar(pcm, label=cbar_label)
    ax.set_title(f"{grid.time.values.astype('datetime64[s]')} UTC")

    fig = copy.deepcopy(figure_options["template"])
    axes = fig.axes[:-1]
    cbar_ax = fig.axes[-1]

    for i in range(len(member_objects)):
        ax = axes[i]
        grid_i = grid[f"{member_objects[i]}_grid"]
        mask_i = mask[f"{member_objects[i]}_mask"]
        pcm = horizontal.grid(grid_i, ax, grid_options, add_colorbar=False)
        if mask_i is not None:
            horizontal.mask(mask_i, ax, grid_options)
        if boundaries is not None:
            horizontal.add_domain_boundary(ax, boundaries)

    cbar_label = grid_i.attrs["long_name"].title() + f" [{grid_i.attrs['units']}]"
    fig.colorbar(pcm, cax=cbar_ax, label=cbar_label)
    fig.suptitle(f"{grid.time.values.astype('datetime64[s]')} UTC", y=1.05)
