"""Visualization convenience functions."""

import string
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from itertools import product
from typing import Any, List, Literal


class BaseLayout:
    """Base class for figure layout."""

    def __init__(
        self,
        subplot_width: float = 4.0,  # Width of the subplots in inches
        subplot_height: float = 4.0,  # Height of the subplots in inches
        rows: int = 1,  # Number of rows in the figure
        columns: int = 1,  # Number of columns in the figure
        # Spacing between subplots as fraction of subplot width
        horizontal_spacing: float = 0.5,  # Estimated spacing between subplots in inches
        vertical_spacing: float = 0,  # Estimated spacing between subplots in inches
    ):
        self.subplot_width = subplot_width
        self.subplot_height = subplot_height
        self.rows = rows
        self.columns = columns
        self.horizontal_spacing = horizontal_spacing
        self.vertical_spacing = vertical_spacing
        self.figure_width = None
        self.figure_height = None

    def rescale_figure(self, fig, new_width):
        """Rescale the figure. Currently rought; could be improved."""
        if self.figure_width is None:
            raise ValueError("Layout not yet initialized.")
        aspect_ratio = self.figure_height / self.figure_width
        new_height = new_width * aspect_ratio
        fig.set_size_inches(new_width, new_height)
        fig.canvas.draw()
        self.figure_height = new_height
        self.figure_width = new_width


class Panelled(BaseLayout):
    """Class for basic panelled figure layouts."""

    def __init__(
        self,
        subplot_width: float = 4.0,  # Width of the subplots in inches
        subplot_height: float = 4.0,  # Height of the subplots in inches
        rows: int = 1,  # Number of rows in the figure
        columns: int = 1,  # Number of columns in the figure
        # Spacing between subplots as fraction of subplot width
        horizontal_spacing: float = 0.5,  # Estimated spacing between subplots in inches
        vertical_spacing: float = 0,  # Estimated spacing between subplots in inches)
        colorbar: bool = False,  # Add a colorbar to the figure
        legend_rows: int | None = None,  # Number of rows in the legend
        shared_legends: Literal["columns", "all", None] = None,  # Share legends
        projections: Any | List[Any] | None = None,  # Projections for each subplot
        label_offset_x: float = -0.12,
        label_offset_y: float = 0.06,
    ):
        super().__init__(
            subplot_width,
            subplot_height,
            rows,
            columns,
            horizontal_spacing,
            vertical_spacing,
        )
        self.label_offset_x = label_offset_x
        self.label_offset_y = label_offset_y
        self.colorbar = colorbar
        self.legend_rows = legend_rows
        self.shared_legends = shared_legends
        if not isinstance(projections, list):
            projections = [projections] * self.rows * self.columns
        projections = np.reshape(projections, (self.rows, self.columns))
        self.projections = projections

    def initialize_gridspec(self):
        """Initialize the gridspec."""
        width = self.subplot_width * self.columns
        width += self.horizontal_spacing * (self.columns - 1)
        height = self.subplot_height * self.rows
        height += self.vertical_spacing * (self.rows - 1)
        self.figure_width = width
        self.figure_height = height

        columns = self.columns
        width_ratios = [self.subplot_width] * self.columns
        if self.colorbar:
            # For now assume colorbars are oriented vertically at the right of each row
            columns = columns + 1
            # Assume colorbar axis width is 1 inch
            colorbar_width = 0.25
            width += colorbar_width
            width_ratios = width_ratios + [colorbar_width]

        rows = self.rows
        height_ratios = [self.subplot_height] * self.rows
        if self.legend_rows is not None:
            # Assume 1.5 spaced legend and add an .75 inch padding
            font_inches = plt.rcParams["font.size"] / 72
            legend_height = font_inches * (self.legend_rows + 0.5) * 1.5
            if self.rows == 1:
                legend_height += font_inches * 1 * 1.5
            height += legend_height
            if self.shared_legends == "all":
                rows = rows + 1
                height_ratios = height_ratios + [legend_height]
            else:
                rows = rows * 2
                # adjust height ratios to account for legend rows interleaved with
                # subplot rows
                legend_heights = [legend_height] * len(height_ratios)
                height_ratios = [
                    ratio
                    for pair in zip(height_ratios, legend_heights)
                    for ratio in pair
                ]

        wspace = self.horizontal_spacing / (sum(width_ratios) / len(width_ratios))
        hspace = self.vertical_spacing / (sum(height_ratios) / len(height_ratios))

        self.fig = plt.figure(figsize=(width, height))
        kwargs = {"width_ratios": width_ratios, "height_ratios": height_ratios}
        kwargs.update({"wspace": wspace, "hspace": hspace})
        self.grid_spec = gridspec.GridSpec(rows, columns, **kwargs)

    def initialize_layout(self):
        """Initialize the figure layout."""

        self.initialize_gridspec()

        subplot_axes = []

        # Looping over self.rows and self.columns rather than rows, columns
        # ignores possible colorbar columns and possible legend row
        if self.shared_legends == "columns" or self.shared_legends == None:
            subplot_rows = range(0, 2 * self.rows, 2)
        else:
            subplot_rows = range(self.rows)

        for i, j in product(range(self.rows), range(self.columns)):
            subplot_row = subplot_rows[i]
            proj = self.projections[i, j]
            ax = self.fig.add_subplot(self.grid_spec[subplot_row, j], projection=proj)
            subplot_axes.append(ax)

        colorbar_axes, legend_axes = self.initialize_legend(subplot_axes)
        return self.fig, subplot_axes, colorbar_axes, legend_axes

    def initialize_legend(self, subplot_axes):
        """Initialize the legend and other plot features."""

        colorbar_axes = []
        legend_axes = []

        if self.shared_legends == "columns" or self.shared_legends == None:
            legend_rows = range(1, 2 * self.rows, 2)
        else:
            legend_rows = [-1]

        if self.rows > 1 or self.columns > 1:
            kwargs = {"x_shift": self.label_offset_x, "y_shift": self.label_offset_y}
            make_subplot_labels(subplot_axes, **kwargs)
        if self.colorbar:
            for i in range(self.rows):
                ax = self.fig.add_subplot(self.grid_spec[i, -1])
                colorbar_axes.append(ax)
        if self.legend_rows is not None:
            if self.shared_legends == "all":
                leg_ax = self.fig.add_subplot(self.grid_spec[-1, :])
                leg_ax.axis("off")
                legend_axes = [leg_ax]
            elif self.shared_legends == "columns":
                legend_axes = []
                for i in range(self.rows):
                    legend_row = legend_rows[i]
                    leg_ax = self.fig.add_subplot(self.grid_spec[legend_row, :])
                    leg_ax.axis("off")
                    legend_axes.append(leg_ax)
            else:
                legend_axes = []
                for j in range(self.columns):
                    leg_ax = self.fig.add_subplot(self.grid_spec[-1, j])
                    leg_ax.axis("off")
                    legend_axes.append(leg_ax)
        if self.rows == 1:
            self.suptitle_height = 1
        else:
            self.suptitle_height = 0.935
        return colorbar_axes, legend_axes


def nice_number(value, round_number=False):
    """Get a number for defining axis ranges/ticks"""
    exponent = np.floor(np.log10(value))
    fraction = value / 10**exponent

    if round_number:
        if fraction < 1.5:
            nice_fraction = 1.0
        elif fraction < 2.5:
            nice_fraction = 2.0
        elif fraction < 6:
            nice_fraction = 5.0
        else:
            nice_fraction = 10.0
    else:
        if fraction <= 1:
            nice_fraction = 1.0
        elif fraction <= 2:
            nice_fraction = 2.0
        elif fraction <= 5:
            nice_fraction = 5.0
        else:
            nice_fraction = 10.0

    return nice_fraction * 10**exponent


def nice_bounds(axis_start, axis_end, num_ticks=10):
    """
    Get nice axis bounds and tick spacing for a given axis range.
    """
    axis_width = axis_end - axis_start
    if axis_width == 0:
        nice_tick = 0
    else:
        nice_range = nice_number(axis_width)
        nice_tick = nice_number(nice_range / (num_ticks - 1), round_number=True)
        axis_start = np.floor(axis_start / nice_tick) * nice_tick
        axis_end = np.ceil(axis_end / nice_tick) * nice_tick

    return axis_start, axis_end, nice_tick


def reorder_legend_entries(handles, labels, columns=2):
    """Reorder handles and labels for left-to-right, top-to-bottom order."""
    rows = np.ceil(len(handles) / columns).astype(int)
    new_order = []
    index = 0
    for i in range(len(handles)):
        index = columns * (i % rows) + i // rows
        new_order.append(index)
    reordered_handles = [handles[i] for i in new_order]
    reordered_labels = [labels[i] for i in new_order]
    return reordered_handles, reordered_labels


def reduce_color_depth(filepath, num_colors=256):
    """Reduce color depth of png image."""
    image = Image.open(filepath)
    image = image.convert("P", palette=Image.ADAPTIVE, colors=num_colors)
    image.save(filepath)


def get_extent(grid_options):
    """Get the cartopy extent."""
    lon = np.array(grid_options.longitude)
    lat = np.array(grid_options.latitude)

    lon_range = (lon.max() - lon.min()) * 1.1
    lat_range = (lat.max() - lat.min()) * 1.1

    lon_center = lon.mean()
    lat_center = lat.mean()

    scale = int(2 ** np.round(np.log2(lon_range / lat_range)))
    if scale == 2:
        nice_range = np.max([lon_range, 2 * lat_range])
        lon_min = lon_center - nice_range / 2
        lon_max = lon_center + nice_range / 2
        lat_min = lat_center - nice_range / 4
        lat_max = lat_center + nice_range / 4
    else:
        nice_range = np.max([lat_range, lon_range])
        lon_min = lon_center - nice_range / 2
        lon_max = lon_center + nice_range / 2
        lat_min = np.max([-90, lat_center - nice_range / 2])
        lat_max = np.min([90, lat_center + nice_range / 2])

    return (lon_min, lon_max, lat_min, lat_max), scale


def make_subplot_labels(axes, x_shift=-0.15, y_shift=0, fontsize=12):
    labels = list(string.ascii_lowercase)
    labels = [label + ")" for label in labels]
    for i in range(len(axes)):
        axes[i].text(
            x_shift,
            1.0 + y_shift,
            labels[i],
            transform=axes[i].transAxes,
            fontsize=plt.rcParams["axes.titlesize"],
        )
