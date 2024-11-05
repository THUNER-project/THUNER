"""Visualization convenience functions."""

import string
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


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
    lon = np.array(grid_options["longitude"])
    lat = np.array(grid_options["latitude"])

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
