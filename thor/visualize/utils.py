"""Visualization convenience functions."""

import string
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


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

    # Quick fix for plotting big grids
    # Rescale to ensure equal ranges
    if lon_range > lat_range:
        lat_range = lon_range
    else:
        lon_range = lat_range

    lon_center = lon.mean()
    lat_center = lat.mean()

    lon_min = lon_center - lon_range / 2
    lon_max = lon_center + lon_range / 2
    lat_min = np.max([-90, lat_center - lat_range / 2])
    lat_max = np.min([90, lat_center + lat_range / 2])

    return (lon_min, lon_max, lat_min, lat_max)


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
