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
    return (lon.min(), lon.max(), lat.min(), lat.max())


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
