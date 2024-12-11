"""Methods for visualizing analyses of thuner output."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import thuner.visualize.utils as utils


def windrose(
    ax,
    u: float,
    v: float,
    bins,
    yticks=None,
    label_angle=112.5,
    colormap=None,
    verticalalignment="top",
    horizontalalignment="right",
):
    """Cretae a windrose style figure."""
    speed = np.sqrt(u**2 + v**2)
    direction = np.rad2deg(np.arctan2(v, u))
    heading = (90 - direction) % 360
    if colormap is None:
        colormap = plt.get_cmap("Spectral_r", len(bins))
    edgecolor = plt.rcParams["axes.edgecolor"]
    kwargs = {"normed": True, "opening": 0.8, "edgecolor": edgecolor}
    kwargs.update({"linewidth": 1, "blowto": False, "bins": bins, "cmap": colormap})
    ax.bar(heading, speed, **kwargs)
    if yticks is not None:
        ax.set_yticks(yticks)
        tick_labels = [t + "%" for t in yticks.astype(str)]
        kwargs = {"verticalalignment": verticalalignment}
        kwargs.update({"horizontalalignment": horizontalalignment})
        ax.set_yticklabels(tick_labels, **kwargs)
    ax.set_rlabel_position(label_angle)

    return ax


def windrose_legend(legend_ax, bins, colormap=None, units="m/s", columns=2):
    """Create a legend for a windrose style figure."""
    colors = colormap(np.linspace(0, 1, len(bins)))
    labels = []
    for i in range(len(bins) - 1):
        labels.append(f"[{bins[i]} {units}, {bins[i+1]} {units})")
    labels.append(f"[{bins[-1]} {units}, " + r"$\infty$)")
    edgecolor = plt.rcParams["axes.edgecolor"]
    kwargs = {"linewidth": 1, "edgecolor": edgecolor}
    handles = [Patch(facecolor=colors[i], **kwargs) for i in range(len(labels))]
    kwargs = {"ncol": columns, "fancybox": True, "shadow": True}
    handles, labels = utils.reorder_legend_entries(handles, labels, columns=columns)
    return legend_ax.legend(handles, labels, **kwargs)
