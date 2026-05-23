"""Methods for visualizing analyses of thuner output."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import thuner.visualize.utils as utils
import thuner.visualize.visualize as visualize
import thuner.visualize.horizontal as horizontal
import networkx as nx

__all__ = ["windrose", "windrose_legend"]


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
    ax.bar(
        heading,
        speed,
        normed=True,
        opening=0.8,
        edgecolor=edgecolor,
        linewidth=1,
        blowto=False,
        bins=bins,
        cmap=colormap,
    )
    if yticks is not None:
        ax.set_yticks(yticks)
        tick_labels = [t + "%" for t in yticks.astype(str)]
        ax.set_yticklabels(
            tick_labels,
            verticalalignment=verticalalignment,
            horizontalalignment=horizontalalignment,
        )
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
    handles = [
        Patch(
            facecolor=colors[i],
            linewidth=1,
            edgecolor=edgecolor,
        )
        for i in range(len(labels))
    ]
    handles, labels = utils.reorder_legend_entries(handles, labels, columns=columns)
    return legend_ax.legend(
        handles,
        labels,
        ncol=columns,
        fancybox=True,
        shadow=True,
    )


def parent_graph(output_directory, parent_graph, ax=None, style="presentation"):
    """Visualize a parent graph."""

    # Convert parent graph nodes to ints for visualization
    parent_graph_int = nx.convert_node_labels_to_integers(
        parent_graph,
        label_attribute="label",
    )
    label_dict = {}
    time_dict = {}
    for node in parent_graph_int.nodes(data=True):
        label_dict[node[0]] = node[1]["label"][1]
        time_dict[node[0]] = node[1]["label"][0]

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 16))

    if style == "presentation":
        background_color = "k"
        edge_color = "w"
        node_color = "#333333"
    else:
        background_color = "w"
        edge_color = "k"
        node_color = "lightgray"

    pos = nx.drawing.nx_pydot.graphviz_layout(parent_graph_int, prog="dot")
    nx.draw(
        parent_graph_int,
        pos,
        node_size=150,
        with_labels=False,
        arrows=True,
        ax=ax,
        node_color=node_color,
        edge_color=edge_color,
    )
    ax.collections[0].set_edgecolor(edge_color)
    font = plt.rcParams["font.family"]
    font_size = "6"
    fig.set_facecolor(background_color)
    ax.set_facecolor(background_color)
    nx.draw_networkx_labels(
        parent_graph_int,
        pos,
        font_family=font,
        font_size=font_size,
        labels=label_dict,
        verticalalignment="center",
        horizontalalignment="center",
        ax=ax,
        font_color=edge_color,
    )

    filepath = output_directory / "visualize/split_merge_graph.png"
    plt.savefig(filepath, bbox_inches="tight", dpi=200)
    plt.close()
