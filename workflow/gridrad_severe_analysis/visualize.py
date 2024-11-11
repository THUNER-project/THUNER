from scipy.stats import circmean, circstd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import networkx as nx
import numpy as np
import pickle
import thor.visualize as visualize
import thor.config as config
from thor.log import setup_logger
import utils

logger = setup_logger(__name__)


# Create a dict of quality control attributes for each classification
# Note duration/parents are often appended to these for particular plots
core = ["convective_contained", "anvil_contained", "duration"]
quality_dispatcher = {
    "velocity": core + ["velocity"],
    "relative_velocity": core + ["relative_velocity"],
    "offset": core + ["offset"],
    "orientation": core + ["axis_ratio"],
    "stratiform_offset": core + ["offset", "velocity"],
    "relative_stratiform_offset": core + ["offset", "relative_velocity"],
    "inflow": core + ["velocity", "relative_velocity"],
    "tilt": core + ["offset", "shear"],
    "propagation": core + ["relative_velocity", "shear"],
    "offset_raw": core,
    "area": core,
}


def parent_graph(parent_graph, ax=None, analysis_directory=None):
    """Visualize a parent graph."""

    if analysis_directory is None:
        analysis_directory = get_analysis_directory()

    # Convert parent graph nodes to ints for visualization
    kwargs = {"label_attribute": "label"}
    parent_graph_int = nx.convert_node_labels_to_integers(parent_graph, **kwargs)
    label_dict = {}
    time_dict = {}
    for node in parent_graph_int.nodes(data=True):
        label_dict[node[0]] = node[1]["label"][1]
        time_dict[node[0]] = node[1]["label"][0]

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 16))

    pos = nx.drawing.nx_pydot.graphviz_layout(parent_graph_int, prog="dot")
    kwargs = {"node_size": 150, "with_labels": False, "arrows": True, "ax": ax}
    kwargs.update({"node_color": "lightgray"})
    nx.draw(parent_graph_int, pos, **kwargs)
    ax.collections[0].set_edgecolor("k")
    font = plt.rcParams["font.family"]
    font_size = "6"
    kwargs = {"font_family": font, "font_size": font_size, "labels": label_dict}
    kwargs.update({"verticalalignment": "center", "horizontalalignment": "center"})
    kwargs.update({"ax": ax})
    nx.draw_networkx_labels(parent_graph_int, pos, **kwargs)

    fig = ax.get_figure()
    fig_dict = {"fig": fig, "ax": ax}
    # Pickle the figure/ax handles for use in other figures
    with open(analysis_directory / "visualize/split_merge_graph.pkl", "wb") as f:
        pickle.dump(fig_dict, f)
    filepath = analysis_directory / "visualize/split_merge_graph.png"
    plt.savefig(filepath, bbox_inches="tight")


def get_analysis_directory():
    outputs_directory = config.get_outputs_directory()
    return outputs_directory / "runs/gridrad_severe/analysis"


def windrose(dfs, analysis_directory=None):
    """Create a windrose style plot of system velocity and stratiform offset."""

    if analysis_directory is None:
        analysis_directory = get_analysis_directory()

    quality = dfs["quality"]
    raw_sample = quality[["duration", "parents"]].any(axis=1)

    kwargs = {"subplot_width": 4, "rows": 1, "columns": 2, "projections": "windrose"}
    kwargs.update({"colorbar": False, "legend_rows": 5, "horizontal_spacing": 2})
    panelled_layout = visualize.horizontal.Panelled(**kwargs)
    fig, subplot_axes, colorbar_axes, legend_axes = panelled_layout.initialize_layout()

    names = quality_dispatcher["velocity"]
    quality = dfs["quality"][names].all(axis=1)
    values = []
    for v in ["u", "v"]:
        values.append(dfs["velocities"][v].where(quality & raw_sample).dropna().values)
    u, v = values
    bins = np.arange(5, 30, 5)
    yticks = np.arange(5, 30, 5)
    colormap = plt.get_cmap("Purples", len(bins))
    kwargs = {"bins": bins, "yticks": yticks, "colormap": colormap}
    kwargs.update({"label_angle": -22.5 - 45})
    kwargs.update({"horizontalalignment": "left", "verticalalignment": "top"})
    visualize.analysis.windrose(subplot_axes[0], u, v, **kwargs)
    subplot_axes[0].set_title("System Velocity")
    visualize.analysis.windrose_legend(legend_axes[0], bins, colormap, columns=2)

    names = quality_dispatcher["offset"]
    quality = dfs["quality"][names].all(axis=1)
    values = []
    for v in ["x_offset", "y_offset"]:
        values.append(dfs["group"][v].where(quality & raw_sample).dropna().values)
    x_offset, y_offset = values
    bins = np.arange(10, 60, 10)
    yticks = np.arange(5, 25, 5)
    colormap = plt.get_cmap("Blues", len(bins))
    kwargs = {"bins": bins, "yticks": yticks, "colormap": colormap}
    kwargs.update({"label_angle": -22.5 - 45, "yticks": yticks})
    kwargs.update({"horizontalalignment": "left", "verticalalignment": "top"})
    visualize.analysis.windrose(subplot_axes[1], x_offset, y_offset, **kwargs)
    subplot_axes[1].set_title("Stratiform Offset")
    kwargs = {"columns": 2, "units": "km"}
    visualize.analysis.windrose_legend(legend_axes[1], bins, colormap, **kwargs)
    visualize_directory = analysis_directory / "visualize"
    fig.savefig(visualize_directory / "vel_so_rose.png", bbox_inches="tight")
    fig_dict = {"fig": fig, "axes": subplot_axes, "legend_axes": legend_axes}
    # Pickle the figure/ax handles for use in other figures
    with open(visualize_directory / "velocity_stratiform_offset_roses.pkl", "wb") as f:
        pickle.dump(fig_dict, f)


prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]
color_dict = {"trailing": colors[0], "leading": colors[1]}
color_dict.update({"left": colors[2], "right": colors[4]})
color_dict.update({"front": colors[0], "rear": colors[1]})
color_dict.update({"up-shear": colors[0], "down-shear": colors[1]})
color_dict.update({"shear-perpendicular": colors[5]})
color_dict.update({"centered": colors[7], "ambiguous": colors[6]})


rel_so_formatter = lambda labels: [f"Relative {l.title()} Stratiform" for l in labels]
formatter_dispatcher = {
    "stratiform_offset": lambda labels: [f"{l.title()} Stratiform" for l in labels],
    "relative_stratiform_offset": rel_so_formatter,
    "inflow": lambda labels: [f"{l.title()} Fed" for l in labels],
    "tilt": lambda labels: [f"{l.title()} Tilted" for l in labels],
    "propagation": lambda labels: [f"{l.title()} Propagating" for l in labels],
}


def setup_grid(ax, dy, max_y, max_x=540, min_y=0, scientific=False, circular=False):
    """Set up the grid for time series style plots."""
    max_y = np.ceil(max_y / dy) * dy
    ax.set_ylim(min_y, max_y)
    ax.set_yticks(np.arange(min_y, max_y + dy, dy))
    ax.set_yticks(np.arange(min_y, max_y + dy / 2, dy / 2), minor=True)

    if scientific:
        formatter = mticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1, 1))
        ax.yaxis.set_major_formatter(formatter)

    if circular:
        labels = np.round(np.arange(min_y, max_y + dy, dy) * 180 / np.pi).astype(int)
        ax.set_yticklabels(labels)

    dx = 180
    ax.set_xlim(0, max_x)
    ax.set_xticks(np.arange(0, max_x + dx, dx))
    ax.set_xticks(np.arange(0, max_x + dx / 3, dx / 3), minor=True)
    ax.grid(True, which="major")
    ax.grid(True, which="minor", alpha=0.4)
    ax.set_xlabel("Time since Detection [min]")


def plot_counts_ratios(
    count_ax,
    ratio_ax,
    legend_ax,
    classifications,
    quality,
    classification_name="stratiform_offset",
    legend_formatter=None,
    legend_columns=2,
):
    """Plot counts and ratios of classifications over time."""

    if legend_formatter is None:
        legend_formatter = lambda labels: labels

    max_minutes = 720

    cond = quality[quality_dispatcher[classification_name] + ["duration"]].all(axis=1)
    classifications = classifications.copy()
    # Select values from with minutes index less than xmax
    classifications = classifications.where(cond).dropna()
    cond = classifications.index.get_level_values("minutes") <= max_minutes
    classifications = classifications[cond]

    group = classifications.groupby(["minutes", classification_name])
    counts = group[classification_name].apply(lambda x: x.count())
    ratios = counts / counts.groupby("minutes").transform("sum")

    dcount = 200
    dratio = 0.2

    unstacked = counts.unstack()
    unstacked.plot(ax=count_ax, legend=False, color=color_dict)
    max_count = unstacked.max().max()
    setup_grid(count_ax, dcount, max_count, max_minutes)
    count_ax.set_ylabel("Count [-]")

    unstacked = ratios.unstack()
    unstacked.plot(ax=ratio_ax, legend=False, color=color_dict)
    max_ratio = unstacked.dropna().max().max()
    setup_grid(ratio_ax, dratio, max_ratio, max_minutes)
    ratio_ax.set_ylabel("Ratio [-]")

    handles, labels = count_ax.get_legend_handles_labels()
    labels = legend_formatter(labels)
    kwargs = {"ncol": legend_columns, "facecolor": "white", "framealpha": 1}
    legend_ax.legend(handles, labels, **kwargs)


def plot_attribute(ax, df, quality, name, ylabel=None, circular=False):
    """Plot counts and ratios of classifications over time."""

    if "minutes" not in df.index.names:
        df = utils.get_duration_minutes(df)

    max_minutes = 720

    cond = quality[quality_dispatcher[name] + ["duration"]].all(axis=1)
    df = df.copy()
    # Select values from with minutes index less than xmax
    df = df.where(cond).dropna()
    cond = df.index.get_level_values("minutes") <= max_minutes
    df = df[cond]

    group = df.groupby(["minutes"])

    if circular:
        mean_value = group.apply(lambda x: circmean(x))
    else:
        mean_value = group.apply(lambda x: x.mean())
    minutes = mean_value.index.get_level_values("minutes").values
    mean_value = mean_value.values.flatten()
    if circular:
        std_dev = group.apply(lambda x: circstd(x)).values.flatten()
    else:
        std_dev = group.apply(lambda x: x.std()).values.flatten()
    max_value = (mean_value + std_dev).max()
    min_value = (mean_value - std_dev).min()

    ax.plot(minutes, mean_value, label="Mean")
    kwargs = {"alpha": 0.2, "label": "Standard Deviation"}
    ax.fill_between(minutes, mean_value - std_dev, mean_value + std_dev, **kwargs)
    if circular:
        args = [min_value * 180 / np.pi, max_value * 180 / np.pi, 10]
        min_value, max_value, dvalue = visualize.utils.nice_bounds(*args)
        min_value = min_value * np.pi / 180
        max_value = max_value * np.pi / 180
        dvalue = dvalue * np.pi / 180
    else:
        args = [min_value, max_value, 10]
        min_value, max_value, dvalue = visualize.utils.nice_bounds(*args)
    if not circular:
        min_value = max(0, min_value)
    scientific = False
    if np.log10(max_value) > 2:
        scientific = True
    kwargs = {"max_x": max_minutes, "min_y": min_value, "scientific": scientific}
    kwargs.update({"circular": circular})
    setup_grid(ax, dvalue, max_value, **kwargs)

    if ylabel is not None:
        ax.set_ylabel(ylabel)


def plot_classification_evolution(classifications, quality, analysis_directory=None):
    """Plot the evolution of classifications over time."""

    if analysis_directory is None:
        analysis_directory = get_analysis_directory()
    if "minutes" not in classifications.index.names:
        classifications = utils.get_duration_minutes(classifications)
    if "minutes" not in quality.index.names:
        quality = utils.get_duration_minutes(quality)

    # Relabel the classifications for plotting
    classifications = classifications.copy()

    layout_kwargs = {"subplot_width": 5, "subplot_height": 2.5, "rows": 3, "columns": 2}
    layout_kwargs.update({"colorbar": False, "legend_rows": 6, "vertical_spacing": 0.6})
    layout_kwargs.update({"shared_legends": "columns", "horizontal_spacing": 1})
    layout_kwargs.update({"label_offset_x": -0.175, "label_offset_y": 0.05})
    panelled_layout = visualize.horizontal.Panelled(**layout_kwargs)
    fig, subplot_axes, colorbar_axes, legend_axes = panelled_layout.initialize_layout()

    classification_names = ["stratiform_offset", "relative_stratiform_offset", "inflow"]
    for i, classification_name in enumerate(classification_names):
        args = [subplot_axes[2 * i], subplot_axes[2 * i + 1]]
        args += [legend_axes[i], classifications, quality]
        kwargs = {"classification_name": classification_name}
        legend_formatter = formatter_dispatcher[classification_name]
        kwargs.update({"legend_formatter": legend_formatter})
        plot_counts_ratios(*args, **kwargs, legend_columns=3)

    filepath = analysis_directory / "visualize/stratiform_inflow.png"
    plt.savefig(filepath, bbox_inches="tight")
    pickle_figure(fig, subplot_axes, legend_axes, colorbar_axes, filepath)

    layout_kwargs.update({"rows": 2, "legend_rows": 4})
    panelled_layout = visualize.horizontal.Panelled(**layout_kwargs)
    fig, subplot_axes, colorbar_axes, legend_axes = panelled_layout.initialize_layout()

    classification_names = ["tilt", "propagation"]
    for i, classification_name in enumerate(classification_names):
        args = [subplot_axes[2 * i], subplot_axes[2 * i + 1]]
        args += [legend_axes[i], classifications, quality]
        kwargs = {"classification_name": classification_name}
        legend_formatter = formatter_dispatcher[classification_name]
        kwargs.update({"legend_formatter": legend_formatter})
        plot_counts_ratios(*args, **kwargs, legend_columns=3)

    filepath = analysis_directory / "visualize/tilt_propagation.png"
    plt.savefig(filepath, bbox_inches="tight")
    pickle_figure(fig, subplot_axes, legend_axes, colorbar_axes, filepath)


def pickle_figure(fig, subplot_axes, legend_axes, colorbar_axes, filepath):
    """Pickle the figure/ax handles for use in other figures."""
    fig_dict = {"fig": fig, "axes": subplot_axes, "legend_axes": legend_axes}
    fig_dict.update({"colorbar_axes": colorbar_axes})
    with open(filepath.with_suffix(".pkl"), "wb") as f:
        pickle.dump(fig_dict, f)


def plot_attribute_evolution(dfs, analysis_directory=None):
    """Plot the evolution of attributes over time."""

    dfs = dfs.copy()
    velocities = dfs["velocities"]
    group = dfs["group"]
    quality = dfs["quality"]
    convective = dfs["convective_core"]
    anvil = dfs["anvil_core"]
    ellipse = dfs["ellipse"]

    if analysis_directory is None:
        analysis_directory = get_analysis_directory()

    logger.info("Calculating minutes.")
    for df in [velocities, group, quality, convective, anvil, ellipse]:
        if "minutes" not in df.index.names:
            df = utils.get_duration_minutes(df)

    ground_relative_speed = velocities[["u", "v"]].apply(np.linalg.norm, axis=1)
    flow_relative_speed = velocities[["u_relative", "v_relative"]].apply(
        np.linalg.norm, axis=1
    )
    offset = group[["x_offset", "y_offset"]].apply(np.linalg.norm, axis=1)
    convective_area = convective["area"]
    anvil_area = anvil["area"]
    orientation = ellipse["orientation"]

    layout_kwargs = {"subplot_width": 3.5, "subplot_height": 2.5, "rows": 2}
    layout_kwargs.update({"columns": 3})
    layout_kwargs.update({"colorbar": False, "legend_rows": 2, "vertical_spacing": 0.8})
    layout_kwargs.update({"shared_legends": "all", "horizontal_spacing": 1})
    layout_kwargs.update({"label_offset_x": -0.2, "label_offset_y": 0.075})

    logger.info("Plotting.")

    panelled_layout = visualize.horizontal.Panelled(**layout_kwargs)
    fig, subplot_axes, colorbar_axes, legend_axes = panelled_layout.initialize_layout()

    attributes = [ground_relative_speed, flow_relative_speed, offset]
    attributes += [convective_area, anvil_area, orientation]
    names = ["velocity", "relative_velocity", "offset_raw", "area", "area"]
    names += ["orientation"]
    ylabels = [r"Ground-Rel. Speed [ms$^{-1}$]", r"Flow-Rel. Speed [ms$^{-1}$]"]
    ylabels += [r"Stratiform Offset [km]", r"Convective Area [km$^2$]"]
    ylabels += [r"Stratiform Area [km$^2$]", r"Orientation [deg]"]
    circulars = [False] * 5 + [True]
    all_args = [subplot_axes, attributes, names, ylabels, circulars]

    for i in range(len(attributes)):
        args = [arr[i] for arr in all_args]
        args.insert(2, quality)
        plot_attribute(*args)

    handles, labels = subplot_axes[5].get_legend_handles_labels()
    kwargs = {"ncol": 2, "loc": "center", "facecolor": "white", "framealpha": 1}
    legend_axes[0].legend(handles, labels, **kwargs)
    filepath = analysis_directory / "visualize/attributes.png"
    plt.savefig(filepath, bbox_inches="tight")
    pickle_figure(fig, subplot_axes, legend_axes, colorbar_axes, filepath)


# shear = velocities[["u_shear", "v_shear"]]
# shear_direction = np.arctan2(shear["v_shear"], shear["u_shear"])
# angles_1 = np.arccos(np.cos(shear_direction-orientation))
# angles_2 = np.arccos(np.cos(shear_direction-(orientation+np.pi)))
# angles = np.minimum(angles_1, angles_2)
