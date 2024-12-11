from pathlib import Path
from scipy.stats import circmean, circstd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import networkx as nx
import numpy as np
import pandas as pd
import pickle
import thuner.visualize as visualize
import thuner.config as config
from thuner.log import setup_logger
import utils

logger = setup_logger(__name__)


# Create a dict of quality control attributes for each classification
# Note duration/parents are often appended to these for particular plots
core = ["convective_contained", "anvil_contained", "initially_contained", "duration"]
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
    "cape": core,
    "ake": core,
    "R": core,
    "ambient_wind": core,
}


def parent_graph(parent_graph, ax=None, analysis_directory=None):
    """Visualize a parent graph."""

    plt.close("all")
    if analysis_directory is None:
        analysis_directory = utils.get_analysis_directory()

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


def windrose(dfs, analysis_directory=None):
    """Create a windrose style plot of system velocity and stratiform offset."""

    plt.close("all")
    if analysis_directory is None:
        analysis_directory = utils.get_analysis_directory()

    quality = dfs["quality"]
    raw_sample = quality[["duration", "parents", "children"]].any(axis=1)

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
    filepath = analysis_directory / "visualize/rose/vel_so_rose.png"
    save_figure(filepath, fig, subplot_axes, colorbar_axes, legend_axes)


def save_figure(filepath, fig, subplot_axes, colorbar_axes, legend_axes):
    """Save figure to png and with pickle."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, bbox_inches="tight")
    fig_dict = {"fig": fig, "subplot_axes": subplot_axes}
    fig_dict.update({"colorbar_axes": colorbar_axes, "legend_axes": legend_axes})
    with open(filepath.with_suffix(".pkl"), "wb") as f:
        pickle.dump(fig_dict, f)


prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]

offset_color_dict = {"trailing": colors[0], "leading": colors[1]}
offset_color_dict.update({"left": colors[2], "right": colors[4]})

inflow_color_dict = {"front": colors[0], "rear": colors[1]}
inflow_color_dict.update({"left": colors[2], "right": colors[4]})

tilt_color_dict = {"up-shear": colors[0], "down-shear": colors[1]}
tilt_color_dict.update({"shear-perpendicular": colors[5]})

propagation_color_dict = {"up-shear": colors[1], "down-shear": colors[0]}
propagation_color_dict.update({"shear-perpendicular": colors[5]})

color_dicts = {
    "stratiform_offset": offset_color_dict,
    "relative_stratiform_offset": offset_color_dict,
    "inflow": inflow_color_dict,
    "tilt": tilt_color_dict,
    "propagation": propagation_color_dict,
}

rel_so_formatter = lambda labels: [f"Relative {l.title()} Stratiform" for l in labels]
formatter_dispatcher = {
    "stratiform_offset": lambda labels: [f"{l.title()} Stratiform" for l in labels],
    "relative_stratiform_offset": rel_so_formatter,
    "inflow": lambda labels: [f"{l.title()} Fed" for l in labels],
    "tilt": lambda labels: [f"{l.title()} Tilted" for l in labels],
    "propagation": lambda labels: [f"{l.title()} Propagating" for l in labels],
}


def plot_pie_inset(data, longitude, latitude, ax, width, offsets, colors):
    """Create a pie chart in an inset axis at a given location."""
    start_lon = longitude - width / 2 + offsets[0]
    start_lat = latitude - width / 2 + offsets[1]
    bounds = [start_lon, start_lat, width, width]
    ax_sub = ax.inset_axes(bounds=bounds, transform=ax.transData, zorder=1)
    wedge_props = {"edgecolor": "black", "linewidth": 1, "antialiased": True}
    kwargs = {"colors": colors, "wedgeprops": wedge_props, "normalize": True}
    patches, texts = ax_sub.pie(data, **kwargs)
    [p.set_zorder(2) for p in patches]
    return patches, texts


def pie_map(dfs, classification_name, analysis_directory=None, block_size=5):
    """Create a map of classification ratios using pie charts."""

    plt.close("all")
    if analysis_directory is None:
        analysis_directory = utils.get_analysis_directory()

    mcs = dfs["mcs_core"].copy()
    classification = dfs["classification"].copy()
    classification["latitude_block"] = mcs["latitude"] // block_size * block_size
    classification["longitude_block"] = mcs["longitude"] // block_size * block_size

    cond = dfs["quality"][quality_dispatcher[classification_name]].all(axis=1)
    classification = classification.where(cond).dropna()

    block_names = ["latitude_block", "longitude_block"]
    group_names = block_names + [classification_name]
    group = classification.groupby(group_names)
    counts = group[classification_name].apply(lambda x: x.count())
    ratios = counts / counts.groupby(block_names).transform("sum")
    totals = counts.groupby(block_names).apply("sum")
    max_total = totals.max()

    extent = [-125, -70, 24, 51]
    kwargs = {"extent": extent, "rows": 1, "columns": 1, "subplot_width": 11}
    kwargs.update({"legend_rows": 1, "colorbar": False})
    layout = visualize.horizontal.PanelledUniformMaps(**kwargs)
    fig, subplot_axes, colorbar_axes, legend_axes = layout.initialize_layout()

    min_obs = 100

    for lat, lon in [(i[0], i[1]) for i in ratios.index]:
        ratio = ratios.loc[(lat, lon)]
        total = totals.loc[(lat, lon)]

        if total < min_obs:
            continue
        lon += block_size / 2
        lat += block_size / 2
        lon = (lon + 180) % 360 - 180
        colors = [color_dicts[classification_name][c] for c in ratio.index]

        scale = 0.6
        width = block_size * scale
        width += (total - min_obs) / (max_total - min_obs) * (1 - scale) * block_size

        args = [ratio, lon, lat, subplot_axes[0], width, [0, 0]]
        plot_pie_inset(*args, colors=colors)

    labels = color_dicts[classification_name].keys()
    colors = color_dicts[classification_name].values()
    labels = formatter_dispatcher[classification_name](labels)
    # Generate recatngle patches with black borders for the legend
    args = [(0, 0), 1, 1]
    kwargs = {"edgecolor": "black"}
    patches = [plt.Rectangle(*args, facecolor=color, **kwargs) for color in colors]
    kwargs = {"ncol": 2, "loc": "center", "facecolor": "white", "framealpha": 1}
    legend_axes[0].legend(patches, labels, **kwargs)

    filepath = analysis_directory / f"visualize/pie/{classification_name}.png"
    save_figure(filepath, fig, subplot_axes, colorbar_axes, legend_axes)


def field_map(
    field, name, mcs, quality, cmap, norm, label, analysis_directory=None, extend=None
):
    """Visualize a field on a map as a pcolormesh."""

    plt.close("all")
    if analysis_directory is None:
        analysis_directory = utils.get_analysis_directory()

    block_size = 5

    field = field.copy()
    mcs = mcs.copy()
    field["latitude_block"] = mcs["latitude"] // block_size * block_size
    field["longitude_block"] = mcs["longitude"] // block_size * block_size

    cond = quality[quality_dispatcher[name]].all(axis=1)
    field = field.where(cond).dropna()

    block_names = ["latitude_block", "longitude_block"]
    group = field.groupby(block_names)
    group_median = group[name].apply(lambda x: x.median())
    group_total = group[name].apply(lambda x: x.count())

    extent = [-125, -70, 24, 51]
    kwargs = {"extent": extent, "rows": 1, "columns": 1, "subplot_width": 10}
    kwargs.update({"legend_rows": 1, "colorbar": True, "border_zorder": 2})
    kwargs.update({"coastline_zorder": 3, "grid_zorder": 3})
    layout = visualize.horizontal.PanelledUniformMaps(**kwargs)
    fig, subplot_axes, colorbar_axes, legend_axes = layout.initialize_layout()

    lats, lons = [group_median.index.get_level_values(b).values for b in block_names]
    # Create array spanning lats and lons
    lats, lons = np.unique(sorted(lats)), np.unique(sorted(lons))
    median_array = np.full((len(lats), len(lons)), np.nan)

    min_obs = 100

    for lat, lon in [(i[0], i[1]) for i in group_median.index]:
        i, j = lats.tolist().index(lat), lons.tolist().index(lon)
        if group_total.loc[lat, lon] < min_obs:
            continue
        median_array[i, j] = group_median.loc[lat, lon]

    median_array.max()

    LONS, LATS = np.meshgrid(lons, lats)
    LONS, LATS = LONS + block_size / 2, LATS + block_size / 2
    kwargs = {"cmap": cmap, "norm": norm, "zorder": 1}
    pcm = subplot_axes[0].pcolormesh(LONS, LATS, median_array, **kwargs)
    cbar = plt.colorbar(pcm, cax=colorbar_axes[0], extend=extend)
    cbar.set_label(label)

    filepath = analysis_directory / f"visualize/heatmap/{name}.png"
    save_figure(filepath, fig, subplot_axes, colorbar_axes, legend_axes)


def field_maps(dfs, analysis_directory=None):
    """Visualize fields related to mesoscale structure."""

    if analysis_directory is None:
        analysis_directory = utils.get_analysis_directory()

    shear = utils.get_shear(dfs["profile"])
    ake = (1 / 2) * (shear["u"] ** 2 + shear["v"] ** 2)
    ake.name = "ake"
    ake = pd.DataFrame(ake)
    cape = dfs["tag"][["cape"]]
    R = cape["cape"] / ake["ake"]
    R.name = "R"
    R = pd.DataFrame(R)

    mcs = dfs["mcs_core"].copy()
    quality = dfs["quality"].copy()

    all_levels = [
        np.arange(0, 2000 + 250, 250),
        np.arange(0, 450 + 50, 50),
        np.arange(0, 40 + 5, 5),
    ]
    all_cmap_names = ["Reds", "Reds", "Spectral_r"]
    all_names = ["cape", "ake", "R"]
    all_fields = [cape, ake, R]
    all_labels = ["CAPE [J kg$^{-1}$]", r"$\frac{1}{2}(\Delta u)^2$ [J kg$^{-1}$]"]
    all_labels += ["$R$ [-]"]
    all_extend = ["neither", "neither", "max"]
    all_pad_start = [2, 2, 0]

    for i in range(len(all_levels)):
        args = [all_levels[i], all_cmap_names[i]]
        kwargs = {"pad_start": all_pad_start[i], "extend": all_extend[i]}
        cmap, norm = visualize.visualize.discrete_cmap_norm(*args, **kwargs)
        args = [all_fields[i], all_names[i], mcs, quality, cmap, norm, all_labels[i]]
        field_map(*args, extend=all_extend[i])


def profile_grid(x_min, x_max, dx, y_min, y_max, dy, ax, x_label=True, y_label=True):
    """Set up the ticks for a plot."""

    ax.set_yticks(np.arange(y_min, y_max + dy, dy))
    ax.set_yticks(np.arange(y_min, y_max + dy / 2, dy / 2), minor=True)
    ax.set_yticklabels((np.arange(y_min, y_max + dy, dy) / 1e3).astype(int))
    if y_label:
        ax.set_ylabel("Altitude [km]")
    ax.set_xticks(np.arange(x_min, x_max + dx, dx))
    ax.set_xticks(np.arange(x_min, x_max + dx / 2, dx / 2), minor=True)
    if x_label:
        ax.set_xlabel("Wind speed [ms$^{-1}$]")
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(x_min, x_max)
    ax.grid(which="major")
    ax.grid(which="minor", alpha=0.4)


def set_scientific(ax, axis="y"):
    """Set the axis to scientific notation."""
    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    if axis == "x":
        coord_ax = ax.xaxis
    elif axis == "y":
        coord_ax = ax.yaxis
    else:
        raise ValueError("Axis must be 'x' or 'y'.")
    coord_ax.set_major_formatter(formatter)


def evolution_grid(ax, dy, y_max, x_max=540, min_y=0, scientific=False, circular=False):
    """Set up the grid for time series style plots."""
    y_max = np.ceil(y_max / dy) * dy
    ax.set_ylim(min_y, y_max)
    ax.set_yticks(np.arange(min_y, y_max + dy, dy))
    ax.set_yticks(np.arange(min_y, y_max + dy / 2, dy / 2), minor=True)

    if scientific:
        set_scientific(ax)

    if circular:
        labels = np.round(np.arange(min_y, y_max + dy, dy) * 180 / np.pi).astype(int)
        ax.set_yticklabels(labels)

    dx = 180
    ax.set_xlim(0, x_max)
    ax.set_xticks(np.arange(0, x_max + dx, dx))
    ax.set_xticks(np.arange(0, x_max + dx / 3, dx / 3), minor=True)
    ax.grid(True, which="major")
    ax.grid(True, which="minor", alpha=0.4)
    ax.set_xlabel("Time since Initiation [min]")


def acute_angle_hist(ax, angles_degrees):
    """Plot a histogram of acute angles, e.g. smallest angle between two vectors."""
    bins = np.arange(0, 100, 10)
    ax.hist(angles_degrees, bins=bins, density=True)
    ax.set_xticks(bins)
    ax.set_xlabel("Shear/Orientation Angle [degrees]")
    ax.set_ylabel("Density [-]")
    # Format y ticks in scientific notation
    set_scientific(ax)

    ax.grid(True, which="major")
    ax.grid(True, which="minor", alpha=0.4)


def shear_orientation_angles(dfs, analysis_directory=None):
    """Plot the distribution of angles between shear and orientation."""

    plt.close("all")
    if analysis_directory is None:
        analysis_directory = utils.get_analysis_directory()

    velocities = dfs["velocities"]
    quality = dfs["quality"]
    orientation = dfs["ellipse"]["orientation"]
    classification = dfs["classification"]

    shear = velocities[["u_shear", "v_shear"]]
    shear_direction = np.arctan2(shear["v_shear"], shear["u_shear"])
    angles_1 = np.arccos(np.cos(shear_direction - orientation))
    angles_2 = np.arccos(np.cos(shear_direction - (orientation + np.pi)))
    angles = np.minimum(angles_1, angles_2)

    upshear = classification["tilt"] == "up-shear"
    downshear = classification["tilt"] == "down-shear"

    layout_kwargs = {"subplot_width": 5, "subplot_height": 3.5, "rows": 1, "columns": 2}
    layout_kwargs.update({"colorbar": False, "legend_rows": 0, "vertical_spacing": 0.6})
    layout_kwargs.update({"shared_legends": "columns", "horizontal_spacing": 1})
    layout_kwargs.update({"label_offset_x": -0.125, "label_offset_y": 0.05})
    panelled_layout = visualize.horizontal.Panelled(**layout_kwargs)
    fig, subplot_axes, colorbar_axes, legend_axes = panelled_layout.initialize_layout()

    angles_degrees = angles * 180 / np.pi
    cond = quality[quality_dispatcher["orientation"] + ["shear"]].all(axis=1)
    acute_angle_hist(subplot_axes[0], angles_degrees.where(downshear & cond).dropna())
    subplot_axes[0].set_title("Down-shear Tilted")
    acute_angle_hist(subplot_axes[1], angles_degrees.where(upshear & cond).dropna())
    subplot_axes[1].set_title("Up-shear Tilted")

    filepath = analysis_directory / "visualize/angle/shear_orientation.png"
    save_figure(filepath, fig, subplot_axes, colorbar_axes, legend_axes)


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

    cond = quality[quality_dispatcher[classification_name]].all(axis=1)
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
    unstacked.plot(ax=count_ax, legend=False, color=color_dicts[classification_name])
    max_count = unstacked.max().max()
    evolution_grid(count_ax, dcount, max_count, max_minutes)
    count_ax.set_ylabel("Count [-]")

    unstacked = ratios.unstack()
    unstacked.plot(ax=ratio_ax, legend=False, color=color_dicts[classification_name])
    max_ratio = unstacked.dropna().max().max()
    evolution_grid(ratio_ax, dratio, max_ratio, max_minutes)
    ratio_ax.set_ylabel("Ratio [-]")

    handles, labels = count_ax.get_legend_handles_labels()
    labels = legend_formatter(labels)
    kwargs = {"ncol": legend_columns, "facecolor": "white", "framealpha": 1}
    legend_ax.legend(handles, labels, **kwargs)


def plot_attribute(ax, df, quality, name, ylabel=None, circular=False):
    """Plot counts and ratios of classifications over time."""

    plt.close("all")
    if analysis_directory is None:
        analysis_directory = utils.get_analysis_directory()

    if "minutes" not in df.index.names:
        df = utils.get_duration_minutes(df)

    max_minutes = 720

    cond = quality[quality_dispatcher[name]].all(axis=1)
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
    kwargs = {"x_max": max_minutes, "min_y": min_value, "scientific": scientific}
    kwargs.update({"circular": circular})
    evolution_grid(ax, dvalue, max_value, **kwargs)

    if ylabel is not None:
        ax.set_ylabel(ylabel)


def plot_classification_evolution(classifications, quality, analysis_directory=None):
    """Plot the evolution of classifications over time."""

    plt.close("all")
    if analysis_directory is None:
        analysis_directory = utils.get_analysis_directory()
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

    filepath = analysis_directory / "visualize/evolution/stratiform_inflow.png"
    save_figure(filepath, fig, subplot_axes, colorbar_axes, legend_axes)

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

    filepath = analysis_directory / "visualize/evolution/tilt_propagation.png"
    save_figure(filepath, fig, subplot_axes, colorbar_axes, legend_axes)


def plot_attribute_evolution(dfs, analysis_directory=None):
    """Plot the evolution of attributes over time."""

    plt.close("all")
    if analysis_directory is None:
        analysis_directory = utils.get_analysis_directory()

    dfs = dfs.copy()
    velocities = dfs["velocities"]
    group = dfs["group"]
    quality = dfs["quality"]
    convective = dfs["convective_core"]
    anvil = dfs["anvil_core"]
    ellipse = dfs["ellipse"]

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
    filepath = analysis_directory / "visualize/evolution/attributes.png"
    save_figure(filepath, fig, subplot_axes, colorbar_axes, legend_axes)


def component_wind_profile(mean_winds, std_winds, ax, linestyle="-", color="tab:blue"):
    """Plot horizontal wind component profile."""
    kwargs = {"linestyle": linestyle, "color": color, "label": "Mean"}
    ax.plot(mean_winds, mean_winds.index, **kwargs)
    args = [mean_winds.index, mean_winds - std_winds]
    args += [mean_winds + std_winds]
    kwargs = {"alpha": 0.2, "linewidth": 1, "color": color}
    kwargs.update({"label": "Standard Deviation"})
    ax.fill_betweenx(*args, **kwargs)


def wind_profile(mean_winds, std_winds, ax):
    "Plot wind profile."
    component_wind_profile(mean_winds["u"], std_winds["u"], ax)
    component_wind_profile(mean_winds["v"], std_winds["v"], ax, linestyle="--")
    profile_grid(-10, 35, 5, 0, 16e3, 2e3, ax, y_label=True)


def wind_profiles(dfs):
    """Contrast wind profiles across categories."""

    plt.close("all")
    analysis_directory = utils.get_analysis_directory()

    profile = dfs["profile"].copy()
    classification = dfs["classification"].copy()
    quality = dfs["quality"].copy()
    winds = profile[["u", "v"]].xs(slice(0, 16e3), level="altitude", drop_level=False)
    cond = quality[quality_dispatcher["ambient_wind"]].all(axis=1)

    tilt_cond = quality[quality_dispatcher["tilt"]].all(axis=1)
    downshear = classification["tilt"] == "down-shear"
    upshear = classification["tilt"] == "up-shear"

    layout_kwargs = {"subplot_width": 3, "subplot_height": 3.5, "rows": 1, "columns": 3}
    layout_kwargs.update({"colorbar": False, "vertical_spacing": 0.75})
    layout_kwargs.update({"legend_rows": 2})
    layout_kwargs.update({"shared_legends": "all", "horizontal_spacing": 0.85})
    layout_kwargs.update({"label_offset_x": -0.15, "label_offset_y": 0.075})
    panelled_layout = visualize.horizontal.Panelled(**layout_kwargs)
    fig, subplot_axes, colorbar_axes, legend_axes = panelled_layout.initialize_layout()

    all_conds = [cond, tilt_cond & downshear, tilt_cond & upshear]
    all_titles = ["All", "Down-shear Tilted", "Up-shear Tilted"]
    for i, cond in enumerate(all_conds):
        quality_winds = winds.where(cond).dropna()
        group = quality_winds.groupby("altitude")
        mean_winds, std_winds = group.mean(), group.std()
        wind_profile(mean_winds, std_winds, subplot_axes[i])
        subplot_axes[i].set_title(all_titles[i])

    handles, labels = subplot_axes[0].get_legend_handles_labels()
    handles = [handles[i] for i in [0, 2, 1]]
    labels = ["$u$ Mean", "$v$ Mean", "Standard Deviation"]
    legend_axes[0].legend(handles, labels, loc="center", ncol=3, framealpha=1)

    filepath = analysis_directory / "visualize/profile/wind.png"
    save_figure(filepath, fig, subplot_axes, colorbar_axes, legend_axes)


def cape_ake_R(dfs):
    """Contrast shear, CAPE and R."""

    plt.close("all")
    analysis_directory = utils.get_analysis_directory()

    shear = utils.get_shear(dfs["profile"])
    ake = (1 / 2) * (shear["u"] ** 2 + shear["v"] ** 2)
    ake.name = "ake"
    ake = pd.DataFrame(ake)
    cape = dfs["tag"][["cape"]]
    R = cape["cape"] / ake["ake"]
    R.name = "R"
    R = pd.DataFrame(R)

    quality = dfs["quality"].copy()
    classification = dfs["classification"].copy()

    # Quality control criteria same for cape, ake, R
    cond = quality[quality_dispatcher["cape"]].all(axis=1)
    tilt_cond = quality[quality_dispatcher["tilt"]].all(axis=1)
    rel_so_cond = quality[quality_dispatcher["relative_stratiform_offset"]].all(axis=1)
    upshear = classification["tilt"] == "up-shear"
    downshear = classification["tilt"] == "down-shear"
    leading = classification["relative_stratiform_offset"] == "leading"
    trailing = classification["relative_stratiform_offset"] == "trailing"
    all_names = ["cape", "ake", "R"]
    all_fields = [cape, ake, R]
    all_labels = ["CAPE [J kg$^{-1}$]", r"$\frac{1}{2}(\Delta u)^2$ [J kg$^{-1}$]"]
    all_labels += ["$R$ [-]"]
    all_bins = [np.arange(0, 4000 + 250, 250), np.arange(0, 1000 + 50, 50)]
    all_bins += [np.arange(0, 60 + 2.5, 2.5)]

    layout_kwargs = {"subplot_width": 3, "subplot_height": 3.5, "rows": 1, "columns": 3}
    layout_kwargs.update({"colorbar": False, "vertical_spacing": 0.75})
    layout_kwargs.update({"legend_rows": 2})
    layout_kwargs.update({"shared_legends": "all", "horizontal_spacing": 1})
    layout_kwargs.update({"label_offset_x": -0.15, "label_offset_y": 0.075})
    panelled_layout = visualize.horizontal.Panelled(**layout_kwargs)

    fig, subplot_axes, colorbar_axes, legend_axes = panelled_layout.initialize_layout()

    line_kwargs = {"linestyle": "--", "linewidth": 2, "label": "Median"}
    for i in range(len(all_names)):

        group_cond = cond & tilt_cond & rel_so_cond & upshear & trailing
        field = all_fields[i].where(group_cond).dropna()
        subplot_axes[i].hist(field, bins=all_bins[i], density=True, alpha=0.5)
        median = field[all_names[i]].median()
        subplot_axes[i].axvline(median, color="tab:blue", **line_kwargs)

        group_cond = cond & tilt_cond & rel_so_cond & downshear & leading
        field = all_fields[i].where(group_cond).dropna()
        subplot_axes[i].hist(field, bins=all_bins[i], density=True, alpha=0.5)
        median = field[all_names[i]].median()
        subplot_axes[i].axvline(median, color="tab:orange", **line_kwargs)

        set_histogram_grid(subplot_axes[i], all_bins[i], all_labels[i])

    filepath = analysis_directory / "visualize/profile/cape_ake_R.png"
    save_figure(filepath, fig, subplot_axes, colorbar_axes, legend_axes)


def set_histogram_grid(ax, bins, label):
    """Convenience function to setup histogram grid."""
    ax.set_xticks(bins[::4])
    ax.set_xticks(bins[::2], minor=True)
    ax.set_xlabel(label)
    ax.set_ylabel("Density [-]")

    # Format y ticks in scientific notation
    set_scientific(ax)
    ax.grid(True, which="major")
    ax.grid(True, which="minor", alpha=0.4)
