"""Methods supporting analysis of GridRad Severe data."""

import yaml
from pathlib import Path
import pandas as pd
import thor.write as write
import glob
import networkx as nx
import numpy as np
import thor.config as config
import thor.attribute as attribute
import thor.data as data
import thor.match as match
import thor.log as log


logger = log.setup_logger(__name__)


def aggregate_runs(base_local=None):
    """Aggregate data across runs."""

    if base_local is None:
        base_local = config.get_outputs_directory()

    pattern = str(base_local / "runs/gridrad_severe/gridrad_2010*[!.tar.gz]")
    run_directories = sorted(glob.glob(pattern))
    analysis_directory = base_local / "runs/gridrad_severe/analysis"
    attributes_directory = analysis_directory / "attributes/aggregated"

    names = ["core", "group", "core", "core", "ellipse"]
    names += ["profile", "tag", "classification"]
    names += ["quality", "velocities"]
    dataset_names = ["mcs_core", "group", "convective_core", "anvil_core", "ellipse"]
    dataset_names += ["profile", "tag", "classification"]
    dataset_names += ["quality", "velocities"]
    subdirectories = ["attributes/mcs"] * 2 + ["attributes/mcs/convective"]
    subdirectories += ["attributes/mcs/anvil"]
    subdirectories += ["attributes/mcs/convective"]
    subdirectories += ["attributes/mcs/era5_pl", "attributes/mcs/era5_sl"]
    subdirectories += ["analysis"] * 3
    dfs = {}
    metadata = {}

    for i, name in enumerate(names):
        print(f"Processing {name}")
        df_list = []
        # Read the first file to get the metadata
        filepath = Path(run_directories[0]) / subdirectories[i] / f"{name}.csv"
        attr = attribute.utils.read_metadata_yml(filepath.with_suffix(".yml"))
        for directory in run_directories:
            options_filepath = Path(directory) / "options/data.yml"
            with open(options_filepath, "r") as f:
                data_options = data.option.DataOptions(**yaml.safe_load(f))
            gridrad_options = data_options.dataset_by_name("gridrad")
            event_start = gridrad_options.event_start

            filepath = Path(directory) / subdirectories[i] / f"{name}.csv"
            df = attribute.utils.read_attribute_csv(filepath)
            df["event_start"] = event_start
            df_list.append(df)

        event_start = attr["time"].copy()
        event_start["description"] = "GridRad Severe event start date."
        attr["event_start"] = event_start
        df = pd.concat(df_list)
        df = df.reset_index().set_index(["time", "universal_id", "event_start"])
        df = df.sort_index()

        filepath = attributes_directory / f"{dataset_names[i]}.csv"
        write.attribute.write_csv(filepath, df, attr)
        dfs[dataset_names[i]] = df
        metadata[dataset_names[i]] = attr
    return dfs, metadata


def load_aggregated_runs(attributes_directory=None):
    """Load aggregated runs."""

    if attributes_directory is None:
        base_local = config.get_outputs_directory()
        analysis_directory = base_local / "runs/gridrad_severe/analysis"
        attributes_directory = analysis_directory / "attributes/aggregated"
    dfs, metadata = {}, {}
    for csv in glob.glob(str(attributes_directory / "*.csv")):
        df = attribute.utils.read_attribute_csv(csv)
        md = attribute.utils.read_metadata_yml(Path(csv).with_suffix(".yml"))
        dfs[Path(csv).stem] = df
        metadata[Path(csv).stem] = md
    return dfs, metadata


def relabel_all(dfs, analysis_directory=None):
    """Relabel objects based on paths through connected components."""

    if analysis_directory is None:
        outputs_directory = config.get_outputs_directory()
        analysis_directory = outputs_directory / "runs/gridrad_severe/analysis"

    longest_directory = analysis_directory / "longest_paths"
    longest_directory.mkdir(exist_ok=True, parents=True)
    for name in dfs.keys():
        (longest_directory / name).mkdir(exist_ok=True, parents=True)

    core = dfs["mcs_core"]
    event_starts = sorted(core.index.get_level_values("event_start").unique())

    non_core = [name for name in dfs.keys() if name != "mcs_core"]
    object_count = 0
    for event_start in event_starts:
        logger.info(f"Re-labelling event {event_start}.")
        core = dfs["mcs_core"].xs(event_start, level="event_start")
        parent_graph = match.utils.get_parent_graph(core)
        component_subgraphs = match.utils.get_component_subgraphs(parent_graph)
        paths = [nx.dag_longest_path(c) for c in component_subgraphs]
        new_core = match.utils.get_new_objects(core, component_subgraphs, object_count)
        new_core["event_start"] = event_start
        filepath = longest_directory / f"mcs_core/{event_start.strftime('%Y%m%d')}.csv"
        write.attribute.write_csv(filepath, new_core)
        for name in non_core:
            df = dfs[name]
            event_df = df.xs(event_start, level="event_start")
            args = [event_df, paths, object_count]
            new_event_df = match.utils.get_new_objects(*args)
            new_event_df["event_start"] = event_start
            filepath = longest_directory
            filepath = filepath / f"{name}/{event_start.strftime('%Y%m%d')}.csv"
            write.attribute.write_csv(filepath, new_event_df)
        object_count += len(component_subgraphs)


def recalculate_duration_check(dfs, analysis_options):
    """Recalculate the duration check using the new objects."""

    # Recalculate duration check
    quality = dfs["quality"]
    velocities = dfs["velocities"]

    # Check the duration of the system is sufficiently long
    # First get the duration of each object from the velocity dataframe
    dummy_df = pd.DataFrame(index=velocities.index)
    dummy_df.index.names = velocities.index.names
    time_group = velocities.reset_index().groupby("universal_id")["time"]
    duration = time_group.agg(lambda x: x.max() - x.min())
    duration_check = duration >= np.timedelta64(analysis_options.min_duration, "m")

    duration_check.name = "duration"
    dummy_df = velocities[[]].reset_index()
    duration_check = dummy_df.merge(duration_check, on="universal_id", how="left")
    duration_check = duration_check.set_index(velocities.index.names)
    quality = quality.drop(columns="duration")
    quality = quality.merge(duration_check, left_index=True, right_index=True)
    dfs["quality"] = quality
    return dfs


def aggregate_relabelled(metadata, relabelled_directory, clean_up=False):
    """Aggregate relabelled dfs."""
    for name in metadata.keys():
        md = metadata[name]
        if "parents" in md:
            md.pop("parents")
        args = [relabelled_directory / name, name, md]
        write.attribute.aggregate_directory(*args, clean_up=clean_up)
        # Overwrite metadata to remove parent
        write.attribute.write_metadata(relabelled_directory / f"{name}.yml", md)


def get_duration_minutes(df: pd.DataFrame) -> pd.DataFrame:
    """Get the duration of each object in minutes."""
    index_columns = df.index.names.copy()
    df = df.copy().reset_index()
    df["time"] = df["time"].dt.round("min")
    minutes = df.groupby("universal_id")["time"]
    minutes = minutes.transform(lambda x: (x - x.min()))
    minutes = (minutes.dt.total_seconds() / 60).astype(int)
    df["minutes"] = minutes
    index_columns.insert(1, "minutes")
    df = df.set_index(index_columns)
    return df
