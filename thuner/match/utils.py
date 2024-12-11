"""General matching convenience functions."""

from itertools import product
import pandas as pd
import numpy as np
import networkx as nx
from thuner.log import setup_logger


logger = setup_logger(__name__)


def get_masks(object_tracks, object_options, matched=False, num_previous=1):
    """Get the appropriate current and previous masks for matching."""
    mask_type = matched * "_matched" + "_mask"
    current_mask = object_tracks[f"current{mask_type}"]
    previous_masks = [
        object_tracks[f"previous{mask_type}s"][-i] for i in range(1, num_previous + 1)
    ]
    masks = [current_mask] + previous_masks
    if "grouping" in object_options.model_fields:
        matched_object = object_options.tracking.matched_object
        for i in range(len(masks)):
            if masks[i] is not None:
                masks[i] = masks[i][f"{matched_object}_mask"]
    return masks


def get_grids(object_tracks, object_options, num_previous=1):
    """Get the appropriate current and previous grids for matching."""
    current_grid = object_tracks["current_grid"]
    previous_grids = [
        object_tracks["previous_grids"][-i] for i in range(1, num_previous + 1)
    ]
    grids = [current_grid] + previous_grids
    if "grouping" in object_options.model_fields:
        matched_object = object_options.tracking.matched_object
        for i in range(len(grids)):
            if grids[i] is not None:
                grids[i] = grids[i][f"{matched_object}_grid"]
    return grids


def parents_to_list(parents_str):
    """Convert a parent str to a list of parent ids as ints."""
    if not isinstance(parents_str, str) or parents_str == "NA":
        return []
    return [int(p) for p in parents_str.split(" ")]


def get_parent_graph(df):
    """
    Create a parent graph from a DataFrame of objects. DataFrame must have columns
    "time", "universal_id", and "parents".
    """

    if "event_start" in df.columns:
        # Check whether event_start column is present; this column is used for GridRad data
        message = (
            "DataFrame should not have event_start column; take cross section first."
        )
        raise ValueError(message)

    # Create a directed graph to capture the object parent/child relationship
    parent_graph = nx.DiGraph()
    # Loop backwards through array. Create new objects, looking up parents as needed
    times = sorted(np.unique(df.reset_index().time))

    for i in range(len(times) - 1, 0, -1):
        time = times[i]
        previous_time = times[i - 1]
        universal_ids = df.xs(time, level="time").reset_index()["universal_id"]
        universal_ids = universal_ids.values
        previous_ids = df.xs(previous_time, level="time").reset_index()
        previous_ids = previous_ids["universal_id"].values
        for obj_id in universal_ids:
            node = tuple([time, obj_id])
            if obj_id in previous_ids:
                # Add edge to same object at previous time
                previous_node = tuple([previous_time, obj_id])
                parent_graph.add_edge(previous_node, node)
            # Add edges to parents (if any) at previous time
            parents = parents_to_list(df.loc[node].parents)
            for parent in parents:
                parent_node = tuple([previous_time, parent])
                parent_graph.add_edge(parent_node, node)

    mapping = {}
    for node in list(parent_graph.nodes):
        time = str(node[0].astype("datetime64[s]")).replace(":", "").replace("-", "")
        new_node = (time, node[1])
        mapping[node] = new_node

    # Relabel node names to be tuples of strings
    parent_graph_str = nx.relabel_nodes(parent_graph, mapping)
    return parent_graph_str


def get_component_subgraphs(parent_graph):
    """Get connected components from a parent graph."""
    undirected_graph = parent_graph.to_undirected()
    components = nx.algorithms.connected.connected_components(undirected_graph)
    return [parent_graph.subgraph(c).copy() for c in components]


# Too slow to iterate over all sources and targets.
# For now, simply take longest path using dag_longest_path
def get_paths(component_subgraph):
    """Get the shortest path from sources to targets in a connected component."""
    sources, targets = get_sources_targets(component_subgraph)
    all_paths, all_path_lengths = [], []
    # Iterating over all the sources and targets still seems too slow.
    for source, target in product(sources, targets):
        # For our application all paths between source and target are the same length.
        # Thus for efficiency, simply take the first path found between source and
        # target, acknowledging there may be more than one such path.
        path_generator = nx.all_simple_paths(component_subgraph, source, target)
        try:
            simple_path = next(path_generator)
        except StopIteration:
            continue
        # Exclude 'paths' of length 1
        if len(simple_path) < 2:
            continue
        # simple_path = [sorted(p) for p in [simple_path] if len(p) > 0]
        # path_lengths = [len(p) for p in simple_path]
        all_paths.append(simple_path)
        all_path_lengths.append(len(simple_path))
    sorted_paths, sorted_lengths = [], []
    for length, path in sorted(zip(all_path_lengths, all_paths)):
        sorted_lengths.append(length)
        sorted_paths.append(path)
    shortest_path = sorted_paths[0]
    longest_path = sorted_paths[-1]
    median_path = sorted_paths[len(sorted_paths) // 2]
    return shortest_path, longest_path, median_path


def get_sources_targets(component_subgraph):
    """Get sources and targets for each connected component."""
    sources = []
    targets = []
    for node in component_subgraph.nodes:
        if component_subgraph.out_degree(node) == 0:
            targets.append(node)
        if component_subgraph.in_degree(node) == 0:
            sources.append(node)
    return sources, targets


def get_component_paths(component_subgraph):
    """Get all paths from sources to targets in a connected component."""
    # Get sources/targets of a component subgraph
    sources, targets = get_sources_targets(component_subgraph)
    all_simple_paths = []
    all_path_lengths = []
    for source, target in product(sources, targets):
        simple_paths = nx.all_simple_paths(component_subgraph, source, target)
        simple_paths = [sorted(p) for p in list(simple_paths) if len(p) > 0]
        path_lengths = [len(p) for p in simple_paths]
        all_simple_paths += simple_paths
        all_path_lengths += path_lengths
    return all_simple_paths, all_path_lengths


def get_new_objects(df, paths, object_count=0):
    """Get new objects based on the split merge history."""

    df = df.copy()
    index_names = df.index.names
    non_index_names = list(set(index_names) - set(["time", "universal_id"]))
    for name in non_index_names:
        df = df.reset_index(level=name, drop=False)
    new_objs = []
    for i, path in enumerate(paths):
        # Extract the relevant rows
        new_obj = df.loc[path].reset_index()
        new_obj["universal_id"] = i + 1 + object_count
        new_objs.append(new_obj)
    new_df = pd.concat(new_objs, axis=0)
    new_df = new_df.set_index(index_names)
    if "parents" in new_df.columns:
        new_df = new_df.drop(columns=["parents"])
    return new_df
