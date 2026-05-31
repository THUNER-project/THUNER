"""General matching convenience functions."""

from itertools import product
import pandas as pd
import numpy as np
import xarray as xr
import networkx as nx
from pydantic import BaseModel, Field, ConfigDict
from thuner.log import setup_logger


logger = setup_logger(__name__)


def get_masks(object_tracks, object_options, matched=False, num_previous=1):
    """
    Get the appropriate current and next masks for matching and visualization.
    """
    mask_type = matched * "matched_" + "mask"
    next_mask = getattr(object_tracks, f"next_{mask_type}")
    pre_masks = getattr(object_tracks, f"{mask_type}s")
    masks = [pre_masks[-i] for i in range(1, num_previous + 1)]
    all_masks = [next_mask] + masks
    if "grouping" in object_options.__class__.model_fields:
        matched_object = object_options.tracking.matched_object
        for i in range(len(all_masks)):
            if all_masks[i] is not None:
                all_masks[i] = all_masks[i][f"{matched_object}_mask"]
    return all_masks


def get_grids(object_tracks, object_options, num_previous=1):
    """
    Get the appropriate current and next grids for matching and visualization.
    """
    next_grid = object_tracks.next_grid
    grids = [object_tracks.grids[-i] for i in range(1, num_previous + 1)]
    all_grids = [next_grid] + grids
    if "grouping" in object_options.__class__.model_fields:
        matched_object = object_options.tracking.matched_object
        for i in range(len(all_grids)):
            if all_grids[i] is not None:
                all_grids[i] = all_grids[i][f"{matched_object}_grid"]
    return all_grids


# Match records are stored in "pixel" (gridcell) coordinates. Flows are reconstructed in
# cartesian or geographic coordinates as required.
ArrayLike = np.ndarray | list


def relabel_parents(ids, parents, universal_ids):
    """Relabel a list of per-object parent local ids with their universal ids."""
    new_parents = []
    for object_parents in parents:
        new_object_parents = []
        for obj_id in object_parents:
            universal_obj_id = universal_ids[ids == obj_id][0]
            new_object_parents.append(universal_obj_id)
        new_parents.append(new_object_parents)
    return new_parents


class MatchRecord(BaseModel):
    """
    Record of the matches between objects in the current and next masks for a single
    matching iteration.

    This base class holds only the identity and lineage fields that every matching
    method must produce, namely the local and universal object ids and the parent
    relationships used to track splits and merges. The shared id-assignment and matched
    mask logic depends only on these fields. Method specific quantities (object
    geometry, flow vectors, cost diagnostics, ...) belong on subclasses such as
    :class:`TintMatchRecord`.
    """

    # Records hold numpy arrays and lists of box dictionaries.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    ids: ArrayLike = Field(
        default_factory=list,
        description="Local ids of objects in the current mask (1-indexed).",
    )
    next_ids: ArrayLike = Field(
        default_factory=list,
        description="For each current object, the local id of the matched object in "
        "the next mask, or 0 if the object died.",
    )
    universal_ids: ArrayLike = Field(
        default_factory=list,
        description="Persistent (universal) id of each current object.",
    )
    parents: list = Field(
        default_factory=list,
        description="Universal ids of each current object's parents, i.e. the objects "
        "it split from or merged with at the previous time.",
    )
    next_parents: list = Field(
        default_factory=list,
        description="Local ids of each next object's parents (relabelled to universal "
        "ids once ids are assigned).",
    )

    def assign_ids(self, object_tracks, object_options, previous_record):
        """
        Assign local ids, universal ids and parents after matching.

        ``previous_record`` is the match record from the previous matching iteration.
        When it holds no objects, this is the first matchable iteration and universal
        ids are assigned from scratch; otherwise universal ids are carried over for
        matched objects and freshly minted for new ones.
        """
        previous_mask = get_masks(object_tracks, object_options)[1]
        total = int(np.max(previous_mask.values))
        ids = np.arange(1, total + 1)

        if len(previous_record.ids) == 0:
            logger.info("New matchable objects. Initializing match record.")
            universal_ids = np.arange(
                object_tracks.object_count + 1, object_tracks.object_count + total + 1
            )
            object_tracks.object_count += total
            self.parents = [[] for _ in range(total)]
        else:
            logger.info("Updating match record.")
            universal_ids = []
            for previous_id in ids:
                # Carry over the universal id if the object was matched last iteration.
                if previous_id in previous_record.next_ids:
                    cond = previous_record.next_ids == previous_id
                    index = np.argwhere(cond)[0, 0]
                    universal_ids.append(previous_record.universal_ids[index])
                else:
                    object_tracks.object_count += 1
                    universal_ids.append(object_tracks.object_count)
            universal_ids = np.array(universal_ids, dtype=int)
            # The previous iteration's next-object parents are this iteration's parents.
            self.parents = previous_record.next_parents

        self.ids = ids
        self.universal_ids = universal_ids
        self.next_parents = relabel_parents(ids, self.next_parents, universal_ids)

    def get_matched_mask(
        self, object_tracks, object_options, grid_options, current_ids=None
    ):
        """Build the matched mask for the next grid by relabelling it with universal ids."""
        next_mask = get_masks(object_tracks, object_options)[0]
        if current_ids is None:
            current_ids = np.unique(next_mask.values)
            current_ids = current_ids[current_ids != 0]
        universal_id_dict = dict(zip(self.next_ids, self.universal_ids))

        # Not all objects in the next mask appear in next_ids; these are new objects,
        # unmatched with any current object. Their universal ids will be created in the
        # next tracking iteration, but to build the matched mask now we preemptively
        # assign them (without incrementing object_count, which happens next iteration).
        unmatched_ids = [i for i in current_ids if i not in self.next_ids]
        new_universal_ids = np.arange(
            object_tracks.object_count + 1,
            object_tracks.object_count + len(unmatched_ids) + 1,
        )
        universal_id_dict.update(dict(zip(unmatched_ids, new_universal_ids)))
        universal_id_dict[0] = 0

        def replace_values(data_array, value_dict):
            series = pd.Series(data_array.ravel())
            return series.map(value_dict).values.reshape(data_array.shape)

        if grid_options.name == "cartesian":
            core_dims = [["y", "x"]]
        elif grid_options.name == "geographic":
            core_dims = [["latitude", "longitude"]]
        else:
            raise ValueError("Grid name must be 'cartesian' or 'geographic'.")

        next_matched_mask = xr.apply_ufunc(
            replace_values,
            object_tracks.next_mask,
            kwargs={"value_dict": universal_id_dict},
            input_core_dims=core_dims,
            output_core_dims=core_dims,
            vectorize=True,
        )
        # Shift the previous iteration's next matched mask into the deque, then store
        # this iteration's next matched mask.
        object_tracks.matched_masks.append(object_tracks.next_matched_mask)
        object_tracks.next_matched_mask = next_matched_mask


class TintMatchRecord(MatchRecord):
    """
    Match record for the TINT/MINT approach, adding the object geometry, flow vectors,
    search boxes and cost-function diagnostics produced by
    :func:`thuner.match.tint.get_matches`.
    """

    areas: ArrayLike = Field(
        default_factory=list,
        description="Gridcell-area weighted area of each current object in km^2.",
    )
    centers: ArrayLike = Field(
        default_factory=list,
        description="Gridcell-area weighted center of each current object in pixels.",
    )
    next_centers: ArrayLike = Field(
        default_factory=list,
        description="Center of each matched next object in pixel coordinates.",
    )
    displacements: ArrayLike = Field(
        default_factory=list,
        description="Displacement of each current object from the previous time, in "
        "pixels.",
    )
    next_displacements: ArrayLike = Field(
        default_factory=list,
        description="Displacement from each current object to its matched next object, "
        "in pixels.",
    )
    flows: ArrayLike = Field(
        default_factory=list,
        description="Local (phase correlation) flow vector for each current object.",
    )
    corrected_flows: ArrayLike = Field(
        default_factory=list,
        description="Corrected flow vector for each current object, in pixels.",
    )
    global_flows: ArrayLike = Field(
        default_factory=list,
        description="Global flow vector for each current object, in pixels.",
    )
    flow_boxes: ArrayLike = Field(
        default_factory=list,
        description="Box used for local flow estimation for each current object.",
    )
    global_flow_boxes: ArrayLike = Field(
        default_factory=list,
        description="Box used for global flow estimation for each current object.",
    )
    search_boxes: ArrayLike = Field(
        default_factory=list,
        description="Search box used to find candidate matches for each current object.",
    )
    cases: ArrayLike = Field(
        default_factory=list,
        description="Flow correction case applied to each current object.",
    )
    costs: ArrayLike = Field(
        default_factory=list,
        description="Matching cost for each current object, in km.",
    )
    distances: ArrayLike = Field(
        default_factory=list,
        description="Distance term of the cost function for each current object, in km.",
    )
    area_differences: ArrayLike = Field(
        default_factory=list,
        description="Area-difference term of the cost function for each current object.",
    )
    overlap_areas: ArrayLike = Field(
        default_factory=list,
        description="Overlap-area term of the cost function for each current object.",
    )


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
        ids = df.xs(previous_time, level="time").reset_index()
        ids = ids["universal_id"].values
        for obj_id in universal_ids:
            node = tuple([time, obj_id])
            if obj_id in ids:
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
