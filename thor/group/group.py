"""Module for grouping objects into new objects."""

import copy
import numpy as np
import xarray as xr
import networkx
from networkx.algorithms.components.connected import connected_components
import thor.detect.preprocess as preprocess


def group(track_input_records, tracks, level_index, obj, object_options, grid_options):
    """Group objects into new objects."""

    input_record = track_input_records[object_options["dataset"]]
    grid = input_record["current_grid"]
    altitudes = [grid.altitude.values.min(), grid.altitude.values.max()]
    processed_grid = preprocess.vertical_max(
        grid, {"detection": {"altitudes": altitudes}}
    )
    tracks[level_index][obj]["processed_grid"] = processed_grid

    mask = get_connected_components(tracks, object_options)

    current_mask = copy.deepcopy(tracks[level_index][obj]["current_mask"])
    if current_mask is not None:
        tracks[level_index][obj]["previous_masks"].append(current_mask)
    tracks[level_index][obj]["current_mask"] = mask


def get_connected_components(tracks, object_options):
    """Calculate connected components from a dictionary of masks."""

    member_objects = object_options["grouping"]["member_objects"]
    member_levels = object_options["grouping"]["member_levels"]

    # Relabel objects in mask so that object numbers are unique
    current_max = 0
    masks = []
    for obj, level in zip(member_objects, member_levels):
        mask = tracks[level][obj]["current_mask"]
        new_mask = mask.copy()
        new_mask = new_mask.where(mask == 0, mask + current_max)
        masks.append(new_mask.values)
        current_max += np.max(mask.values)

    # Create graph for objects that overlap at different vertical levels.
    overlap_graph = networkx.Graph()
    overlap_graph.add_nodes_from(set(range(1, current_max)))

    # Create edges between the objects that overlap vertically, assuming member objects
    # listed in increasing altitude.
    for i in range(len(masks) - 1):
        # Determine the objects in frame i.
        objects = set(np.unique(masks[i])).difference({0})
        for j in objects:
            # Determine the objects in frame i + 1 that overlap with object j.
            overlap = np.logical_and(masks[i] == j, masks[i + 1] > 0)
            overlap_objects = set(masks[i + 1][overlap].flatten())
            # If objects overlap, add edge between object j and first
            # object from overlap set
            for k in overlap_objects:
                overlap_graph.add_edges_from([(j, k)])

    # Initialize a new mask ds to represent the grouped object.
    mask_da_list = []
    for obj, level in zip(member_objects, member_levels):
        mask_da = xr.full_like(tracks[level][obj]["current_mask"], 0, dtype=int)
        mask_da_list.append(mask_da)

    # Create new objects based on connected components
    new_objs = list(connected_components(overlap_graph))
    for i in range(len(new_objs)):
        # Require that components span all member objects
        if not component_span(masks, new_objs[i]):
            continue
        # Create new grouped objects
        for j in range(len(mask_da_list)):
            mask_da_list[j] = mask_da_list[j].where(
                ~np.isin(masks[j], list(new_objs[i])), i + 1
            )

    return xr.Dataset({da.name: da for da in mask_da_list})


def component_span(masks, new_objs):
    """Check if connected component spans all member objects."""

    in_mask = []
    for i in range(len(masks)):
        in_mask.append(any([j in masks[i] for j in new_objs]))

    return all(in_mask)
