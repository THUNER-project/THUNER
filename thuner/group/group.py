"""Module for grouping objects into new objects."""

import copy
import numpy as np
import xarray as xr
import networkx as nx
from networkx.algorithms.components.connected import connected_components
from thuner.utils import get_time_interval


def group(
    track_input_records,
    tracks,
    level_index,
    obj,
    dataset_options,
    object_options,
    grid_options,
):
    """Group objects into new objects."""

    dataset = track_input_records[object_options.dataset].dataset
    if tracks.levels[level_index].objects[obj].gridcell_area is None:
        tracks.levels[level_index].objects[obj].gridcell_area = dataset["gridcell_area"]
    member_objects = object_options.grouping.member_objects
    member_levels = object_options.grouping.member_levels

    grid_dict = {}
    for member_obj, member_level in zip(member_objects, member_levels):
        member_grid = tracks.levels[member_level].objects[member_obj].next_grid
        grid_dict[f"{member_obj}_grid"] = member_grid

    # Store the domain boundaries associated with the consituent masks
    grid = xr.Dataset(grid_dict)
    mask = get_connected_components(tracks, object_options)

    previous_mask = copy.deepcopy(tracks.levels[level_index].objects[obj].next_mask)
    tracks.levels[level_index].objects[obj].masks.append(previous_mask)
    tracks.levels[level_index].objects[obj].next_mask = mask

    previous_grid = copy.deepcopy(tracks.levels[level_index].objects[obj].next_grid)
    tracks.levels[level_index].objects[obj].grids.append(previous_grid)
    tracks.levels[level_index].objects[obj].next_grid = grid

    tracks.levels[level_index].objects[obj].previous_time_interval = copy.deepcopy(
        tracks.levels[level_index].objects[obj].next_time_interval
    )
    tracks.levels[level_index].objects[obj].next_time_interval = get_time_interval(
        grid, previous_grid
    )


def get_connected_components(tracks, object_options):
    """Calculate connected components from a dictionary of masks."""

    member_objects = object_options.grouping.member_objects
    member_levels = object_options.grouping.member_levels

    # Relabel objects in mask so that object numbers are unique
    current_max = 0
    masks = []
    for obj, level in zip(member_objects, member_levels):
        mask = tracks.levels[level].objects[obj].next_mask
        new_mask = mask.copy()
        new_mask = new_mask.where(mask == 0, mask + current_max)
        masks.append(new_mask.values)
        current_max += np.max(mask.values)

    # Create graph for objects that overlap at different vertical levels.
    overlap_graph = nx.Graph()
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
        mask_da = xr.full_like(
            tracks.levels[level].objects[obj].next_mask, 0, dtype=int
        )
        mask_da_list.append(mask_da)

    # Create new objects based on connected components
    new_objs = list(connected_components(overlap_graph))
    # Create a counter, as some of the connected components will be rejected
    new_obj_counter = 0
    for i in range(len(new_objs)):
        # Require that components span all member objects
        if not component_span(masks, list(new_objs[i])):
            continue
        # Require total areas of member objects are above thresholds after grouping
        if not check_areas(masks, tracks, object_options, list(new_objs[i])):
            continue
        # Create new grouped objects
        new_obj_counter += 1
        for j in range(len(mask_da_list)):
            mask_da_list[j] = mask_da_list[j].where(
                ~np.isin(masks[j], list(new_objs[i])), new_obj_counter
            )
    grouped_mask = xr.Dataset({da.name: da for da in mask_da_list})
    return grouped_mask


def check_areas(masks, tracks, object_options, objs):
    """
    Check if the areas of the member objects after grouping are above the threshold.
    """
    member_objects = object_options.grouping.member_objects
    member_levels = object_options.grouping.member_levels
    member_min_areas = object_options.grouping.member_min_areas
    for j in range(len(masks)):
        mem_obj = tracks.levels[member_levels[j]].objects[member_objects[j]]
        mask_j = np.isin(masks[j], objs)
        if mem_obj.gridcell_area.where(mask_j).sum() < member_min_areas[j]:
            return False
    return True


def component_span(masks, new_objs):
    """Check if connected component spans all member objects."""

    in_mask = []
    for i in range(len(masks)):
        in_mask.append(any([j in masks[i] for j in new_objs]))

    return all(in_mask)
