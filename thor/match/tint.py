"""Perform matching using TINT approach."""

import copy
import numpy as np
from thor.match.correlate import get_local_flow
from thor.object.object import get_bounding_box, get_object_center
from thor.utils import get_cartesian_displacement
from thor.log import setup_logger

logger = setup_logger(__name__)


def initialize_object_record(matches, parents, object_tracks):
    """Initialize record of object properties in previous and current masks."""

    total_previous_objects = np.max(object_tracks["previous_masks"][-1])

    previous_ids = np.arange(1, total_previous_objects + 1)
    universal_ids = np.arange(
        object_tracks["object_count"] + 1, total_previous_objects + 1
    )
    object_tracks["object_count"] += total_previous_objects

    object_record = {
        "previous_ids": previous_ids,
        "universal_ids": universal_ids,
        "current_ids": matches,
        "parents": parents,
    }
    [
        object_record["center_displacements"],
        object_record["previous_centers"],
        object_record["current_centers"],
    ] = get_center_displacements(object_tracks)


def update_object_record(
    matches,
    parents,
    object_tracks,
    # data_dic, pairs, old_objects, counter, old_obj_merge, interval, params, grid_obj
):
    """Update record of object properties in previous and current masks."""

    previous_object_record = copy.deepcopy(object_tracks["current_object_record"])
    total_previous_objects = np.max(object_tracks["previous_masks"][-1])
    previous_ids = np.arange(1, total_previous_objects + 1)
    universal_ids = np.array([], dtype=int)

    for previous_id in np.arange(1, total_previous_objects + 1):
        # Check if object was matched in previous iteration
        if id in previous_object_record["current_ids"]:
            index = np.argwhere(previous_object_record["current_ids"] == previous_id)
            index = index[0, 0]
            # Append the previously created universal id corresponding to previous_id
            universal_ids = np.append(
                universal_ids, previous_object_record["universal_ids"][index]
            )
        else:
            uid = object_tracks["object_count"] + 1
            object_tracks["object_count"] += 1
            universal_ids = np.append(universal_ids, uid)
            # Check if new object split from old?

    object_record = {
        "previous_ids": previous_ids,
        "universal_ids": universal_ids,
        "current_ids": matches,
        "parents": parents,
    }
    [
        object_record["center_displacements"],
        object_record["previous_centers"],
        object_record["current_centers"],
    ] = get_center_displacements(object_tracks)


def get_center_displacements(object_tracks):
    current_object_record = object_tracks["current_object_record"]
    """Get the centroid displacement vectors."""
    total_current_objects = len(current_object_record["previous_ids"])
    center_displacements = np.ones((total_current_objects, 2)) * np.nan
    previous_centers = np.ones((total_current_objects, 2)) * np.nan
    current_centers = np.ones((total_current_objects, 2)) * np.nan
    previous_mask = object_tracks["previous_masks"][-1]
    current_mask = object_tracks["current_mask"]
    previous_grid = object_tracks["previous_grids"][-1]
    current_grid = object_tracks["current_grid"]

    for i in range(total_current_objects):
        if (current_object_record["previous_ids"][i] > 0) and (
            current_object_record["current_ids"][i] > 0
        ):
            previous_center = get_object_center(
                current_object_record["previous_ids"][i],
                previous_mask,
                gridcell_area=object_tracks["gridcell_area"],
                grid=previous_grid,
            )
            current_center = get_object_center(
                current_object_record["current_ids"][i],
                current_mask,
                gridcell_area=object_tracks["gridcell_area"],
                grid=current_grid,
            )
            center_displacements[i, :] = get_cartesian_displacement(
                previous_center[0],
                previous_center[1],
                current_center[0],
                current_center[1],
            )
            previous_centers[i, :] = previous_center
            current_centers[i, :] = current_center
        else:
            logger.debug("Unmatched object in previous or current mask.")

    return center_displacements, previous_centers, current_centers


def get_area_change(area1, area2):
    """Returns change in size of an echo as the ratio of the larger size to
    the smaller, minus 1."""
    if area1 >= area2:
        return area1 / area2 - 1
    else:
        return area2 / area1 - 1


def match_all_objects(input_record, object_tracks, object_options, grid_options):
    """Matches all the objects in previous mask to those in current mask."""
    previous_object_total = np.max(object_tracks["previous_masks"][-1])
    current_object_total = np.max(object_tracks["current_mask"])
    latitudes = object_tracks["current_grid"].latitude.values
    longitudes = object_tracks["current_grid"].longitude.values

    if (previous_object_total == 0) or (current_object_total == 0):
        logger.info("No objects in at least one of previous or current masks.")
        return

    matches = np.full(
        (previous_object_total, np.max((previous_object_total, current_object_total))),
        np.inf,
        dtype=float,
    )
    u_flows = []
    v_flows = []

    for obj_id in np.arange(1, previous_object_total + 1):
        bounding_box = get_bounding_box(obj_id, object_tracks["previous_mask"][-1])
        flow = get_local_flow(bounding_box, object_tracks, object_options, grid_options)
        previous_center_row = int(
            np.round((bounding_box["row_min"] + bounding_box["row_max"]) / 2)
        )
        previous_center_col = int(
            np.round((bounding_box["col_min"] + bounding_box["col_max"]) / 2)
        )
        current_center_row = previous_center_row + flow[0]
        current_center_col = previous_center_col + flow[1]
        previous_center_lat = latitudes[previous_center_row]
        previous_center_lon = longitudes[previous_center_col]
        current_center_lat = latitudes[current_center_row]
        current_center_lon = longitudes[current_center_col]

        flow_meters = get_cartesian_displacement(
            previous_center_lat,
            previous_center_lon,
            current_center_lat,
            current_center_lon,
        )
        [v_flow, u_flow] = np.array(flow_meters) / object_tracks["time_interval"]
        u_flows.append(u_flow)
        v_flows.append(v_flow)

        search_box = predict_search_extent(obj1_extent, shift, params, record.grid_size)
        search_box = check_search_box(search_box, data_dic["frame_new"].shape)
        objs_found = find_objects(search_box, data_dic["frame_new"])
        disparity = get_disparity_all(
            objs_found, data_dic["frame_new"], search_box, obj1_extent
        )
        obj_match = save_obj_match(obj_id1, objs_found, disparity, obj_match, params)

    return obj_match, u_shift, v_shift


# def correct_flow(local_flow, current_objects, obj_id1, global_shift, record, params):
#     """Takes in flow vector based on local phase correlation (see
#     get_std_flow) and compares it to the last headings of the object and
#     the global_shift vector for that timestep. Corrects accordingly.
#     Note: At the time of this function call, current_objects has not yet been
#     updated for the current frame and frame_new, so the current_idss in current_objects
#     correspond to the objects in the current frame."""
#     global_shift = clip_shift(global_shift, record, params)

#     # Note last_heads is defined using object centers! These jump around a lot
#     # when tracking large objects and should therefore probably not be used
#     # when tracking MCS systems!

#     if current_objects is None:
#         last_heads = None
#     else:
#         obj_index = current_objects["current_ids"] == obj_id1
#         last_heads = current_objects["last_heads"][obj_index].flatten()
#         last_heads = np.round(last_heads * record.interval_ratio, 2)
#         if len(last_heads) == 0:
#             last_heads = None

#     if last_heads is None:
#         if shifts_disagree(local_shift, global_shift, record, params["MAX_SHIFT_DISP"]):
#             case = 0
#             corrected_shift = global_shift
#         else:
#             case = 1
#             corrected_shift = (local_shift + global_shift) / 2

#     elif shifts_disagree(local_shift, last_heads, record, params["MAX_SHIFT_DISP"]):
#         if shifts_disagree(local_shift, global_shift, record, params["MAX_SHIFT_DISP"]):
#             case = 2
#             corrected_shift = last_heads
#         else:
#             case = 3
#             corrected_shift = local_shift

#     else:
#         case = 4
#         # corrected_shift = (local_shift + last_heads)/2
#         if shifts_disagree(
#             local_shift, global_shift, record, params["MAX_SHIFT_DISP_ALT"]
#         ):
#             corrected_shift = global_shift
#         else:
#             corrected_shift = local_shift

#     corrected_shift = np.round(corrected_shift, 2)

#     record.count_case(case)
#     record.record_shift(corrected_shift, global_shift, last_heads, local_shift, case)
#     return corrected_shift
