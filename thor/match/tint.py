"""Perform matching using TINT approach."""

import numpy as np
from scipy import optimize
from thor.match.correlate import get_local_flow, convert_flow_cartesian
from thor.match.utils import get_masks, get_grids
import thor.object.object as thor_object
import thor.object.box as box
from thor.utils import geodesic_distance
from thor.log import setup_logger

logger = setup_logger(__name__)


def get_costs_matrix(object_tracks, object_options, grid_options):
    """Get the costs matrix used to match objects between previous and current masks."""
    current_mask, previous_mask = get_masks(object_tracks, object_options)
    previous_total = np.max(previous_mask.values)
    current_total = np.max(current_mask.values)
    latitudes = current_mask.latitude.values
    longitudes = previous_mask.longitude.values

    if (previous_total == 0) or (current_total == 0):
        logger.info("No objects in at least one of previous or current masks.")
        return

    max_cost = object_options["tracking"]["options"]["max_cost"]

    matrix_shape = [previous_total, np.max([previous_total, current_total])]
    costs_matrix = np.full(matrix_shape, max_cost, dtype=float)
    current_lats_matrix = np.full(matrix_shape, np.nan, dtype=float)
    current_lons_matrix = np.full(matrix_shape, np.nan, dtype=float)
    flow_velocities = []
    flows = []
    previous_centers = []
    bounding_boxes = []
    flow_boxes = []
    search_boxes = []
    previous_displacements = []

    # Get the object record before updating it
    previous_object_record = object_tracks["object_record"]
    # Get the matched "current" ids and displacements from the previous object record.
    # These variables correspond to objects in the current mask of previous iteration.
    # These variables now correspond to the subset of objects in the "previous mask"
    # of the current iteration which were matched during the previous iteration.
    matched_previous_ids = previous_object_record["matched_current_ids"]
    matched_previous_displacements = previous_object_record["current_displacements"]

    for previous_id in np.arange(1, previous_total + 1):
        bounding_box = box.get_bounding_box(previous_id, previous_mask)
        bounding_boxes.append(bounding_box)
        flow, flow_box = get_local_flow(
            bounding_box, object_tracks, object_options, grid_options
        )
        flows.append(flow)
        flow_boxes.append(flow_box)
        flow_meters = convert_flow_cartesian(flow, bounding_box, latitudes, longitudes)
        flow_velocity = np.array(flow_meters) / object_tracks["time_interval"]
        flow_velocities.append(flow_velocity)

        if previous_id in matched_previous_ids:
            previous_displacement = matched_previous_displacements[
                matched_previous_ids == previous_id
            ].flatten()
        else:
            previous_displacement = np.array([np.nan, np.nan])
        previous_displacements.append(previous_displacement)

        search_margin = object_options["tracking"]["options"]["search_margin"]
        grid_spacing = grid_options["geographic_spacing"]
        row_margin = int(np.ceil(search_margin / grid_spacing[0]))
        col_margin = int(np.ceil(search_margin / grid_spacing[1]))
        search_box = bounding_box.copy()
        search_box = box.expand_box(search_box, row_margin, col_margin)
        search_box = box.shift_box(search_box, flow[0], flow[1])
        search_box = box.clip_box(search_box, current_mask.shape)
        search_boxes.append(search_box)
        current_ids = thor_object.find_objects(search_box, current_mask)
        costs, previous_lat, previous_lon, current_lats, current_lons = get_costs(
            current_ids, previous_id, object_tracks, object_options
        )
        costs_matrix[previous_id - 1, current_ids - 1] = costs
        current_lats_matrix[previous_id - 1, current_ids - 1] = current_lats
        current_lons_matrix[previous_id - 1, current_ids - 1] = current_lons
        previous_centers.append([previous_lat, previous_lon])

    costs_data = {
        "costs_matrix": costs_matrix,
        "current_lats_matrix": current_lats_matrix,
        "current_lons_matrix": current_lons_matrix,
        "flows": np.array(flows),
        "flow_velocities": np.array(flow_velocities),
        "previous_centers": np.array(previous_centers),
        "previous_displacements": np.array(previous_displacements),
        "bounding_boxes": np.array(bounding_boxes),
        "flow_boxes": np.array(flow_boxes),
        "search_boxes": np.array(search_boxes),
    }
    return costs_data


def get_costs(current_ids, previous_id, object_tracks, object_options):
    """
    Caculate the cost function for all objects found within the search box. Note that
    this cost function is subtly different to that described by Raut et al. (2021), as
    distances are calculated from objects within the search box to the centre
    of the search box. This inconsistency suggests the R and Python versions of TINT
    are slightly different."""

    costs = np.array([])
    current_lats = np.array([])
    current_lons = np.array([])

    current_mask, previous_mask = get_masks(object_tracks, object_options)
    current_grid, previous_grid = get_grids(object_tracks, object_options)
    gridcell_area = object_tracks["gridcell_area"]

    previous_lat, previous_lon, previous_area = thor_object.get_object_center(
        previous_id, previous_mask, gridcell_area, previous_grid
    )

    if len(current_ids) == 0:
        return None, previous_lat, previous_lon, current_lats, current_lons

    for current_id in current_ids:

        current_lat, current_lon, current_area = thor_object.get_object_center(
            current_id, current_mask, gridcell_area, current_grid
        )
        distance = geodesic_distance(
            previous_lon, previous_lat, current_lon, current_lat
        )
        cost = distance / 1e3 + np.sqrt(np.abs(current_area - previous_area))
        costs = np.append(costs, cost)
        current_lats = np.append(current_lats, current_lat)
        current_lons = np.append(current_lons, current_lon)
    return costs, previous_lat, previous_lon, current_lats, current_lons


def get_matches(object_tracks, object_options, grid_options):
    """Matches objects into pairs given a disparity matrix and removes
    bad matches. Bad matches have a disparity greater than the maximum
    threshold."""

    max_cost = object_options["tracking"]["options"]["max_cost"]
    costs_data = get_costs_matrix(object_tracks, object_options, grid_options)
    costs_matrix = costs_data["costs_matrix"]
    current_lats_matrix = costs_data["current_lats_matrix"]
    current_lons_matrix = costs_data["current_lons_matrix"]
    try:
        matches = optimize.linear_sum_assignment(costs_matrix)
    except ValueError:
        logger.debug("Could not solve matching problem.")
    parents = []
    costs = []
    current_centers = []
    for i in matches[0]:
        cost = costs_matrix[i, matches[1][i]]
        current_lat = current_lats_matrix[i, matches[1][i]]
        current_lon = current_lons_matrix[i, matches[1][i]]
        parents.append(None)
        costs.append(cost)
        current_centers.append([current_lat, current_lon])
        if cost >= max_cost:
            # Set to -1 if object has died (or merged)
            matches[1][i] = -1
    matches = matches[1] + 1  # Recall ids are 1 indexed
    match_data = costs_data.copy()
    del match_data["costs_matrix"]
    del match_data["current_lats_matrix"]
    del match_data["current_lons_matrix"]
    match_data["matched_current_centers"] = np.array(current_centers)
    match_data["current_displacements"] = (
        match_data["matched_current_centers"] - match_data["previous_centers"]
    ) / grid_options["geographic_spacing"]
    match_data["matched_current_ids"] = matches
    match_data["parents"] = np.array(parents)
    match_data["costs"] = np.array(costs)

    return match_data


# def correct_flow_tint(local_flow, object_record, global_flow):
#     """Correct the local flow vector using the TINT approach."""
#     """Correct the local flow vector using the MINT approach."""
#     global_shift = clip_shift(global_shift, record, params)

#     if current_objects is None:
#         last_heads = None
#     else:
#         obj_index = current_objects["id2"] == obj_id1
#         last_heads = current_objects["last_heads"][obj_index].flatten()
#         last_heads = np.round(last_heads * record.interval_ratio, 2)
#         if len(last_heads) == 0:
#             last_heads = None

#     if last_heads is None:
#         if shifts_disagree(local_shift, global_shift, record, params):
#             case = 0
#             corrected_shift = global_shift
#         else:
#             case = 1
#             corrected_shift = (local_shift + global_shift) / 2

#     elif shifts_disagree(local_shift, last_heads, record, params):
#         if shifts_disagree(local_shift, global_shift, record, params):
#             case = 2
#             corrected_shift = last_heads
#         else:
#             case = 3
#             corrected_shift = local_shift

#     else:
#         case = 4
#         corrected_shift = (local_shift + last_heads) / 2

#     corrected_shift = np.round(corrected_shift, 2)

#     record.count_case(case)
#     record.record_shift(corrected_shift, global_shift, last_heads, local_shift, case)
#     return corrected_shift


def correct_local_flow(local_flow, global_flow, displacement, interval_ratio):
    """Correct the local flow vector using the MINT approach."""
    if displacement == np.array([np.nan, np.nan]) or interval_ratio is None:
        displacement = None
    else:
        # Scale the displacement vector by the ratio of time intervals for the current
        # and previous iteration.
        scaled_displacement = displacement * interval_ratio

    if last_heads is None:
        if shifts_disagree(local_shift, global_shift, record, params["MAX_SHIFT_DISP"]):
            case = 0
            corrected_shift = global_shift
        else:
            case = 1
            corrected_shift = (local_shift + global_shift) / 2

    elif shifts_disagree(local_shift, last_heads, record, params["MAX_SHIFT_DISP"]):
        if shifts_disagree(local_shift, global_shift, record, params["MAX_SHIFT_DISP"]):
            case = 2
            corrected_shift = last_heads
        else:
            case = 3
            corrected_shift = local_shift

    else:
        case = 4
        # corrected_shift = (local_shift + last_heads)/2
        if shifts_disagree(
            local_shift, global_shift, record, params["MAX_SHIFT_DISP_ALT"]
        ):
            corrected_shift = global_shift
        else:
            corrected_shift = local_shift

    corrected_shift = np.round(corrected_shift, 2)

    record.count_case(case)
    record.record_shift(corrected_shift, global_shift, last_heads, local_shift, case)
    return corrected_shift
