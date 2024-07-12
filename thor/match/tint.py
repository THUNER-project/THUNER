"""Perform matching using TINT approach."""

import numpy as np
from scipy import optimize
from thor.match.correlate import get_flow, convert_flow_cartesian
from thor.match.utils import get_masks
import thor.object.object as thor_object
import thor.object.box as box
from thor.utils import get_cartesian_displacement, geodesic_distance
from thor.log import setup_logger

logger = setup_logger(__name__)


def get_costs_data(object_tracks, object_options, grid_options):
    """Get the costs matrix used to match objects between previous and current masks."""
    current_mask, previous_mask = get_masks(object_tracks, object_options)
    previous_total = np.max(previous_mask.values)
    current_total = np.max(current_mask.values)
    latitudes = grid_options["latitude"]
    longitudes = grid_options["longitude"]
    grid_spacing = grid_options["geographic_spacing"]
    shape = current_mask.shape
    local_flow_margin = object_options["tracking"]["options"]["local_flow_margin"]
    global_flow_margin = object_options["tracking"]["options"]["global_flow_margin"]

    if object_options["tracking"]["options"]["unique_global_flow"]:
        unique_global_flow_box = box.get_unique_global_flow_box(
            latitudes, longitudes, global_flow_margin, grid_spacing, shape
        )
        unique_global_flow, unique_global_flow_box = get_flow(
            unique_global_flow_box,
            object_tracks,
            object_options,
            grid_options,
            global_flow_margin,
        )

    max_cost = object_options["tracking"]["options"]["max_cost"]
    gridcell_area = object_tracks["gridcell_area"]

    matrix_shape = [previous_total, np.max([previous_total, current_total])]
    costs_matrix = np.full(matrix_shape, max_cost, dtype=float)
    current_rows_matrix = np.full(matrix_shape, np.nan, dtype=float)
    current_cols_matrix = np.full(matrix_shape, np.nan, dtype=float)
    distances_matrix = np.full(matrix_shape, np.nan, dtype=float)
    area_differences_matrix = np.full(matrix_shape, np.nan, dtype=float)
    overlap_areas_matrix = np.full(matrix_shape, np.nan, dtype=float)

    flows = []
    global_flows = []
    corrected_flows = []
    cases = []
    areas = []
    previous_centers = []
    bounding_boxes = []
    flow_boxes = []
    global_flow_boxes = []
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
    previous_ids = np.arange(1, previous_total + 1)

    search_margin = object_options["tracking"]["options"]["search_margin"]
    spacing = grid_options["geographic_spacing"]

    for previous_id in previous_ids:
        # Get the object bounding box and local flow
        try:
            bounding_box = box.get_bounding_box(previous_id, previous_mask)
        except ValueError:
            logger.debug(f"Object {previous_id} not found in previous mask.")
        bounding_boxes.append(bounding_box)
        flow, flow_box = get_flow(
            bounding_box, object_tracks, object_options, grid_options, local_flow_margin
        )
        flows.append(flow)
        flow_boxes.append(flow_box)
        # Get the global flow
        if object_options["tracking"]["options"]["unique_global_flow"]:
            global_flow = unique_global_flow
            global_flow_box = unique_global_flow_box
        else:
            global_flow, global_flow_box = get_flow(
                bounding_box,
                object_tracks,
                object_options,
                grid_options,
                global_flow_margin,
            )
        global_flows.append(global_flow)
        global_flow_boxes.append(global_flow_box)
        # Get the previous object center, displacement and area
        if previous_id in matched_previous_ids:
            previous_displacement = matched_previous_displacements[
                matched_previous_ids == previous_id
            ].flatten()
        else:
            previous_displacement = np.array([np.nan, np.nan])
        previous_displacements.append(previous_displacement)
        previous_row, previous_col, area = thor_object.get_object_center(
            previous_id, previous_mask, gridcell_area
        )[:3]
        previous_center = [previous_row, previous_col]
        previous_centers.append(previous_center)
        areas.append(area)
        # Get the corrected flow
        corrected_flow, case = correct_local_flow(
            flow_box,  # Use this for flow vector origin
            flow,
            previous_center,  # Use this for displacement vector origin
            previous_displacement,
            global_flow,
            grid_options,
            object_tracks,
            object_options,
        )
        corrected_flows.append(corrected_flow)
        cases.append(case)
        # Get the search box, objects in search box, and evaluate cost function
        search_box = box.get_search_box(
            bounding_box,
            np.ceil(corrected_flow).astype(int),
            latitudes,
            longitudes,
            search_margin,
            spacing,
            current_mask.shape,
        )
        search_boxes.append(search_box)
        current_ids = thor_object.find_objects(search_box, current_mask)
        object_costs_data = get_object_costs_data(
            current_ids, previous_id, object_tracks, object_options
        )
        i = previous_id - 1
        j = current_ids - 1
        costs_matrix[i, j] = object_costs_data["costs"]
        distances_matrix[i, j] = object_costs_data["distances"]
        area_differences_matrix[i, j] = object_costs_data["area_differences"]
        overlap_areas_matrix[i, j] = object_costs_data["overlap_areas"]
        current_rows_matrix[i, j] = object_costs_data["current_rows"]
        current_cols_matrix[i, j] = object_costs_data["current_cols"]

    costs_data = {
        "costs_matrix": costs_matrix,
        "current_rows_matrix": current_rows_matrix,
        "current_cols_matrix": current_cols_matrix,
        "distances_matrix": distances_matrix,
        "area_differences_matrix": area_differences_matrix,
        "overlap_areas_matrix": overlap_areas_matrix,
        "previous_ids": previous_ids,
        "flow_boxes": np.array(flow_boxes),  # Use for flow vector origin
        "search_boxes": np.array(search_boxes),
        "flows": np.array(flows),
        "corrected_flows": np.array(corrected_flows),
        "cases": np.array(cases),
        "global_flows": np.array(global_flows),
        "global_flow_boxes": np.array(global_flow_boxes),
        "previous_centers": np.array(previous_centers),  # Displacement vector origin
        "previous_displacements": np.array(previous_displacements),
    }
    return costs_data


def get_object_costs_data(current_ids, previous_id, object_tracks, object_options):
    """
    Caculate the cost function for all objects found within the search box, associated with
    the specific object previous_id. Note that this cost function is subtly different
    to that described by Raut et al. (2021), noting we have ignored the term associated with
    distances from the object to the center of the search box."""

    costs = []
    distances = []
    area_differences = []
    overlap_areas = []
    current_rows = []
    current_cols = []

    current_mask, previous_mask = get_masks(object_tracks, object_options)
    gridcell_area = object_tracks["gridcell_area"]

    previous_row, previous_col, previous_area, previous_lat, previous_lon = (
        thor_object.get_object_center(previous_id, previous_mask, gridcell_area)
    )

    for current_id in current_ids:

        current_row, current_col, current_area, current_lat, current_lon = (
            thor_object.get_object_center(current_id, current_mask, gridcell_area)
        )
        current_rows.append(current_row)
        current_cols.append(current_col)
        distance = geodesic_distance(
            previous_lon, previous_lat, current_lon, current_lat
        )
        distance = distance / 1e3
        distances.append(distance)
        area_difference = np.sqrt(np.abs(current_area - previous_area))
        area_differences.append(area_difference)
        overlap_cond = np.logical_and(
            current_mask == current_id, previous_mask == previous_id
        )
        overlap_area = np.sqrt(gridcell_area.where(overlap_cond).sum())
        overlap_areas.append(overlap_area)
        cost = distance + area_difference - overlap_area
        costs.append(cost)

    object_costs_data = {
        "costs": np.array(costs),
        "distances": np.array(distances),
        "area_differences": np.array(area_differences),
        "overlap_areas": np.array(overlap_areas),
        "previous_row": previous_row,
        "previous_col": previous_col,
        "current_rows": np.array(current_rows),
        "current_cols": np.array(current_cols),
    }

    return object_costs_data


def get_matches(object_tracks, object_options, grid_options):
    """Matches objects into pairs given a costs matrix and removes
    bad matches. Bad matches have a cost greater than the maximum
    cost."""

    max_cost = object_options["tracking"]["options"]["max_cost"]
    costs_data = get_costs_data(object_tracks, object_options, grid_options)
    costs_matrix = costs_data["costs_matrix"]
    current_rows_matrix = costs_data["current_rows_matrix"]
    current_cols_matrix = costs_data["current_cols_matrix"]
    distances_matrix = costs_data["distances_matrix"]
    area_differences_matrix = costs_data["area_differences_matrix"]
    overlap_areas_matrix = costs_data["overlap_areas_matrix"]
    try:
        matches = optimize.linear_sum_assignment(costs_matrix)
    except ValueError:
        logger.debug("Could not solve matching problem.")
    parents = []
    costs = []
    distances = []
    area_differences = []
    overlap_areas = []
    current_centers = []
    for i in matches[0]:
        cost = costs_matrix[i, matches[1][i]]
        costs.append(cost)
        current_row = current_rows_matrix[i, matches[1][i]]
        current_col = current_cols_matrix[i, matches[1][i]]
        current_centers.append([current_row, current_col])
        distance = distances_matrix[i, matches[1][i]]
        distances.append(distance)
        area_difference = area_differences_matrix[i, matches[1][i]]
        area_differences.append(area_difference)
        overlap_area = overlap_areas_matrix[i, matches[1][i]]
        overlap_areas.append(overlap_area)
        parents.append(None)
        if cost >= max_cost:
            # Set to -1 if object has died (or merged)
            matches[1][i] = -1
    matches = matches[1] + 1  # Recall ids are 1 indexed. Dead objects now set to zero
    match_data = costs_data.copy()
    del match_data["costs_matrix"]
    del match_data["current_rows_matrix"]
    del match_data["current_cols_matrix"]
    del match_data["distances_matrix"]
    del match_data["area_differences_matrix"]
    del match_data["overlap_areas_matrix"]
    match_data["matched_current_centers"] = np.array(current_centers)
    match_data["current_displacements"] = (
        match_data["matched_current_centers"] - match_data["previous_centers"]
    )
    match_data["matched_current_ids"] = matches
    match_data["parents"] = np.array(parents)
    match_data["costs"] = np.array(costs)
    match_data["distances"] = np.array(distances)
    match_data["area_differences"] = np.array(area_differences)
    match_data["overlap_areas"] = np.array(overlap_areas)

    return match_data


def correct_local_flow(
    flow_box,
    local_flow,
    previous_center,
    displacement,
    global_flow,
    grid_options,
    object_tracks,
    object_options,
):
    """Correct the local flow vector."""

    spacing = grid_options["geographic_spacing"]
    lats = grid_options["latitude"]
    lons = grid_options["longitude"]
    current_time_interval = object_tracks["current_time_interval"]
    previous_time_interval = object_tracks["previous_time_interval"]

    flow_box_center = box.get_center(flow_box)
    flow_box_lat = lats[flow_box_center[0]]
    flow_box_lon = lons[flow_box_center[1]]
    local_cartesian_flow = convert_flow_cartesian(
        local_flow, flow_box_lat, flow_box_lon, spacing
    )
    local_flow_velocity = local_cartesian_flow / current_time_interval
    # Note both global and local flows are calculated in geographic coordinates.
    # Still makes sense to calculate the "global" flow velocity associated with a given object
    # by calculating the cartesian flow at the object location using the global flow.
    global_cartesian_flow = convert_flow_cartesian(
        global_flow, flow_box_lat, flow_box_lon, spacing
    )
    global_flow_velocity = global_cartesian_flow / current_time_interval

    previous_row, previous_col = [previous_center[0], previous_center[1]]
    previous_lat, previous_lon = (lats[previous_row], lons[previous_col])
    center_velocity = get_center_velocity(
        displacement, previous_lat, previous_lon, spacing, previous_time_interval
    )
    corrected_flow, case = determine_case(
        local_flow,
        local_flow_velocity,
        global_flow,
        global_flow_velocity,
        displacement,
        center_velocity,
        object_options,
    )

    return corrected_flow, case


def get_center_velocity(
    displacement, previous_lat, previous_lon, spacing, previous_time_interval
):
    """Get the velocity using the object center."""
    displacement_exists = ~np.all(np.isnan(displacement))
    if ~displacement_exists or previous_time_interval is None:
        center_velocity = None
    else:
        if np.all(displacement == np.array([0, 0])):
            center_velocity = np.array([0, 0])
        else:
            previous_previous_lat = previous_lat - displacement[0] * spacing[0]
            previous_previous_lon = previous_lon - displacement[1] * spacing[1]
            cartesian_displacement = get_cartesian_displacement(
                previous_previous_lat, previous_previous_lon, previous_lat, previous_lon
            )
            center_velocity = cartesian_displacement / previous_time_interval
    return center_velocity


def determine_case(
    local_flow,
    local_flow_velocity,
    global_flow,
    global_flow_velocity,
    displacement,
    center_velocity,
    object_options,
):
    """Determine the case for the TINT/MINT flow correction. Note that geographic
    coordinates require that we convert flows (i.e. the displacements in "pixel"
    coordinates) to cartesian coordinates to compare vectors consistently."""
    tracking_options = object_options["tracking"]["options"]
    max_velocity_diff = tracking_options["max_velocity_diff"]
    max_velocity_mag = tracking_options["max_velocity_mag"]
    method = object_options["tracking"]["method"]

    # Check for bad velocities
    bad_local = np.sqrt((local_flow_velocity**2).sum()) > max_velocity_mag
    bad_global = np.sqrt((global_flow_velocity**2).sum()) > max_velocity_mag
    bad_center_velocity = (
        center_velocity is not None
        and np.sqrt((center_velocity**2).sum()) > max_velocity_mag
    )
    if bad_global and not bad_local:
        logger.debug(
            "Bad global flow. " "Setting global to local while correcting local flow."
        )
        global_flow = local_flow
        global_flow_velocity = local_flow_velocity
    elif (
        bad_global
        and bad_local
        and not bad_center_velocity
        and center_velocity is not None
    ):
        logger.debug(
            "Bad global and local flow. "
            "Setting both to center while correcting local flow."
        )
        global_flow = displacement
        global_flow_velocity = center_velocity
        local_flow = displacement
        local_flow_velocity = center_velocity
    elif bad_local and bad_global and (bad_center_velocity or center_velocity is None):
        logger.debug(
            "Bad local, global, and center velocities. "
            "Setting local and global to zero, and center to None, "
            "while correcting local flow."
        )
        global_flow = np.array([0, 0])
        global_flow_velocity = np.array([0, 0])
        local_flow = np.array([0, 0])
        local_flow_velocity = np.array([0, 0])
        displacement = np.array([np.nan, np.nan])
        center_velocity = None

    if center_velocity is None:
        if velocities_disagree(
            local_flow_velocity, global_flow_velocity, max_velocity_diff
        ):
            # If there is no displacement, trust the global flow
            # if local and global flow velocities disagree.
            case = 0
            corrected_flow = global_flow.astype(int)
        else:
            # Otherwise, average the local and global flows.
            case = 1
            corrected_flow = (local_flow + global_flow) / 2
    elif velocities_disagree(local_flow_velocity, center_velocity, max_velocity_diff):
        if velocities_disagree(
            local_flow_velocity, global_flow_velocity, max_velocity_diff
        ):
            # If the local flow velocity disagrees with the center velocity and the
            # global flow velocity, trust the center displacement.
            case = 2
            corrected_flow = displacement.astype(int)
        else:
            # Otherwise, if the local flow velocity agrees with both the center velocity
            # and the global flow velocity, trust the local flow velocity.
            case = 3
            corrected_flow = local_flow.astype(int)
    else:
        if method == "mint":
            # In the MINT method, we are typically matching large objects, and
            # center velocities (calculated from the displacement of object centers)
            # are often unreliable. We also want to use the local flow for object
            # velocity.
            max_velocity_diff_alt = tracking_options["max_velocity_diff_alt"]
            if velocities_disagree(
                local_flow_velocity, global_flow_velocity, max_velocity_diff_alt
            ):
                # If the local flow velocity greatly agrees with the center velocity
                # but greatly disagrees with the global flow velocity, trust the global
                # flow.
                case = 4
                corrected_flow = global_flow.astype(int)
            else:
                # Otherwise, trust the local flow.
                case = 5
                corrected_flow = local_flow.astype(int)
        elif method == "tint":
            # In the TINT method, when the local flow velocity agrees with the center
            # velocity, average the local flow and displacement.
            case = 6
            corrected_flow = (local_flow + displacement) / 2
    return corrected_flow, case


def velocities_disagree(velocity_1, velocity_2, max_velocity_diff):
    """Check if vector difference of flow velocities greater than max_velocity_diff."""

    vector_difference = np.sqrt((velocity_1**2 + velocity_2**2).sum())
    return vector_difference > max_velocity_diff
