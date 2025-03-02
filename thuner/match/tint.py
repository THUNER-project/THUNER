"""
Perform matching using the TINT/MINT approach. The methods are discussed in the 
following papers.

Leese et al. (1971), An automated technique for obtaining cloud motion from 
geosynchronous satellite data using cross correlation. 
https://dx.doi.org/10.1175/1520-0450(1971)010<0118:AATFOC>2.0.CO;2

Dixon & Wiener (1993), TITAN: Thunderstorm Identification, Tracking, Analysis, and
Nowcasting—a radar-based methodology. 
https://dx.doi.org/10.1175/1520-0426(1993)010<0785:TTITAA>2.0.CO;2

Fridlind et al. (2019), Use of polarimetric radar measurements to constrain simulated 
convective cell evolution: a pilot study with Lagrangian tracking.
https://dx.doi.org/10.5194/amt-12-2979-2019

Raut et al. (2021), An adaptive tracking algorithm for convection in simulated and 
remote sensing data.
https://dx.doi.org/10.1175/JAMC-D-20-0119.1

Short et al. (2023), Objectively diagnosing characteristics of mesoscale organization 
from radar reflectivity and ambient winds.
https://dx.doi.org/10.1175/MWR-D-22-0146.1

"""

import numpy as np
from scipy import optimize
from thuner.match.correlate import get_flow
from thuner.match.utils import get_masks
import thuner.match.object as thuner_object
import thuner.match.box as box
from thuner.log import setup_logger
import thuner.grid as grid


logger = setup_logger(__name__)


def get_costs_data(object_tracks, object_options, grid_options):
    """Get the costs matrix used to match objects between current and next masks."""
    next_mask, current_mask = get_masks(object_tracks, object_options)
    current_total = np.max(current_mask.values)
    next_total = np.max(next_mask.values)
    local_flow_margin = object_options.tracking.local_flow_margin
    global_flow_margin = object_options.tracking.global_flow_margin

    if object_options.tracking.unique_global_flow:
        args = [global_flow_margin, grid_options]
        unique_global_flow_box = get_unique_global_flow_box(*args)
        args = [unique_global_flow_box, object_tracks, object_options, grid_options]
        args += [global_flow_margin]
        unique_global_flow, unique_global_flow_box = get_flow(*args)

    max_cost = object_options.tracking.max_cost
    gridcell_area = object_tracks.gridcell_area

    matrix_shape = [current_total, np.max([current_total, next_total])]
    costs_matrix = np.full(matrix_shape, max_cost, dtype=float)
    next_rows_matrix = np.full(matrix_shape, np.nan, dtype=float)
    next_cols_matrix = np.full(matrix_shape, np.nan, dtype=float)
    distances_matrix = np.full(matrix_shape, np.nan, dtype=float)
    area_differences_matrix = np.full(matrix_shape, np.nan, dtype=float)
    overlap_areas_matrix = np.full(matrix_shape, np.nan, dtype=float)

    flows = []
    global_flows = []
    corrected_flows = []
    cases = []
    areas = []
    centers = []
    bounding_boxes = []
    flow_boxes = []
    global_flow_boxes = []
    search_boxes = []
    displacements = []

    # Get the match record before updating it
    previous_match_record = object_tracks.match_record
    # Get the next_ids and next_displacements from the previous iteration.
    matched_ids = previous_match_record["next_ids"]
    matched_displacements = previous_match_record["next_displacements"]
    current_ids = np.arange(1, current_total + 1)

    search_margin = object_options.tracking.search_margin

    for current_id in current_ids:
        # Get the object bounding box and local flow
        bounding_box = box.get_bounding_box(current_id, current_mask)
        bounding_boxes.append(bounding_box)
        args = [bounding_box, object_tracks, object_options, grid_options]
        args += [local_flow_margin]
        flow, flow_box = get_flow(*args)

        flows.append(flow)
        flow_boxes.append(flow_box)
        # Get the global flow
        if object_options.tracking.unique_global_flow:
            global_flow = unique_global_flow
            global_flow_box = unique_global_flow_box
        else:
            args = [bounding_box, object_tracks, object_options, grid_options]
            args += [global_flow_margin]
            global_flow, global_flow_box = get_flow(*args)
        global_flows.append(global_flow)
        global_flow_boxes.append(global_flow_box)
        # Get the previous object center, displacement and area
        if current_id in matched_ids:
            displacement = matched_displacements[matched_ids == current_id].flatten()
        else:
            displacement = np.array([np.nan, np.nan])
        displacements.append(displacement)
        current_row, current_col, current_area = thuner_object.get_object_center(
            current_id, current_mask, grid_options, gridcell_area
        )
        current_center = [current_row, current_col]
        centers.append(current_center)
        areas.append(current_area)
        # Get the corrected flow
        args = [flow_box, flow, current_center, displacement, global_flow, grid_options]
        args += [object_tracks, object_options]
        logger.debug(f"Correcting flow for object {current_id}.")
        corrected_flow, case = correct_local_flow(*args)
        corrected_flows.append(corrected_flow)
        cases.append(case)
        # Get the search box, objects in search box, and evaluate cost function
        logger.debug(f"Getting search box for object {current_id}.")
        int_corrected_flow = np.ceil(corrected_flow).astype(int)
        search_box = box.get_search_box(
            bounding_box, int_corrected_flow, search_margin, grid_options
        )
        search_boxes.append(search_box)
        next_ids = thuner_object.find_objects(search_box, next_mask)
        object_costs_data = get_object_costs_data(
            next_ids, current_id, object_tracks, object_options, grid_options
        )
        i = current_id - 1
        j = next_ids - 1
        costs_matrix[i, j] = object_costs_data["costs"]
        distances_matrix[i, j] = object_costs_data["distances"]
        area_differences_matrix[i, j] = object_costs_data["area_differences"]
        overlap_areas_matrix[i, j] = object_costs_data["overlap_areas"]
        next_rows_matrix[i, j] = object_costs_data["next_rows"]
        next_cols_matrix[i, j] = object_costs_data["next_cols"]

    costs_data = {
        "costs_matrix": costs_matrix,
        "next_rows_matrix": next_rows_matrix,
        "next_cols_matrix": next_cols_matrix,
        "distances_matrix": distances_matrix,
        "area_differences_matrix": area_differences_matrix,
        "overlap_areas_matrix": overlap_areas_matrix,
        "current_ids": current_ids,
        "flow_boxes": np.array(flow_boxes),  # Use for flow vector origin
        "search_boxes": np.array(search_boxes),
        "flows": np.array(flows),
        "corrected_flows": np.array(corrected_flows),
        "cases": np.array(cases),
        "global_flows": np.array(global_flows),
        "global_flow_boxes": np.array(global_flow_boxes),
        "centers": np.array(centers),  # Displacement vector origin
        "displacements": np.array(displacements),
        "areas": np.array(areas),
    }
    return costs_data


def get_object_costs_data(
    next_ids, current_id, object_tracks, object_options, grid_options
):
    """
    Caculate the cost function for all objects found within the search box, associated with
    the specific object current_id. Note that this cost function is subtly different
    to that described by Raut et al. (2021), noting we have ignored the term associated with
    distances from the object to the center of the search box."""

    costs = []
    distances = []
    area_differences = []
    overlap_areas = []
    next_rows = []
    next_cols = []

    next_mask, current_mask = get_masks(object_tracks, object_options)
    gridcell_area = object_tracks.gridcell_area

    current_row, current_col, current_area = thuner_object.get_object_center(
        current_id, current_mask, grid_options, gridcell_area
    )

    for next_id in next_ids:

        next_row, next_col, next_area = thuner_object.get_object_center(
            next_id, next_mask, grid_options, gridcell_area
        )
        next_rows.append(next_row)
        next_cols.append(next_col)
        distance = grid.get_distance(
            current_row, current_col, next_row, next_col, grid_options
        )
        distance = distance / 1e3
        distances.append(distance)
        area_difference = np.sqrt(np.abs(next_area - current_area))
        area_differences.append(area_difference)
        overlap_cond = np.logical_and(next_mask == next_id, current_mask == current_id)
        overlap_area = np.sqrt(gridcell_area.where(overlap_cond).sum())
        overlap_areas.append(overlap_area)
        cost = distance + area_difference - overlap_area
        costs.append(cost)

    object_costs_data = {
        "costs": np.array(costs),
        "distances": np.array(distances),
        "area_differences": np.array(area_differences),
        "overlap_areas": np.array(overlap_areas),
        "current_row": current_row,
        "current_col": current_col,
        "next_rows": np.array(next_rows),
        "next_cols": np.array(next_cols),
    }

    return object_costs_data


def get_matches(object_tracks, object_options, grid_options):
    """Matches objects into pairs given a costs matrix and removes
    bad matches. Bad matches have a cost greater than the maximum
    cost."""

    logger.debug("Getting tint matches.")
    max_cost = object_options.tracking.max_cost
    costs_data = get_costs_data(object_tracks, object_options, grid_options)
    costs_matrix = costs_data["costs_matrix"]
    next_rows_matrix = costs_data["next_rows_matrix"]
    next_cols_matrix = costs_data["next_cols_matrix"]
    distances_matrix = costs_data["distances_matrix"]
    area_differences_matrix = costs_data["area_differences_matrix"]
    overlap_areas_matrix = costs_data["overlap_areas_matrix"]
    try:
        matches = optimize.linear_sum_assignment(costs_matrix)
    except ValueError:
        logger.debug("Could not solve matching problem.")
    # Initialize parents to be the same length as the number of current objects
    next_mask = get_masks(object_tracks, object_options)[0]
    next_total = np.max(next_mask.values)
    parents = [[] for i in range(next_total)]
    costs = []
    distances = []
    area_differences = []
    overlap_areas = []
    next_centers = []
    for i in matches[0]:
        cost = costs_matrix[i, matches[1][i]]
        costs.append(cost)
        next_row = next_rows_matrix[i, matches[1][i]]
        next_col = next_cols_matrix[i, matches[1][i]]
        next_centers.append([next_row, next_col])
        distance = distances_matrix[i, matches[1][i]]
        distances.append(distance)
        area_difference = area_differences_matrix[i, matches[1][i]]
        area_differences.append(area_difference)
        overlap_area = overlap_areas_matrix[i, matches[1][i]]
        overlap_areas.append(overlap_area)
        if cost >= max_cost:
            logger.debug(f"Cost {cost} exceeds max_cost {max_cost}.")
            matches[1][i] = -1
        # Determine children of objects as unmatched objects possessing area overlap > 0
        children = np.argwhere(overlap_areas_matrix[i] > 0).flatten()
        # Remove the matched object from the list of children
        if matches[1][i] in children:
            children = children[children != matches[1][i]]
        # Append object i to the list of parents of each child, recalling objects are 1 indexed
        for child in children:
            if (child not in matches[1]) or (matches[1][i] == -1):
                # Only append object i+1 as a parent of child if either the child is
                # unmatched, i.e. a new object in the next mask and hence a "split",
                # or if object i+1 is unmatched, i.e. a dead object in the current mask
                # and hence a "merge". If neither of these conditions are met, while
                # overlap may occur, both objects are matched, and hence no split/merge
                # has occurred.
                parents[child].append(i + 1)
    matches = matches[1] + 1  # Recall ids are 1 indexed. Dead objects now set to zero
    match_data = costs_data.copy()
    del match_data["costs_matrix"]
    del match_data["next_rows_matrix"]
    del match_data["next_cols_matrix"]
    del match_data["distances_matrix"]
    del match_data["area_differences_matrix"]
    del match_data["overlap_areas_matrix"]
    match_data["next_centers"] = np.array(next_centers)
    match_data["next_displacements"] = (
        match_data["next_centers"] - match_data["centers"]
    )
    match_data["next_ids"] = matches
    # For each object in the next mask, we record the ids of "parent" objects from the current mask
    match_data["next_parents"] = parents
    match_data["costs"] = np.array(costs)
    match_data["distances"] = np.array(distances)
    match_data["area_differences"] = np.array(area_differences)
    match_data["overlap_areas"] = np.array(overlap_areas)

    return match_data


def correct_local_flow(
    flow_box,
    local_flow,
    current_center,
    displacement,
    global_flow,
    grid_options,
    object_tracks,
    object_options,
):
    """Correct the local flow vector."""

    logger.debug("Correcting local flow.")
    next_time_interval = object_tracks.next_time_interval
    previous_time_interval = object_tracks.previous_time_interval
    flow_box_center = box.get_center(flow_box)
    local_flow_cartesian = grid.pixel_to_cartesian_vector(
        flow_box_center[0], flow_box_center[1], local_flow, grid_options
    )
    local_flow_velocity = local_flow_cartesian / next_time_interval  # in m/s
    # Note both global and local flows are calculated in geographic coordinates.
    # If global flow unique, still makes sense to calculate the "global" flow velocity
    # associated with a given object by calculating the cartesian displacement at the object
    # location, using the global flow vector.
    global_flow_cartesian = grid.pixel_to_cartesian_vector(
        flow_box_center[0], flow_box_center[1], global_flow, grid_options
    )
    global_flow_velocity = global_flow_cartesian / next_time_interval
    current_row, current_col = current_center[0], current_center[1]
    center_velocity = get_center_velocity(
        current_row, current_col, displacement, previous_time_interval, grid_options
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
    current_row, current_col, displacement, previous_time_interval, grid_options
):
    """Get the velocity using the object center."""
    if previous_time_interval is None:
        return None
    if np.any(np.isnan(displacement)):
        return None
    if np.all(displacement == np.array([0, 0])):
        return np.array([0, 0])
    previous_previous_row = current_row - int(displacement[0])
    previous_previous_col = current_col - int(displacement[1])
    displacement_cartesian = grid.pixel_to_cartesian_vector(
        previous_previous_row, previous_previous_col, displacement, grid_options
    )
    return displacement_cartesian / previous_time_interval


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
    tracking_options = object_options.tracking
    max_velocity_diff = tracking_options.max_velocity_diff
    max_velocity_mag = tracking_options.max_velocity_mag

    # Check for bad velocities
    bad_local = np.sqrt((local_flow_velocity**2).sum()) > max_velocity_mag
    bad_global = np.sqrt((global_flow_velocity**2).sum()) > max_velocity_mag
    bad_center_velocity = (
        center_velocity is not None
        and np.sqrt((center_velocity**2).sum()) > max_velocity_mag
    )
    if bad_global and not bad_local:
        logger.debug("Bad global flow. Setting global to local while correcting.")
        global_flow = local_flow
        global_flow_velocity = local_flow_velocity
    elif (
        bad_global
        and bad_local
        and not bad_center_velocity
        and center_velocity is not None
    ):
        message = "Bad global and local flow. "
        message += "Setting global to center while correcting."
        logger.debug(message)
        global_flow = displacement
        global_flow_velocity = center_velocity
        local_flow = displacement
        local_flow_velocity = center_velocity
    elif bad_local and bad_global and (bad_center_velocity or center_velocity is None):
        message = "Bad local, global, and center velocities. "
        message += "Setting local and global to zero, and center to None, "
        message += "while correcting local flow."
        logger.debug(message)
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
        if tracking_options.name == "mint":
            logger.debug("Using mint method.")
            # In the MINT method, we are typically matching large objects, and
            # center velocities (calculated from the displacement of object centers)
            # are often unreliable. We also want to use the local flow for object
            # velocity.
            max_velocity_diff_alt = tracking_options.max_velocity_diff_alt
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
        elif tracking_options.name == "tint":
            logger.debug("Using tint method.")

            # In the TINT method, when the local flow velocity agrees with the center
            # velocity, average the local flow and displacement.
            case = 6
            corrected_flow = (local_flow + displacement) / 2
    return corrected_flow, case


def velocities_disagree(velocity_1, velocity_2, max_velocity_diff):
    """Check if vector difference of flow velocities greater than max_velocity_diff."""

    vector_difference = np.sqrt((velocity_1**2 + velocity_2**2).sum())
    return vector_difference > max_velocity_diff


def get_unique_global_flow_box(global_flow_margin, grid_options):
    """Set the unique global flow box to the center of the domain."""
    shape = grid_options.shape
    row, col = np.floor(np.array(shape) / 2).astype(int)
    if grid_options.name == "cartesian":
        spacing = grid_options.cartesian_spacing
        # Note that the global flow margin is in km, but spacing is in m.
        radius = np.ceil(global_flow_margin * 1e3 / np.array(spacing)).astype(int)
    elif grid_options.name == "geographic":
        spacing = grid_options.geographic_spacing
        lats = grid_options.latitude
        lons = grid_options.longitude
        lat = lats[row]
        lon = lons[col] % 360
        radius = box.get_geographic_margins(lat, lon, global_flow_margin, grid_options)

    global_flow_box = box.create_box(
        row - radius[0], row + radius[0], col - radius[1], col + radius[1]
    )
    global_flow_box = box.clip_box(global_flow_box, shape)
    missing_pixels = np.max(
        [
            global_flow_box["row_min"],
            shape[0] - 1 - global_flow_box["row_max"],
            global_flow_box["col_min"],
            shape[1] - 1 - global_flow_box["col_max"],
        ]
    )
    if missing_pixels > 0:
        logger.warning(
            f"Unique global flow box under spans grid rows or columns by "
            f"up to {missing_pixels} pixels."
        )
    return global_flow_box
