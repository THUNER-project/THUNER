"""
Methods for defining property options associated with detected objects, and for 
measuring such properties.
"""

import numpy as np
from thor.log import setup_logger
import thor.grid as grid
from thor.object.object import get_object_center

logger = setup_logger(__name__)


def coordinates_from_object_record(
    attribute_name, time, object_tracks, object_options, grid_options
):
    """Get coordinate from object record created by the matching process."""

    pixel_coordinates = object_tracks["object_record"]["previous_centers"]
    latitude, longitude = grid_options["latitude"], grid_options["longitude"]
    latitudes = []
    longitudes = []
    for pixel_coordinate in pixel_coordinates:
        if grid_options["name"] == "geographic":
            latitudes.append(latitude[pixel_coordinate[0]])
            longitudes.append(longitude[pixel_coordinate[1]])
        elif grid_options["name"] == "cartesian":
            latitudes.append(latitude[pixel_coordinate[0], pixel_coordinate[1]])
            longitudes.append(longitude[pixel_coordinate[0], pixel_coordinate[1]])
    return latitudes, longitudes


def areas_from_object_record(
    attribute_name, time, object_tracks, object_options, grid_options
):
    """Get area from object record created by the matching process."""
    areas = object_tracks["object_record"]["previous_areas"]
    attribute_options = object_options["attribute"]["core"]
    data_type = attribute_options[attribute_name]["data_type"]
    return np.round(areas, 1).astype(data_type)


def velocities_from_object_record(
    attribute_name, time, object_tracks, object_options, grid_options
):
    """Get velocity from object record created by the matching process."""
    centers = object_tracks["object_record"]["previous_centers"]
    # Get the "shift" vectors, i.e. the distance vector between the object's previous and
    # current centers in pixel units.
    if attribute_name in ["u_flow", "v_flow"]:
        shifts = object_tracks["object_record"]["corrected_flows"]
    elif attribute_name in ["u_displacement", "v_displacement"]:
        shifts = object_tracks["object_record"]["current_displacements"]
    else:
        raise ValueError(f"Attribute {attribute_name} not recognised.")
    v_list, u_list = [[] for i in range(2)]
    time_interval = object_tracks["current_time_interval"]
    for i, shift in enumerate(shifts):
        if np.any(np.isnan(np.array(shift))):
            v_list.append(np.nan), u_list.append(np.nan)
            logger.info(f"Object {i} has a nan {attribute_name}.")
            continue
        row, col = centers[i]
        distance = grid.pixel_to_cartesian_vector(row, col, shift, grid_options)
        v, u = np.array(distance) / time_interval
        v_list.append(v), u_list.append(u)
    attribute_options = object_options["attribute"]["core"]
    data_type = attribute_options[attribute_name]["data_type"]
    v_list = np.round(np.array(v_list), 1).astype(data_type)
    u_list = np.round(np.array(u_list), 1).astype(data_type)
    return v_list, u_list


def coordinates_from_mask(
    attribute_name, time, object_tracks, object_options, grid_options
):
    """Get object coordinate from mask."""
    mask = object_tracks["previous_masks"][-1]
    gridcell_area = object_tracks["gridcell_area"]
    ids = ids_from_mask(attribute_name, object_tracks, object_options)

    lats, lons = [], []
    for obj_id in ids:
        row, col = get_object_center(obj_id, mask, grid_options, gridcell_area)[:2]
        if grid_options["name"] == "geographic":
            lats.append(grid_options["latitude"][row])
            lons.append(grid_options["longitude"][col])
        elif grid_options["name"] == "cartesian":
            lats.append(grid_options["latitude"][row, col])
            lons.append(grid_options["longitude"][row, col])

    attribute_options = object_options["attribute"]["core"]
    data_type = attribute_options[attribute_name]["data_type"]
    lats = np.round(np.array(lats), 4).astype(data_type)
    lons = np.round(np.array(lons), 4).astype(data_type)
    return lats, lons


def areas_from_mask(attribute_name, time, object_tracks, object_options, grid_options):
    """Get object area from mask."""
    mask = object_tracks["previous_masks"][-1]
    gridcell_area = object_tracks["gridcell_area"]
    ids = ids_from_mask(attribute_name, object_tracks, object_options)

    areas = []
    for obj_id in ids:
        area = get_object_center(obj_id, mask, grid_options, gridcell_area)[2]
        areas.append(area)

    attribute_options = object_options["attribute"]["core"]
    data_type = attribute_options[attribute_name]["data_type"]
    areas = np.round(np.array(areas), 1).astype(data_type)
    return areas


def ids_from_mask(attribute_name, object_tracks, object_options):
    """Get object ids from a labelled mask."""
    previous_mask = object_tracks["previous_masks"][-1]
    if previous_mask is None:
        return None
    else:
        ids = np.unique(previous_mask)
        ids = sorted(list(ids[ids != 0]))
        return ids


def ids_from_object_record(attribute_name, object_tracks, object_options):
    """Get object ids from the object record."""
    object_record_names = {"universal_id": "universal_ids", "id": "previous_ids"}
    ids = object_tracks["object_record"][object_record_names[attribute_name]]
    return ids


get_attributes_dispatcher = {
    "coordinates_from_object_record": coordinates_from_object_record,
    "coordinates_from_mask": coordinates_from_mask,
    "areas_from_object_record": areas_from_object_record,
    "areas_from_mask": areas_from_mask,
    "velocities_from_object_record": velocities_from_object_record,
    "ids_from_mask": ids_from_mask,
    "ids_from_object_record": ids_from_object_record,
}


def record_coordinates(
    core_attributes,
    attribute_options,
    time,
    object_tracks,
    object_options,
    grid_options,
):
    """Record object coordinates."""
    keys = core_attributes.keys()
    if not "latitude" in keys or not "longitude" in keys:
        message = "Both latitude and longitude must be specified."
        raise ValueError(message)
    func = attribute_options["latitude"]["method"]["function"]
    lon_func = attribute_options["longitude"]["method"]["function"]
    if func != lon_func:
        message = "Functions for acquring latitude and longitude must be the same."
        raise ValueError(message)
    get = get_attributes_dispatcher.get(func)
    if get is None:
        message = f"Function {func} for obtaining lat and lon not recognised."
        raise ValueError(message)

    lats, lons = get("latitude", time, object_tracks, object_options, grid_options)
    core_attributes["latitude"] += list(lats)
    core_attributes["longitude"] += list(lons)


def record_velocities(
    core_attributes,
    attribute_options,
    time,
    object_tracks,
    object_options,
    grid_options,
    velocity_type="flow",
):
    """Record object coordinates."""
    keys = core_attributes.keys()
    if not f"u_{velocity_type}" in keys or not f"v_{velocity_type}" in keys:
        message = "Both u and v compononents must be specified."
        raise ValueError(message)
    func = attribute_options[f"u_{velocity_type}"]["method"]["function"]
    lon_func = attribute_options[f"v_{velocity_type}"]["method"]["function"]
    if func != lon_func:
        message = "Functions for acquring u and v velocities must be the same."
        raise ValueError(message)
    get = get_attributes_dispatcher.get(func)
    if get is None:
        message = f"Function {func} for obtaining u and v not recognised."
        raise ValueError(message)

    v, u = get(f"u_{velocity_type}", time, object_tracks, object_options, grid_options)
    core_attributes[f"v_{velocity_type}"] += list(v)
    core_attributes[f"u_{velocity_type}"] += list(u)


def get_ids(time, object_tracks, object_options):
    """Get object ids."""

    attribute_options = object_options["attribute"]["core"]
    if attribute_options is None:
        return
    if "universal_id" in object_options["attribute"]["core"]:
        id_type = "universal_id"
    elif "id" in object_options["attribute"]["core"]:
        id_type = "id"

    func = attribute_options[id_type]["method"]["function"]
    get = get_attributes_dispatcher.get(func)
    ids = get(id_type, object_tracks, object_options)

    if ids is not None:
        ids = np.array(ids).astype(attribute_options[id_type]["data_type"])
    return id_type, ids


def record(time, object_tracks, object_options, grid_options):
    """Get object attributes."""

    core_attributes = object_tracks["attribute"]["core"]
    attribute_options = object_options["attribute"]["core"]
    id_type, ids = get_ids(time, object_tracks, object_options)
    # If no objects, return
    if ids is None or len(ids) == 0:
        return

    time_data_type = attribute_options["time"]["data_type"]
    times = np.array([time for i in range(len(ids))]).astype(time_data_type)

    core_attributes["time"] += list(times)
    core_attributes[id_type] += list(ids)
    keys = core_attributes.keys()

    record_func_args = [core_attributes, attribute_options, time, object_tracks]
    record_func_args += [object_options, grid_options]

    if "latitude" in keys or "longitude" in keys:
        record_coordinates(*record_func_args)
    if "u_flow" in keys or "v_flow" in keys:
        record_velocities(*record_func_args, velocity_type="flow")
    if "u_displacement" in keys or "v_displacement" in keys:
        record_velocities(*record_func_args, velocity_type="displacement")

    # Get the remaining attributes
    processed_attributes = ["time", id_type, "latitude", "longitude"]
    processed_attributes += ["u_flow", "v_flow", "u_displacement", "v_displacement"]
    remaining_attributes = [attr for attr in keys if attr not in processed_attributes]
    for name in remaining_attributes:
        func = attribute_options[name]["method"]["function"]
        get = get_attributes_dispatcher.get(func)
        if get is not None:
            attributes = get(name, time, object_tracks, object_options, grid_options)
            core_attributes[name] += list(attributes)
        else:
            message = f"Function {func} for obtaining attribute {name} not recognised."
            raise ValueError(message)
