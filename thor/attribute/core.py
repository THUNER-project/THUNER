"""
Methods for defining property options associated with detected objects, and for 
measuring such properties.
"""

import numpy as np
import xarray as xr
from thor.log import setup_logger
import thor.grid as grid
from thor.object.object import get_object_center
import thor.grid as grid
import thor.attribute.utils as utils

logger = setup_logger(__name__)


# Convenience functions for creating default core attribute options dictionaries
def default(names=None, tracked=True, matched=None, grouped=False):
    """Create a dictionary of default core attributes."""

    if matched is None:
        matched = tracked

    if names is None:
        names = ["time", "latitude", "longitude"]
        if not grouped:
            names += ["area"]
        if matched:
            names += ["universal_id"]
        else:
            names += ["id"]
        if tracked:
            names += ["u_flow", "v_flow"]
            names += ["u_displacement", "v_displacement"]

    attributes_dict = {}
    for name in names:
        attributes_dict[name] = attribute_dispatcher[name](name, tracked=tracked)

    return attributes_dict


def identity(name="id", method=None, description=None, tracked=True):
    """
    Options for id attribute.
    """
    attribute_dict = {}
    data_type = int
    precision = None
    units = None
    if method is None:
        if tracked:
            method = {"function": "ids_from_object_record"}
        else:
            method = {"function": "ids_from_mask"}
    if description is None:
        description = f"{name} taken from object record or object mask. "
        description += "Unlike uid, id is not necessarily unique across time steps."
    attribute_dict.update({"name": name, "method": method, "data_type": data_type})
    attribute_dict.update({"precision": precision, "description": description})
    attribute_dict.update({"units": units})

    return attribute_dict


def coordinate(name, method=None, description=None, tracked=True):
    """
    Options for coordinate attributes.
    """
    attribute_dict = {}
    data_type = float
    precision = 4
    if name == "latitude":
        units = "degrees_north"
    elif name == "longitude":
        units = "degrees_east"
    else:
        raise ValueError(f"Coordinate must be 'latitude' or 'longitude'.")
    if method is None:
        if tracked:
            method = {"function": "coordinates_from_object_record"}
        else:
            method = {"function": "coordinates_from_mask"}
    if description is None:
        description = f"{name} position taken from the object_record or object mask; "
        description += f"usually a gridcell area weighted mean over the object mask."
    attribute_dict.update({"name": name, "method": method, "data_type": data_type})
    attribute_dict.update({"precision": precision, "description": description})
    attribute_dict.update({"units": units})

    return attribute_dict


def velocity(name, method=None, description=None, tracked=True):
    """
    Options for velocity attributes. Velocities only defined for tracked objects.
    """
    attribute_dict = {}
    data_type = float
    precision = 1
    units = "m/s"
    if not tracked:
        message = f"Velocity attribute {name} only defined for tracked objects."
        raise ValueError(message)

    if method is None:
        method = {"function": "velocities_from_object_record"}
    if description is None:
        description = f"{name} velocities taken from the matching process."
    attribute_dict.update({"name": name, "method": method, "data_type": data_type})
    attribute_dict.update({"precision": precision, "description": description})
    attribute_dict.update({"units": units})

    return attribute_dict


def area(name="area", method=None, description=None, tracked=True):
    """
    Options for area attribute.
    """
    attribute_dict = {}
    data_type = float
    precision = 1
    units = "km^2"
    if method is None:
        if tracked:
            method = {"function": "areas_from_object_record"}
        else:
            method = {"function": "areas_from_mask"}
    if description is None:
        description = f"Object area taken from the object_record or object mask."
    attribute_dict.update({"name": name, "method": method, "data_type": data_type})
    attribute_dict.update({"precision": precision, "description": description})
    attribute_dict.update({"units": units})

    return attribute_dict


def time(name="time", method=None, description=None, tracked=True):
    """
    Options for time attribute.
    """
    attribute_dict = {}
    if method is None:
        method = {"function": None}
    if description is None:
        description = f"Time taken from the tracking process."
    data_type = "datetime64[s]"
    units = None
    precision = None
    attribute_dict.update({"name": name, "method": method, "data_type": data_type})
    attribute_dict.update({"precision": precision, "description": description})
    attribute_dict.update({"units": units})

    return attribute_dict


attribute_dispatcher = {
    "id": identity,
    "universal_id": identity,
    "latitude": coordinate,
    "longitude": coordinate,
    "u_flow": velocity,
    "v_flow": velocity,
    "u_displacement": velocity,
    "v_displacement": velocity,
    "area": area,
    "time": time,
}


# Functions for obtaining and recording core attributes
def coordinates_from_object_record(
    name, time, object_tracks, attribute_options, grid_options, member_object=None
):
    """
    Get coordinate from object record created by the matching process to avoid
    redundant calculation.
    """

    pixel_coordinates = object_tracks["object_record"]["previous_centers"]
    latitude, longitude = grid_options["latitude"], grid_options["longitude"]
    latitudes, longitudes = [], []
    for pixel_coordinate in pixel_coordinates:
        if grid_options["name"] == "geographic":
            latitudes.append(latitude[pixel_coordinate[0]])
            longitudes.append(longitude[pixel_coordinate[1]])
        elif grid_options["name"] == "cartesian":
            latitudes.append(latitude[pixel_coordinate[0], pixel_coordinate[1]])
            longitudes.append(longitude[pixel_coordinate[0], pixel_coordinate[1]])
    return latitudes, longitudes


def areas_from_object_record(
    name, time, object_tracks, attribute_options, grid_options, member_object=None
):
    """
    Get area from object record created by the matching process to avoid redundant
    calculation.
    """
    areas = object_tracks["object_record"]["previous_areas"]
    data_type = attribute_options[name]["data_type"]
    return areas.astype(data_type)


def velocities_from_object_record(
    name, time, object_tracks, attribute_options, grid_options, member_object=None
):
    """Get velocity from object record created by the matching process."""
    centers = object_tracks["object_record"]["previous_centers"]
    # Get the "shift" vectors, i.e. the distance vector between the object's previous and
    # current centers in pixel units.
    if name in ["u_flow", "v_flow"]:
        shifts = object_tracks["object_record"]["corrected_flows"]
    elif name in ["u_displacement", "v_displacement"]:
        shifts = object_tracks["object_record"]["current_displacements"]
    else:
        raise ValueError(f"Attribute {name} not recognised.")
    v_list, u_list = [[] for i in range(2)]
    time_interval = object_tracks["current_time_interval"]
    for i, shift in enumerate(shifts):
        if np.any(np.isnan(np.array(shift))):
            v_list.append(np.nan), u_list.append(np.nan)
            logger.info(f"Object {i} has a nan {name}.")
            continue
        row, col = centers[i]
        distance = grid.pixel_to_cartesian_vector(row, col, shift, grid_options)
        v, u = np.array(distance) / time_interval
        v_list.append(v), u_list.append(u)
    data_type = attribute_options[name]["data_type"]
    v_list = np.array(v_list).astype(data_type)
    u_list = np.array(u_list).astype(data_type)
    return v_list, u_list


def coordinates_from_mask(
    name, time, object_tracks, attribute_options, grid_options, member_object=None
):
    """Get object coordinate from mask."""
    mask = utils.get_previous_mask(attribute_options, object_tracks)
    # If examining just a member of a grouped object, get masks for that object
    if member_object is not None and isinstance(mask, xr.Dataset):
        mask = mask[f"{member_object}_mask"]
    gridcell_area = object_tracks["gridcell_area"]
    ids = ids_from_mask(name, object_tracks, attribute_options, member_object)

    lats, lons = [], []
    for obj_id in ids:
        row, col = get_object_center(obj_id, mask, grid_options, gridcell_area)[:2]
        if grid_options["name"] == "geographic":
            lats.append(grid_options["latitude"][row])
            lons.append(grid_options["longitude"][col])
        elif grid_options["name"] == "cartesian":
            lats.append(grid_options["latitude"][row, col])
            lons.append(grid_options["longitude"][row, col])

    data_type = attribute_options[name]["data_type"]
    lats = np.array(lats).astype(data_type)
    lons = np.array(lons).astype(data_type)
    return lats, lons


def areas_from_mask(
    name, time, object_tracks, attribute_options, grid_options, member_object=None
):
    """Get object area from mask."""
    mask = utils.get_previous_mask(attribute_options, object_tracks)
    # If examining just a member of a grouped object, get masks for that object
    if member_object is not None and isinstance(mask, xr.Dataset):
        mask = mask[f"{member_object}_mask"]

    gridcell_area = object_tracks["gridcell_area"]
    ids = ids_from_mask(name, object_tracks, attribute_options)

    areas = []
    for obj_id in ids:
        area = get_object_center(obj_id, mask, grid_options, gridcell_area)[2]
        areas.append(area)

    data_type = attribute_options[name]["data_type"]
    areas = np.array(areas).astype(data_type)
    return areas


def ids_from_mask(attribute_name, object_tracks, attribute_options, member_object=None):
    """Get object ids from a labelled mask."""
    previous_mask = utils.get_previous_mask(attribute_options, object_tracks)
    if previous_mask is None:
        return None
    if member_object is not None:
        previous_mask = previous_mask[f"{member_object}_mask"]
    if isinstance(previous_mask, xr.Dataset):
        ids = []
        for variable in list(previous_mask.data_vars):
            ids += np.unique(previous_mask[variable].values).tolist()
        ids = np.unique(ids)
        ids = sorted(ids[ids != 0])
    elif isinstance(previous_mask, xr.DataArray):
        ids = np.unique(previous_mask)
        ids = sorted(list(ids[ids != 0]))
    return ids


def ids_from_object_record(name, object_tracks, attribute_options, member_object=None):
    """Get object ids from the object record to avoid recalculating."""
    object_record_names = {"universal_id": "universal_ids", "id": "previous_ids"}
    ids = object_tracks["object_record"][object_record_names[name]]
    return ids


# Dispatch dictionary for getting core attributes
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
    attributes,
    options,
    time,
    object_tracks,
    grid_options,
    member_object=None,
):
    """Record object coordinates."""
    keys = attributes.keys()
    if not "latitude" in keys or not "longitude" in keys:
        message = "Both latitude and longitude must be specified."
        raise ValueError(message)
    func = options["latitude"]["method"]["function"]
    lon_func = options["longitude"]["method"]["function"]
    if func != lon_func:
        message = "Functions for acquring latitude and longitude must be the same."
        raise ValueError(message)
    get = get_attributes_dispatcher.get(func)
    if get is None:
        message = f"Function {func} for obtaining lat and lon not recognised."
        raise ValueError(message)

    args = ["latitude", time, object_tracks, options, grid_options, member_object]
    lats, lons = get(*args)
    attributes["latitude"] += list(lats)
    attributes["longitude"] += list(lons)


def record_velocities(
    attributes,
    options,
    time,
    object_tracks,
    grid_options,
    velocity_type="flow",
    member_object=None,
):
    """Record object coordinates."""
    keys = attributes.keys()
    if not f"u_{velocity_type}" in keys or not f"v_{velocity_type}" in keys:
        message = "Both u and v compononents must be specified."
        raise ValueError(message)
    func = options[f"u_{velocity_type}"]["method"]["function"]
    lon_func = options[f"v_{velocity_type}"]["method"]["function"]
    if func != lon_func:
        message = "Functions for acquring u and v velocities must be the same."
        raise ValueError(message)
    get = get_attributes_dispatcher.get(func)
    if get is None:
        message = f"Function {func} for obtaining u and v not recognised."
        raise ValueError(message)

    u_str = f"u_{velocity_type}"
    v, u = get(u_str, time, object_tracks, options, grid_options, member_object)
    attributes[f"v_{velocity_type}"] += list(v)
    attributes[f"u_{velocity_type}"] += list(u)


def get_ids(time, object_tracks, attribute_options, member_object):
    """Get object ids."""

    if attribute_options is None:
        return
    if "universal_id" in attribute_options:
        id_type = "universal_id"
    elif "id" in attribute_options:
        id_type = "id"

    func = attribute_options[id_type]["method"]["function"]
    get = get_attributes_dispatcher.get(func)
    ids = get(id_type, object_tracks, attribute_options, member_object)

    if ids is not None:
        ids = np.array(ids).astype(attribute_options[id_type]["data_type"])
    return id_type, ids


# Record core attributes
def record(
    time,
    input_records,
    attributes,
    object_tracks,
    object_options,
    attribute_options,
    grid_options,
    member_object=None,
):
    """Get object core attributes."""
    id_type, ids = get_ids(time, object_tracks, attribute_options, member_object)
    # If no objects, return
    if ids is None or len(ids) == 0:
        return

    time_data_type = attribute_options["time"]["data_type"]
    previous_time = object_tracks["previous_times"][-1]
    times = np.array([previous_time for i in range(len(ids))]).astype(time_data_type)

    attributes["time"] += list(times)
    attributes[id_type] += list(ids)
    keys = attributes.keys()

    args = [attributes, attribute_options, time, object_tracks]
    args += [grid_options]

    # Two dimensionality checks
    if "latitude" in keys or "longitude" in keys:
        record_coordinates(*args, member_object=member_object)
    if "u_flow" in keys or "v_flow" in keys:
        record_velocities(*args, velocity_type="flow", member_object=member_object)
    if "u_displacement" in keys or "v_displacement" in keys:
        args_dict = {"velocity_type": "displacement", "member_object": member_object}
        record_velocities(*args, **args_dict)

    # Get the remaining attributes
    processed_attributes = ["time", id_type, "latitude", "longitude"]
    processed_attributes += ["u_flow", "v_flow", "u_displacement", "v_displacement"]
    remaining_attributes = [attr for attr in keys if attr not in processed_attributes]
    for name in remaining_attributes:
        attr_function = attribute_options[name]["method"]["function"]
        get_attr = get_attributes_dispatcher.get(attr_function)
        if get_attr is not None:
            args = [name, time, object_tracks, attribute_options, grid_options]
            attribute = get_attr(*args, member_object=member_object)
            attributes[name] += list(attribute)
        else:
            message = f"Function {attr_function} for obtaining attribute {name} not recognised."
            raise ValueError(message)
