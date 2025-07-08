"""
Convenience classes describing core object attributes like position and velocity,
"""

import numpy as np
import xarray as xr
from thuner.log import setup_logger
import thuner.grid as grid
from thuner.match.object import get_object_center
import thuner.grid as grid
import thuner.attribute.utils as utils
from thuner.option.attribute import Attribute, AttributeGroup, AttributeType
from thuner.utils import Retrieval
from thuner.attribute.utils import get_ids

logger = setup_logger(__name__)


__all__ = [
    "time_from_tracks",
    "coordinates_from_match_record",
    "echo_top_height_from_mask",
    "areas_from_match_record",
    "parents_from_match_record",
    "velocities_from_match_record",
    "ids_from_mask",
    "ids_from_match_record",
    "coordinates_from_mask",
    "areas_from_mask",
    "default_tracked",
    "default_member",
    "record_id",
    "mask_id",
    "record_universal_id",
    "mask_universal_id",
    "parents",
    "latitude",
    "longitude",
    "coordinates_record",
    "coordinates_mask",
    "u_flow",
    "v_flow",
    "flow_velocity",
    "u_displacement",
    "v_displacement",
    "displacement_velocity",
    "areas_record",
    "areas_mask",
    "echo_top_height",
    "time",
]


def time_from_tracks(attribute: Attribute, object_tracks):
    """Get time from object tracks."""

    current_time = object_tracks.times[-1]
    array_length = len(object_tracks.match_record["ids"])
    time = np.array([current_time for i in range(array_length)])
    return {"time": list(time.astype(attribute.data_type))}


# Functions for obtaining and recording attributes
def coordinates_from_match_record(
    attribute_group: AttributeGroup, object_tracks, grid_options
):
    """
    Get coordinate from match record created by the matching process to avoid
    redundant calculation.
    """

    names = [attr.name for attr in attribute_group.attributes]
    if "latitude" not in names or "longitude" not in names:
        raise ValueError("Attribute names should be 'latitude' and 'longitude'.")

    pixel_coordinates = object_tracks.match_record["centers"]
    latitude = np.array(grid_options.latitude)
    longitude = np.array(grid_options.longitude)
    latitudes, longitudes = [], []
    for pixel_coordinate in pixel_coordinates:
        if grid_options.name == "geographic":
            latitudes.append(latitude[pixel_coordinate[0]])
            longitudes.append(longitude[pixel_coordinate[1]])
        elif grid_options.name == "cartesian":
            latitudes.append(latitude[pixel_coordinate[0], pixel_coordinate[1]])
            longitudes.append(longitude[pixel_coordinate[0], pixel_coordinate[1]])
    return {"latitude": list(latitudes), "longitude": list(longitudes)}


def echo_top_height_from_mask(
    attribute: Attribute, object_tracks, input_records, object_options, threshold
):
    """Get echo top height from the object mask and dataset."""
    input_records
    object_dataset = object_options.dataset
    try:
        input_record = input_records.track[object_dataset]
    except KeyError:
        input_record = input_records.tag[object_dataset]
    grid = input_record.grids[-1]
    core_attributes = object_tracks.current_attributes.attribute_types["core"]
    if "universal_id" in core_attributes.keys():
        id_type = "universal_id"
        mask = object_tracks.matched_masks[-1]
    else:
        id_type = "id"
        mask = object_tracks.masks[-1]
    ids = np.array(core_attributes[id_type])
    echo_top_heights = []
    for obj_id in ids:
        obj_mask = mask == obj_id
        # If obj_mask is a Dataset, sum over all data variables
        if isinstance(obj_mask, xr.Dataset):
            obj_mask = obj_mask.to_array().sum(dim="variable") > 0
        # Broadcast obj_mask over the altitude dimension of grid
        obj_mask = obj_mask.broadcast_like(grid)
        obj_grid = grid.where(obj_mask)
        echo_top_height = obj_grid.altitude.where(obj_grid >= threshold).max().values
        echo_top_heights.append(echo_top_height)
    echo_top_heights = np.array(echo_top_heights).astype(attribute.data_type).tolist()
    return {attribute.name: echo_top_heights}


def areas_from_match_record(attribute: Attribute, object_tracks):
    """
    Get area from match record created by the matching process to avoid redundant
    calculation.
    """

    areas = np.array(object_tracks.match_record["areas"])
    return {attribute.name: list(areas.astype(attribute.data_type))}


def parents_from_match_record(attribute: Attribute, object_tracks):
    """Get parent ids from the match record to avoid recalculating."""

    parents = object_tracks.match_record["parents"]
    parents_str = []
    for obj_parents in parents:
        if len(obj_parents) == 0:
            parents_str.append("")
        else:
            obj_parents_str = " ".join([str(parent) for parent in obj_parents])
            parents_str.append(obj_parents_str)
    if len(parents_str) != len(parents):
        print("Bad string.")
    return {attribute.name: list(parents_str)}


def velocities_from_match_record(
    attribute_group: AttributeGroup, object_tracks, grid_options
):
    """Get velocity from match record created by the matching process."""
    names = sorted([attr.name for attr in attribute_group.attributes], reverse=True)
    centers = object_tracks.match_record["centers"]
    # Get the "shift" vectors, i.e. the distance vector between the object's current and
    # next centers in pixel units.
    if "u_flow" in names:
        shifts = object_tracks.match_record["corrected_flows"]
    elif "u_displacement" in names:
        shifts = object_tracks.match_record["next_displacements"]
    else:
        raise ValueError(f"Attributes {', '.join(names)} not recognised.")
    v_list, u_list = [[] for i in range(2)]
    time_interval = object_tracks.next_time_interval
    for i, shift in enumerate(shifts):
        if np.any(np.isnan(np.array(shift))):
            v_list.append(np.nan), u_list.append(np.nan)
            logger.debug(f"Object {i} has a nan {attribute_group.name}.")
            continue
        row, col = centers[i]
        distance = grid.pixel_to_cartesian_vector(row, col, shift, grid_options)
        v, u = np.array(distance) / time_interval
        v_list.append(v), u_list.append(u)
    # Take the data type from the first attribute for simplicity
    data_type = attribute_group.attributes[0].data_type
    v_list = np.array(v_list).astype(data_type).tolist()
    u_list = np.array(u_list).astype(data_type).tolist()
    return dict(zip(names, [v_list, u_list]))


def coordinates_from_mask(
    attribute_group: AttributeGroup,
    object_tracks,
    grid_options,
    matched,
    member_object,
):
    """Get object coordinate from mask."""
    mask = utils.get_current_mask(object_tracks, matched)
    # If examining just a member of a grouped object, get masks for that object
    if member_object is not None and isinstance(mask, xr.Dataset):
        mask = mask[f"{member_object}_mask"]
    gridcell_area = object_tracks.gridcell_area

    ids = get_ids(object_tracks, matched, member_object)

    latitude, longitude = [], []
    for obj_id in ids:
        args = [obj_id, mask, grid_options, gridcell_area]
        row, col = get_object_center(*args)[:2]
        lats, lons = np.array(grid_options.latitude), np.array(grid_options.longitude)
        if grid_options.name == "geographic":
            latitude.append(lats[row])
            longitude.append(lons[col])
        elif grid_options.name == "cartesian":
            latitude.append(lats[row, col])
            longitude.append(lons[row, col])

    data_type = attribute_group.attributes[0].data_type
    latitude = np.array(latitude).astype(data_type).tolist()
    longitude = np.array(longitude).astype(data_type).tolist()
    return {"latitude": latitude, "longitude": longitude}


def areas_from_mask(object_tracks, attribute, grid_options, member_object, matched):
    """Get object area from mask."""
    mask = utils.get_current_mask(object_tracks, matched)
    # If examining just a member of a grouped object, get masks for that object
    if member_object is not None and isinstance(mask, xr.Dataset):
        mask = mask[f"{member_object}_mask"]

    gridcell_area = object_tracks.gridcell_area
    ids = get_ids(object_tracks, matched, member_object)

    areas = []
    for obj_id in ids:
        args = [obj_id, mask, grid_options, gridcell_area]
        area = get_object_center(*args)[2]
        areas.append(area)

    areas = np.array(areas).astype(attribute.data_type)
    return {attribute.name: areas.tolist()}


def ids_from_mask(
    attribute: Attribute, object_tracks, member_object: str, matched: bool
):
    """Get object ids from a labelled mask."""
    current_mask = utils.get_current_mask(object_tracks, matched)
    if current_mask is None:
        return None
    if member_object is not None:
        current_mask = current_mask[f"{member_object}_mask"]
    if isinstance(current_mask, xr.Dataset):
        ids = []
        for variable in list(current_mask.data_vars):
            ids += np.unique(current_mask[variable].values).tolist()
        ids = np.unique(ids)
        ids = sorted(ids[ids != 0])
    elif isinstance(current_mask, xr.DataArray):
        ids = np.unique(current_mask)
        ids = sorted(list(ids[ids != 0]))
    ids = list(np.array(ids).astype(attribute.data_type))
    return {attribute.name: ids}


def ids_from_match_record(attribute: Attribute, object_tracks):
    """Get object ids from the match record to avoid recalculating."""
    match_record_names = {"universal_id": "universal_ids", "id": "ids"}
    ids = object_tracks.match_record[match_record_names[attribute.name]]
    ids = np.array(ids).astype(attribute.data_type).tolist()
    return {attribute.name: ids}


def record_id():
    """Convenience function to build a record ID attribute."""
    _desc = "id taken from match record."
    kwargs = {"name": "id", "data_type": int, "description": _desc}
    _func = ids_from_match_record
    _retieval = Retrieval(function=_func, keyword_arguments={"matched": False})
    kwargs.update({"retrieval": _retieval})
    return Attribute(**kwargs)


def mask_id():
    """Convenience function to build a mask ID attribute."""
    _desc = "id taken from object mask."
    kwargs = {"name": "id", "data_type": int, "description": _desc}
    _func = ids_from_mask
    _retrieval = Retrieval(function=_func, keyword_arguments={"matched": False})
    kwargs.update({"retrieval": _retrieval})
    return Attribute(**kwargs)


def record_universal_id():
    """Convenience function to build a record universal ID attribute."""
    _desc = "universal_id taken from match record."
    kwargs = {"name": "universal_id", "data_type": int, "description": _desc}
    _func = ids_from_match_record
    _retrieval = Retrieval(function=_func, keyword_arguments={"matched": True})
    kwargs.update({"retrieval": _retrieval})
    return Attribute(**kwargs)


def mask_universal_id():
    """Convenience function to build a mask universal ID attribute."""
    _desc = "universal_id taken from object mask."
    kwargs = {"name": "universal_id", "data_type": int, "description": _desc}
    _func = ids_from_mask
    _retrieval = Retrieval(function=_func, keyword_arguments={"matched": True})
    kwargs.update({"retrieval": _retrieval})
    return Attribute(**kwargs)


def parents():
    """Convenience function to build a parents attribute."""
    _desc = "parents taken from match record."
    kwargs = {"name": "parents", "data_type": str, "description": _desc}
    _func = parents_from_match_record
    _retrieval = Retrieval(function=_func)
    kwargs.update({"retrieval": _retrieval})
    return Attribute(**kwargs)


def latitude():
    """Convenience function to build a latitude attribute."""
    kwargs = {"name": "latitude", "data_type": float, "precision": 4}
    _desc = "Latitude position of the object."
    kwargs.update({"units": "degrees_north", "description": _desc})
    return Attribute(**kwargs)


def longitude():
    """Convenience function to build a longitude attribute."""
    kwargs = {"name": "longitude", "data_type": float, "precision": 4}
    _desc = "Longitude position of the object."
    kwargs.update({"units": "degrees_east", "description": _desc})
    return Attribute(**kwargs)


def coordinates_record():
    """Convenience function to build a coordinates record attribute group."""
    attributes_list = [latitude(), longitude()]
    _retrieval = Retrieval(function=coordinates_from_match_record)
    _desc = "Coordinates taken from the match_record."
    kwargs = {"name": "coordinates", "description": _desc}
    kwargs.update({"attributes": attributes_list, "retrieval": _retrieval})
    return AttributeGroup(**kwargs)


def coordinates_mask():
    """Convenience function to build a coordinates mask attribute group."""
    attributes_list = [latitude(), longitude()]
    _func = coordinates_from_mask
    _retrieval = Retrieval(function=_func, keyword_arguments={"matched": True})
    _desc = "Coordinates taken from the object mask."
    kwargs = {"name": "coordinates", "description": _desc}
    kwargs.update({"attributes": attributes_list, "retrieval": _retrieval})
    return AttributeGroup(**kwargs)


def u_flow():
    """Convenience function to build a zonal flow velocity attribute."""
    kwargs = {"name": "u_flow", "data_type": float, "precision": 1}
    _desc = "Zonal velocity from cross correlation."
    kwargs.update({"units": "m/s", "description": _desc})
    return Attribute(**kwargs)


def v_flow():
    """Convenience function to build a meridional flow velocity attribute."""
    kwargs = {"name": "v_flow", "data_type": float, "precision": 1}
    _desc = "Meridional velocity from cross correlation."
    kwargs.update({"units": "m/s", "description": _desc})
    return Attribute(**kwargs)


def flow_velocity():
    """Convenience function to build a flow velocity attribute group."""
    attributes_list = [u_flow(), v_flow()]
    _retrieval = Retrieval(function=velocities_from_match_record)
    _desc = "Flow velocities from match record."
    kwargs = {"name": "flow_velocity", "description": _desc}
    kwargs.update({"attributes": attributes_list, "retrieval": _retrieval})
    return AttributeGroup(**kwargs)


def u_displacement():
    """Convenience function to build a zonal displacement velocity attribute."""
    kwargs = {"name": "u_displacement", "data_type": float, "precision": 1}
    _desc = "Zonal centroid displacement velocity."
    kwargs.update({"units": "m/s", "description": _desc})
    return Attribute(**kwargs)


def v_displacement():
    """Convenience function to build a meridional displacement velocity attribute."""
    kwargs = {"name": "v_displacement", "data_type": float, "precision": 1}
    _desc = "Meridional centroid displacement velocity."
    kwargs.update({"units": "m/s", "description": _desc})
    return Attribute(**kwargs)


def displacement_velocity():
    """Convenience function to build a displacement velocity attribute group."""
    attributes_list = [u_displacement(), v_displacement()]
    _retrieval = Retrieval(function=velocities_from_match_record)
    _desc = "Displacement velocities from match record."
    kwargs = {"name": "displacement_velocity", "description": _desc}
    kwargs.update({"attributes": attributes_list, "retrieval": _retrieval})
    return AttributeGroup(**kwargs)


def areas_record():
    """Convenience function to build an areas record attribute."""
    kwargs = {"name": "area", "data_type": float, "precision": 1}
    _desc = "Area taken from the match record."
    kwargs.update({"units": "km^2", "description": _desc})
    _func = areas_from_match_record
    _retrieval = Retrieval(function=_func)
    kwargs.update({"retrieval": _retrieval})
    return Attribute(**kwargs)


def areas_mask():
    """Convenience function to build an areas mask attribute."""
    kwargs = {"name": "area", "data_type": float, "precision": 1}
    _desc = "Area taken from the object mask."
    kwargs.update({"units": "km^2", "description": _desc})
    _func = areas_from_mask
    _retrieval = Retrieval(function=_func, keyword_arguments={"matched": True})
    kwargs.update({"retrieval": _retrieval})
    return Attribute(**kwargs)


# class AreasMask(Attribute):
#     """Object areas taken from object masks."""

#     name: str = "area"
#     data_type: type = float
#     precision: int = 1
#     units: str = "km^2"
#     description: str = "Area taken from the object mask."
#     retrieval: Retrieval | None = Retrieval(
#         function=areas_from_mask, keyword_arguments={"matched": True}
#     )


def echo_top_height():
    """Convenience function to build an echo top height attribute."""
    kwargs = {"name": "echo_top_height", "data_type": float, "precision": 1}
    _desc = "Echo top height taken from the object mask."
    kwargs.update({"units": "m", "description": _desc})
    _func = echo_top_height_from_mask
    _retrieval = Retrieval(function=_func, keyword_arguments={"threshold": 15})
    kwargs.update({"retrieval": _retrieval})
    return Attribute(**kwargs)


# class EchoTopHeight(Attribute):
#     """Echo top heights."""

#     name: str = "echo_top_height"
#     data_type: type = float
#     precision: int = 1
#     units: str = "m"
#     description: str = "Echo top height of the object."
#     retrieval: Retrieval | None = Retrieval(
#         function=echo_top_height_from_mask, keyword_arguments={"threshold": 15}
#     )


def time():
    """Convenience function to build a time attribute."""
    kwargs = {"name": "time", "data_type": np.datetime64}
    _desc = "Time taken from the tracking process."
    _retrieval = Retrieval(function=time_from_tracks)
    kwargs.update({"units": "yyyy-mm-dd hh:mm:ss", "description": _desc})
    kwargs.update({"retrieval": _retrieval})
    return Attribute(**kwargs)


# class Time(Attribute):
#     """Time taken from the tracking process."""

#     name: str = "time"
#     data_type: type = np.datetime64
#     units: str = "yyyy-mm-dd hh:mm:ss"
#     description: str = "Time taken from the tracking process."
#     retrieval: Retrieval | None = Retrieval(function=time_from_tracks)


def default_tracked():
    """Create the default core attribute type for tracked objects."""
    attributes_list = [time(), record_universal_id(), parents(), coordinates_record()]
    attributes_list += [areas_record(), flow_velocity(), displacement_velocity()]
    description = "Core attributes of tracked object, e.g. position and velocities."
    kwargs = {"name": "core", "attributes": attributes_list, "description": description}
    return AttributeType(**kwargs)


def default_member():
    """Create the default core attribute type for member, i.e. component, objects."""
    attributes_list = [time(), record_universal_id(), coordinates_mask(), areas_mask()]
    description = "Core attributes of a member object, e.g. position and velocities."
    kwargs = {"name": "core", "attributes": attributes_list, "description": description}
    return AttributeType(**kwargs)


def retrieve_core(
    attributes_list=[time(), latitude(), longitude()], matched=True, member_object=None
):
    """Get core attributes list for use with other attribute types."""
    if matched:
        attributes_list += [record_universal_id()]
    else:
        attributes_list += [record_id()]
    # Replace retrieval for the core attributes with attribute_from_core function
    kwargs = {"function": utils.attribute_from_core}
    kwargs.update({"keyword_arguments": {"member_object": member_object}})
    for attribute in attributes_list:
        attribute.retrieval = Retrieval(**kwargs)
    return attributes_list
