"""
Core attributes.
"""

import numpy as np
import xarray as xr
from thuner.log import setup_logger
import thuner.grid as grid
from thuner.object.object import get_object_center
import thuner.grid as grid
import thuner.attribute.utils as utils
from thuner.option.attribute import Retrieval, Attribute, AttributeGroup, AttributeType

logger = setup_logger(__name__)


def time_from_tracks(object_tracks, attribute: Attribute):
    """Get time from object tracks."""

    previous_time = object_tracks["previous_times"][-1]
    array_length = len(object_tracks["object_record"]["previous_ids"])
    time = np.array([previous_time for i in range(array_length)])
    return {"time": list(time.astype(attribute.data_type))}


# Functions for obtaining and recording attributes
def coordinates_from_object_record(
    attribute_group: AttributeGroup, object_tracks, grid_options
):
    """
    Get coordinate from object record created by the matching process to avoid
    redundant calculation.
    """

    names = [attr.name for attr in attribute_group.attributes]
    if "latitude" not in names or "longitude" not in names:
        raise ValueError("Attribute names should be 'latitude' and 'longitude'.")

    pixel_coordinates = object_tracks["object_record"]["previous_centers"]
    latitude, longitude = grid_options.latitude, grid_options.longitude
    latitudes, longitudes = [], []
    for pixel_coordinate in pixel_coordinates:
        if grid_options.name == "geographic":
            latitudes.append(latitude[pixel_coordinate[0]])
            longitudes.append(longitude[pixel_coordinate[1]])
        elif grid_options.name == "cartesian":
            latitudes.append(latitude[pixel_coordinate[0], pixel_coordinate[1]])
            longitudes.append(longitude[pixel_coordinate[0], pixel_coordinate[1]])
    return {"latitude": list(latitudes), "longitude": list(longitudes)}


def areas_from_object_record(attribute: Attribute, object_tracks):
    """
    Get area from object record created by the matching process to avoid redundant
    calculation.
    """

    areas = object_tracks["object_record"]["previous_areas"]
    return {attribute.name: list(areas.astype(attribute.data_type))}


def parents_from_object_record(attribute: Attribute, object_tracks):
    """Get parent ids from the object record to avoid recalculating."""

    parents = object_tracks["object_record"]["previous_parents"]
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


def velocities_from_object_record(
    attribute_group: AttributeGroup, object_tracks, grid_options
):
    """Get velocity from object record created by the matching process."""
    names = sorted([attr.name for attr in attribute_group.attributes], reverse=True)
    centers = object_tracks["object_record"]["previous_centers"]
    # Get the "shift" vectors, i.e. the distance vector between the object's previous and
    # current centers in pixel units.
    if "u_flow" in names:
        shifts = object_tracks["object_record"]["corrected_flows"]
    elif "u_displacement" in names:
        shifts = object_tracks["object_record"]["current_displacements"]
    else:
        raise ValueError(f"Attributes {', '.join(names)} not recognised.")
    v_list, u_list = [[] for i in range(2)]
    time_interval = object_tracks["current_time_interval"]
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
    mask = utils.get_previous_mask(object_tracks, matched)
    # If examining just a member of a grouped object, get masks for that object
    if member_object is not None and isinstance(mask, xr.Dataset):
        mask = mask[f"{member_object}_mask"]
    gridcell_area = object_tracks["gridcell_area"]
    ids = ids_from_mask(object_tracks, member_object, matched)

    latitude, longitude = [], []
    for obj_id in ids:
        args = [obj_id, mask, grid_options, gridcell_area]
        row, col = get_object_center(*args)[:2]
        if grid_options.name == "geographic":
            latitude.append(grid_options.latitude[row])
            longitude.append(grid_options.longitude[col])
        elif grid_options.name == "cartesian":
            latitude.append(grid_options.latitude[row, col])
            longitude.append(grid_options.longitude[row, col])

    data_type = attribute_group.attributes[0].data_type
    latitude = np.array(latitude).astype(data_type).tolist()
    longitude = np.array(longitude).astype(data_type).tolist()
    return {"latitude": latitude, "longitude": longitude}


def areas_from_mask(object_tracks, attribute, grid_options, member_object):
    """Get object area from mask."""
    mask = utils.get_previous_mask(attribute, object_tracks)
    # If examining just a member of a grouped object, get masks for that object
    if member_object is not None and isinstance(mask, xr.Dataset):
        mask = mask[f"{member_object}_mask"]

    gridcell_area = object_tracks["gridcell_area"]
    ids = ids_from_mask(object_tracks, attribute, member_object)

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
    previous_mask = utils.get_previous_mask(object_tracks, matched)
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
    ids = list(np.array(ids).astype(attribute.data_type))
    return {attribute.name: ids}


def ids_from_object_record(attribute: Attribute, object_tracks):
    """Get object ids from the object record to avoid recalculating."""
    object_record_names = {"universal_id": "universal_ids", "id": "previous_ids"}
    ids = object_tracks["object_record"][object_record_names[attribute.name]]
    ids = np.array(ids).astype(attribute.data_type).tolist()
    return {attribute.name: ids}


# Define convenience attributes
retrieval_kwargs = {"function": ids_from_object_record}
retrieval_kwargs.update({"keyword_arguments": {"matched": False}})
retrieval = Retrieval(**retrieval_kwargs)
description = "id taken from object record."
kwargs = {"name": "id", "data_type": int, "description": description}
ids_record = Attribute(retrieval=retrieval, **kwargs)

description = "id taken from object mask."
kwargs.update({"description": description})
retrieval_kwargs = {"matched": False}
retrieval = Retrieval(function=ids_from_mask, keyword_arguments=retrieval_kwargs)
kwargs.update({"retrieval": retrieval})
ids_mask = Attribute(**kwargs)

description = "universal_id taken from object record."
retrieval_kwargs = {"matched": True}
retrieval = Retrieval(function=ids_from_mask, keyword_arguments=retrieval_kwargs)
kwargs.update({"name": "universal_id", "description": description})
kwargs.update({"retrieval": retrieval})
universal_ids_record = Attribute(**kwargs)

description = "universal_id taken from object mask."
retrieval_kwargs = {"matched": True}
retrieval = Retrieval(function=ids_from_mask, keyword_arguments=retrieval_kwargs)
kwargs.update({"description": description, "retrieval": retrieval})
universal_ids_mask = Attribute(**kwargs)

parent_description = "parent objects as space separated list of universal_ids."
kwargs = {"name": "parents", "data_type": str, "description": parent_description}
parents = Attribute(retrieval=Retrieval(function=parents_from_object_record), **kwargs)

kwargs = {"data_type": float, "precision": 4, "retrieval": None, "name": "latitude"}
kwargs.update({"description": "Latitude position of the object."})
kwargs.update({"units": "degrees_north"})
latitude = Attribute(**kwargs)
kwargs.update({"name": "longitude", "units": "degrees_east"})
kwargs.update({"description": "Longitude position of the object."})
longitude = Attribute(**kwargs)
description = f"Coordinates taken from the object_record or object mask; "
description += f"usually a gridcell area weighted mean over the object mask."
kwargs = {"name": "coordinate", "description": description}
kwargs.update({"attributes": [latitude, longitude]})
kwargs.update({"retrieval": Retrieval(function=coordinates_from_object_record)})
coordinates_record = AttributeGroup(**kwargs)

keyword_arguments = {"matched": False}
retrieval_kwargs = {"function": coordinates_from_mask}
retrieval_kwargs.update({"keyword_arguments": keyword_arguments})
kwargs.update({"retrieval": Retrieval(**retrieval_kwargs)})
coordinates_mask = AttributeGroup(**kwargs)

kwargs = {"data_type": float, "precision": 1, "retrieval": None, "units": "m/s"}
description = "velocity from cross correlation."
u_flow = Attribute(name="u_flow", description="Zonal " + description, **kwargs)
v_flow = Attribute(name="v_flow", description="Meridional " + description, **kwargs)
retrieval = Retrieval(function=velocities_from_object_record)
kwargs = {"name": "flow_velocity", "description": "Flow velocities from object record."}
kwargs.update({"attributes": [u_flow, v_flow], "retrieval": retrieval})
flow_velocity = AttributeGroup(**kwargs)

kwargs = {"data_type": float, "precision": 1, "retrieval": None, "units": "m/s"}
description = " centroid displacement velocity."
kwargs.update({"description": "Zonal " + description})
u_displacement = Attribute(name="u_displacement", **kwargs)
kwargs.update({"description": "Meridional " + description})
v_displacement = Attribute(name="v_displacement", **kwargs)
kwargs = {"name": "displacement_velocity", "description": "Displacement velocities."}
kwargs.update({"attributes": [u_displacement, v_displacement], "retrieval": retrieval})
displacement_velocity = AttributeGroup(**kwargs)

kwargs = {"name": "area", "data_type": float, "precision": 1, "units": "km^2"}
kwargs.update({"description": "Area taken from the object record."})
kwargs.update({"retrieval": Retrieval(function=areas_from_object_record)})
areas_record = Attribute(**kwargs)

kwargs.update({"description": "Area taken from the object mask."})
kwargs.update({"retrieval": Retrieval(function=areas_from_mask)})
areas_mask = Attribute(**kwargs)

kwargs = {"data_type": np.datetime64, "precision": None}
kwargs.update({"units": "yyyy-mm-dd hh:mm:ss"})
kwargs.update({"description": "Time taken from the tracking process."})
retrieval = Retrieval(function=time_from_tracks)
kwargs.update({"name": "time", "retrieval": retrieval})
time = Attribute(**kwargs)


# Convenience function for creating default core attribute type
def default(matched=True, tracked=True, grouped=False):
    """Create the default core attribute type."""
    attributes_list = [time]
    if matched:
        # If the object is matched, take core properties from the object record
        attributes_list += [universal_ids_record, coordinates_record, parents]
        if not grouped:
            attributes_list += [areas_record]
    else:
        # If the object is not matched, take core properties from the object mask
        attributes_list += [ids_mask, coordinates_mask]
        if not grouped:
            attributes_list += [areas_mask]
    if tracked:
        attributes_list += [flow_velocity, displacement_velocity]
    description = "Core attributes of the object, e.g. position and velocities."
    kwargs = {"name": "core", "attributes": attributes_list, "description": description}
    return AttributeType(**kwargs)


def retrieve_core(attributes_list=[time, latitude, longitude], matched=True):
    """Get core attributes list for use with other attribute types."""
    if matched:
        attributes_list += [universal_ids_record]
    else:
        attributes_list += [ids_record]
    # Replace retrieval for the core attributes with attribute_from_core function
    new_attributes_list = []
    for attribute in attributes_list:
        new_attribute = attribute.model_copy(deep=True)
        new_attribute.retrieval = Retrieval(function=utils.attribute_from_core)
        new_attributes_list.append(new_attribute)
    return new_attributes_list
