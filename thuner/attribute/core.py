"""
Core attributes.
"""

import numpy as np
import xarray as xr
from thuner.log import setup_logger
import thuner.grid as grid
import thuner.object.object as thuner_object
import thuner.grid as grid
import thuner.attribute.utils as utils
from thuner.option.attribute import Retrieval, Attribute, AttributeGroup, AttributeType

logger = setup_logger(__name__)


# attribute_dispatcher = {
#     "id": identity,
#     "universal_id": identity,
#     "parents": parents,
#     "latitude": coordinate,
#     "longitude": coordinate,
#     "u_flow": velocity,
#     "v_flow": velocity,
#     "u_displacement": velocity,
#     "v_displacement": velocity,
#     "area": area,
#     "time": time,
# }


def time_from_tracks(object_tracks, attribute: Attribute):
    """Get time from object tracks."""
    previous_time = object_tracks["previous_times"][-1]
    array_length = len(object_tracks["previous_ids"])
    times = np.array([previous_time for i in range(len(array_length))])
    return times.astype(attribute.data_type)


# Functions for obtaining and recording attributes
def coordinates_from_object_record(
    attribute_group: AttributeGroup, object_tracks, grid_options, member_object
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


def areas_from_object_record(
    attributes, attribute: Attribute, object_tracks, grid_options, member_object
):
    """
    Get area from object record created by the matching process to avoid redundant
    calculation.
    """

    areas = object_tracks["object_record"]["previous_areas"]
    return {attribute.name: areas.astype(attribute.data_type)}


def parents_from_object_record(
    attributes, attribute: Attribute, object_tracks, grid_options, member_object
):
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
    return {attribute.name: parents_str}


def velocities_from_object_record(
    attributes,
    attribute_group: AttributeGroup,
    object_tracks,
    grid_options,
    member_object,
):
    """Get velocity from object record created by the matching process."""
    names = sorted([attr.name for attr in attribute_group.attributes], reverse=True)
    if "u_flow" not in names or "v_flow" not in names:
        raise ValueError("Unrecognised attribute names.")
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
    v_list = np.array(v_list).astype(data_type)
    u_list = np.array(u_list).astype(data_type)
    return dict(zip(names, [v_list, u_list]))
    # return v_list, u_list


def coordinates_from_mask(
    attributes,
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

    lats, lons = [], []
    for obj_id in ids:
        args = [obj_id, mask, grid_options, gridcell_area]
        row, col = thuner_object.get_object_center(*args)[:2]
        if grid_options.name == "geographic":
            lats.append(grid_options.latitude[row])
            lons.append(grid_options.longitude[col])
        elif grid_options.name == "cartesian":
            lats.append(grid_options.latitude[row, col])
            lons.append(grid_options.longitude[row, col])

    data_type = attribute_group.attributes[0].data_type
    lats = np.array(lats).astype(data_type)
    lons = np.array(lons).astype(data_type)
    return lats, lons


def areas_from_mask(object_tracks, attribute_options, grid_options, member_object):
    """Get object area from mask."""
    mask = utils.get_previous_mask(attribute_options, object_tracks)
    # If examining just a member of a grouped object, get masks for that object
    if member_object is not None and isinstance(mask, xr.Dataset):
        mask = mask[f"{member_object}_mask"]

    gridcell_area = object_tracks["gridcell_area"]
    ids = ids_from_mask(object_tracks, attribute_options, member_object)

    areas = []
    for obj_id in ids:
        args = [obj_id, mask, grid_options, gridcell_area]
        area = thuner_object.get_object_center(*args)[2]
        areas.append(area)

    areas = np.array(areas).astype(attribute_options["area"]["data_type"])
    return areas


def ids_from_mask(object_tracks, member_object, matched):
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
    return ids


def ids_from_object_record(name, object_tracks):
    """Get object ids from the object record to avoid recalculating."""
    object_record_names = {"universal_id": "universal_ids", "id": "previous_ids"}
    ids = object_tracks["object_record"][object_record_names[name]]
    return ids


# Define convenience attributes
description = "id taken from object record."
kwargs = {"name": "id", "data_type": int, "description": description}
retrieval_kwargs = {"matched": False}
retrieval = Retrieval(
    function=ids_from_object_record, keyword_arguments=retrieval_kwargs
)
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
kwargs.update(
    {"name": "universal_id", "description": description, "retrieval": retrieval}
)
universal_ids_record = Attribute(**kwargs)

description = "universal_id taken from object mask."
retrieval_kwargs = {"matched": True}
retrieval = Retrieval(function=ids_from_mask, keyword_arguments=retrieval_kwargs)
kwargs.update({"description": description, "retrieval": retrieval})
universal_ids_mask = Attribute(**kwargs)

parent_description = "parent objects as space separated list of universal_ids."
kwargs = {"name": "parents", "data_type": str, "description": parent_description}
parents = Attribute(retrieval=Retrieval(function=parents_from_object_record), **kwargs)


# def identity(name="id", method=None, description=None, tracked=True):
#     """
#     Options for id attribute.
#     """
#     data_type = int
#     precision = None
#     units = None
#     if method is None:
#         if tracked:
#             method = {"function": "ids_from_object_record"}
#         else:
#             method = {"function": "ids_from_mask"}
#     if description is None:
#         description = f"{name} taken from object record or object mask. "
#         description += "Unlike uid, id is not necessarily unique across time steps."
#     args = [name, method, data_type, precision, description, units]
#     return utils.get_attribute_dict(*args)


# def parents(name="parents", method=None, description=None, tracked=True):
#     """
#     Options for parents attribute. Store a space separated list of parent ids as a str.
#     """
#     data_type = str
#     precision = None
#     units = None
#     if method is None:
#         method = {"function": "parents_from_object_record"}
#     if description is None:
#         description = f"{name} taken from object record."
#     args = [name, method, data_type, precision, description, units]
#     return utils.get_attribute_dict(*args)


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


# def coordinate(name, method=None, description=None, tracked=True):
#     """
#     Options for coordinate attributes.
#     """
#     data_type = float
#     precision = 4
#     if name == "latitude":
#         units = "degrees_north"
#     elif name == "longitude":
#         units = "degrees_east"
#     else:
#         raise ValueError(f"Coordinate must be 'latitude' or 'longitude'.")
#     if method is None:
#         if tracked:
#             method = {"function": "coordinates_from_object_record"}
#         else:
#             method = {"function": "coordinates_from_mask"}
#     if description is None:
#         description = f"{name} taken from object record or object mask. "
#         description += "Unlike uid, id is not necessarily unique across time steps."

#     args = [name, method, data_type, precision, description, units]
#     return utils.get_attribute_dict(*args)


# def velocity(name, method=None, description=None, tracked=True):
#     """
#     Options for velocity attributes. Velocities only defined for tracked objects.
#     """
#     data_type = float
#     precision = 1
#     units = "m/s"
#     if not tracked:
#         message = f"Velocity attribute {name} only defined for tracked objects."
#         raise ValueError(message)

#     if method is None:
#         method = {"function": "velocities_from_object_record"}
#     if description is None:
#         description = f"{name} velocities taken from the matching process."
#     args = [name, method, data_type, precision, description, units]
#     return utils.get_attribute_dict(*args)


# def area(name="area", method=None, description=None, tracked=True):
#     """
#     Options for area attribute.
#     """
#     data_type = float
#     precision = 1
#     units = "km^2"
#     if method is None:
#         if tracked:
#             method = {"function": "areas_from_object_record"}
#         else:
#             method = {"function": "areas_from_mask"}
#     if description is None:
#         description = f"Object area taken from the object_record or object mask."
#     args = [name, method, data_type, precision, description, units]
#     return utils.get_attribute_dict(*args)


kwargs = {"name": "area", "data_type": float, "precision": 1, "units": "km^2"}
kwargs.update({"description": "Area taken from the object record."})
kwargs.update({"retrieval": Retrieval(function=areas_from_object_record)})
areas_record = Attribute(**kwargs)

kwargs.update({"description": "Area taken from the object mask."})
kwargs.update({"retrieval": Retrieval(function=areas_from_mask)})
areas_mask = Attribute(**kwargs)


# def time(name="time", method=None, description=None, tracked=True):
#     """
#     Options for time attribute.
#     """
#     if method is None:
#         method = {"function": None}
#     if description is None:
#         description = f"Time taken from the tracking process."
#     data_type = "datetime64[s]"
#     units = None
#     precision = None
#     args = [name, method, data_type, precision, description, units]
#     return utils.get_attribute_dict(*args)


kwargs = {"data_type": np.datetime64, "precision": None, "units": "yyyy-mm-dd hh:mm:ss"}
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
    for attribute in attributes_list:
        attribute.retrieval = Retrieval(function=utils.attribute_from_core)
    return attributes_list


# Convenience functions for creating default core attribute options dictionaries
# def default(names=None, tracked=True, matched=None, grouped=False):
#     """Create a dictionary of default core attributes."""

#     if matched is None:
#         matched = tracked

#     if names is None:
#         names = ["time", "latitude", "longitude"]
#         if not grouped:
#             names += ["area"]
#         if matched:
#             names += ["universal_id"]
#         else:
#             names += ["id"]
#         if tracked:
#             names += ["parents", "u_flow", "v_flow"]
#             names += ["u_displacement", "v_displacement"]

#     attributes_dict = {}
#     for name in names:
#         attributes_dict[name] = attribute_dispatcher[name](name, tracked=tracked)

#     return attributes_dict


# Dispatch dictionary for getting core attributes
get_attributes_dispatcher = {
    "coordinates_from_object_record": coordinates_from_object_record,
    "coordinates_from_mask": coordinates_from_mask,
    "areas_from_object_record": areas_from_object_record,
    "areas_from_mask": areas_from_mask,
    "velocities_from_object_record": velocities_from_object_record,
    "ids_from_mask": ids_from_mask,
    "ids_from_object_record": ids_from_object_record,
    "parents_from_object_record": parents_from_object_record,
}


def record_coordinates(
    attributes, attribute_options, object_tracks, grid_options, member_object
):
    """Record object coordinates."""
    keys = attributes.keys()
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
    from_mask_args = [object_tracks, attribute_options, grid_options, member_object]
    args_dispatcher = {
        "coordinates_from_object_record": [object_tracks, grid_options],
        "coordinates_from_mask": from_mask_args,
    }
    args = args_dispatcher.get(func)
    lats, lons = get(*args)
    attributes["latitude"] += list(lats)
    attributes["longitude"] += list(lons)


def record_velocities(
    attributes, attribute_options, object_tracks, grid_options, velocity_type
):
    """Record object coordinates."""
    keys = attributes.keys()
    if not f"u_{velocity_type}" in keys or not f"v_{velocity_type}" in keys:
        message = "Both u and v compononents must be specified."
        raise ValueError(message)
    func = attribute_options[f"u_{velocity_type}"]["method"]["function"]
    lon_func = attribute_options[f"v_{velocity_type}"]["method"]["function"]
    if func != lon_func:
        message = "Functions for acquring u and v velocities must be the same."
        raise ValueError(message)
    get_velocities = get_attributes_dispatcher.get(func)
    if get_velocities is None:
        message = f"Function {func} for obtaining u and v not recognised."
        raise ValueError(message)

    object_tracks, attribute_options, grid_options

    name = "u_" + velocity_type
    velocities_args = [name, object_tracks, attribute_options, grid_options]
    args_dispatcher = {"velocities_from_object_record": velocities_args}
    args = args_dispatcher.get(func)
    v, u = get_velocities(*args)
    attributes[f"v_{velocity_type}"] += list(v)
    attributes[f"u_{velocity_type}"] += list(u)


def get_ids(object_tracks, attribute_options, member_object):
    """Get object ids."""

    if attribute_options is None:
        return
    if "universal_id" in attribute_options:
        id_type = "universal_id"
    elif "id" in attribute_options:
        id_type = "id"

    arguments_dispatcher = {
        "ids_from_mask": [object_tracks, attribute_options, member_object],
        "ids_from_object_record": [id_type, object_tracks],
    }

    func = attribute_options[id_type]["method"]["function"]
    get = get_attributes_dispatcher.get(func)
    args = arguments_dispatcher.get(func)
    ids = get(*args)

    if ids is not None:
        ids = np.array(ids).astype(attribute_options[id_type]["data_type"])
    return id_type, ids


# Record core attributes
# def record(
#     attributes,
#     object_tracks,
#     attribute_options,
#     grid_options,
#     member_object=None,
# ):

#     # attributes, attribute_options, object_tracks, grid_options, member_object

#     """Get object core attributes."""
#     id_type, ids = get_ids(object_tracks, attribute_options, member_object)
#     # If no objects, return
#     if ids is None or len(ids) == 0:
#         return

#     time_data_type = attribute_options["time"]["data_type"]
#     previous_time = object_tracks["previous_times"][-1]
#     times = np.array([previous_time for i in range(len(ids))]).astype(time_data_type)

#     attributes["time"] += list(times)
#     attributes[id_type] += list(ids)
#     keys = attributes.keys()

#     # Need a better way of handling attributes like lat lon which come in groups
#     if "latitude" in keys and "longitude" in keys:
#         args = [attributes, attribute_options, object_tracks, grid_options]
#         args += [member_object]
#         record_coordinates(*args)
#     if "u_flow" in keys and "v_flow" in keys:
#         args = [attributes, attribute_options, object_tracks, grid_options, "flow"]
#         record_velocities(*args)
#     if "u_displacement" in keys and "v_displacement" in keys:
#         args = [attributes, attribute_options, object_tracks, grid_options]
#         args += ["displacement"]
#         record_velocities(*args)

#     # Get the remaining attributes
#     # First create a dispatcher for the arguments to the various get attribute functions
#     areas_args = [object_tracks, attribute_options, grid_options, member_object]
#     arguments_dispatcher = {
#         "areas_from_object_record": [object_tracks, attribute_options],
#         "parents_from_object_record": [object_tracks, attribute_options],
#         "areas_from_mask": areas_args,
#     }
#     processed_attributes = ["time", id_type, "latitude", "longitude"]
#     processed_attributes += ["u_flow", "v_flow", "u_displacement", "v_displacement"]
#     remaining_attributes = [attr for attr in keys if attr not in processed_attributes]
#     for name in remaining_attributes:
#         attr_function = attribute_options[name]["method"]["function"]
#         get_attr = get_attributes_dispatcher.get(attr_function)
#         if get_attr is None:
#             message = f"Function {attr_function} for obtaining attribute {name} not recognised."
#             raise ValueError(message)
#         args = arguments_dispatcher.get(attr_function)
#         if args is None:
#             message = f"Arguments for function {attr_function} not specified in "
#             message += "arguments_dispatcher."
#             raise ValueError(message)
#         attribute = get_attr(*args)
#         attributes[name] += list(attribute)
