"""
Functions for defining property options associated with grouped objects, and for 
measuring such properties. Note the distinction between the attributes associated with 
this module, and functions like record_detected and record_grouped in the attributes
module, which collect the attributes associated with grouped and detected objects 
respectively.
"""

from thuner.log import setup_logger
import thuner.attribute.core as core
import numpy as np
import thuner.grid as grid
import thuner.attribute.utils as utils

logger = setup_logger(__name__)


# Convenience functions for defining default attribute options
def offset(name, method=None, description=None):
    """
    Specify options for a core property, typically obtained from the matching process.
    """
    data_type = float
    precision = 1
    units = "km"
    if method is None:
        method = {"function": "offset_from_centers"}
        # List objects to calculate offset between, with the offset vector pointing
        # from the first object in the list to the second object
        method["args"] = {"objects": ["convective", "anvil"]}
    if description is None:
        description = f"{name} of one member object center from another."
    args = [name, method, data_type, precision, description, units]
    return utils.get_attribute_dict(*args)


def default(names=None, matched=True):
    """Create a dictionary of default attribute options of grouped objects."""

    if names is None:
        names = ["time", "latitude", "longitude", "x_offset", "y_offset"]
    if matched:
        id_type = "universal_id"
    else:
        id_type = "id"
    names += [id_type]
    core_method = {"function": "attribute_from_core"}
    attributes = {}
    # Reuse core attributes, just replace the default functions method
    attributes["time"] = core.time(method=core_method)
    attributes["latitude"] = core.coordinate("latitude", method=core_method)
    attributes["longitude"] = core.coordinate("longitude", method=core_method)
    attributes[id_type] = core.identity(id_type, method=core_method)
    attributes["x_offset"] = offset("x_offset")
    attributes["y_offset"] = offset("y_offset")

    return attributes


# Functions for obtaining and recording attributes
def offset_from_centers(name, object_tracks, attribute_options):
    """Calculate offset between object centers."""
    member_attributes = object_tracks["current_attributes"]["member_objects"]
    objects = attribute_options[name]["method"]["args"]["objects"]
    grouped_object = object_tracks["name"]
    if len(objects) != 2:
        raise ValueError("Offset calculation requires two objects.")
    core_attributes = object_tracks["current_attributes"][grouped_object]["core"]
    if "universal_id" in core_attributes.keys():
        id_type = "universal_id"
    else:
        id_type = "id"
    ids = np.array(core_attributes[id_type])
    ids_1 = np.array(member_attributes[objects[0]]["core"][id_type])
    ids_2 = np.array(member_attributes[objects[1]]["core"][id_type])
    lats1 = np.array(member_attributes[objects[0]]["core"]["latitude"])
    lons1 = np.array(member_attributes[objects[0]]["core"]["longitude"])
    lats2 = np.array(member_attributes[objects[1]]["core"]["latitude"])
    lons2 = np.array(member_attributes[objects[1]]["core"]["longitude"])

    lats3, lons3, lats4, lons4 = [], [], [], []
    # Re-order the arrays so that the ids match
    for i in range(len(ids)):
        lats3.append(lats1[ids_1 == ids[i]][0])
        lons3.append(lons1[ids_1 == ids[i]][0])
        lats4.append(lats2[ids_2 == ids[i]][0])
        lons4.append(lons2[ids_2 == ids[i]][0])

    args = [lats3, lons3, lats4, lons4]
    y_offsets, x_offsets = grid.geographic_to_cartesian_displacement(*args)
    # Convert to km
    y_offsets, x_offsets = y_offsets / 1000, x_offsets / 1000
    return y_offsets, x_offsets


get_attributes_dispatcher = {
    "attribute_from_core": utils.attribute_from_core,
    "offset_from_centers": offset_from_centers,
}


def record_offsets(attributes, attribute_options, object_tracks):
    """Record offset."""
    keys = attributes.keys()
    if "x_offset" not in keys or "y_offset" not in keys:
        message = "Both x_offset and y_offset must be specified."
        raise ValueError(message)
    get_offsets_function = attribute_options["x_offset"]["method"]["function"]
    y_get_offsets_function = attribute_options["y_offset"]["method"]["function"]
    if get_offsets_function != y_get_offsets_function:
        message = "Functions for acquring y_offset and x_offset must be the same."
        raise ValueError(message)
    get_offsets = get_attributes_dispatcher.get(get_offsets_function)
    if get_offsets is None:
        message = f"Function {get_offsets_function} for obtaining x_offset and y_offset not recognised."
        raise ValueError(message)

    offset_args = ["y_offset", object_tracks, attribute_options]
    args_dispatcher = {"offset_from_centers": offset_args}
    args = args_dispatcher[get_offsets_function]
    y_offsets, x_offsets = get_offsets(*args)
    attributes["y_offset"] += list(y_offsets)
    attributes["x_offset"] += list(x_offsets)


def record(
    attributes,
    object_tracks,
    attribute_options,
    member_object=None,
):
    """Get group object attributes."""
    # Get core attributes
    core_attributes = ["time", "id", "universal_id", "latitude", "longitude"]
    core_attributes += ["u_flow", "v_flow", "u_displacement", "v_displacement"]
    keys = attributes.keys()
    core_attributes = [attr for attr in core_attributes if attr in keys]
    # Get the appropriate core attributes
    for name in core_attributes:
        attr_function = attribute_options[name]["method"]["function"]
        get_attr = get_attributes_dispatcher.get(attr_function)
        if get_attr is None:
            message = f"Function {attr_function} for obtaining attribute {name} not recognised."
            raise ValueError(message)
        attr = get_attr(name, object_tracks, member_object)
        attributes[name] += list(attr)

    if attributes["time"] is None or len(attributes["time"]) == 0:
        return

    # Get non-core attributes
    if "x_offset" in keys and "y_offset" in keys:
        record_offsets(attributes, attribute_options, object_tracks)
