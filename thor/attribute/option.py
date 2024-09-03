"""Manage properties dictionaries."""

from thor.log import setup_logger
import numpy as np

logger = setup_logger(__name__)


def core_attributes(names=None, tracked=True):
    """Create a dictionary of core attributes for detected objects."""

    if names is None:
        names = ["time", "latitude", "longitude", "area"]
        if tracked:
            names += ["universal_id", "u_flow", "v_flow"]
            names += ["u_displacement", "v_displacement"]
        else:
            names += ["id"]
    core_attributes = {name: basic_attribute(name, tracked=tracked) for name in names}
    return core_attributes


def id_attribute(name="id", method=None, description=None, tracked=True):
    """
    Options for id attribute.
    """
    if method is None:
        if tracked:
            method = {"function": "ids_from_object_record"}
            source = "object record."
        else:
            method = {"function": "ids_from_mask"}
            source = "object mask."
    if description is None:
        description = f"{name} taken from {source} "
        description += "Unlike uid, id is not necessarily unique across time steps."
    return method, description, int


def coordinate_attribute(name, method=None, description=None, tracked=True):
    """
    Options for coordinate attributes.
    """
    if method is None:
        if tracked:
            method = {"function": "coordinates_from_object_record"}
            source = "object record."
        else:
            method = {"function": "coordinates_from_mask"}
            source = "object mask."
    if description is None:
        description = f"{name} position taken from the {source}; "
        description += f"usually a gridcell area weighted mean over the object mask."
    return method, description, float


def velocity_attribute(name, method=None, description=None, tracked=True):
    """
    Options for velocity attributes. Velocities only defined for tracked objects.
    """
    if not tracked:
        message = f"Velocity attribute {name} only defined for tracked objects."
        raise ValueError(message)

    if method is None:
        method = {"function": "velocities_from_object_record"}
    if description is None:
        description = f"{name} velocities taken from the matching process."
    return method, description, float


def area_attribute(name="area", method=None, description=None, tracked=True):
    """
    Options for area attribute.
    """
    if method is None:
        if tracked:
            method = {"function": "areas_from_object_record"}
            source = "object record."
        else:
            method = {"function": "areas_from_mask"}
            source = "object mask."
    if description is None:
        description = f"Object area taken from the {source}."
    return method, description, float


basic_attribute_dispatcher = {
    "id": id_attribute,
    "universal_id": id_attribute,
    "latitude": coordinate_attribute,
    "longitude": coordinate_attribute,
    "u_flow": velocity_attribute,
    "v_flow": velocity_attribute,
    "u_displacement": velocity_attribute,
    "v_displacement": velocity_attribute,
    "area": area_attribute,
}


def basic_attribute(name, method=None, description=None, tracked=True, data_type=None):
    """
    Specify options for a core property, typically obtained from the matching process.
    """

    if name == "time":
        if method is None:
            method = {"function": None}
        if description is None:
            description = f"Time taken from the tracking process."
        if data_type is None:
            data_type = "datetime64[s]"
    else:
        if name in basic_attribute_dispatcher.keys():
            get_attr_options = basic_attribute_dispatcher[name]
            method, description, data_type = get_attr_options(
                name, method, description, tracked
            )
        elif method is None or description is None or data_type is None:
            message = f"Property {name} not recognised. Please specify method and description."
            raise ValueError(message)
    attribute_options = {
        "name": name,
        "method": method,
        "data_type": data_type,
        "description": description,
    }
    return attribute_options
