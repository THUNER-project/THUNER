"""Manage properties dictionaries."""

from thor.log import setup_logger

logger = setup_logger(__name__)


def core_attributes(core_attribute_names=None):
    """Create a dictionary of core attributes for detected objects."""

    if core_attribute_names is None:
        core_attribute_names = ["time", "universal_id", "id"]
        core_attribute_names += ["latitude", "longitude", "area", "u_flow", "v_flow"]
        core_attribute_names += ["u_displacement", "v_displacement"]
    core_attributes = {name: basic_attribute(name) for name in core_attribute_names}
    return core_attributes


def basic_attribute(name, method=None, description=None):
    """
    Specify options for a core property, typically obtained from the matching process.
    """

    if name == "time":
        if method is None:
            method = {"function": None}
        if description is None:
            description = f"Time taken from the tracking process."
    elif name == "universal_id":
        if method is None:
            method = {"function": "attribute_from_object_record"}
        if description is None:
            description = f"{name} taken from the matching process."
    elif name == "id":
        if method is None:
            method = {"function": "id_from_mask"}
        if description is None:
            description = f"{name} taken from the object mask."
            description += "Unlike uid, id is not necessarily unique across time steps."
    elif name == "latitude" or name == "longitude":
        if method is None:
            method = {"function": "attribute_from_object_record"}
        if description is None:
            description = f"{name} position taken from the matching process;"
            description += f"typically this is a gridcell area weighted mean over the "
            description += "object mask."
    elif (name == "u" or name == "v") and method is None:
        if method is None:
            method = {"function": "attribute_from_object_record"}
        if description is None:
            description = f"{name} velocity taken from the matching process;"
            description += f"typically this is a quality controlled cross correlation."
    elif name == "area":
        if method is None:
            method = {"function": "attribute_from_object_record"}
        if description is None:
            description = f"Object area taken from the matching process."
    else:
        if method is None or description is None:
            message = f"Property {name} not recognised. Please specify method and description."
            raise ValueError(message)

    attribute_options = {
        "name": name,
        "method": method,
        "description": description,
    }
    return attribute_options
