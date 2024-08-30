"""
Methods for defining property options associated with detected objects, and for 
measuring such properties.
"""

from thor.log import setup_logger

logger = setup_logger(__name__)


def basic_property(name, method=None, description=None):
    """
    Specify options for a location property, typically obtained from the matching
    process.
    """

    if name == "latitude" or name == "longitude":
        if method is None:
            method = {"function": "location_from_match"}
        if description is None:
            description = f"{name} position taken from the matching process;"
            description += f"typically this is a gridcell area weighted mean over the "
            description += "object mask."
    elif (name == "u" or name == "v") and method is None:
        if method is None:
            method = {"function": "velocity_from_match"}
        if description is None:
            description = f"{name} velocity taken from the matching process;"
            description += f"typically this is a quality controlled cross correlation."
    elif name == "area":
        if method is None:
            method = {"function": "area_from_match"}
        if description is None:
            description = f"Object area taken from the matching process."
    else:
        if method is None or description is None:
            message = f"Property {name} not recognised. Please specify method and description."
            raise ValueError(message)

    property_options = {
        "name": name,
        "method": method,
        "description": description,
    }
    return property_options
