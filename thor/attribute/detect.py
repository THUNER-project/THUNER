"""
Methods for defining property options associated with detected objects, and for 
measuring such properties.
"""

from thor.log import setup_logger

logger = setup_logger(__name__)


def attribute_from_object_record(
    attribute_name, time, object_tracks, object_options, grid_options
):
    """Get object attribute from object record created by the matching process."""
    if attribute_name is "latitude" or attribute_name is "longitude":
        coordinates = object_tracks["object_record"]["previous_centers"]


def coordinate_from_mask(
    attribute_name, time, object_tracks, object_options, grid_options
):
    """Get object coordinate from mask."""
    pass


get_coordinate_dispatcher = {
    "attribute_from_object_record": attribute_from_object_record,
    "atribute_from_mask": coordinate_from_mask,
}


def get_coordinate(attribute_name, time, object_tracks, object_options):
    """Get object coordinate."""
    attribute_options = object_options["attribute"]["detect"][attribute_name]
    get_coord = get_coordinate_dispatcher[attribute_options["function"]]
    coordinates = get_coord(attribute_name, time, object_tracks, object_options)
    return coordinates


def get_area(time, object_tracks, object_options):
    """Get object area."""
    pass


def get_velocity(attribute_name, object_tracks, object_options):
    """Get object velocity."""
    pass


get_attribute_dispatcher = {
    "latitude": get_coordinate,
    "longitude": get_coordinate,
    "area": get_area,
    "u": get_velocity,
    "v": get_velocity,
}


def get(time, object_tracks, object_options):
    """Get object attributes."""
    if object_options["attribute"]["detect"] is None:
        return

    if "universal_id" in object_options["attribute"]["detect"]:
        id_type = "universal_id"
        id = object_tracks["object_record"]["universal_ids"]
    elif "id" in object_options["attribute"]["detect"]:
        id_type = "id"
        id = object_tracks["object_record"]["previous_ids"]

    times = [time] * len(id)

    detected_attributes = object_tracks["attribute"]["detect"]
    for attribute_name in detected_attributes.keys():
        if attribute_name in get_attribute_dispatcher.keys():
            get_attribute = get_attribute_dispatcher[attribute_name]
            attribute = get_attribute(time, object_tracks, object_options)
            object_tracks["attribute"][attribute_name] = attribute
        else:
            raise ValueError(f"Attribute {attribute_name} not recognised.")
