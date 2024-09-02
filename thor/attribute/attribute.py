"""Methods for getting object attributes."""

from thor.attribute import detect
from thor.attribute import tag
from thor.attribute import profile
from thor.attribute import group
from thor.attribute import hybrid

get_attributes_dispatcher = {
    "detect": detect.get,
    "tag": tag.get,
    "profile": profile.get,
    "group": group.get,
    "hybrid": hybrid.get,
}


def get(time, object_tracks, object_options):
    """Get object attributes."""
    if object_options["attribute"] is None:
        return

    # Get the object attributes of each type, e.g. detect, tag, profile
    for attributes_type in object_options["attribute"].keys():
        get_attributes = get_attributes_dispatcher(attributes_type)
        attributes = get_attributes(object_tracks, object_options)
        object_tracks["attribute"][attributes_type] = attributes
