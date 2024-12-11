"""Functions for getting object attributes."""

from thuner.attribute import core, group, profile, utils, quality, tag, ellipse
from thuner.log import setup_logger

logger = setup_logger(__name__)

record_dispatcher = {
    "core": core.record,
    "group": group.record,
    "profile": profile.record,
    "quality": quality.record,
    "tag": tag.record,
    "ellipse": ellipse.record,
}


def get_record_arguments(
    attributes_type,
    input_records,
    attributes,
    object_tracks,
    object_options,
    attribute_options,
    grid_options,
    member_object=None,
):
    """Get the arguments for the specific record functions."""
    core_arguments = [attributes, object_tracks, attribute_options, grid_options]
    core_arguments += [member_object]
    group_arguments = [attributes, object_tracks, attribute_options, member_object]
    profile_arguments = [input_records, attributes, object_tracks, attribute_options]
    profile_arguments += [grid_options, member_object]
    quality_arguments = [input_records, attributes, object_tracks, object_options]
    quality_arguments += [attribute_options, member_object]
    tag_arguments = [input_records, attributes, object_tracks, attribute_options]
    tag_arguments += [member_object]
    ellipse_arguments = [attributes, object_tracks, attribute_options, grid_options]
    ellipse_arguments += [member_object]
    argument_dispatcher = {
        "core": core_arguments,
        "group": group_arguments,
        "profile": profile_arguments,
        "quality": quality_arguments,
        "tag": tag_arguments,
        "ellipse": ellipse_arguments,
    }
    return argument_dispatcher[attributes_type]


# @memory_profile
def record_detected(time, input_records, object_tracks, object_options, grid_options):
    """Get detected object attributes."""
    all_attribute_options = object_options.attributes
    # Get the object attributes of each type, e.g. core, tag, profile
    for attributes_type in all_attribute_options.keys():
        attribute_options = all_attribute_options[attributes_type]
        attributes = object_tracks["current_attributes"][attributes_type]
        record_func = record_dispatcher[attributes_type]
        args = [input_records, attributes, object_tracks, object_options]
        args += [attribute_options, grid_options]
        args = get_record_arguments(attributes_type, *args)
        record_func(*args)


# @memory_profile
# But what if a member object is also a grouped object?
def record_grouped(time, input_records, object_tracks, object_options, grid_options):
    """Get object attributes."""
    # First get the attributes of each member object
    member_options = object_options.attributes["member_objects"]
    member_attributes = object_tracks["current_attributes"]["member_objects"]
    for obj in member_attributes.keys():
        obj_attributes = member_attributes[obj]
        for attributes_type in obj_attributes.keys():
            attribute_options = member_options[obj][attributes_type]
            attributes = obj_attributes[attributes_type]
            record_func = record_dispatcher[attributes_type]
            args = [input_records, attributes, object_tracks, object_options]
            args += [attribute_options, grid_options, obj]
            args = get_record_arguments(attributes_type, *args)
            record_func(*args)
    # Now get attributes of the grouped object
    obj = list(object_options.attributes.keys() - {"member_objects"})[0]
    obj_attributes = object_tracks["current_attributes"][obj]
    for attributes_type in obj_attributes.keys():
        attribute_options = object_options.attributes[obj][attributes_type]
        attributes = obj_attributes[attributes_type]
        record_func = record_dispatcher[attributes_type]
        args = [input_records, attributes, object_tracks, object_options]
        args += [attribute_options, grid_options, obj]
        args = get_record_arguments(attributes_type, *args)
        record_func(*args)


def append_attribute_type(current_attributes, attributes, attributes_type):
    """
    Append current_attributes dictionary to attributes dictionary for a given
    attribute type.
    """
    for attr in current_attributes[attributes_type].keys():
        if attributes_type == "profile" or attributes_type == "tag":
            for dataset in current_attributes[attributes_type][attr].keys():
                attr_list = attributes[attributes_type][attr][dataset]
                attr_list += current_attributes[attributes_type][attr][dataset]
        else:
            attr_list = attributes[attributes_type][attr]
            attr_list += current_attributes[attributes_type][attr]


def append_detected(object_tracks):
    """
    Append current_attributes dictionary to attributes dictionary for detected objects.
    """
    attributes = object_tracks["attributes"]
    current_attributes = object_tracks["current_attributes"]
    for attributes_type in current_attributes.keys():
        append_attribute_type(current_attributes, attributes, attributes_type)


def append_grouped(object_tracks):
    """
    Append current_attributes dictionary to attributes dictionary grouped objects.
    """
    member_attributes = object_tracks["attributes"]["member_objects"]
    current_member_attributes = object_tracks["current_attributes"]["member_objects"]
    # First append attributes for member objects
    for obj in member_attributes.keys():
        for attributes_type in member_attributes[obj].keys():
            attr = member_attributes[obj]
            current_attr = current_member_attributes[obj]
            append_attribute_type(current_attr, attr, attributes_type)
    # Now append attributes for grouped object
    obj = list(object_tracks["attributes"].keys() - {"member_objects"})[0]
    attributes = object_tracks["attributes"][obj]
    current_attributes = object_tracks["current_attributes"][obj]
    for attributes_type in current_attributes.keys():
        append_attribute_type(current_attributes, attributes, attributes_type)


def record(time, input_records, object_tracks, object_options, grid_options):
    """Get object attributes."""
    logger.info("Recording object attributes.")
    if object_options.attributes is None:
        return

    # Reset the "current" attributes dictionary, i.e. the attributes associated with the
    # objects in the "previous" grid. Note out naming convention is that objects are
    # identified in the "current" (corresponding to "time") and matched with the
    # objects previously identified in the "previous" grid. The iteration corresponding
    # to time then records the attributes of the objects identified in the "previous"
    # grid. The name "current_attributes" is thus perhaps misleading.
    object_tracks["current_attributes"] = utils.initialize_attributes(object_options)

    if "detection" in object_options.model_fields:
        record_func = record_detected
        append_func = append_detected
    elif "grouping" in object_options.model_fields:
        record_func = record_grouped
        append_func = append_grouped
    else:
        message = "Object indentification method must be specified, i.e. "
        message += "'detection' or 'grouping'."
        raise ValueError(message)

    record_func(time, input_records, object_tracks, object_options, grid_options)
    append_func(object_tracks)
