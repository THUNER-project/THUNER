"""Methods for getting object attributes."""

from thor.attribute import core, group, utils
from thor.log import setup_logger

logger = setup_logger(__name__)

record_dispatcher = {"core": core.record, "group": group.record}


def record_detected(time, object_tracks, object_options, grid_options):
    """Get detected object attributes."""
    attribute_options = object_options["attributes"]
    # Get the object attributes of each type, e.g. core, tag, profile
    for attributes_type in attribute_options.keys():
        options = attribute_options[attributes_type]
        attributes = object_tracks["current_attributes"][attributes_type]
        record_func = record_dispatcher[attributes_type]
        args = [time, attributes, object_tracks, object_options, options, grid_options]
        record_func(*args)


def record_grouped(time, object_tracks, object_options, grid_options):
    """Get object attributes."""
    # First get the attributes of each member object
    member_options = object_options["attributes"]["member_objects"]
    member_attributes = object_tracks["current_attributes"]["member_objects"]
    for obj in member_attributes.keys():
        obj_attributes = member_attributes[obj]
        for attribute_type in obj_attributes.keys():
            options = member_options[obj][attribute_type]
            attributes = obj_attributes[attribute_type]
            record_func = record_dispatcher[attribute_type]
            record_args = [time, attributes, object_tracks, object_options, options]
            record_args += [grid_options]
            record_func(*record_args, member_object=obj)
    # Now get attributes of the grouped object
    obj = list(object_options["attributes"].keys() - {"member_objects"})[0]
    obj_attributes = object_tracks["current_attributes"][obj]
    for attribute_type in obj_attributes.keys():
        options = object_options["attributes"][obj][attribute_type]
        attributes = obj_attributes[attribute_type]
        record_func = record_dispatcher[attribute_type]
        record_args = [time, attributes, object_tracks, object_options, options]
        record_args += [grid_options]
        record_func(*record_args, member_object=obj)


def append_detected(object_tracks):
    """
    Append current_attributes dictionary to attributes dictionary for detected objects.
    """
    attributes = object_tracks["attributes"]
    current_attributes = object_tracks["current_attributes"]
    for attribute_type in current_attributes.keys():
        for attr in current_attributes[attribute_type].keys():
            attr_list = attributes[attribute_type][attr]
            attr_list += current_attributes[attribute_type][attr]


def append_grouped(object_tracks):
    """
    Append current_attributes dictionary to attributes dictionary grouped objects.
    """
    member_attributes = object_tracks["attributes"]["member_objects"]
    current_member_attributes = object_tracks["current_attributes"]["member_objects"]
    # First append attributes for member objects
    for obj in member_attributes.keys():
        for attribute_type in member_attributes[obj].keys():
            for attr in member_attributes[obj][attribute_type].keys():
                attr_list = member_attributes[obj][attribute_type][attr]
                attr_list += current_member_attributes[obj][attribute_type][attr]
    # Now append attributes for grouped object
    obj = list(object_tracks["attributes"].keys() - {"member_objects"})[0]
    attributes = object_tracks["attributes"][obj]
    current_attributes = object_tracks["current_attributes"][obj]
    for attribute_type in current_attributes.keys():
        for attr in current_attributes[attribute_type].keys():
            attr_list = attributes[attribute_type][attr]
            attr_list += current_attributes[attribute_type][attr]


def record(time, object_tracks, object_options, grid_options):
    """Get object attributes."""
    logger.info("Recording object attributes.")
    if object_options["attributes"] is None:
        return

    # Reset the "current" attributes dictionary, i.e. the attributes associated with the
    # objects in the "previous" grid. Note out naming convention is that objects are
    # identified in the "current" (corresponding to "time") and matched with the
    # objects previously identified in the "previous" grid. The iteration corresponding
    # to time then records the attributes of the objects identified in the "previous"
    # grid. The name "current_attributes" is thus perhaps misleading.
    object_tracks["current_attributes"] = utils.initialize_attributes(object_options)

    if "detection" in object_options:
        record_func = record_detected
        append_func = append_detected
    elif "grouping" in object_options:
        record_func = record_grouped
        append_func = append_grouped
    else:
        message = "Object indentification method must be specified, i.e. "
        message += "'detection' or 'grouping'."
        raise ValueError(message)

    record_func(time, object_tracks, object_options, grid_options)
    append_func(object_tracks)
