"""Functions for getting object attributes."""

import inspect
from thuner.log import setup_logger
from thuner.option.track import AnyObjectOptions
from thuner.option.attribute import Attribute, AttributeGroup, AttributeType
from thuner.utils import camel_to_snake, filter_arguments

logger = setup_logger(__name__)


def _initialize_attribute_type(attribute_type: AttributeType):
    """Initialize attributes lists for a given attribute type."""
    attributes = {}
    for attr in attribute_type.attributes:
        if isinstance(attr, AttributeGroup):
            for attr_attr in attr.attributes:
                attributes[attr_attr.name] = []
        elif isinstance(attr, Attribute):
            attributes[attr.name] = []
        else:
            raise ValueError(f"Unknown type {attr.type}.")
    return attributes


def initialize_attributes(object_options):
    """Initialize attributes lists for grouped objects."""
    # First initialize attributes for member objects
    attributes = object_options.attributes
    member_attributes = attributes.member_attributes
    object_name = object_options.name
    attributes_record = {"member_objects": {}, object_name: {}}
    member_attributes_record = {}
    # Initialize attributes for member objects if appropriate
    if member_attributes is not None:
        for obj in member_attributes.keys():
            attribute_types = member_attributes[obj].attribute_types
            member_attributes_record[obj] = {t.name: {} for t in attribute_types}
            for attribute_type in attribute_types:
                attr = _initialize_attribute_type(attribute_type)
                member_attributes_record[obj][attribute_type.name] = attr
        attributes_record["member_objects"] = member_attributes_record
    # Now initialize the objects attributes
    obj = object_name
    for attribute_type in attributes.attribute_types:
        attr = _initialize_attribute_type(attribute_type)
        attributes_record[obj][attribute_type.name] = attr
    return attributes_record


def retrieve_attribute(general_kwargs, attribute, member_object=None):
    # Get the retrieval function and arguments for the attribute
    func_kwargs = general_kwargs.copy()
    keyword_arguments = attribute.retrieval.keyword_arguments
    func_kwargs.update(keyword_arguments)
    # Retrieval functions expect either "attribute" or "attribute_group"
    # keyword arguments. Infer correct argument name from attribute type.
    if isinstance(attribute, Attribute):
        func_kwargs.update({"attribute": attribute})
    elif isinstance(attribute, AttributeGroup):
        func_kwargs.update({"attribute_group": attribute})
    else:
        raise ValueError(f"attribute must be instance of Attribute or AttributeGroup.")
    func_kwargs.update({"member_object": member_object})
    func = attribute.retrieval.function
    # Filter out arguments not expected by the function
    # Doing this here avoids cluttering retrieval function definitions
    func_kwargs = filter_arguments(func, func_kwargs)
    # Retrieve the attribute
    return func(**func_kwargs)


def record(
    time, input_records, object_tracks, object_options: AnyObjectOptions, grid_options
):
    """Record object attributes."""

    if "object_record" not in object_tracks.keys():
        no_objects = True
    else:
        no_objects = len(object_tracks["object_record"]["previous_ids"]) == 0
    if object_options.attributes is None or no_objects:
        return

    logger.info(f"Recording {object_options.name} attributes.")

    # Specify the keyword arguments common to all attribute retrieval functions
    kwargs = {"time": time, "input_records": input_records}
    kwargs.update({"object_tracks": object_tracks, "object_options": object_options})
    kwargs.update({"grid_options": grid_options})

    # For grouped objects, get the member object attributes first
    if object_options.attributes.member_attributes is not None:
        member_attribute_options = object_options.attributes.member_attributes
        member_attributes = object_tracks["current_attributes"]["member_objects"]
        for obj in member_attribute_options.keys():
            for attribute_type in member_attribute_options[obj].attribute_types:
                for attribute in attribute_type.attributes:
                    attr = retrieve_attribute(kwargs, attribute, member_object=obj)
                    member_attributes[obj][attribute_type.name].update(attr)

    # Now get the attributes of the object itself
    obj = object_options.attributes.name
    obj_attributes = object_tracks["current_attributes"][obj]
    for attribute_type in object_options.attributes.attribute_types:
        for attribute in attribute_type.attributes:
            attr = retrieve_attribute(kwargs, attribute)
            obj_attributes[attribute_type.name].update(attr)

    # Append the current attributes to the attributes dictionary
    append(object_tracks)


def append_attribute_type(current_attributes, attributes, attributes_type):
    """
    Append current_attributes dictionary to attributes dictionary for a given
    attribute type.
    """
    for attr in current_attributes[attributes_type].keys():
        attr_list = attributes[attributes_type][attr]
        attr_list += current_attributes[attributes_type][attr]


def append(object_tracks):
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
