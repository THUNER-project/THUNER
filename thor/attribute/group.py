"""
Methods for defining property options associated with detected objects, and for 
measuring such properties.
"""

import numpy as np
from thor.log import setup_logger
import thor.grid as grid
from thor.object.object import get_object_center
import thor.attribute.core as core

logger = setup_logger(__name__)


def record(
    time, attributes, object_tracks, object_options, attribute_options, grid_options
):
    """Get object attributes."""
    matched_obj = object_options["tracking"]["options"]["matched_object"]

    # First get the core attributes of each member object
    for obj in attributes["member_objects"].keys():
        obj_attributes = attributes["member_objects"][obj]["core"]
        options = attribute_options["member_objects"][obj]["core"]
        # Need to tell core to look at member_obj in the mask dataset
        record_args = [time, obj_attributes, object_tracks, object_options, options]
        record_args += [grid_options, obj]
        core.record(*record_args)
    # Now get core attributes of the grouped object
    obj = list(attribute_options.keys() - {"member_objects"})[0]
    obj_attributes = attributes[obj]["core"]
    options = attribute_options[obj]["core"]
    record_args = [time, obj_attributes, object_tracks, object_options, options]
    record_args += [grid_options, matched_obj]
    core.record(*record_args)
