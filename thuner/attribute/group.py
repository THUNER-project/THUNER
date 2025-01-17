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
from thuner.option.attribute import Retrieval, Attribute, AttributeType, AttributeGroup

logger = setup_logger(__name__)


def offset_from_centers(object_tracks, attribute_group: AttributeGroup, objects):
    """Calculate offset between object centers."""
    member_attributes = object_tracks["current_attributes"]["member_objects"]
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
    data_type = attribute_group.attributes[0].data_type
    y_offsets = y_offsets.astype(data_type).tolist()
    x_offsets = x_offsets.astype(data_type).tolist()
    return {"y_offset": y_offsets, "x_offset": x_offsets}


kwargs = {"name": "x_offset", "data_type": float, "precision": 1, "units": "km"}
description = " offset of one object from another in km."
kwargs.update({"description": "x " + description})
x_offset = Attribute(**kwargs)
kwargs.update({"name": "y_offset", "description": "y " + description})
y_offset = Attribute(**kwargs)

keyword_arguments = {"objects": ["convective", "anvil"]}
retrieval = Retrieval(function=offset_from_centers, keyword_arguments=keyword_arguments)
kwargs = {"name": "offset", "description": "Offset of one object from another"}
kwargs.update({"retrieval": retrieval, "attributes": [x_offset, y_offset]})
offset = AttributeGroup(**kwargs)


# Convenience functions for creating default ellipse attribute type
def default(matched=True):
    """Create the default group attribute type."""

    attributes_list = core.retrieve_core(attributes_list=[core.time], matched=matched)
    attributes_list.append(offset)
    description = "Attributes associated with grouped objects, e.g. offset of "
    description += "stratiform echo from convective echo."
    kwargs = {"name": "group", "attributes": attributes_list}
    kwargs.update({"description": description})

    return AttributeType(**kwargs)
