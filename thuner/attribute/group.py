"""
Describe and retrieve attributes associated with grouped objects.
"""

from thuner.log import setup_logger
import thuner.attribute.core as core
import numpy as np
import thuner.grid as grid
from thuner.option.attribute import Attribute, AttributeType, AttributeGroup
from thuner.utils import Retrieval
from thuner.attribute.utils import get_current_mask, get_ids

logger = setup_logger(__name__)

__all__ = [
    "offset_from_centers",
    "default",
    "XOffset",
    "YOffset",
    "Offset",
]


def offset_from_centers(object_tracks, attribute_group: AttributeGroup, objects):
    """Calculate offset between object centers."""
    member_attributes = object_tracks.current_attributes.member_attributes
    if len(objects) != 2:
        raise ValueError("Offset calculation requires two objects.")
    core_attributes = object_tracks.current_attributes.attribute_types["core"]
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


def members_from_masks(
    object_options,
    tracks,
    matched: bool,
    members_matched: list[bool],
):
    """Retrieve ids of the member objects comprising the grouped object."""

    object_name = object_options.name
    object_level = object_options.hierarchy_level
    object_tracks = tracks.levels[object_level].objects[object_name]

    member_objects = object_options.grouping.member_objects
    member_levels = object_options.grouping.member_levels

    grouped_mask = get_current_mask(object_tracks, matched)
    ids = get_ids(object_tracks, matched, None)
    member_object_ids = {f"{member_obj}_ids": [] for member_obj in member_objects}
    for object_id in ids:
        for i, obj in enumerate(member_objects):
            level = member_levels[i]
            member_tracks = tracks.levels[level].objects[obj]
            mask = get_current_mask(member_tracks, members_matched[i])
            mask_name = obj + "_mask"
            member_ids = mask.where(grouped_mask[mask_name] == object_id)
            member_ids = np.unique(member_ids.values.flatten())
            member_ids = member_ids[np.logical_not(np.isnan(member_ids))]
            # Create a space separated string of ids
            member_ids = " ".join([str(int(id)) for id in member_ids])
            member_object_ids[f"{obj}_ids"].append(member_ids)

    # Rename the keys to match the attribute names
    return member_object_ids


class XOffset(Attribute):
    """Zonal offset between member objects."""

    name: str = "x_offset"
    data_type: type = float
    precision: int = 1
    units: str = "km"
    description: str = "x offset of one object from another in km."


class YOffset(Attribute):
    """Meridional offset between member objects."""

    name: str = "y_offset"
    data_type: type = float
    precision: int = 1
    units: str = "km"
    description: str = "y offset of one object from another in km."


class Offset(AttributeGroup):
    """Attribute describing horizontal offset vector between objects."""

    name: str = "offset"
    description: str = "Offset of one object from another."
    attributes: list[Attribute] = [XOffset(), YOffset()]
    retrieval: Retrieval = Retrieval(
        function=offset_from_centers,
        keyword_arguments={"objects": ["convective", "anvil"]},
    )


class MemberIDs(Attribute):
    """Member IDs of the group."""

    name: str = ""  # Replace this with the actual object name
    data_type: type = str
    description: str = "IDs of the member objects in the grouped object."


def build_membership_attribute_group(
    member_objects=["convective", "middle", "anvil"],
    matched=True,
    members_matched=[True, False, False],
):
    """
    Create attribute options for each member object of a group, and an encompassing
    attribute group. Note the object ordering implicit in members_matched should match
    the ordering of objects in the grouped objects grouping options.
    """
    attributes = []
    for i, obj in enumerate(member_objects):
        id_type = "universal_id" if members_matched[i] else "id"
        name = f"{obj}_ids"
        description = f"Space seperated list of the {id_type}s of the {obj} objects "
        description += "in the group."
        kwargs = {"name": name, "data_type": str, "description": description}
        attributes.append(Attribute(**kwargs))
    description = f"Attribute group for the attributes describing member object ids of "
    description += "a grouped object."
    kwargs = {"function": members_from_masks}
    kwargs["keyword_arguments"] = {"matched": matched}
    kwargs["keyword_arguments"]["members_matched"] = members_matched

    retrieval = Retrieval(**kwargs)

    kwargs = {"name": "member_objects", "attributes": attributes}
    kwargs.update({"retrieval": retrieval, "description": description})
    return AttributeGroup(**kwargs)


# Convenience functions for creating default group attribute type
def default(matched=True):
    """Create the default group attribute type."""

    attributes_list = core.retrieve_core(attributes_list=[core.Time()], matched=matched)
    attributes_list.append(Offset())
    description = "Attributes associated with grouped objects, e.g. offset of "
    description += "stratiform echo from convective echo."
    kwargs = {"name": "group", "attributes": attributes_list}
    kwargs.update({"description": description})
    return AttributeType(**kwargs)
