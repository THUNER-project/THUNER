"""Functions for working with attributes related to quality control."""

import xarray as xr
from thuner.log import setup_logger
import thuner.attribute.core as core
import thuner.attribute.utils as utils
from thuner.option.attribute import Retrieval, Attribute, AttributeType


logger = setup_logger(__name__)


def overlap_from_mask(
    input_records,
    object_tracks,
    object_options,
    member_object=None,
    matched=True,
):
    """Get boundary overlap from mask."""

    if "dataset" not in object_options.model_fields:
        raise ValueError("Dataset must be specified in object_options.")
    object_dataset = object_options.dataset
    input_record = input_records["track"][object_dataset]
    boundary_mask = input_record["previous_boundary_masks"][-1]

    mask = utils.get_previous_mask(object_tracks, matched=matched)
    # If examining just a member of a grouped object, get masks for that object
    if member_object is not None and isinstance(mask, xr.Dataset):
        mask = mask[f"{member_object}_mask"]

    areas = object_tracks["gridcell_area"]

    if matched:
        ids = object_tracks["object_record"]["universal_ids"]
    else:
        ids = object_tracks["object_record"]["previous_ids"]

    overlaps = []
    for obj_id in ids:
        if boundary_mask is None:
            overlaps.append(0)
        else:
            obj_mask = mask == obj_id
            overlap = (obj_mask * boundary_mask) == True
            area_fraction = areas.where(overlap).sum() / areas.where(obj_mask).sum()
            overlaps.append(float(area_fraction))

    return {"boundary_overlap": overlaps}


kwargs = {"name": "boundary_overlap", "data_type": float, "precision": 4}
kwargs.update({"description": "Fraction of object area comprised of boundary pixels."})
kwargs.update({"retrieval": Retrieval(function=overlap_from_mask)})
boundary_overlap = Attribute(**kwargs)


# Convenience functions for creating default ellipse attribute type
def default(matched=True, member_object=None):
    """Create the default quality attribute type."""

    # Copy boundary overlap and add the appropriate member object argument
    new_boundary_overlap = boundary_overlap.model_copy(deep=True)
    new_boundary_overlap.retrieval.keyword_arguments["member_object"] = member_object
    attributes_list = core.retrieve_core(attributes_list=[core.time], matched=matched)
    attributes_list.append(new_boundary_overlap)
    description = "Attributes associated with quality control, e.g. boundary overlap."
    kwargs = {"name": "quality", "attributes": attributes_list}
    kwargs.update({"description": description})

    return AttributeType(**kwargs)
