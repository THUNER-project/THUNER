"""Functions for working with attributes related to quality control."""

import xarray as xr
from thuner.log import setup_logger
import thuner.attribute.core as core
import thuner.attribute.utils as utils
from thuner.option.attribute import Attribute, AttributeType
from thuner.utils import Retrieval


logger = setup_logger(__name__)

__all__ = ["overlap_from_mask", "default", "boundary_overlap"]


def overlap_from_mask(
    input_records,
    object_tracks,
    object_options,
    member_object=None,
    matched=True,
):
    """Get boundary overlap from mask."""

    if "dataset" not in object_options.__class__.model_fields:
        raise ValueError("Dataset must be specified in object_options.")
    object_dataset = object_options.dataset
    try:
        input_record = input_records.track[object_dataset]
    except KeyError:
        input_record = input_records.tag[object_dataset]
    boundary_mask = input_record.boundary_masks[-1]

    mask = utils.get_current_mask(object_tracks, matched=matched)
    # If examining just a member of a grouped object, get masks for that object
    if member_object is not None and isinstance(mask, xr.Dataset):
        mask = mask[f"{member_object}_mask"]

    areas = object_tracks.gridcell_area

    if matched:
        ids = object_tracks.match_record["universal_ids"]
    else:
        ids = object_tracks.match_record["ids"]

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


def boundary_overlap():
    """Convenience function to build a BoundaryOverlap attribute."""
    kwargs = {"name": "boundary_overlap", "data_type": float, "precision": 4}
    _desc = "Fraction of object area comprised of boundary gridcells."
    _retrieval = Retrieval(function=overlap_from_mask)
    kwargs.update({"description": _desc, "retrieval": _retrieval})
    return Attribute(**kwargs)


# class BoundaryOverlap(Attribute):
#     """Fraction of object pixels overlapping the domain boundary."""

#     name: str = "boundary_overlap"
#     data_type: type = float
#     precision: int = 4
#     description: str = "Fraction of object area comprised of boundary gridcells."
#     retrieval: Retrieval = Retrieval(function=overlap_from_mask)


# Convenience functions for creating default ellipse attribute type
def default(matched=True, member_object=None):
    """Create the default quality attribute type."""

    _attributes = core.retrieve_core(attributes_list=[core.time()], matched=matched)
    _boundary_overlap = boundary_overlap()
    _boundary_overlap.retrieval.keyword_arguments["member_object"] = member_object
    _attributes.append(_boundary_overlap)
    description = "Attributes associated with quality control, e.g. boundary overlap."
    kwargs = {"name": "quality", "attributes": _attributes}
    kwargs.update({"description": description})
    return AttributeType(**kwargs)
