"""Functions for working with attributes related to quality control."""

from thuner.log import setup_logger
import thuner.attribute.core as core
import thuner.attribute.utils as utils
import xarray as xr

logger = setup_logger(__name__)


# Convenience functions for defining default attribute options
def boundary_overlap():
    """
    Options for the boundary_overlap fraction attribute.
    """
    name = "boundary_overlap"
    method = None
    data_type = float
    precision = 4
    units = None
    description = "Fraction of object area comprised on boundary pixels."
    args = [name, method, data_type, precision, description, units]
    return utils.get_attribute_dict(*args)


def default(names=None, matched=True):
    """Create a dictionary of default quality control attribute options."""

    if names is None:
        names = ["time", "boundary_overlap"]
    if matched:
        id_type = "universal_id"
    else:
        id_type = "id"
    core_method = {"function": "attribute_from_core"}
    attributes = {}
    # Reuse core attributes, just replace the default functions method
    attributes["time"] = core.time(method=core_method)
    attributes[id_type] = core.identity(id_type, method=core_method)
    if "boundary_overlap" in names:
        attributes["boundary_overlap"] = boundary_overlap()

    return attributes


get_attributes_dispatcher = {"attribute_from_core": utils.attribute_from_core}


def record_boundary_overlaps(
    input_records,
    attributes,
    attribute_options,
    object_tracks,
    object_options,
    member_object=None,
):
    """Get boundary overlap from mask."""

    if "universal_id" in attributes.keys():
        id_type = "universal_id"
    elif "id" in attributes.keys():
        id_type = "id"
    else:
        message = "No id attribute found in attributes."
        raise ValueError(message)
    ids = attributes[id_type]

    if "dataset" not in object_options.model_fields:
        message = "Dataset must be specified in object_options for which domain "
        message += "boundary is defined."
        raise ValueError(message)
    object_dataset = object_options.dataset
    input_record = input_records["track"][object_dataset]
    boundary_mask = input_record["previous_boundary_masks"][-1]

    mask = utils.get_previous_mask(attribute_options, object_tracks)
    # If examining just a member of a grouped object, get masks for that object
    if member_object is not None and isinstance(mask, xr.Dataset):
        mask = mask[f"{member_object}_mask"]

    areas = object_tracks["gridcell_area"]

    overlaps = []
    for obj_id in ids:
        if boundary_mask is None:
            overlaps.append(0)
        else:
            obj_mask = mask == obj_id
            overlap = (obj_mask * boundary_mask) == True
            area_fraction = areas.where(overlap).sum() / areas.where(obj_mask).sum()
            overlaps.append(float(area_fraction.values))

    boundary_overlaps = {"boundary_overlap": overlaps}
    attributes.update(boundary_overlaps)


def record(
    input_records,
    attributes,
    object_tracks,
    object_options,
    attribute_options,
    member_object=None,
):
    """Get group object attributes."""
    # Get core attributes
    core_attributes = ["time", "id", "universal_id"]
    keys = attributes.keys()
    core_attributes = [attr for attr in core_attributes if attr in keys]
    remaining_attributes = [attr for attr in keys if attr not in core_attributes]
    # Get the appropriate core attributes
    for name in core_attributes:
        attr_function = attribute_options[name]["method"]["function"]
        get_attr = get_attributes_dispatcher.get(attr_function)
        if get_attr is None:
            message = f"Function {attr_function} for obtaining attribute {name} not recognised."
            raise ValueError(message)
        # Should add an arguments dispatcher here
        attr = get_attr(name, object_tracks, member_object)
        attributes[name] += list(attr)

    if attributes["time"] is None or len(attributes["time"]) == 0:
        return

    # Get non-core attributes
    if "boundary_overlap" in remaining_attributes:
        args = [input_records, attributes, attribute_options, object_tracks]
        args += [object_options, member_object]
        record_boundary_overlaps(*args)
