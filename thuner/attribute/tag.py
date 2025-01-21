from thuner.log import setup_logger
import thuner.attribute.core as core
import xarray as xr
from thuner.option.attribute import Retrieval, Attribute, AttributeGroup, AttributeType

logger = setup_logger(__name__)


def setup_interp(
    attribute_group: AttributeGroup,
    input_records,
    object_tracks,
    dataset,
    member_object=None,
):
    name = object_tracks["name"]
    excluded = ["time", "id", "universal_id", "latitude", "longitude", "altitude"]
    excluded += ["time_offset"]
    attributes = attribute_group.attributes
    names = [attr.name for attr in attributes if attr.name not in excluded]
    tag_input_records = input_records["tag"]
    previous_time = object_tracks["previous_times"][-1]

    # Get object centers
    if member_object is None:
        core_attributes = object_tracks["current_attributes"][name]["core"]
    else:
        core_attributes = object_tracks["current_attributes"]["member_objects"]
        core_attributes = core_attributes[member_object]["core"]

    ds = tag_input_records[dataset]["dataset"]
    ds["longitude"] = ds["longitude"] % 360
    return name, names, ds, core_attributes, previous_time


# Functions for obtaining and recording attributes
def from_centers(
    attribute_group: AttributeGroup,
    input_records,
    object_tracks,
    dataset,
    member_object=None,
):
    """
    Calculate profile from object centers.

    Parameters
    ----------
    names : list of str
        Names of attributes to calculate.
    """

    # Note the attributes being recorded correspond to objects identified in the
    # previous timestep.
    args = [attribute_group, input_records, object_tracks, dataset, member_object]
    name, names, ds, core_attributes, previous_time = setup_interp(*args)
    tags = ds[names]
    lats_da = xr.DataArray(core_attributes["latitude"], dims="points")
    lons_da = xr.DataArray(core_attributes["longitude"], dims="points")
    previous_time = object_tracks["previous_times"][-1]

    # Convert object lons to 0-360
    lons_da = lons_da % 360
    kwargs = {"latitude": lats_da, "longitude": lons_da}
    kwargs.update({"time": previous_time.astype("datetime64[ns]")})
    kwargs.update({"method": "linear"})
    tags = tags.interp(**kwargs)

    tag_dict = {name: [] for name in names}
    for name in names:
        tag_dict[name] += list(tags[name].values)
    return tag_dict


kwargs = {"name": "cape", "data_type": float, "precision": 1, "units": "J/kg"}
description = "Convective available potential energy at the object center."
kwargs.update({"description": description})
cape = Attribute(**kwargs)

description = "Convective inhibition at the object center."
kwargs.update({"description": description, "name": "cin"})
cin = Attribute(**kwargs)

kwargs = {"name": "tags_center", "attributes": [cape, cin]}
description = "Tag attributes associated with object centers, e.g. cape and cin."
kwargs.update({"description": description})
keyword_arguments = {"center_type": "area_weighted"}
retrieval = Retrieval(function=from_centers, keyword_arguments=keyword_arguments)
kwargs.update({"retrieval": retrieval})
tag_center = AttributeGroup(**kwargs)


def default(dataset, matched=True):
    """Create the default tag attribute type."""

    attributes_list = core.retrieve_core(matched=matched)
    new_tag_center = tag_center.model_copy(deep=True)
    new_tag_center.retrieval.keyword_arguments.update({"dataset": dataset})
    attributes_list += [new_tag_center]
    description = "Tag attributes, e.g. cape and cin."
    kwargs = {"name": f"{dataset}_tag", "attributes": attributes_list}
    kwargs.update({"description": description})

    return AttributeType(**kwargs)
