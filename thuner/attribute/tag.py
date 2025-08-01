"""
Describe and collect "tags" from tagging datasets.
"""

import numpy as np
from thuner.log import setup_logger
import thuner.attribute.core as core
import thuner.attribute.utils as utils
import xarray as xr
from thuner.option.attribute import Attribute, AttributeGroup, AttributeType
from thuner.utils import Retrieval

logger = setup_logger(__name__)

__all__ = [
    "from_centers",
    "from_masks",
    "default",
    "CAPE",
    "CIN",
    "tag_center",
]


# Functions for obtaining and recording attributes
def from_centers(
    attribute_group: AttributeGroup,
    input_records,
    object_tracks,
    dataset: str,
    time_offsets: list[int],
    member_object: str | None = None,
):
    """
    Get tags from object centers.
    """

    args = [attribute_group, input_records, object_tracks, dataset, member_object]
    name, names, ds, core_attributes, current_time = utils.setup_interp(*args)
    tags = ds[names]
    lats_da = xr.DataArray(core_attributes["latitude"], dims="points")
    lons_da = xr.DataArray(core_attributes["longitude"], dims="points")
    lons_da = lons_da % 360
    current_time = object_tracks.times[-1]

    # Convert object lons to 0-360
    if "id" in core_attributes.keys():
        id_name = "id"
    elif "universal_id" in core_attributes.keys():
        id_name = "universal_id"
    else:
        message = "No id or universal_id found in core attributes."
        raise ValueError(message)
    ids = core_attributes[id_name]

    tag_dict = {name: [] for name in names}
    coordinates = ["time", "time_offset", id_name, "latitude", "longitude"]
    tag_dict.update({name: [] for name in coordinates})
    # Setup interp kwargs
    kwargs = {"latitude": lats_da, "longitude": lons_da, "method": "linear"}
    for offset in time_offsets:
        interp_time = current_time + np.timedelta64(offset, "m")
        kwargs.update({"time": interp_time.astype("datetime64[ns]")})
        tags_time = tags.interp(**kwargs)
        for name in names:
            tag_dict[name] += list(tags_time[name].values)
        tag_dict["time_offset"] += [offset] * len(core_attributes["latitude"])
        tag_dict["latitude"] += core_attributes["latitude"]
        tag_dict["longitude"] += core_attributes["longitude"]
        tag_dict["time"] += [current_time] * len(core_attributes["latitude"])
        tag_dict[id_name] += ids
    return tag_dict


def from_masks(
    attribute_group: AttributeGroup,
    input_records,
    object_tracks,
    dataset: str,
    time_offsets: list[int],
    member_object: str | None = None,
):
    """
    Get tags from masks.
    """

    args = [attribute_group, input_records, object_tracks, dataset, member_object]
    name, names, ds, core_attributes, current_time = utils.setup_interp(*args)
    tags = ds[names]
    latitude, longitude = core_attributes["latitude"], core_attributes["longitude"]

    current_time = object_tracks.times[-1]

    # Convert object lons to 0-360
    if "id" in core_attributes.keys():
        matched = False
        id_name = "id"
    elif "universal_id" in core_attributes.keys():
        id_name = "universal_id"
        matched = True
    else:
        message = "No id or universal_id found in core attributes."
        raise ValueError(message)
    ids = core_attributes[id_name]
    mask = utils.get_current_mask(object_tracks, matched=matched)
    stacked_mask = mask.stack(points=["latitude", "longitude"])

    tag_dict = {name: [] for name in names}
    coordinates = ["time", "time_offset", id_name, "latitude", "longitude"]
    tag_dict.update({name: [] for name in coordinates})
    # Setup interp kwargs
    for offset in time_offsets:
        interp_time = current_time + np.timedelta64(offset, "m")
        tags_time = tags.interp(time=interp_time.astype("datetime64[ns]"))
        tags_time = tags_time.stack(points=["latitude", "longitude"])
        for i in range(len(ids)):
            points = utils.get_nearest_points(stacked_mask, ids[i], ds)
            tag = tags_time.sel(points=points).mean(dim="points")
            for name in names:
                tag_dict[name] += [tag[name].values.tolist()]
            tag_dict["time_offset"] += [offset]
            tag_dict["latitude"] += [core_attributes["latitude"][i]]
            tag_dict["longitude"] += [core_attributes["longitude"][i]]
            tag_dict["time"] += [current_time]
            tag_dict[id_name] += [ids[i]]

    return tag_dict


def CAPE():
    """Convenience function to build a CAPE attribute."""
    kwargs = {"name": "cape", "data_type": float, "precision": 1, "units": "J/kg"}
    kwargs.update({"description": "Convective available potential energy."})
    _retrieval = Retrieval(function=from_centers)
    kwargs.update({"retrieval": _retrieval})
    return Attribute(**kwargs)


# class CAPE(Attribute):
#     """Convective available potential energy (CAPE)."""

#     name: str = "cape"
#     data_type: type = float
#     precision: int = 1
#     units: str = "J/kg"
#     description: str = "Convective available potential energy."


def CIN():
    """Convenience function to build a CIN attribute."""
    kwargs = {"name": "cin", "data_type": float, "precision": 1, "units": "J/kg"}
    kwargs.update({"description": "Convective inhibition."})
    _retrieval = Retrieval(function=from_centers)
    kwargs.update({"retrieval": _retrieval})
    return Attribute(**kwargs)


# class CIN(Attribute):
#     """Convective inhibition."""

#     name: str = "cin"
#     data_type: type = float
#     precision: int = 1
#     units: str = "J/kg"
#     description: str = "Convective inhibition."


def tag_center(dataset):
    """Convenience function to build a TagCenter attribute group."""
    _ret_kwargs = {"center_type": "area_weighted", "time_offsets": [-120, -60, 0]}
    _ret_kwargs.update({"dataset": dataset})
    _retrieval = Retrieval(function=from_centers, keyword_arguments=_ret_kwargs)
    _time, _lat, _lon = core.time(), core.latitude(), core.longitude()
    _time.retrieval, _lat.retrieval, _lon.retrieval = None, None, None
    _attributes = [_time, utils.time_offset(), _lat, _lon, CAPE(), CIN()]
    _desc = "Tags, e.g. CAPE and CIN, at object centers"
    kwargs = {"name": "tags_center", "retrieval": _retrieval}
    kwargs.update({"attributes": _attributes, "description": _desc})
    return AttributeGroup(**kwargs)


# class TagCenter(AttributeGroup):
#     """Tags at the object center."""

#     name: str = "tags_center"
#     retrieval: Retrieval = Retrieval(
#         function=from_centers,
#         keyword_arguments={
#             "center_type": "area_weighted",
#             "time_offsets": [-120, -60, 0],
#         },
#     )
#     attributes: list[Attribute] = [
#         core.Time(retrieval=None),
#         utils.TimeOffset(),
#         core.Latitude(retrieval=None),
#         core.Longitude(retrieval=None),
#         CAPE(),
#         CIN(),
#     ]
#     description: str = "Tags at object centers, e.g. cape and cin."


def default(dataset: str, matched=True):
    """Create the default tag attribute type."""

    _tag_center = tag_center(dataset)
    # Add the appropriate ID attribute
    if matched:
        _record_id = core.record_universal_id()
        _record_id.retrieval = None
        _tag_center.attributes.insert(2, _record_id)
    else:
        _record_id = core.record_id()
        _record_id.retrieval = None
        _tag_center.attributes.insert(2, _record_id)
    # Add the appropriate dataset keyword argument pair
    description = "Tag attributes, e.g. cape and cin, at object center."
    kwargs = {"name": f"{dataset}_tag", "attributes": [_tag_center]}
    kwargs.update({"description": description})
    return AttributeType(**kwargs)
