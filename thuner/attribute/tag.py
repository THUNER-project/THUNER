import numpy as np
from thuner.log import setup_logger
import thuner.attribute.core as core
from thuner.attribute.utils import setup_interp, TimeOffset
import xarray as xr
from thuner.option.attribute import Retrieval, Attribute, AttributeGroup, AttributeType

logger = setup_logger(__name__)


# Functions for obtaining and recording attributes
def from_centers(
    attribute_group: AttributeGroup,
    input_records,
    object_tracks,
    dataset,
    time_offsets,
    member_object=None,
):
    """
    Calculate profile from object centers.

    Parameters
    ----------
    names : list of str
        Names of attributes to calculate.
    """

    args = [attribute_group, input_records, object_tracks, dataset, member_object]
    name, names, ds, core_attributes, current_time = setup_interp(*args)
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


class CAPE(Attribute):
    name: str = "cape"
    data_type: type = float
    precision: int = 1
    units: str = "J/kg"
    description: str = "Convective available potential energy."


class CIN(Attribute):
    name: str = "cin"
    data_type: type = float
    precision: int = 1
    units: str = "J/kg"
    description: str = "Convective inhibition."


class TagCenter(AttributeGroup):
    name: str = "tags_center"
    retrieval: Retrieval = Retrieval(
        function=from_centers,
        keyword_arguments={
            "center_type": "area_weighted",
            "time_offsets": [-120, -60, 0],
        },
    )
    attributes: list[Attribute] = [
        core.Time(retrieval=None),
        TimeOffset(),
        core.Latitude(retrieval=None),
        core.Longitude(retrieval=None),
        CAPE(),
        CIN(),
    ]
    description: str = "Tags at object centers, e.g. cape and cin."


def default(dataset, matched=True):
    """Create the default tag attribute type."""

    tag_center = TagCenter()
    # Add the appropriate ID attribute
    if matched:
        tag_center.attributes.insert(2, core.RecordUniversalID(retrieval=None))
    else:
        tag_center.attributes.insert(2, core.RecordID(retrieval=None))
    # Add the appropriate dataset attribute
    tag_center.retrieval.keyword_arguments.update({"dataset": dataset})
    description = "Tag attributes, e.g. cape and cin, at object center."
    kwargs = {"name": f"{dataset}_tag", "attributes": [tag_center]}
    kwargs.update({"description": description})
    return AttributeType(**kwargs)
