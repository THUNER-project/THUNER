"""
Functions for defining attribute options associated with vertical profile attributes.
"""

import numpy as np
import xarray as xr
from thuner.log import setup_logger
import thuner.attribute.core as core
from thuner.attribute.tag import setup_interp
from thuner.option.attribute import Retrieval, Attribute, AttributeGroup, AttributeType


logger = setup_logger(__name__)


# Functions for obtaining and recording attributes
def from_centers(
    attribute_group: AttributeGroup,
    input_records,
    object_tracks,
    grid_options,
    dataset,
    time_offsets,
    member_object=None,
):
    """
    Calculate profile from object centers. Lookup core attributes, and extend to match
    length of profile attributes.

    Parameters
    ----------
    names : list of str
        Names of attributes to calculate.
    """

    # Note the attributes being recorded correspond to objects identified in the
    # previous timestep.
    args = [attribute_group, input_records, object_tracks, dataset, member_object]
    name, names, ds, core_attributes, previous_time = setup_interp(*args)

    if "pressure" not in ds.coords or "geopotential" not in ds.data_vars:
        raise ValueError("Dataset must contain pressure levels and geopotential.")

    logger.debug(f"Interpolating from pressure levels to altitude using geopotential.")
    # Convert tag lons to 0-360
    ds["longitude"] = ds["longitude"] % 360
    profiles = ds[names + ["geopotential"]]

    latitude, longitude = core_attributes["latitude"], core_attributes["longitude"]
    time = core_attributes["time"]
    lats_da = xr.DataArray(core_attributes["latitude"], dims="points")
    lons_da = xr.DataArray(core_attributes["longitude"], dims="points")

    # Convert object lons to 0-360
    lons_da = lons_da % 360
    for offset in time_offsets:
        interp_time = previous_time + np.timedelta64(offset, "m")
        kwargs = {"latitude": lats_da, "longitude": lons_da}
        kwargs.update({"time": interp_time.astype("datetime64[ns]")})
        kwargs.update({"method": "linear"})
        profiles = profiles.interp(**kwargs)

        profiles["altitude"] = profiles["geopotential"] / 9.80665
        new_altitudes = np.array(grid_options.altitude)
        profile_dict = {name: [] for name in names}

        for i in range(len(profiles.points)):
            profile = profiles.isel(points=i)
            profile = profile.swap_dims({"pressure": "altitude"})
            profile = profile.drop_vars(["geopotential"])
            profile = profile.interp(altitude=new_altitudes)
            profile = profile.reset_coords("pressure")
            for name in names:
                profile_dict[name] += list(profile[name].values)

    return profile_dict


kwargs = {"name": "altitude", "data_type": float, "precision": 1, "units": "m"}
kwargs.update({"description": "Altitude coordinate of profile."})
altitude = Attribute(**kwargs)

kwargs = {"name": "time_offset", "data_type": int, "units": "min"}
description = "Time offset in minutes of tagging dataset from object detection time."
kwargs.update({"description": description})
time_offset = Attribute(**kwargs)

kwargs = {"name": "u", "data_type": float, "precision": 1, "units": "m/s"}
description = " profile taken at the object center."
kwargs.update({"description": "u " + description})
u = Attribute(**kwargs)
kwargs.update({"name": "v", "description": "v " + description})
v = Attribute(**kwargs)

kwargs = {"name": "temperature", "data_type": float, "precision": 2, "units": "K"}
kwargs.update({"description": "temperature" + description})
temperature = Attribute(**kwargs)

kwargs = {"name": "pressure", "data_type": float, "precision": 1, "units": "hPa"}
kwargs.update({"description": "pressure" + description})
pressure = Attribute(**kwargs)

kwargs = {"name": "relative_humidity", "data_type": float, "precision": 1, "units": "%"}
kwargs.update({"description": "relative humidity" + description})
relative_humidity = Attribute(**kwargs)

# Create a convenience attribute group, as they are typically all retrieved at once
keyword_arguments = {"center_type": "area_weighted", "time_offsets": [-120, -60, 0]}
retrieval = Retrieval(function=from_centers, keyword_arguments=keyword_arguments)

attribute_list = [
    core.time.model_copy(deep=True),
    core.universal_ids_record.model_copy(deep=True),
    core.latitude.model_copy(deep=True),
    core.longitude.model_copy(deep=True),
]
for attr in attribute_list:
    attr.retrieval = None
attribute_list += [time_offset, altitude, u, v, temperature]
attribute_list += [pressure, relative_humidity]
kwargs = {"name": "profiles", "attributes": attribute_list, "retrieval": retrieval}
kwargs.update({"description": "Environmental profiles at object center."})
profile_center_matched = AttributeGroup(**kwargs)
attribute_list[1] = core.ids_record.model_copy(deep=True)
attribute_list[1].retrieval = None
kwargs.update({"attributes": attribute_list})
profile_center_unmatched = AttributeGroup(**kwargs)


def default(dataset, matched=True):
    """Create the default profile attribute type."""
    if matched:
        profile_center = profile_center_matched.model_copy(deep=True)
    else:
        profile_center = profile_center_unmatched.model_copy(deep=True)
    profile_center.retrieval.keyword_arguments.update({"dataset": dataset})
    description = "Attributes corresponding to profiles taken from a tagging dataset, "
    description += "e.g. ambient winds, temperature and humidity."
    kwargs = {"name": f"{dataset}_profile", "attributes": [profile_center]}
    kwargs.update({"description": description, "dataset": dataset})

    return AttributeType(**kwargs)
