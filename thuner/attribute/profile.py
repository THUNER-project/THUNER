"""
Record vertical profiles from tagging datasets associated with meteorological objects.
"""

import numpy as np
import xarray as xr
from thuner.log import setup_logger
import thuner.attribute.core as core
import thuner.attribute.utils as utils
from thuner.option.attribute import Attribute, AttributeGroup, AttributeType
from thuner.utils import Retrieval
from thuner.option.grid import GridOptions

logger = setup_logger(__name__)


# __all__ = [
#     "from_centers",
#     "from_masks",
#     "default",
#     "Altitude",
#     "U",
#     "V",
#     "Temperature",
#     "Pressure",
#     "RelativeHumidity",
#     "ProfileCenter",
# ]


__all__ = [
    "from_centers",
    "from_masks",
    "default",
    "altitude",
    "u_wind",
    "v_wind",
    "temperature",
    "pressure",
    "relative_humidity",
    "profile_center",
]


# Functions for obtaining and recording attributes
def from_centers(
    attribute_group: AttributeGroup,
    input_records,
    object_tracks,
    grid_options: GridOptions,
    dataset: str,
    time_offsets: list[int],
    member_object: str | None = None,
):
    """
    Calculate profile from object centers. Lookup core attributes, and extend to match
    length of profile attributes.
    """

    args = [attribute_group, input_records, object_tracks, dataset, member_object]
    name, names, ds, core_attributes, current_time = utils.setup_interp(*args)

    if "pressure" not in ds.coords or "geopotential" not in ds.data_vars:
        raise ValueError("Dataset must contain pressure levels and geopotential.")

    logger.debug(f"Interpolating from pressure levels to altitude using geopotential.")
    ds["longitude"] = ds["longitude"] % 360
    profiles = ds[names + ["geopotential"]]

    latitude, longitude = core_attributes["latitude"], core_attributes["longitude"]
    lats_da = xr.DataArray(core_attributes["latitude"], dims="points")
    lons_da = xr.DataArray(core_attributes["longitude"], dims="points")
    lons_da = lons_da % 360

    if "id" in core_attributes.keys():
        id_name = "id"
    elif "universal_id" in core_attributes.keys():
        id_name = "universal_id"
    else:
        message = "No id or universal_id found in core attributes."
        raise ValueError(message)
    ids = core_attributes[id_name]

    profile_dict = {name: [] for name in names}
    coordinates = ["time", "time_offset", id_name, "altitude", "latitude", "longitude"]
    profile_dict.update({name: [] for name in coordinates})
    # Setup interp kwargs
    kwargs = {"latitude": lats_da, "longitude": lons_da, "method": "linear"}
    for offset in time_offsets:
        # Interp to given time
        interp_time = current_time + np.timedelta64(offset, "m")
        kwargs.update({"time": interp_time.astype("datetime64[ns]")})
        profile_time = profiles.interp(**kwargs)

        profile_time["altitude"] = profile_time["geopotential"] / 9.80665
        new_altitudes = np.array(grid_options.altitude)

        for i in range(len(profile_time.points)):
            profile = profile_time.isel(points=i)
            profile = _interp_profile(profile, new_altitudes)
            args = [names, profile, profile_dict, offset, current_time, id_name]
            args += [ids[i], latitude[i], longitude[i]]
            profile_dict = _append_profile_dict(*args)

    return profile_dict


def from_masks(
    attribute_group: AttributeGroup,
    input_records,
    object_tracks,
    grid_options: GridOptions,
    dataset: str,
    time_offsets: list[int],
    member_object: str | None = None,
):
    """
    Calculate profile from object centers. Lookup core attributes, and extend to match
    length of profile attributes.

    Parameters
    ----------
    names : list of str
        Names of attributes to calculate.
    """

    args = [attribute_group, input_records, object_tracks, dataset, member_object]
    name, names, ds, core_attributes, current_time = utils.setup_interp(*args)

    latitude, longitude = core_attributes["latitude"], core_attributes["longitude"]

    if "pressure" not in ds.coords or "geopotential" not in ds.data_vars:
        raise ValueError("Dataset must contain pressure levels and geopotential.")

    logger.debug(f"Interpolating from pressure levels to altitude using geopotential.")
    ds["longitude"] = ds["longitude"] % 360
    profiles = ds[names + ["geopotential"]]

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

    profile_dict = {name: [] for name in names}
    coordinates = ["time", "time_offset", id_name, "altitude", "latitude", "longitude"]
    profile_dict.update({name: [] for name in coordinates})
    # Setup interp kwargs
    for offset in time_offsets:
        # Interp to given time
        interp_time = current_time + np.timedelta64(offset, "m")
        profile_time = profiles.interp(time=interp_time.astype("datetime64[ns]"))
        profile_time = profile_time.stack(points=["latitude", "longitude"])

        profile_time["altitude"] = profile_time["geopotential"] / 9.80665
        new_altitudes = np.array(grid_options.altitude)

        for i in range(len(ids)):
            points = utils.get_nearest_points(stacked_mask, ids[i], ds)
            profile_list = []
            for j in range(len(points)):
                profile = profile_time.sel(points=points[j])
                profile = _interp_profile(profile, new_altitudes).drop("points")
                profile_list.append(profile)
            all_profiles = xr.concat(profile_list, dim="profiles")
            profile = all_profiles.mean(dim="profiles")
            args = [names, profile, profile_dict, offset, current_time, id_name]
            args += [ids[i], latitude[i], longitude[i]]
            profile_dict = _append_profile_dict(*args)
    return profile_dict


def _append_profile_dict(
    names, profile, profile_dict, offset, current_time, id_name, id_number, lat, lon
):
    """Append profile values to profile dictionary."""
    for name in names:
        profile_dict[name] += list(profile[name].values)
    profile_dict["altitude"] += list(profile["altitude"].values)
    profile_dict["time_offset"] += [offset] * len(profile["altitude"])
    profile_dict["latitude"] += [lat] * len(profile["altitude"])
    profile_dict["longitude"] += [lon] * len(profile["altitude"])
    profile_dict["time"] += [current_time] * len(profile["altitude"])
    profile_dict[id_name] += [id_number] * len(profile["altitude"])
    return profile_dict


def _interp_profile(profile: xr.DataArray | xr.Dataset, new_altitudes: np.ndarray):
    """Interpolate a profile to new altitudes."""
    profile = profile.swap_dims({"pressure": "altitude"})
    profile = profile.drop_vars(["geopotential"])
    profile = profile.interp(altitude=new_altitudes)
    profile = profile.reset_coords("pressure")
    return profile


def altitude():
    """Convenience function to build an altitude attribute."""
    kwargs = {"name": "altitude", "data_type": float, "precision": 1}
    kwargs.update({"units": "m", "description": "Altitude coordinate of profile."})
    return Attribute(**kwargs)


# class Altitude(Attribute):
#     """Altitude coordinate of profile."""

#     name: str = "altitude"
#     data_type: type = float
#     precision: int = 1
#     units: str = "m"
#     description: str = "Altitude coordinate of profile."


def u_wind():
    """Convenience function to build a u wind component attribute."""
    kwargs = {"name": "u", "data_type": float, "precision": 1}
    kwargs.update({"units": "m/s", "description": "u component of wind."})
    return Attribute(**kwargs)


# class U(Attribute):
#     """Zonal winds."""

#     name: str = "u"
#     data_type: type = float
#     precision: int = 1
#     units: str = "m/s"
#     description: str = "u component of wind."


def v_wind():
    """Convenience function to build a v wind component attribute."""
    kwargs = {"name": "v", "data_type": float, "precision": 1}
    kwargs.update({"units": "m/s", "description": "v component of wind."})
    return Attribute(**kwargs)


# class V(Attribute):
#     """Meridional winds."""

#     name: str = "v"
#     data_type: type = float
#     precision: int = 1
#     units: str = "m/s"
#     description: str = "v component of wind."


def temperature():
    """Convenience function to build a temperature attribute."""
    kwargs = {"name": "temperature", "data_type": float, "precision": 2}
    kwargs.update({"units": "K", "description": "Temperature profile."})
    return Attribute(**kwargs)


# class Temperature(Attribute):
#     """Temperature in Kelvin."""

#     name: str = "temperature"
#     data_type: type = float
#     precision: int = 2
#     units: str = "K"
#     description: str = "Temperature profile."


def pressure():
    """Convenience function to build a pressure attribute."""
    kwargs = {"name": "pressure", "data_type": float, "precision": 1}
    kwargs.update({"units": "hPa", "description": "Pressure profile."})
    return Attribute(**kwargs)


# class Pressure(Attribute):
#     """Pressure in hPa."""

#     name: str = "pressure"
#     data_type: type = float
#     precision: int = 1
#     units: str = "hPa"
#     description: str = "Pressure profile."


def relative_humidity():
    """Convenience function to build a relative humidity attribute."""
    kwargs = {"name": "relative_humidity", "data_type": float, "precision": 1}
    kwargs.update({"units": "%", "description": "Relative humidity profile."})
    return Attribute(**kwargs)


# class RelativeHumidity(Attribute):
#     """Relative humidity as percentage."""

#     name: str = "relative_humidity"
#     data_type: type = float
#     precision: int = 1
#     units: str = "%"
#     description: str = "Relative humidity profile."


def profile_center(dataset: str):
    """Convenience function to build a profile center attribute group."""

    _time, _lat, _lon = core.time(), core.latitude(), core.longitude()
    _time.retrieval, _lat.retrieval, _lon.retrieval = None, None, None

    _attributes = [_time, utils.time_offset(), _lat, _lon, altitude()]
    _attributes += [u_wind(), v_wind(), temperature(), pressure(), relative_humidity()]
    _ret_kwargs = {"center_type": "area_weighted", "time_offsets": [-120, -60, 0]}
    _ret_kwargs.update({"dataset": dataset})
    _retrieval = Retrieval(function=from_centers, keyword_arguments=_ret_kwargs)
    _desc = "Environmental profiles at object center."
    kwargs = {"name": "profiles", "attributes": _attributes, "retrieval": _retrieval}
    kwargs.update({"description": _desc})
    return AttributeGroup(**kwargs)


# Create a convenience attribute group, as they are typically all retrieved at once
# class ProfileCenter(AttributeGroup):
#     """Attribute group describing profiles obtained at the object center."""

#     name: str = "profiles"
#     attributes: list[Attribute] = [
#         core.Time(retrieval=None),
#         utils.TimeOffset(),
#         core.Latitude(retrieval=None),
#         core.Longitude(retrieval=None),
#         Altitude(),
#         U(),
#         V(),
#         Temperature(),
#         Pressure(),
#         RelativeHumidity(),
#     ]
#     retrieval: Retrieval = Retrieval(
#         function=from_centers,
#         keyword_arguments={
#             "center_type": "area_weighted",
#             "time_offsets": [-120, -60, 0],
#         },
#     )
#     description: str = "Environmental profiles at object center."


def default(dataset, matched=True):
    """Create the default profile attribute type."""
    _profile_center = profile_center(dataset)
    # Add the appropriate ID attribute
    if matched:
        _record_id = core.record_universal_id()
        _record_id.retrieval = None
        _profile_center.attributes.insert(2, _record_id)
    else:
        _record_id = core.record_id()
        _record_id.retrieval = None
        _profile_center.attributes.insert(2, _record_id)
    # Add the appropriate dataset keyword argument pair
    description = "Attributes corresponding to profiles taken from a tagging dataset, "
    description += "e.g. ambient winds, temperature and humidity."
    kwargs = {"name": f"{dataset}_profile", "attributes": [_profile_center]}
    kwargs.update({"description": description, "dataset": dataset})
    return AttributeType(**kwargs)
