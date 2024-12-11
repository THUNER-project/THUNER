"""
Functions for defining attribute options associated with vertical profile attributes.
"""

import numpy as np
from itertools import chain
from thuner.log import setup_logger
import thuner.attribute.core as core
import thuner.attribute.utils as utils
import xarray as xr

logger = setup_logger(__name__)


# Convenience functions for defining default attribute options
def altitude():
    """
    Altitude attribute
    """
    name = "altitude"
    method = None
    data_type = float
    precision = 1
    units = "m"
    description = "Altitude coordinate of profile."
    args = [name, method, data_type, precision, description, units]
    return utils.get_attribute_dict(*args)


def hour_offset():
    """
    Hour offset attribute
    """
    name = "hour_offset"
    method = None
    data_type = float
    precision = 1
    units = "hours"
    description = "Hour offset from the object time."
    args = [name, method, data_type, precision, description, units]
    return utils.get_attribute_dict(*args)


def wind(dataset, name, method=None, description=None):
    """
    Specify options for a core property, typically obtained from the matching process.
    """
    data_type = float
    precision = 1
    units = "m/s"
    if method is None:
        method = {"function": "from_centers", "dataset": dataset}
        method["args"] = {"center_type": "area_weighted"}
    if description is None:
        description = f"Vertical {name} profile, typically at the object center."
    args = [name, method, data_type, precision, description, units]
    return utils.get_attribute_dict(*args)


def temperature(dataset, name, method=None, description=None):
    """
    Specify options for a core property, typically obtained from the matching process.
    """
    data_type = float
    precision = 2
    units = "K"
    if method is None:
        method = {"function": "from_centers", "dataset": dataset}
        method["args"] = {"center_type": "area_weighted"}
    if description is None:
        description = f"Vertical {name} profile, typically at the object center."
    args = [name, method, data_type, precision, description, units]
    return utils.get_attribute_dict(*args)


def pressure(dataset, name, method=None, description=None):
    """
    Specify options for a core property, typically obtained from the matching process.
    """
    data_type = float
    precision = 1
    units = "hPa"
    if method is None:
        method = {"function": "from_centers", "dataset": dataset}
        method["args"] = {"center_type": "area_weighted"}
    if description is None:
        description = f"Vertical {name} profile, typically at the object center."
    args = [name, method, data_type, precision, description, units]
    return utils.get_attribute_dict(*args)


def relative_humidity(dataset, name="relative_humidity", method=None, description=None):
    """
    Specify options for a core property, typically obtained from the matching process.
    """
    data_type = float
    precision = 1
    units = "%"
    if method is None:
        method = {"function": "from_centers", "dataset": dataset}
        method["args"] = {"center_type": "area_weighted"}
    if description is None:
        description = f"Vertical {name} profile, typically at the object center."
    args = [name, method, data_type, precision, description, units]
    return utils.get_attribute_dict(*args)


# Modify below approach to allow for multiple tagging/profile datasets.
# Simply create another function to call the one below.
def dataset_default(dataset, names=None, matched=True):
    """Create a dictionary of default attribute options for a specified dataset."""

    if names is None:
        names = ["time", "latitude", "longitude", "altitude"]
        names += ["temperature", "relative_humidity", "u", "v"]
    if matched:
        id_type = "universal_id"
    else:
        id_type = "id"
    core_method = {"function": "attribute_from_core"}
    attributes = {}
    # Reuse core attributes, just replace the default functions method
    attributes["time"] = core.time(method=core_method)
    attributes["latitude"] = core.coordinate("latitude", method=core_method)
    attributes["longitude"] = core.coordinate("longitude", method=core_method)
    attributes["altitude"] = altitude()
    attributes["pressure"] = pressure(dataset, "pressure")
    # attributes["hour_offset"] = hour_offset()
    attributes[id_type] = core.identity(id_type, method=core_method)
    if "relative_humidity" in names:
        attributes["relative_humidity"] = relative_humidity(dataset)
    if "temperature" in names:
        attributes["temperature"] = temperature(dataset, "temperature")
    if "u" in names and "v" in names:
        attributes["u"] = wind(dataset, "u")
        attributes["v"] = wind(dataset, "v")
    return attributes


def default(datasets, names=None, matched=True):
    """Create a dictionary of default attribute options across all datasets."""
    attributes = {ds: dataset_default(ds, names, matched) for ds in datasets}
    return attributes


def from_pressure_levels(names, previous_time, lats, lons, ds, grid_options):
    """Get vertical profiles from data on pressure levels."""

    if "pressure" not in ds.coords or "geopotential" not in ds.data_vars:
        raise ValueError("Dataset must contain pressure levels or geopotential.")

    logger.debug(f"Interpolating from pressure levels to altitude using geopotential.")
    # Convert tag lons to 0-360
    ds["longitude"] = ds["longitude"] % 360
    profiles = ds[names + ["geopotential"]]

    lats_da = xr.DataArray(lats, dims="points")
    lons_da = xr.DataArray(lons, dims="points")

    # Convert object lons to 0-360
    lons_da = lons_da % 360
    kwargs = {"latitude": lats_da, "longitude": lons_da}
    kwargs.update({"time": previous_time.astype("datetime64[ns]")})
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


interpolate_dispatcher = {
    "era5_pl": from_pressure_levels,
}


# Functions for obtaining and recording attributes
def from_centers(names, input_records, attributes, object_tracks, method, grid_options):
    """
    Calculate profile from object centers.

    Parameters
    ----------
    names : list of str
        Names of attributes to calculate.
    """

    # Note the attributes being recorded correspond to objects identified in the
    # previous timestep.
    tag_input_records = input_records["tag"]
    previous_time = object_tracks["previous_times"][-1]
    lats = attributes["latitude"]
    lons = attributes["longitude"]
    ds = tag_input_records[method["dataset"]]["dataset"]
    interp = interpolate_dispatcher.get(method["dataset"])
    profiles = interp(names, previous_time, lats, lons, ds, grid_options)
    return profiles


get_attributes_dispatcher = {"attribute_from_core": utils.attribute_from_core}
get_profiles_dispatcher = {"from_centers": from_centers}


def record_profiles(
    names, input_records, attributes, object_tracks, method, grid_options
):
    """Get profiles from tag datasets."""
    method = utils.tuple_to_dict(method)
    get_prof = get_profiles_dispatcher.get(method["function"])
    if get_prof is None:
        message = f"Function {method['function']} for obtaining profiles "
        message += "not recognised."
        raise ValueError(message)
    from_centers_args = [names, input_records, attributes, object_tracks]
    from_centers_args += [method, grid_options]
    args_dispatcher = {"from_centers": from_centers_args}
    args = args_dispatcher[method["function"]]
    profiles = get_prof(*args)
    attributes.update(profiles)


def record(
    input_records,
    attributes,
    object_tracks,
    attribute_options,
    grid_options,
    member_object=None,
):
    for dataset in attribute_options.keys():
        dataset_record(
            input_records,
            attributes[dataset],
            object_tracks,
            attribute_options[dataset],
            grid_options,
            member_object,
        )


def dataset_record(
    input_records,
    attributes,
    object_tracks,
    attribute_options,
    grid_options,
    member_object=None,
):
    """Get group object attributes."""
    # Get core attributes
    previous_time = object_tracks["previous_times"][-1]
    if previous_time is None:
        return
    core_attributes = ["time", "id", "universal_id", "latitude", "longitude"]
    keys = attributes.keys()
    core_attributes = [attr for attr in core_attributes if attr in keys]
    remaining_attributes = [
        attr
        for attr in keys
        if attr not in core_attributes and attr not in ["altitude", "hour_offset"]
    ]
    altitude = grid_options.altitude
    # Get the appropriate core attributes
    for name in core_attributes:
        attr_function = attribute_options[name]["method"]["function"]
        get_attr = get_attributes_dispatcher.get(attr_function)
        if get_attr is None:
            message = f"Function {attr_function} for obtaining attribute {name} not recognised."
            raise ValueError(message)
        attr = get_attr(name, object_tracks, member_object)
        attributes[name] += list(attr)

    if attributes["time"] is None or len(attributes["time"]) == 0:
        return

    number_objects = len(attributes["time"])
    altitude_attribute = list(altitude) * number_objects
    attributes["altitude"] += list(altitude_attribute)

    # Get profiles efficiently by processing attributes with same method together
    profile_attributes = {key: attribute_options[key] for key in remaining_attributes}
    grouped_by_method = utils.group_by_method(profile_attributes)
    for method in grouped_by_method.keys():
        names = grouped_by_method[method]
        args = [names, input_records, attributes, object_tracks, method]
        args += [grid_options]
        record_profiles(*args)

    # Repeat the core attributes to account for the repeating altitude attribute
    for name in core_attributes:
        attr = attributes[name]
        rep_attr = chain.from_iterable([[element] * len(altitude) for element in attr])
        attributes[name] = list(rep_attr)
