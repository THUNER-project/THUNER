from thor.log import setup_logger
import thor.attribute.core as core
import thor.attribute.utils as utils
import xarray as xr

logger = setup_logger(__name__)


def cape(dataset, name, method=None, description=None):
    """
    Specify options for a specific potential energy type attribute.
    """
    data_type = float
    precision = 1
    units = "J/kg"
    if method is None:
        method = {"function": "from_centers", "dataset": dataset}
        method["args"] = {"center_type": "area_weighted"}
    if description is None:
        description = f"{name} at the object center."
    args = [name, method, data_type, precision, description, units]
    return utils.get_attribute_dict(*args)


def default(dataset, names=None, matched=True):
    """Create a dictionary of default attribute options of grouped objects."""

    if names is None:
        names = ["time", "latitude", "longitude", "cape", "cin"]
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
    attributes[id_type] = core.identity(id_type, method=core_method)
    if "cape" in names:
        attributes["cape"] = cape(dataset, "cape")
    if "cin" in names:
        attributes["cin"] = cape(dataset, "cin")

    return attributes


# Methods for obtaining and recording attributes
def from_centers(names, input_records, attributes, object_tracks, method):
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

    tags = ds[names]
    lats_da = xr.DataArray(lats, dims="points")
    lons_da = xr.DataArray(lons, dims="points")
    args_dict = {"latitude": lats_da, "longitude": lons_da}
    args_dict.update({"time": previous_time.astype("datetime64[ns]")})
    args_dict.update({"method": "linear"})
    tags = tags.interp(**args_dict)

    tag_dict = {name: [] for name in names}
    for name in names:
        tag_dict[name] += list(tags[name].values)
    return tag_dict


get_attributes_dispatcher = {"attribute_from_core": utils.attribute_from_core}
get_profiles_dispatcher = {"from_centers": from_centers}


def record_tags(names, input_records, attributes, object_tracks, method):
    """Record tags."""
    method = utils.tuple_to_dict(method)
    get_tag = get_profiles_dispatcher.get(method["function"])
    if get_tag is None:
        message = f"Function {method['function']} for obtaining tags "
        message += "not recognised."
        raise ValueError(message)
    from_centers_args = [names, input_records, attributes, object_tracks, method]
    args_dispatcher = {"from_centers": from_centers_args}
    args = args_dispatcher[method["function"]]
    tags = get_tag(*args)
    attributes.update(tags)


def record(
    time,
    input_records,
    attributes,
    object_tracks,
    attribute_options,
    grid_options,
    member_object=None,
):
    """Get group object attributes."""
    # Get core attributes
    core_attributes = ["time", "id", "universal_id", "latitude", "longitude"]
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
        attr = get_attr(name, object_tracks, member_object)
        attributes[name] += list(attr)

    if attributes["time"] is None or len(attributes["time"]) == 0:
        return

    # Get tags
    tag_attributes = {key: attribute_options[key] for key in remaining_attributes}
    grouped_by_method = utils.group_by_method(tag_attributes)
    for method in grouped_by_method.keys():
        names = grouped_by_method[method]
        record_tags(names, input_records, attributes, object_tracks, method)
