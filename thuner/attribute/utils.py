"""General utilities for object attributes."""

from pydantic import ValidationError
import yaml
from pathlib import Path
import pandas as pd
import xarray as xr
from pydantic import BaseModel, model_validator
import numpy as np
from thuner.option.attribute import Attribute, AttributeGroup, AttributeType, Attributes
from thuner.log import setup_logger

logger = setup_logger(__name__)


def get_nearest_points(
    stacked_mask: xr.DataArray | xr.Dataset,
    id_number: int,
    ds: xr.Dataset | xr.DataArray,
):
    """
    Get the nearest points in a tagging dataset to a given object id.

    Parameters
    ----------
    stacked_mask : xarray.DataArray
        mask containing object ids with latitude and longitude stacked into a new
        dimension called 'points'.
    id_number : int
        Object ID number to get from stacked_mask.
    ds : xarray.Dataset | xarray.DataArray
        Tagging dataset containing latitude and longitude.

    Returns
    -------
    list[tuple]
        List of tuples containing the latitude and longitude of the nearest points.
    """
    points = stacked_mask.where(stacked_mask == id_number, drop=True).points.values
    lats, lons = zip(*points)
    lats_da = xr.DataArray(list(lats), dims="points")
    lons_da = xr.DataArray(list(lons), dims="points") % 360
    ds_grid = ds[["latitude", "longitude"]]
    ds_points = ds_grid.sel(latitude=lats_da, longitude=lons_da, method="nearest")
    ds_lats = ds_points.latitude.values.tolist()
    ds_lons = ds_points["longitude"].values.tolist()
    return list(set(zip(ds_lats, ds_lons)))


def _init_attr_type(attribute_type: AttributeType):
    """Initialize attributes lists for a given attribute type."""
    attributes = {}
    for attr in attribute_type.attributes:
        if isinstance(attr, AttributeGroup):
            for attr_attr in attr.attributes:
                attributes[attr_attr.name] = []
        elif isinstance(attr, Attribute):
            attributes[attr.name] = []
        else:
            raise ValueError(f"Unknown type {attr.type}.")
    return attributes


class AttributesRecord(BaseModel):
    """
    Class for storing attributes recorded during the tracking process
    """

    # Allow arbitrary types in the class.
    class Config:
        arbitrary_types_allowed = True

    attribute_options: Attributes
    name: str = None
    attribute_types: dict | None = None
    member_attributes: dict | None = None

    @model_validator(mode="after")
    def _check_name(cls, values):
        if values.name is None:
            values.name = values.attribute_options.name
        elif values.name != values.attribute_options.name:
            raise ValueError("Name must match attribute_options name.")
        return values

    @model_validator(mode="after")
    def _initialize_attributes(cls, values):
        options = values.attribute_options
        if options is None:
            return values
        values.attribute_types = {}
        for attr_type in options.attribute_types:
            values.attribute_types[attr_type.name] = _init_attr_type(attr_type)
        if options.member_attributes is not None:
            values.member_attributes = {}
            for obj, obj_attributes in options.member_attributes.items():
                obj_attr = {}
                for attr_type in obj_attributes.attribute_types:
                    obj_attr[attr_type.name] = _init_attr_type(attr_type)
                values.member_attributes[obj] = obj_attr
        return values


# Mapping of string representations to actual data types
string_to_data_type = {
    "float": float,
    "int": int,
    "datetime64[s]": "datetime64[s]",
    "bool": bool,
    "str": str,
}


class TimeOffset(Attribute):
    name: str = "time_offset"
    data_type: type = int
    units: str = "min"
    description: str = "Time offset in minutes from object detection time."


def setup_interp(
    attribute_group: AttributeGroup,
    input_records,
    object_tracks,
    dataset: str,
    member_object: str = None,
):
    name = object_tracks.name
    excluded = ["time", "id", "universal_id", "latitude", "longitude", "altitude"]
    excluded += ["time_offset"]
    attributes = attribute_group.attributes
    names = [attr.name for attr in attributes if attr.name not in excluded]
    tag_input_records = input_records.tag
    current_time = object_tracks.times[-1]

    # Get object centers
    if member_object is None:
        core_attributes = object_tracks.current_attributes.attribute_types["core"]
    else:
        core_attributes = object_tracks.current_attributes.member_attributes
        core_attributes = core_attributes[member_object]["core"]

    ds = tag_input_records[dataset].dataset
    ds["longitude"] = ds["longitude"] % 360
    return name, names, ds, core_attributes, current_time


def get_current_mask(object_tracks, matched=False):
    """Get the appropriate previous mask."""
    if matched:
        mask_type = "matched_masks"
    else:
        mask_type = "masks"
    mask = getattr(object_tracks, mask_type)[-1]
    return mask


def attribute_from_core(attribute, object_tracks, member_object):
    """Get attribute from core object properties."""
    # Check if grouped object
    current_attributes = object_tracks.current_attributes
    if member_object is not None and member_object is not object_tracks.name:
        member_attr = current_attributes.member_attributes
        attr = member_attr[member_object]["core"][attribute.name]
    else:
        core_attr = current_attributes.attribute_types["core"]
        attr = core_attr[attribute.name]
    return {attribute.name: attr}


def attributes_dataframe(recorded_attributes, attribute_type):
    """Create a pandas DataFrame from object attributes dictionary."""

    data_types = get_data_type_dict(attribute_type)
    data_types.pop("time")
    try:
        df = pd.DataFrame(recorded_attributes).astype(data_types)
    except:
        pass
    multi_index = ["time"]
    if "time_offset" in recorded_attributes.keys():
        multi_index.append("time_offset")
    if "universal_id" in recorded_attributes.keys():
        id_index = "universal_id"
    else:
        id_index = "id"
    multi_index.append(id_index)
    if "altitude" in recorded_attributes.keys():
        multi_index.append("altitude")
    df.set_index(multi_index, inplace=True)
    df.sort_index(inplace=True)
    return df


def read_metadata_yml(filepath):
    """Read metadata from a yml file."""
    with open(filepath, "r") as file:
        kwargs = yaml.safe_load(file)
        try:
            attribute_type = AttributeType(**kwargs)
        except ValidationError:
            logger.warning("Invalid metadata file found for %s.", filepath)
            attribute_type = None
    return attribute_type


def get_indexes(attribute_type: AttributeType):
    """Get the indexes for the attribute DataFrame."""
    all_indexes = ["time", "time_offset", "event_start", "universal_id", "id"]
    all_indexes += ["altitude"]
    indexes = []
    for attribute in attribute_type.attributes:
        if isinstance(attribute, AttributeGroup):
            for attr in attribute.attributes:
                if attr.name in all_indexes:
                    indexes.append(attr.name)
        else:
            if attribute.name in all_indexes:
                indexes.append(attribute.name)
    return indexes


def read_attribute_csv(filepath, attribute_type=None, columns=None, times=None):
    """
    Read a CSV file and return a DataFrame.

    Parameters
    ----------
    filepath : str
        Filepath to the CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the CSV data.
    """

    filepath = Path(filepath)

    data_types = None
    if attribute_type is None:
        try:
            meta_path = filepath.with_suffix(".yml")
            attribute_type = read_metadata_yml(meta_path)
            data_types = get_data_type_dict(attribute_type)
        except FileNotFoundError:
            logger.warning("No metadata file found for %s.", filepath)
        except ValidationError:
            logger.warning("Invalid metadata file found for %s.", filepath)
        except AttributeError:
            logger.warning("Invalid metadata file found for %s.", filepath)

    if attribute_type is None:
        message = "No metadata; loading entire dataframe and data types not enforced."
        logger.warning(message)
        return pd.read_csv(filepath, na_values=["", "NA"], keep_default_na=True)

    # Get attributes with np.datetime64 data type
    time_attrs = []
    for attribute in attribute_type.attributes:
        if isinstance(attribute, AttributeGroup):
            for attr in attribute.attributes:
                if attr.data_type is np.datetime64:
                    time_attrs.append(attr.name)
        else:
            if attribute.data_type is np.datetime64:
                time_attrs.append(attribute.name)

    indexes = get_indexes(attribute_type)
    if columns is None:
        columns = get_names(attribute_type)
    all_columns = indexes + [col for col in columns if col not in indexes]
    data_types = get_data_type_dict(attribute_type)
    # Remove time columns as pd handles these separately
    for name in time_attrs:
        data_types.pop(name, None)
    if times is not None:
        kwargs = {"usecols": ["time"], "parse_dates": time_attrs}
        kwargs.update({"na_values": ["", "NA"], "keep_default_na": True})
        index_df = pd.read_csv(filepath, **kwargs)
        row_numbers = index_df[~index_df["time"].isin(times)].index.tolist()
        # Increment row numbers by 1 to account for header
        row_numbers = [i + 1 for i in row_numbers]
    else:
        row_numbers = None

    kwargs = {"usecols": all_columns, "dtype": data_types, "parse_dates": time_attrs}
    kwargs.update({"skiprows": row_numbers})
    kwargs.update({"na_values": ["", "NA"], "keep_default_na": True})
    df = pd.read_csv(filepath, **kwargs)
    df = df.set_index(indexes)
    return df


def get_names(attribute_type: AttributeType):
    """Get the names of the attributes in the attribute type."""
    names = []
    for attribute in attribute_type.attributes:
        if isinstance(attribute, AttributeGroup):
            for attr in attribute.attributes:
                names.append(attr.name)
        else:
            names.append(attribute.name)
    return names


def get_precision_dict(attribute_type: AttributeType):
    """Get precision dictionary for attribute options."""
    precision_dict = {}
    for attribute in attribute_type.attributes:
        if isinstance(attribute, AttributeGroup):
            for attr in attribute.attributes:
                if attr.data_type == float:
                    precision_dict[attr.name] = attr.precision
        else:
            if attribute.data_type == float:
                precision_dict[attribute.name] = attribute.precision
    return precision_dict


def get_data_type_dict(attribute_type: AttributeType):
    """Get precision dictionary for attribute options."""
    data_type_dict = {}
    for attribute in attribute_type.attributes:
        if isinstance(attribute, AttributeGroup):
            # If the attribute is a group, get data type for each attribute in group
            for attr in attribute.attributes:
                data_type_dict[attr.name] = attr.data_type
        else:
            data_type_dict[attribute.name] = attribute.data_type
    return data_type_dict
