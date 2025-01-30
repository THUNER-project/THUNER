"""General utilities for object attributes."""

import yaml
from pathlib import Path
import pandas as pd
import numpy as np
from thuner.option.attribute import AttributeGroup, AttributeType
from thuner.log import setup_logger

logger = setup_logger(__name__)

# Mapping of string representations to actual data types
string_to_data_type = {
    "float": float,
    "int": int,
    "datetime64[s]": "datetime64[s]",
    "bool": bool,
    "str": str,
}


def get_previous_mask(object_tracks, matched=False):
    """Get the appropriate previous mask."""
    if matched:
        mask_type = "previous_matched_masks"
    else:
        mask_type = "previous_masks"
    mask = object_tracks[mask_type][-1]
    return mask


def attribute_from_core(attribute, object_tracks, member_object):
    """Get attribute from core object properties."""
    # Check if grouped object
    object_name = object_tracks["name"]
    if object_name in object_tracks["current_attributes"]:
        if member_object is not None and member_object is not object_name:
            member_attr = object_tracks["current_attributes"]["member_objects"]
            attr = member_attr[member_object]["core"][attribute.name]
        else:
            core_attr = object_tracks["current_attributes"][object_name]["core"]
            attr = core_attr[attribute.name]
    else:
        attr = object_tracks["current_attributes"]["core"][attribute.name]
    return {attribute.name: attr}


def attributes_dataframe(recorded_attributes, attribute_type):
    """Create a pandas DataFrame from object attributes dictionary."""

    data_types = get_data_type_dict(attribute_type)
    data_types.pop("time")
    df = pd.DataFrame(recorded_attributes).astype(data_types)
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
        attribute_type = AttributeType(**kwargs)
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
