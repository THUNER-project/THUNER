"""General utilities for object attributes."""

import yaml
import pandas as pd
import dask.dataframe as dd
from collections import defaultdict
from thor.log import setup_logger

logger = setup_logger(__name__)

# Mapping of string representations to actual data types
string_to_data_type = {
    "float": float,
    "int": int,
    "datetime64[s]": "datetime64[s]",
    "bool": bool,
    "str": str,
}


def get_attribute_dict(name, method, data_type, precision, description, units):
    """Create a dictionary of attribute options."""
    attribute_dict = {}
    attribute_dict.update({"name": name, "method": method, "data_type": data_type})
    attribute_dict.update({"precision": precision, "description": description})
    attribute_dict.update({"units": units})
    return attribute_dict


def get_previous_mask(attribute_options, object_tracks):
    """Get the appropriate previous mask."""
    if "universal_id" in attribute_options.keys():
        mask_type = "previous_matched_masks"
    elif "id" in attribute_options.keys():
        mask_type = "previous_masks"
    else:
        message = "Either universal_id or id must be specified as an attribute."
        raise ValueError(message)
    mask = object_tracks[mask_type][-1]
    return mask


def dict_to_tuple(d):
    """Recursively convert a dictionary to a tuple of key-value pairs."""
    return tuple(
        (k, dict_to_tuple(v) if isinstance(v, dict) else v) for k, v in d.items()
    )


def tuple_to_dict(t):
    """Recursively convert a tuple of key-value pairs back to a dictionary."""
    return {k: tuple_to_dict(v) if isinstance(v, tuple) else v for k, v in t}


def group_by_method(attributes):
    """Group attributes dictionary by method key."""
    grouped = defaultdict(list)
    for key, value in attributes.items():
        method = value["method"]
        method_tuple = dict_to_tuple(method)
        grouped[method_tuple].append(key)
    return grouped


def attribute_from_core(name, object_tracks, member_object):
    """Get attribute from core object properties."""
    # Check if grouped object
    object_name = object_tracks["name"]
    if object_name in object_tracks["current_attributes"]:
        if member_object is not None and member_object is not object_name:
            member_attr = object_tracks["current_attributes"]["member_objects"]
            attr = member_attr[member_object]["core"][name]
        else:
            attr = object_tracks["current_attributes"][object_name]["core"][name]
    else:
        attr = object_tracks["current_attributes"]["core"][name]
    return attr


def initialize_attributes_detected(object_options):
    """Initialize attributes lists for detected objects."""
    attribute_types = object_options["attributes"].keys()
    attributes_dict = {t: {} for t in attribute_types}
    for attribute_type in attribute_types:
        if attribute_type == "tag" or attribute_type == "profile":
            datasets = object_options["attributes"][attribute_type].keys()
            attributes_dict[attribute_type] = {ds: {} for ds in datasets}
            for dataset in object_options["attributes"][attribute_type].keys():
                all_options = object_options["attributes"][attribute_type]
                attribute_options = all_options[dataset]
                attributes = {attr: [] for attr in attribute_options.keys()}
                attributes_dict[attribute_type][dataset] = attributes
        else:
            attribute_options = object_options["attributes"][attribute_type]
            attributes = {attr: [] for attr in attribute_options.keys()}
            attributes_dict[attribute_type] = attributes
    return attributes_dict


def initialize_attributes_grouped(object_options):
    """Initialize attributes lists for grouped objects."""
    # First initialize attributes for member objects
    member_options = object_options["attributes"]["member_objects"]
    object_name = object_options["name"]
    attributes_dict = {"member_objects": {}, object_name: {}}
    member_attributes = attributes_dict["member_objects"]
    for obj in member_options.keys():
        member_types = member_options[obj].keys()
        member_attributes[obj] = {t: {} for t in member_types}
        for attribute_type in member_options[obj].keys():
            if attribute_type == "tag" or attribute_type == "profile":
                datasets = member_options[obj][attribute_type].keys()
                member_attributes[obj][attribute_type] = {ds: {} for ds in datasets}
                for dataset in datasets:
                    attribute_options = member_options[obj][attribute_type][dataset]
                    attributes = {attr: [] for attr in attribute_options.keys()}
                    member_attributes[obj][attribute_type][dataset] = attributes
            else:
                attribute_options = member_options[obj][attribute_type]
                attributes = {attr: [] for attr in attribute_options.keys()}
                member_attributes[obj][attribute_type] = attributes
    # Now initialize attributes for grouped object
    obj = list(object_options["attributes"].keys() - {"member_objects"})[0]
    for attribute_type in object_options["attributes"][obj].keys():
        if attribute_type == "tag" or attribute_type == "profile":
            datasets = object_options["attributes"][obj][attribute_type].keys()
            attributes_dict[obj][attribute_type] = {ds: {} for ds in datasets}
            for dataset in datasets:
                obj_attribute_options = object_options["attributes"][obj]
                attribute_options = obj_attribute_options[attribute_type][dataset]
                attributes = {attr: [] for attr in attribute_options.keys()}
                attributes_dict[obj][attribute_type][dataset] = attributes
        else:
            attribute_options = object_options["attributes"][obj][attribute_type]
            attributes = {attr: [] for attr in attribute_options.keys()}
            attributes_dict[obj][attribute_type] = attributes
    return attributes_dict


def initialize_attributes(object_options):
    """Initialize attributes lists for object tracks."""
    if "detection" in object_options:
        init_func = initialize_attributes_detected
    elif "grouping" in object_options:
        init_func = initialize_attributes_grouped
    else:
        message = "Object indentification method must be specified, i.e. "
        message += "'detection' or 'grouping'."
        raise ValueError(message)
    return init_func(object_options)


def attributes_dataframe(attributes, options):
    """Create a pandas DataFrame from object attributes dictionary."""
    data_types = {name: options[name]["data_type"] for name in options.keys()}
    df = pd.DataFrame(attributes).astype(data_types)
    if "universal_id" in attributes.keys():
        id_index = "universal_id"
    else:
        id_index = "id"
    multi_index = ["time", id_index]
    if "altitude" in attributes.keys():
        multi_index.append("altitude")
    df.set_index(multi_index, inplace=True)
    df.sort_index(inplace=True)
    return df


def read_metadata_yml(filepath):
    """Read metadata from a yml file."""
    with open(filepath, "r") as file:
        attribute_options = yaml.safe_load(file)
        for key in attribute_options.keys():
            data_type = attribute_options[key]["data_type"]
            attribute_options[key]["data_type"] = string_to_data_type[data_type]
    return attribute_options


def get_indexes(attribute_options):
    """Get the indexes for the attribute DataFrame."""
    indexes = ["time"]
    if "universal_id" in attribute_options.keys():
        id_index = "universal_id"
    elif "id" in attribute_options.keys():
        id_index = "id"
    else:
        ValueError("No object id column found in attribute options.")
    indexes.append(id_index)
    if "altitude" in attribute_options.keys():
        indexes.append("altitude")
    return indexes


def read_attribute_csv(filepath, attribute_options=None, columns=None):
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

    if attribute_options is None:
        try:
            meta_path = filepath.with_suffix(".yml")
            attribute_options = read_metadata_yml(meta_path)
        except FileNotFoundError:
            logger.warning("No metadata file found for %s.", filepath)

    if attribute_options is None:
        message = "No metadata; loading entire dataframe and data types not enforced."
        logger.warning(message)
        return pd.read_csv(filepath)

    indexes = get_indexes(attribute_options)
    if columns is None:
        columns = list(attribute_options.keys())
    all_columns = indexes + [col for col in columns if col not in indexes]
    data_types = {name: attribute_options[name]["data_type"] for name in all_columns}
    # Remove time column as pd handles this separately
    data_types.pop("time", None)
    args_dict = {"usecols": all_columns, "dtype": data_types, "parse_dates": ["time"]}
    df = pd.read_csv(filepath, **args_dict)
    df = df.set_index(indexes)
    return df


def get_precision_dict(attribute_options):
    """Get precision dictionary for attribute options."""
    precision_dict = {}
    for key in attribute_options.keys():
        if attribute_options[key]["data_type"] == float:
            precision_dict[key] = attribute_options[key]["precision"]
    return precision_dict
