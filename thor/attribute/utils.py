"""General utilities for object attributes."""

import yaml
import pandas as pd
import numpy as np
from thor.log import setup_logger

logger = setup_logger(__name__)

# Mapping of string representations to actual data types
string_to_data_type = {
    "float": float,
    "int": int,
    "datetime64[s]": "datetime64[s]",
}


def initialize_attributes_detected(object_options):
    """Initialize attributes lists for detected objects."""
    attributes_dict = {}
    for key in object_options["attributes"].keys():
        attribute_options = object_options["attributes"][key]
        attributes = {attr: [] for attr in attribute_options.keys()}
        attributes_dict[key] = attributes
    return attributes_dict


def initialize_attributes_grouped(object_options):
    """Initialize attributes lists for grouped objects."""
    # First initialize attributes for member objects
    member_options = object_options["attributes"]["member_objects"]
    object_name = object_options["name"]
    attributes_dict = {"member_objects": {}, object_name: {}}
    member_attributes = attributes_dict["member_objects"]
    for obj in member_options.keys():
        member_attributes[obj] = {}
        for attribute_type in member_options[obj].keys():
            attribute_options = member_options[obj][attribute_type]
            attributes = {attr: [] for attr in attribute_options.keys()}
            member_attributes[obj][attribute_type] = attributes
    # Now initialize attributes for grouped object
    obj = list(object_options["attributes"].keys() - {"member_objects"})[0]
    for attribute_type in object_options["attributes"][obj].keys():
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
    df.set_index(["time", id_index], inplace=True)
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


def read_attribute_csv(filepath, attribute_options=None):
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
            stem = filepath.stem
            meta_path = filepath.with_stem(f"{stem}_metadata").with_suffix(".yml")
            attribute_options = read_metadata_yml(meta_path)
        except FileNotFoundError:
            logger.warning("No metadata file found for %s.", filepath)
    if attribute_options is not None:
        keys = attribute_options.keys()
        keys = [key for key in keys if key != "time"]
        data_types = {name: attribute_options[name]["data_type"] for name in keys}
        df = pd.read_csv(filepath, dtype=data_types, parse_dates=["time"])
    else:
        logger.warning("No metadata; data types not enforced.")
        df = pd.read_csv(filepath)
    indexes = ["time"]
    if "universal_id" in df.columns:
        id_index = "universal_id"
    elif "id" in df.columns:
        id_index = "id"
    else:
        ValueError("No object id column found in CSV file.")
    indexes.append(id_index)
    if "altitude" in df.columns:
        indexes.append("altitude")
    df = df.set_index(indexes)

    return df


def get_precision_dict(attribute_options):
    """Get precision dictionary for attribute options."""
    precision_dict = {}
    for key in attribute_options.keys():
        if attribute_options[key]["data_type"] == float:
            precision_dict[key] = attribute_options[key]["precision"]
    return precision_dict
