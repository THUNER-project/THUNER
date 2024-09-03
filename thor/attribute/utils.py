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


def initialize_attributes(object_tracks, object_options):
    object_tracks["attribute"] = {}
    for key in object_options["attribute"].keys():
        object_tracks["attribute"][key] = {
            attr: [] for attr in object_options["attribute"][key].keys()
        }


def create_dataframe(attribute_type, object_tracks, object_options):
    """Create a pandas DataFrame from object attributes."""
    attributes = object_tracks["attribute"][attribute_type]
    options = object_options["attribute"][attribute_type]
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
