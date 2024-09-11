"""Methods for writing object masks."""

import yaml
import glob
import shutil
import copy
from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
from thor.utils import format_time
from thor.log import setup_logger
import thor.attribute.utils as utils

logger = setup_logger(__name__)
data_type_to_string = {v: k for k, v in utils.string_to_data_type.items()}


# Create custom yaml dumper to disable aliasing
class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True  # Create data type conversion dictionary


def write_setup(object_tracks, object_options, output_directory):
    """Setup to write for object attributes."""
    object_name = object_options["name"]
    base_filepath = output_directory / f"attributes/{object_name}/"

    last_write_time = object_tracks["last_write_time"]
    write_interval = np.timedelta64(object_options["write_interval"], "h")

    last_write_str = format_time(last_write_time, filename_safe=False, day_only=False)
    next_write_time = last_write_time + write_interval
    current_str = format_time(next_write_time, filename_safe=False, day_only=False)
    message = f"Writing {object_name} attributes from {last_write_str} to "
    message += f"{current_str}, inclusive and non-inclusive, respectively."
    logger.info(message)

    return base_filepath, last_write_str


def write_detected(object_tracks, object_options, output_directory):
    """
    Write attributes to file.
    """
    args = [object_tracks, object_options, output_directory]
    base_filepath, last_write_str = write_setup(*args)

    for attribute_type in object_options["attributes"].keys():
        filepath = base_filepath / f"{attribute_type}"
        filepath = filepath / f"{format_time(last_write_str)}.csv"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        attributes = object_tracks["attributes"][attribute_type]
        options = object_options["attributes"][attribute_type]
        df = utils.attributes_dataframe(attributes, options)
        df.to_csv(filepath)


def write_grouped(object_tracks, object_options, output_directory):
    """
    Write group attributes to file.
    """
    write_args = [object_tracks, object_options, output_directory]
    base_filepath, last_write_str = write_setup(*write_args)
    # First write member object core attributes
    member_options = object_options["attributes"]["member_objects"]
    member_attributes = object_tracks["attributes"]["member_objects"]
    for obj in member_options.keys():
        for attribute_type in member_attributes[obj].keys():
            filepath = base_filepath / f"{obj}/{attribute_type}/"
            filepath = filepath / f"{format_time(last_write_str)}.csv"
            filepath.parent.mkdir(parents=True, exist_ok=True)
            attributes = member_attributes[obj][attribute_type]
            options = member_options[obj][attribute_type]
            df = utils.attributes_dataframe(attributes, options)
            df.to_csv(filepath)
    # Now write grouped object core attributes
    obj_attr_options = object_options["attributes"][object_options["name"]]
    obj_attr = object_tracks["attributes"][object_options["name"]]
    for attribute_type in obj_attr.keys():
        filepath = base_filepath / f"{attribute_type}/{format_time(last_write_str)}.csv"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        attributes = obj_attr[attribute_type]
        options = obj_attr_options[attribute_type]
        df = utils.attributes_dataframe(attributes, options)
        df.to_csv(filepath)


def write(object_tracks, object_options, output_directory):
    """Write masks to file."""

    if "detection" in object_options:
        write_func = write_detected
    elif "grouping" in object_options:
        write_func = write_grouped
    else:
        message = "Object indentification method must be specified, i.e. "
        message += "'detection' or 'grouping'."
        raise ValueError(message)

    write_func(object_tracks, object_options, output_directory)
    # Reset attributes lists after writing
    object_tracks["attributes"] = utils.initialize_attributes(object_options)


def write_final(tracks, track_options, output_directory):
    """Write final masks to file."""

    for index, level_options in enumerate(track_options):
        for obj in level_options.keys():
            write(tracks[index][obj], level_options[obj], output_directory)


def write_metadata(filepath, attribute_options):
    """Write metadata to yml file."""
    formatted_options = copy.deepcopy(attribute_options)
    for key in formatted_options.keys():
        data_type = formatted_options[key]["data_type"]
        data_type_string = data_type_to_string[data_type]
        formatted_options[key]["data_type"] = data_type_string
    logger.debug("Saving attribute metadata to %s", filepath)
    with open(filepath, "w") as outfile:
        args = {"default_flow_style": False, "allow_unicode": True, "sort_keys": False}
        args.update({"Dumper": NoAliasDumper})
        yaml.dump(formatted_options, outfile, **args)


def aggregate_directory(directory, attribute_type, attribute_options, clean_up):
    """Aggregate attribute files within a directory into single file."""
    filepaths = glob.glob(str(directory / "*.csv"))
    df_list = []
    index_cols = ["time"]
    if "universal_id" in attribute_options.keys():
        index_cols += ["universal_id"]
    elif "id" in attribute_options.keys():
        index_cols += ["id"]

    for filepath in filepaths:
        df_list.append(pd.read_csv(filepath, index_col=index_cols))
    df = pd.concat(df_list)
    precision_dict = utils.get_precision_dict(attribute_options)
    df = df.round(precision_dict)
    df = df.sort_index()
    # Store aggregated file in parent directory
    df.to_csv(directory.parent / f"{attribute_type}.csv")
    if clean_up:
        shutil.rmtree(directory)


def aggregate_detected(base_path, object_options, clean_up):
    """Aggregate attributes of detected objects."""
    obj_name = object_options["name"]
    for attribute_type in object_options["attributes"].keys():
        directory = base_path / f"{obj_name}/{attribute_type}"
        options = object_options["attributes"][attribute_type]
        aggregate_directory(directory, attribute_type, options, clean_up)
        filepath = Path(directory.parent) / f"{attribute_type}_metadata.yml"
        write_metadata(filepath, options)


def aggregate_grouped(base_path, object_options, clean_up):
    """Aggregate group attributes."""
    member_options = object_options["attributes"]["member_objects"]
    obj_name = object_options["name"]
    # First aggregate core attributes of member objects
    for member_obj in member_options.keys():
        for attribute_type in member_options[member_obj].keys():
            directory = base_path / f"{obj_name}/{member_obj}/{attribute_type}"
            options = member_options[member_obj][attribute_type]
            aggregate_directory(directory, attribute_type, options, clean_up)
            filepath = Path(directory.parent) / f"{attribute_type}_metadata.yml"
            write_metadata(filepath, options)
    # Now aggregate core attributes of grouped object
    for attribute_type in object_options["attributes"][obj_name].keys():
        directory = base_path / f"{obj_name}/{attribute_type}"
        options = object_options["attributes"][obj_name][attribute_type]
        aggregate_directory(directory, attribute_type, options, clean_up)
        filepath = Path(directory.parent) / f"{attribute_type}_metadata.yml"
        write_metadata(filepath, options)


aggregate_dispatcher = {"core": aggregate_detected, "group": aggregate_grouped}


def aggregate(track_options, output_directory, clean_up=True):
    """Aggregate masks into single file."""

    logger.info("Aggregating attribute files.")
    base_path = Path(f"{output_directory}/attributes/")

    for level_options in track_options:
        for obj_name in level_options.keys():
            object_options = level_options[obj_name]

            if "detection" in object_options:
                aggregate_func = aggregate_detected
            elif "grouping" in object_options:
                aggregate_func = aggregate_grouped
            else:
                message = "Object indentification method must be specified, i.e. "
                message += "'detection' or 'grouping'."
                raise ValueError(message)

            aggregate_func(base_path, object_options, clean_up)
