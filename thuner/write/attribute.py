"""Functions for writing object masks."""

import yaml
import glob
import shutil
import copy
from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import multiprocessing
from thuner.utils import format_time
from thuner.log import setup_logger
import thuner.attribute.utils as utils

logger = setup_logger(__name__)
data_type_to_string = {v: k for k, v in utils.string_to_data_type.items()}


# Create custom yaml dumper to disable aliasing
class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True  # Create data type conversion dictionary


def write_setup(object_tracks, object_options, output_directory):
    """Setup to write for object attributes."""
    object_name = object_options.name
    base_directory = output_directory / f"attributes/{object_name}/"

    last_write_time = object_tracks["last_write_time"]
    write_interval = np.timedelta64(object_options.write_interval, "h")

    last_write_str = format_time(last_write_time, filename_safe=False, day_only=False)
    next_write_time = last_write_time + write_interval
    current_str = format_time(next_write_time, filename_safe=False, day_only=False)
    message = f"Writing {object_name} attributes from {last_write_str} to "
    message += f"{current_str}, inclusive and non-inclusive, respectively."
    logger.info(message)

    return base_directory, last_write_str


def write_attributes(directory, last_write_str, attributes, attribute_options):
    """Write attributes to file."""
    directory.mkdir(parents=True, exist_ok=True)
    filepath = directory / f"{format_time(last_write_str)}.csv"
    df = utils.attributes_dataframe(attributes, attribute_options)
    precicion_dict = utils.get_precision_dict(attribute_options)
    df = df.round(precicion_dict)
    df.to_csv(filepath, na_rep="NA")


def write_attribute_type(
    base_directory, last_write_str, attribute_type, attributes, attribute_options
):
    """Write attributes to file."""
    if attribute_type == "tag" or attribute_type == "profile":
        for dataset in attributes.keys():
            # For tag and profile attributes, additional layer of nesting by dataset
            directory = base_directory / f"{dataset}/{attribute_type}"
            attr = attributes[dataset]
            attr_options = attribute_options[dataset]
            write_attributes(directory, last_write_str, attr, attr_options)
    else:
        directory = base_directory / f"{attribute_type}"
        write_attributes(directory, last_write_str, attributes, attribute_options)


def write_detected(object_tracks, object_options, output_directory):
    """
    Write detected object attributes to file.
    """
    args = [object_tracks, object_options, output_directory]
    base_directory, last_write_str = write_setup(*args)

    for attribute_type in object_options.attributes.keys():
        attributes = object_tracks["attributes"][attribute_type]
        options = object_options.attributes[attribute_type]
        write_attribute_type(
            base_directory, last_write_str, attribute_type, attributes, options
        )


def write_grouped(object_tracks, object_options, output_directory):
    """
    Write grouped object attributes to file.
    """
    write_args = [object_tracks, object_options, output_directory]
    base_directory, last_write_str = write_setup(*write_args)
    # Write member object attributes
    member_options = object_options.attributes["member_objects"]
    member_attributes = object_tracks["attributes"]["member_objects"]
    for obj in member_options.keys():
        for attribute_type in member_attributes[obj].keys():
            directory = base_directory / f"{obj}/"
            attributes = member_attributes[obj][attribute_type]
            options = member_options[obj][attribute_type]
            args = [directory, last_write_str, attribute_type, attributes, options]
            write_attribute_type(*args)
    # Write grouped object attributes
    obj_attr_options = object_options.attributes[object_options.name]
    obj_attr = object_tracks["attributes"][object_options.name]
    for attribute_type in obj_attr.keys():
        attributes = obj_attr[attribute_type]
        options = obj_attr_options[attribute_type]
        args = [base_directory, last_write_str, attribute_type, attributes, options]
        write_attribute_type(*args)


def write(object_tracks, object_options, output_directory):
    """Write attributes to file."""

    if "detection" in object_options.model_fields:
        write_func = write_detected
    elif "grouping" in object_options.model_fields:
        write_func = write_grouped
    else:
        message = "Object indentification method must be specified, i.e. "
        message += "'detection' or 'grouping'."
        raise ValueError(message)

    write_func(object_tracks, object_options, output_directory)
    # Reset attributes lists after writing
    object_tracks["attributes"] = utils.initialize_attributes(object_options)


def write_final(tracks, track_options, output_directory):
    """Write final attributes to file."""

    for index, level_options in enumerate(track_options.levels):
        for object_options in level_options.objects:
            obj_name = object_options.name
            write(tracks[index][obj_name], object_options, output_directory)


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


def write_csv(filepath, df, attribute_options=None):
    """Write attribute dataframe to csv."""
    if attribute_options is None:
        df.to_csv(filepath, na_rep="NA")
        logger.debug("No attributes metadata provided. Writing csv without metadata.")
        return
    precision_dict = utils.get_precision_dict(attribute_options)
    df = df.round(precision_dict)
    df = df.sort_index()
    # Make filepath parent directory if it doesn't exist
    filepath.parent.mkdir(parents=True, exist_ok=True)
    logger.debug("Writing attribute dataframe to %s", filepath)
    df.to_csv(filepath, na_rep="NA")
    write_metadata(Path(filepath).with_suffix(".yml"), attribute_options)
    return df


def aggregate_directory(directory, attribute_type, attribute_options, clean_up):
    """Aggregate attribute files within a directory into single file."""
    filepaths = glob.glob(str(directory / "*.csv"))
    df_list = []
    index_cols = ["time"]
    if "universal_id" in attribute_options.keys():
        index_cols += ["universal_id"]
    elif "id" in attribute_options.keys():
        index_cols += ["id"]

    data_types = utils.get_data_type_dict(attribute_options)

    time_attrs = []
    for attr in attribute_options.keys():
        if attribute_options[attr]["data_type"] == "datetime64[s]":
            time_attrs.append(attr)

    for name in time_attrs:
        data_types.pop(name, None)

    for filepath in filepaths:
        date_format = "%Y-%m-%d %H:%M:%S"
        kwargs = {"index_col": index_cols, "na_values": ["", "NA"]}
        kwargs.update({"keep_default_na": True, "dtype": data_types})
        kwargs.update({"parse_dates": time_attrs, "date_format": date_format})
        df_list.append(pd.read_csv(filepath, **kwargs))
    df = pd.concat(df_list, sort=False)
    aggregated_filepath = directory.parent / f"{attribute_type}.csv"
    write_csv(aggregated_filepath, df, attribute_options)
    if clean_up:
        shutil.rmtree(directory)


def aggregate_attribute_type(
    base_directory, attribute_type, attribute_options, clean_up
):
    """Aggregate attributes within a directory."""

    if attribute_type == "tag" or attribute_type == "profile":
        # Additional layer of nesting by dataset for tag and profile attributes
        for dataset in attribute_options.keys():
            directory = base_directory / f"{dataset}/{attribute_type}"
            attr_options = attribute_options[dataset]
            aggregate_directory(directory, attribute_type, attr_options, clean_up)
            filepath = Path(directory.parent) / f"{attribute_type}.yml"
            write_metadata(filepath, attr_options)
    else:
        directory = base_directory / f"{attribute_type}"
        aggregate_directory(directory, attribute_type, attribute_options, clean_up)
        filepath = Path(directory.parent) / f"{attribute_type}.yml"
        write_metadata(filepath, attribute_options)


def aggregate_detected(base_directory, object_options, clean_up):
    """Aggregate attributes directory for detected objects."""
    obj_name = object_options.name
    for attribute_type in object_options.attributes.keys():
        directory = base_directory / f"{obj_name}"
        options = object_options.attributes[attribute_type]
        aggregate_attribute_type(directory, attribute_type, options, clean_up)


def aggregate_grouped(base_directory, object_options, clean_up):
    """Aggregate attributes directory for grouped attributes."""
    member_options = object_options.attributes["member_objects"]
    obj_name = object_options.name
    # First aggregate attributes of member objects
    for member_obj in member_options.keys():
        for attribute_type in member_options[member_obj].keys():
            directory = base_directory / f"{obj_name}/{member_obj}"
            options = member_options[member_obj][attribute_type]
            aggregate_attribute_type(directory, attribute_type, options, clean_up)
    # Now aggregate core attributes of grouped object
    for attribute_type in object_options.attributes[obj_name].keys():
        directory = base_directory / f"{obj_name}"
        options = object_options.attributes[obj_name][attribute_type]
        aggregate_attribute_type(directory, attribute_type, options, clean_up)


def aggregate(track_options, output_directory, clean_up=True):
    """Aggregate attributes into single file."""

    logger.info("Aggregating attribute files.")
    base_directory = Path(f"{output_directory}/attributes/")

    for level_options in track_options.levels:
        for object_options in level_options.objects:

            if "detection" in object_options.model_fields:
                aggregate_func = aggregate_detected
            elif "grouping" in object_options.model_fields:
                aggregate_func = aggregate_grouped
            else:
                message = "Object indentification method must be specified, i.e. "
                message += "'detection' or 'grouping'."
                raise ValueError(message)

            aggregate_func(base_directory, object_options, clean_up)
