"""Functions for writing object masks."""

import yaml
import glob
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from thuner.utils import format_time
from thuner.log import setup_logger
import thuner.attribute.utils as utils
from thuner.option.track import BaseObjectOptions
from thuner.option.attribute import AttributeType

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

    _last_write_time = object_tracks._last_write_time
    write_interval = np.timedelta64(object_options.write_interval, "h")

    last_write_str = format_time(_last_write_time, filename_safe=False, day_only=False)
    next_write_time = _last_write_time + write_interval
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
    date_format = "%Y-%m-%d %H:%M:%S"
    df.to_csv(filepath, na_rep="NA", date_format=date_format)


def write(object_tracks, object_options: BaseObjectOptions, output_directory):
    """Write attributes to file."""

    if object_options.attributes is None:
        return

    write_args = [object_tracks, object_options, output_directory]
    base_directory, last_write_str = write_setup(*write_args)
    member_attribute_options = object_options.attributes.member_attributes
    if member_attribute_options is not None:
        # Write member object attributes
        recorded_member_attr = object_tracks.attributes.member_attributes
        for obj in member_attribute_options.keys():
            for attribute_type in member_attribute_options[obj].attribute_types:
                directory = base_directory / f"{obj}" / f"{attribute_type.name}/"
                rec_attr = recorded_member_attr[obj][attribute_type.name]
                write_attributes(directory, last_write_str, rec_attr, attribute_type)
    # Write object attributes
    obj_attr = object_tracks.attributes.attribute_types
    for attribute_type in object_options.attributes.attribute_types:
        attributes = obj_attr[attribute_type.name]
        directory = base_directory / f"{attribute_type.name}"
        write_attributes(directory, last_write_str, attributes, attribute_type)

    # Reset attributes lists after writing
    attr_options = object_options.attributes
    object_tracks.attributes = utils.AttributesRecord(attribute_options=attr_options)


def write_final(tracks, track_options, output_directory):
    """Write final attributes to file."""

    for index, level_options in enumerate(track_options.levels):
        for object_options in level_options.objects:
            obj_name = object_options.name
            obj_tracks = tracks.levels[index].objects[obj_name]
            write(obj_tracks, object_options, output_directory)


def write_metadata(filepath, attribute_type: AttributeType):
    """Write metadata to yml file."""
    logger.debug("Saving attribute metadata to %s", filepath)
    attribute_type.to_yaml(filepath)


def write_csv(filepath, df, attribute_type=None):
    """Write attribute dataframe to csv."""
    if attribute_type is None:
        date_format = "%Y-%m-%d %H:%M:%S"
        df.to_csv(filepath, na_rep="NA", date_format=date_format)
        logger.debug("No attributes metadata provided. Writing csv without metadata.")
        return
    precision_dict = utils.get_precision_dict(attribute_type)
    df = df.round(precision_dict)
    df = df.sort_index()
    # Make filepath parent directory if it doesn't exist
    filepath.parent.mkdir(parents=True, exist_ok=True)
    logger.debug("Writing attribute dataframe to %s", filepath)
    date_format = "%Y-%m-%d %H:%M:%S"
    df.to_csv(filepath, na_rep="NA", date_format=date_format)
    write_metadata(Path(filepath).with_suffix(".yml"), attribute_type)
    return df


def aggregate_directory(directory, attribute_type: AttributeType, clean_up=True):
    """Aggregate attribute files within a directory into single file."""
    filepaths = glob.glob(str(directory / "*.csv"))
    df_list = []
    names = [attr.name for attr in attribute_type.attributes]
    index_cols = ["time"]
    if ["time_offset"] in names:
        index_cols += ["time_offset"]
    if "universal_id" in names:
        index_cols += ["universal_id"]
    elif "id" in names:
        index_cols += ["id"]

    data_types = utils.get_data_type_dict(attribute_type)
    time_attrs = []
    for name in data_types.keys():
        if data_types[name] is np.datetime64:
            time_attrs.append(name)

    for name in time_attrs:
        data_types.pop(name, None)

    for filepath in filepaths:
        date_format = "%Y-%m-%d %H:%M:%S"
        kwargs = {"index_col": index_cols, "na_values": ["", "NA"]}
        kwargs.update({"keep_default_na": True, "dtype": data_types})
        kwargs.update({"parse_dates": time_attrs, "date_format": date_format})
        df_list.append(pd.read_csv(filepath, **kwargs))
    df = pd.concat(df_list, sort=False)
    aggregated_filepath = directory.parent / f"{attribute_type.name}.csv"
    write_csv(aggregated_filepath, df, attribute_type)
    if clean_up:
        shutil.rmtree(directory)


def aggregate_attribute_type(base_directory, attribute_type: AttributeType, clean_up):
    """Aggregate attributes within a directory."""

    directory = base_directory / f"{attribute_type.name}"
    aggregate_directory(directory, attribute_type, clean_up)
    filepath = Path(directory.parent) / f"{attribute_type.name}.yml"
    write_metadata(filepath, attribute_type)


def aggregate_object(base_directory, object_options, clean_up):
    """Aggregate attributes directory for grouped attributes."""
    if object_options.attributes is None:
        return
    member_options = object_options.attributes.member_attributes
    obj_name = object_options.name
    # First aggregate attributes of member objects
    if member_options is not None:
        for member_obj in member_options.keys():
            for attribute_type in member_options[member_obj].attribute_types:
                directory = base_directory / f"{obj_name}/{member_obj}"
                aggregate_attribute_type(directory, attribute_type, clean_up)
    # Now aggregate attributes of object itself
    for attribute_type in object_options.attributes.attribute_types:
        directory = base_directory / f"{obj_name}"
        aggregate_attribute_type(directory, attribute_type, clean_up)


def aggregate(track_options, output_directory, clean_up=True):
    """Aggregate attribute types of each object."""

    logger.info("Aggregating attribute files.")
    base_directory = Path(f"{output_directory}/attributes/")

    for level_options in track_options.levels:
        for object_options in level_options.objects:
            aggregate_object(base_directory, object_options, clean_up)
