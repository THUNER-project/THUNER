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


data_type_to_string = {v: k for k, v in utils.string_to_data_type.items()}
logger = setup_logger(__name__)


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


def write_core(object_tracks, object_options, output_directory):
    """
    Write core attributes to file.
    """
    write_args = [object_tracks, object_options, output_directory]
    base_filepath, last_write_str = write_setup(*write_args)
    filepath = base_filepath / f"core/{format_time(last_write_str)}.csv"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    attributes = object_tracks["attribute"]["core"]
    options = object_options["attribute"]["core"]
    df = utils.attributes_dataframe(attributes, options)
    df.to_csv(filepath)


def write_group(object_tracks, object_options, output_directory):
    """
    Write group attributes to file.
    """
    write_args = [object_tracks, object_options, output_directory]
    base_filepath, last_write_str = write_setup(*write_args)
    # First write member object core attributes
    member_options = object_options["attribute"]["group"]["member_objects"]
    member_attributes = object_tracks["attribute"]["group"]["member_objects"]
    for obj in member_options.keys():
        filepath = base_filepath / f"{obj}/core/{format_time(last_write_str)}.csv"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        attributes = member_attributes[obj]["core"]
        options = member_options[obj]["core"]
        df = utils.attributes_dataframe(attributes, options)
        df.to_csv(filepath)
    # Now write grouped object core attributes
    filepath = base_filepath / f"core/{format_time(last_write_str)}.csv"
    filepath.parent.mkdir(parents=True, exist_ok=True)
    attributes = object_tracks["attribute"]["group"][object_options["name"]]["core"]
    options = object_options["attribute"]["group"][object_options["name"]]["core"]
    df = utils.attributes_dataframe(attributes, options)
    df.to_csv(filepath)


write_dispatcher = {"core": write_core, "group": write_group}


def write(object_tracks, object_options, output_directory):
    """Write masks to file."""

    for attribute_type in object_options["attribute"].keys():
        write_func = write_dispatcher[attribute_type]
        write_func(object_tracks, object_options, output_directory)
    utils.initialize_attributes(object_tracks, object_options)


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


def aggregate_core(base_path, object_options, clean_up):
    """Aggregate core attributes."""
    options = object_options["attribute"]["core"]
    obj_name = object_options["name"]
    directory = base_path / f"{obj_name}/core"
    aggregate_directory(directory, "core", options, clean_up)
    filepath = Path(directory.parent) / "core_metadata.yml"
    write_metadata(filepath, options)


def aggregate_group(base_path, object_options, clean_up):
    """Aggregate group attributes."""
    member_options = object_options["attribute"]["group"]["member_objects"]
    obj_name = object_options["name"]
    # First aggregate core attributes of member objects
    for member_obj in member_options.keys():
        directory = base_path / f"{obj_name}/{member_obj}/core"
        options = member_options[member_obj]["core"]
        aggregate_directory(directory, "core", options, clean_up)
        filepath = Path(directory.parent) / f"core_metadata.yml"
        write_metadata(filepath, options)
    # Now aggregate core attributes of grouped object
    options = object_options["attribute"]["group"][obj_name]["core"]
    directory = base_path / f"{obj_name}/core"
    aggregate_directory(directory, "core", options, clean_up)
    filepath = Path(directory.parent) / f"core_metadata.yml"
    write_metadata(filepath, options)


aggregate_dispatcher = {"core": aggregate_core, "group": aggregate_group}


def aggregate(track_options, output_directory, clean_up=True):
    """Aggregate masks into single file."""

    logger.info("Aggregating attribute files.")
    base_path = Path(f"{output_directory}/attributes/")

    for level_options in track_options:
        for obj_name in level_options.keys():
            for attribute_type in level_options[obj_name]["attribute"].keys():
                aggregate_func = aggregate_dispatcher[attribute_type]
                aggregate_func(base_path, level_options[obj_name], clean_up)
