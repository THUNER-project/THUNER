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
from thor.attribute.utils import create_dataframe, string_to_data_type


data_type_to_string = {v: k for k, v in string_to_data_type.items()}
logger = setup_logger(__name__)


def write(object_tracks, object_options, output_directory):
    """Write masks to file."""
    object_name = object_options["name"]
    last_write_time = object_tracks["last_write_time"]
    write_interval = np.timedelta64(object_options["write_interval"], "h")

    last_write_str = format_time(last_write_time, filename_safe=False, day_only=False)
    current_str = format_time(
        last_write_time + write_interval, filename_safe=False, day_only=False
    )
    message = f"Writing {object_name} attributes from {last_write_str} to "
    message += f"{current_str}, inclusive and non-inclusive, respectively."
    logger.info(message)

    base_filepath = output_directory / f"attributes/{object_name}/"

    for attribute_type in object_options["attribute"].keys():
        filepath = base_filepath / f"{attribute_type}/{format_time(last_write_str)}.csv"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df = create_dataframe(attribute_type, object_tracks, object_options)
        df.to_csv(filepath)


def write_final(tracks, track_options, output_directory):
    """Write final masks to file."""

    for index, level_options in enumerate(track_options):
        for obj in level_options.keys():
            write(tracks[index][obj], level_options[obj], output_directory)


def aggregate_type(base_path, obj_name, attribute_type, clean_up=True):
    """Aggregate attribute type for object into single file."""
    attr_path = Path(f"{base_path}/{obj_name}/{attribute_type}")
    filepaths = glob.glob(str(attr_path / "*.csv"))
    df_list = []
    for filepath in filepaths:
        df_list.append(pd.read_csv(filepath, index_col=["time"]))
    df = pd.concat(df_list)
    df.to_csv(attr_path.parent / f"{attribute_type}.csv")
    if clean_up:
        shutil.rmtree(attr_path)


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


def aggregate(track_options, output_directory, clean_up=True):
    """Aggregate masks into single file."""

    logger.info("Aggregating attribute files.")
    base_path = Path(f"{output_directory}/attributes/")

    for level_options in track_options:
        for obj_name in level_options.keys():
            for attribute_type in level_options[obj_name]["attribute"].keys():
                aggregate_type(base_path, obj_name, attribute_type, clean_up)
                attribute_options = level_options[obj_name]["attribute"][attribute_type]
                filepath = base_path / f"{obj_name}/{attribute_type}_metadata.yml"
                write_metadata(filepath, attribute_options)
