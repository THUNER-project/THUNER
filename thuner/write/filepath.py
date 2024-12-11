"""Functions for writing filepaths for each dataset to file."""

import pandas as pd
import numpy as np
import thuner.attribute.utils as utils
import thuner.write.attribute as attribute
from thuner.utils import format_time
from thuner.log import setup_logger

logger = setup_logger(__name__)


def write(input_record, output_directory):
    """Write the track input record filepaths and times to a file."""

    if "filepath_list" not in input_record.keys():
        return

    name = input_record["name"]
    write_interval = input_record["write_interval"]
    last_write_time = input_record["last_write_time"]
    last_write_str = format_time(last_write_time, filename_safe=False, day_only=False)
    next_write_time = last_write_time + write_interval
    current_str = format_time(next_write_time, filename_safe=False, day_only=False)

    message = f"Writing {name} filepaths from {last_write_str} to "
    message += f"{current_str}, inclusive and non-inclusive, "
    message += "respectively."
    logger.info(message)

    last_write_str = format_time(last_write_time, filename_safe=True, day_only=False)
    csv_filepath = output_directory / "records/filepaths"
    csv_filepath = csv_filepath / f"{name}/{last_write_str}.csv"
    csv_filepath.parent.mkdir(parents=True, exist_ok=True)

    method = None
    data_type = str
    precision = None
    units = None

    attribute_options = {}
    filepaths = input_record["filepath_list"]
    times = input_record["time_list"]

    description = f"Filepath to {name} data containing the given time."
    filepaths_df = pd.DataFrame({"time": times, name: filepaths})
    args = [name, method, data_type, precision, description, units]
    attribute_options[name] = utils.get_attribute_dict(*args)
    args = ["time", None, np.datetime64, None, "Time.", "UTC"]
    attribute_options["time"] = utils.get_attribute_dict(*args)
    precision_dict = utils.get_precision_dict(attribute_options)
    filepaths_df = filepaths_df.round(precision_dict)
    filepaths_df = filepaths_df.sort_index()
    # Make filepath parent directory if it doesn't exist
    csv_filepath.parent.mkdir(parents=True, exist_ok=True)
    logger.debug("Writing attribute dataframe to %s", csv_filepath)
    filepaths_df.set_index("time", inplace=True)
    filepaths_df.sort_index(inplace=True)
    filepaths_df.to_csv(csv_filepath, na_rep="NA")
    input_record["last_write_time"] = last_write_time + write_interval
    # Empty mask_list after writing
    input_record["time_list"] = []
    input_record["filepath_list"] = []


def write_final(track_input_records, output_directory):
    """Write the track input record filepaths and times to a file."""

    for input_record in track_input_records.values():
        if "filepath_list" not in input_record.keys():
            continue
        write(input_record, output_directory)


def aggregate(track_input_records, output_directory, clean_up=True):
    """Aggregate the track input record filepaths and times to a single file."""

    logger.info("Aggregating filepath records.")

    for input_record in track_input_records.values():
        if "filepath_list" not in input_record.keys():
            continue
        attribute_options = {}
        name = input_record["name"]
        directory = output_directory / f"records/filepaths/{name}"
        description = f"Filepath to {name} data containing the given time."
        args = [name, None, str, None, description, None]
        attribute_options[name] = utils.get_attribute_dict(*args)
        description = "Time taken from the tracking process."
        args = ["time", None, "datetime64[s]", None, description, None]
        attribute_options["time"] = utils.get_attribute_dict(*args)
        attribute.aggregate_directory(directory, name, attribute_options, clean_up)
