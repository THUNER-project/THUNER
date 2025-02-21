"""Functions for writing filepaths for each dataset to file."""

import pandas as pd
import numpy as np
import thuner.attribute.utils as utils
import thuner.write.attribute as attribute
from thuner.utils import format_time
from thuner.log import setup_logger
from thuner.option.attribute import Attribute, AttributeType

logger = setup_logger(__name__)


def write(input_record, output_directory):
    """Write the track input record filepaths and times to a file."""

    if input_record.filepaths is None:
        return

    name = input_record.name
    write_interval = input_record.write_interval
    _last_write_time = input_record._last_write_time
    last_write_str = format_time(_last_write_time, filename_safe=False, day_only=False)
    next_write_time = _last_write_time + write_interval
    current_str = format_time(next_write_time, filename_safe=False, day_only=False)

    message = f"Writing {name} filepaths from {last_write_str} to "
    message += f"{current_str}, inclusive and non-inclusive, "
    message += "respectively."
    logger.info(message)

    last_write_str = format_time(_last_write_time, filename_safe=True, day_only=False)
    csv_filepath = output_directory / "records/filepaths"
    csv_filepath = csv_filepath / f"{name}/{last_write_str}.csv"
    csv_filepath.parent.mkdir(parents=True, exist_ok=True)

    filepaths = input_record._filepath_list
    times = input_record._time_list
    filepaths_df = pd.DataFrame({"time": times, name: filepaths})
    filepaths_df = filepaths_df.sort_index()
    # Make filepath parent directory if it doesn't exist
    csv_filepath.parent.mkdir(parents=True, exist_ok=True)

    logger.debug("Writing attribute dataframe to %s", csv_filepath)
    filepaths_df.set_index("time", inplace=True)
    filepaths_df.sort_index(inplace=True)
    date_format = "%Y-%m-%d %H:%M:%S"
    filepaths_df.to_csv(csv_filepath, na_rep="NA", date_format=date_format)
    input_record._last_write_time = _last_write_time + write_interval

    # Empty mask_list after writing
    input_record._time_list = []
    input_record._filepath_list = []


def write_final(track_input_records, output_directory):
    """Write the track input record filepaths and times to a file."""

    for input_record in track_input_records.values():
        if input_record.filepaths is None:
            continue
        write(input_record, output_directory)


def aggregate(track_input_records, output_directory, clean_up=True):
    """Aggregate the track input record filepaths and times to a single file."""

    logger.info("Aggregating filepath records.")

    for input_record in track_input_records.values():
        if input_record.filepaths is None:
            continue
        name = input_record.name
        directory = output_directory / f"records/filepaths/{name}"

        description = "Time taken from the tracking process."
        kwargs = {"name": "time", "data_type": np.datetime64}
        kwargs.update({"description": description, "units": "UTC"})
        time_attr = Attribute(**kwargs)

        description = f"Filepath to {name} data containing the given time."
        kwargs = {"name": name, "data_type": str, "description": description}
        filepaths_attr = Attribute(**kwargs)

        kwargs = {"name": name, "description": "Filepath to the data."}
        kwargs.update({"attributes": [time_attr, filepaths_attr]})
        attribute_type = AttributeType(**kwargs)

        attribute.aggregate_directory(directory, attribute_type, clean_up)
