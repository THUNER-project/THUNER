"""Write track-input filepath records to the unified zarr store."""

import numpy as np
import pandas as pd

import thuner.write.attribute as attribute
from thuner.utils import format_time
from thuner.log import setup_logger
from thuner.option.attribute import Attribute, AttributeType

logger = setup_logger(__name__)


def _filepath_attribute_type(name):
    """Build the AttributeType describing a filepath records table."""
    description = "Time taken from the tracking process."
    kwargs = {"name": "time", "data_type": np.datetime64}
    kwargs.update({"description": description, "units": "UTC"})
    time_attr = Attribute(**kwargs)

    description = f"Filepath to {name} data containing the given time."
    kwargs = {"name": name, "data_type": str, "description": description}
    filepaths_attr = Attribute(**kwargs)

    kwargs = {"name": name, "description": "Filepath to the data."}
    kwargs.update({"attributes": [time_attr, filepaths_attr]})
    return AttributeType(**kwargs)


def write(input_record, output_directory):
    """Append filepath records to the unified zarr store, then reset the buffer."""

    if input_record.filepaths is None:
        return

    name = input_record.name
    write_interval = input_record.write_interval
    _last_write_time = input_record._last_write_time
    last_write_str = format_time(_last_write_time, filename_safe=False, day_only=False)
    next_write_time = _last_write_time + write_interval
    current_str = format_time(next_write_time, filename_safe=False, day_only=False)

    message = (
        f"Writing {name} filepaths from {last_write_str} to "
        f"{current_str}, inclusive and non-inclusive, respectively."
    )
    logger.info(message)

    filepaths = input_record._filepath_list
    times = input_record._time_list
    if len(times) == 0:
        input_record._last_write_time = _last_write_time + write_interval
        input_record._time_list = []
        input_record._filepath_list = []
        return

    df = pd.DataFrame({"time": times, name: filepaths})
    df.set_index("time", inplace=True)
    df.sort_index(inplace=True)

    store = attribute.store_path(output_directory)
    group = f"records/filepaths/{name}"
    attribute_type = _filepath_attribute_type(name)
    attribute.write_attributes(store, group, df, attribute_type)

    input_record._last_write_time = _last_write_time + write_interval
    input_record._time_list = []
    input_record._filepath_list = []


def write_final(track_input_records, output_directory):
    """Flush remaining filepath records to the unified zarr store."""
    for input_record in track_input_records.values():
        if input_record.filepaths is None:
            continue
        write(input_record, output_directory)
