"""General input data processing."""

from thuner.log import setup_logger
from thuner.utils import time_in_dataset_range


logger = setup_logger(__name__)


def boilerplate_update(
    time, input_record, track_options, dataset_options, grid_options
):
    """Update the dataset."""

    earliest_time = time + dataset_options.start_buffer
    latest_time = time + dataset_options.end_buffer
    cond = not time_in_dataset_range(earliest_time, input_record.dataset)
    cond = cond or not time_in_dataset_range(latest_time, input_record.dataset)
    if cond:
        args = [time, input_record, track_options, grid_options]
        dataset_options.update_input_record(*args)


def update_track_input_records(
    time,
    track_input_records,
    track_options,
    data_options,
    grid_options,
    output_directory,
):
    """Update the input record, i.e. grids and datasets."""
    for name in track_input_records.keys():
        input_record = track_input_records[name]
        dataset_options = data_options.dataset_by_name(name)
        args = [time, input_record, track_options, dataset_options, grid_options]
        boilerplate_update(*args)
        if input_record.next_grid is not None:
            input_record.grids.append(input_record.next_grid)
        if len(dataset_options.fields) > 1:
            raise ValueError("Only one field allowed for track datasets.")
        else:
            field = dataset_options.fields[0]
        next_grid = dataset_options.grid_from_dataset(input_record.dataset, field, time)
        input_record.next_grid = next_grid
        if dataset_options.filepaths is None:
            return
        input_record._time_list.append(time)
        filepath = dataset_options.filepaths[input_record._current_file_index]
        input_record._filepath_list.append(filepath)


def update_tag_input_records(
    time, tag_input_records, track_options, data_options, grid_options
):
    """Update the tag input records."""
    if time is None:
        return
    for name in tag_input_records.keys():
        input_record = tag_input_records[name]
        dataset_options = data_options.dataset_by_name(name)
        args = [time, input_record, track_options, dataset_options, grid_options]
        boilerplate_update(*args)
