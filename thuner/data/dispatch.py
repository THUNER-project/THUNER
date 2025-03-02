"""General input data processing."""

import thuner.data.aura as aura
import thuner.data.era5 as era5
import thuner.data.synthetic as synthetic
import thuner.data.gridrad as gridrad
import thuner.data.utils as utils
from thuner.log import setup_logger
from thuner.utils import time_in_dataset_range


logger = setup_logger(__name__)


update_dataset_dispatcher = {
    "cpol": aura.update_dataset,
    "operational": aura.update_dataset,
    "gridrad": gridrad.update_dataset,
    "era5_pl": era5.update_dataset,
    "era5_sl": era5.update_dataset,
    "synthetic": synthetic.update_dataset,
}

convert_dataset_dispatcher = {
    "cpol": aura.convert_cpol,
    "operational": aura.convert_operational,
    "gridrad": gridrad.convert_gridrad,
    "era5_pl": era5.convert_era5,
    "era5_sl": era5.convert_era5,
    "synthetic": synthetic.convert_synthetic,
}

grid_from_dataset_basic = lambda dataset, variable, time: dataset[variable].sel(
    time=time
)

grid_from_dataset_dispatcher = {
    "cpol": aura.cpol_grid_from_dataset,
    "gridrad": gridrad.gridrad_grid_from_dataset,
    "operational": grid_from_dataset_basic,
    "synthetic": grid_from_dataset_basic,
}


generate_filepaths_dispatcher = {
    "cpol": aura.get_cpol_filepaths,
    "operational": aura.get_operational_filepaths,
    "gridrad": gridrad.get_gridrad_filepaths,
    "era5_pl": era5.get_era5_filepaths,
    "era5_sl": era5.get_era5_filepaths,
    "synthetic": None,
}


get_domain_mask_dispatcher = {
    "gridrad": gridrad.get_domain_mask,
    "cpol": utils.mask_from_input_record,
}


def generate_filepaths(dataset_options):
    """
    Get the filepaths for the dataset.

    Parameters
    ----------
    dataset_options : dict
        Dictionary containing the dataset options. Note this is the
        dictionary for an individual dataset, not the entire data_options
        dictionary.

    Returns
    -------
    list
        List of filepaths to files ready to be converted.

    """

    get_filepaths = generate_filepaths_dispatcher.get(dataset_options.name)
    if get_filepaths is None:
        raise KeyError(f"Filepath generator for {dataset_options['name']} not found.")
    filepaths = get_filepaths(dataset_options)

    return filepaths


def boilerplate_update(
    time, input_record, track_options, dataset_options, grid_options
):
    """Update the dataset."""

    earliest_time = time + dataset_options.start_buffer
    latest_time = time + dataset_options.end_buffer
    cond = not time_in_dataset_range(earliest_time, input_record.dataset)
    cond = cond or not time_in_dataset_range(latest_time, input_record.dataset)
    if cond:
        update_dataset(time, input_record, track_options, dataset_options, grid_options)


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
        grid_from_dataset = grid_from_dataset_dispatcher.get(name)
        if len(dataset_options.fields) > 1:
            raise ValueError("Only one field allowed for track datasets.")
        else:
            field = dataset_options.fields[0]
        input_record.next_grid = grid_from_dataset(input_record.dataset, field, time)
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
        boilerplate_update(
            time,
            input_record,
            track_options,
            data_options.dataset_by_name(name),
            grid_options,
        )


def update_dataset(time, input_record, track_options, dataset_options, grid_options):
    """Update the dataset."""

    updt_dataset = update_dataset_dispatcher.get(dataset_options.name)
    if updt_dataset is None:
        raise KeyError(f"Dataset updater for {dataset_options['name']} not found.")
    updt_dataset(time, input_record, track_options, dataset_options, grid_options)
