"""General input data processing."""

import thor.data.aura as aura
import thor.data.era5 as era5
from thor.log import setup_logger
from thor.utils import time_in_dataset_range


logger = setup_logger(__name__)


update_dataset_dispatcher = {
    "cpol": aura.update_dataset,
    "operational": aura.update_dataset,
    "era5_pl": era5.update_dataset,
    "era5_sl": era5.update_dataset,
}


grid_from_dataset_dispatcher = {
    "cpol": lambda dataset, time: dataset.sel(time=time),
    "operational": lambda dataset, time: dataset.sel(time=time),
}


grid_from_dataset_dispatcher = {
    "cpol": lambda dataset, variable, time: dataset[variable].sel(time=time),
}


generate_times_dispatcher = {
    "cpol": aura.generate_cpol_times,
    # "operational": data.aura.generate_operational_times,
    # "era5": data.era5.generate_era5_times,
}


generate_filepaths_dispatcher = {
    "cpol": aura.generate_cpol_filepaths,
    "operational": aura.generate_operational_filepaths,
    "era5_pl": era5.generate_era5_filepaths,
    "era5_sl": era5.generate_era5_filepaths,
}


check_data_options_dispatcher = {
    "cpol": aura.check_data_options,
    "operational": aura.check_data_options,
    "era5_pl": era5.check_data_options,
    "era5_sl": era5.check_data_options,
}


def check_data_options(data_options):
    """TBA."""
    for name in data_options.keys():
        check_data_options = check_data_options_dispatcher.get(name)
        if check_data_options is None:
            message = "check_data_options function for dataset "
            message += f"{name} not found."
            raise KeyError(message)
        else:
            check_data_options(data_options[name])
        if data_options[name]["filepaths"] is None:
            filepaths = generate_filepaths(data_options[name])
            data_options[name]["filepaths"] = filepaths


def generate_times(filepaths, dataset_name):
    """Universal generate times function."""
    get_times = generate_times_dispatcher.get(dataset_name)
    if get_times is None:
        raise KeyError(f"get_time function for {dataset_name} not found.")
    for filepath in sorted(filepaths):
        times = get_times(filepath)
        for time in times:
            yield time


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

    get_filepaths = generate_filepaths_dispatcher.get(dataset_options["name"])
    if get_filepaths is None:
        raise KeyError(f"Filepath generator for {dataset_options['name']} not found.")
    filepaths = get_filepaths(dataset_options)

    return filepaths


def boilerplate_update(time, input_record, dataset_options, grid_options):
    """Update the dataset."""
    if not time_in_dataset_range(time, input_record["dataset"]):
        update_dataset(time, input_record, dataset_options, grid_options)


def update_track_input_records(time, track_input_records, data_options, grid_options):
    """Update the input record, i.e. grids and datasets."""
    for name in track_input_records.keys():
        input_record = track_input_records[name]
        boilerplate_update(time, input_record, data_options[name], grid_options)
        if input_record["current_grid"] is not None:
            input_record["previous_grids"].append(input_record["current_grid"])
        grid_from_dataset = grid_from_dataset_dispatcher.get(name)
        input_record["dataset"] = input_record["dataset"]
        if len(data_options[name]["fields"]) > 1:
            raise ValueError("Only one field allowed for track datasets.")
        else:
            field = data_options[name]["fields"][0]
        input_record["current_grid"] = grid_from_dataset(
            input_record["dataset"], field, time
        )


def update_tag_input_records(time, tag_input_records, data_options, grid_options):
    for name in tag_input_records.keys():
        input_record = tag_input_records[name]
        boilerplate_update(time, input_record, data_options[name], grid_options)


def update_dataset(time, input_record, dataset_options, grid_options):
    """Update the dataset."""

    updt_dataset = update_dataset_dispatcher.get(dataset_options["name"])
    if updt_dataset is None:
        raise KeyError(f"Dataset updater for {dataset_options['name']} not found.")
    updt_dataset(time, input_record, dataset_options, grid_options)
