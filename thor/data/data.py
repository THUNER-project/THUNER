"""General data processing."""

import thor.data as data
from thor.log import setup_logger
from thor.option import save_options
from pathlib import Path


logger = setup_logger(__name__)


converter_dispathcher = {
    "cpol": data.aura.convert_cpol,
    "operational": data.aura.convert_operational,
}


get_times_dispatcher = {
    "cpol": data.aura.get_cpol_times,
    "operational": data.aura.get_operational_times,
}


generate_filepaths_dispatcher = {
    "cpol": data.aura.generate_cpol_filepaths,
}


def check_component_options(component_options):
    """Check options for converted datasets and masks."""

    if not isinstance(component_options, dict):
        raise TypeError("component_options must be a dictionary.")
    if "save" not in component_options:
        raise KeyError("save key not found in component_options.")
    if "load" not in component_options:
        raise KeyError("load key not found in component_options.")
    if not isinstance(component_options["save"], bool):
        raise TypeError("save key must be a boolean.")
    if not isinstance(component_options["load"], bool):
        raise TypeError("load key must be a boolean.")


def create_options(
    name,
    start,
    end,
    parent_remote=None,
    save_local=False,
    parent_local=None,
    converted_options=None,
    mask_options=None,
    filepaths=None,
):
    """
    Generate dataset options dictionary.

    Parameters
    ----------
    name : str, optional
        The name of the dataset; default is "operational".
    start : datetime.datetime, optional
        The start time of the dataset; default is None.
    end : datetime.datetime, optional
        The end time of the dataset; default is None.
    parent_remote : str, optional
        The remote parent directory; default is None.
    save_local : bool, optional
        Whether to save the raw data locally; default is False.
    parent_local : str, optional
        The local parent directory; default is None.
    options_converted : dict, optional
        Dictionary containing the converted data options; default is None.
    options_mask : dict, optional
        Dictionary containing the mask options; default is None.
    filepaths : list, optional
        List of filepaths to the files to be converted; default is None.

    Returns
    -------
    options : dict
        Dictionary containing the data options.
    """

    if converted_options is None:
        converted_options = {"save": False, "load": False}
    else:
        check_component_options(converted_options)

    if mask_options is None:
        mask_options = {"save": False, "load": False}
    else:
        check_component_options(mask_options)

    options = {
        "name": name,
        "start": start,
        "end": end,
        "parent_remote": parent_remote,
        "save_local": save_local,
        "parent_local": parent_local,
        "converted_options": converted_options,
        "mask_options": mask_options,
        "filepaths": filepaths,
    }

    return options


def save_data_options(
    data_options, filename=None, options_directory=None, append_time=False
):

    if options_directory is None:
        options_directory = Path(__file__).parent.parent / "options/data_options"
    if filename is None:
        filename = "data_options"
        append_time = True
    logger.debug(f"Saving data options to {options_directory / filename}")
    save_options(data_options, filename, options_directory, append_time=append_time)


def generate_times(filepaths, dataset_name):
    get_times = get_times_dispatcher.get(dataset_name)
    if get_times is None:
        raise KeyError(f"get_time function for {dataset_name} not found.")
    for filepath in sorted(filepaths):
        times = get_times(filepath)
        for time in times:
            yield time


def convert(filepath, data_options, grid_options, save=False):

    converter = converter_dispathcher.get(data_options["name"])
    if converter is None:
        raise KeyError(f"Converter for {data_options['name']} not found.")
    ds = converter(filepath, data_options, grid_options, save=save)

    return ds


def consolidate_options(data_options_list):
    options = {}
    for data_options in data_options_list:
        options[data_options["name"]] = data_options
    return options


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
        raise KeyError(f"Converter for {dataset_options['name']} not found.")
    filepaths = get_filepaths(dataset_options)

    return filepaths
