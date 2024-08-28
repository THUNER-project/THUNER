"""Process ERA5 data."""

import time
import threading
from pathlib import Path
import inspect
import tempfile
import numpy as np
import pandas as pd
import xarray as xr
import cdsapi
from thor.log import setup_logger
from thor.utils import format_string_list, get_hour_interval
import thor.data.utils as utils
import thor.data.option as option
import thor.tag as tag
from thor.config import get_outputs_directory


logger = setup_logger(__name__)


def data_options(
    start="2005-11-13T00:00:00",
    end="2005-11-14T00:00:00",
    parent_remote="/g/data/rt52/era5",
    save_local=False,
    parent_local=str(get_outputs_directory() / "input_data/raw/"),
    converted_options=None,
    filepaths=None,
    use="tag",
    mode="reanalysis",
    data_format="pressure-levels",
    pressure_levels=None,
    fields=None,
    start_latitude=None,
    end_latitude=None,
    start_longitude=None,
    end_longitude=None,
    **kwargs,
):
    """
    Generate input options dictionary.

    Parameters
    ----------
    name : str, optional
        The name of the dataset; default is "era5".
    start : str, optional
        The start date and time of the dataset; default is "2005-02-01T00:00:00".
    end : str, optional
        The end date and time of the dataset; default is "2005-02-02T00:00:00".
    mode : str, optional
        The mode of the dataset; default is "reanalysis".
    data_format : str, optional
        The data_format of the dataset; default is "pressure-levels".
    parent : str, optional
        The parent URL; default is "/g/data/rt52/era5/".
    fields : list, optional
        The fields to include in the dataset; default is ["u", "v"].
    save : bool, optional
        Whether to save the dataset; default is True.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    options : dict
        Dictionary containing the input options.
    """

    if data_format == "pressure-levels":
        name = "era5_pl"
    elif data_format == "single-levels":
        name = "era5_sl"

    if fields is None:
        if data_format == "pressure-levels":
            fields = ["u", "v", "z", "r", "t"]
        elif data_format == "single-levels":
            fields = ["cape", "cin"]

    options = option.boilerplate_options(
        name,
        start,
        end,
        parent_remote,
        save_local,
        parent_local,
        converted_options,
        filepaths,
        use=use,
    )

    if data_format == "pressure-levels" and pressure_levels is None:
        if pressure_levels is None:
            pressure_levels = era5_pressure_levels
        pressure_levels = [str(level) for level in pressure_levels]

    options.update(
        {
            "mode": mode,
            "data_format": data_format,
            "fields": fields,
            "pressure_levels": pressure_levels,
            "start_latitude": start_latitude,
            "start_longitude": start_longitude,
            "end_latitude": end_latitude,
            "end_longitude": end_longitude,
        }
    )

    for key, value in kwargs.items():
        options[key] = value

    return options


def check_data_options(options):
    """
    Check the input options.

    Parameters
    ----------
    options : dict
        Dictionary containing the input options.

    Returns
    -------
    options : dict
        Dictionary containing the input options.
    """

    for key in inspect.getfullargspec(data_options).args:
        if key not in options.keys():
            raise ValueError(f"Missing required key {key}")

    min_start = np.datetime64("1959-01-01T00:00:00")
    if np.datetime64(options["start"]) < min_start:
        raise ValueError(f"start must be {min_start} or later.")

    data_formats = ["pressure-levels", "single-levels"]
    if options["data_format"] not in data_formats:
        raise ValueError(
            f"data_format must be one of {format_string_list(data_formats)}."
        )

    if (
        options["data_format"] == "pressure-levels"
        and options["pressure_levels"] is None
    ):
        raise ValueError(
            "pressure_levels must be provided for pressure-levels data_format."
        )

    modes = ["monthly-averaged", "monthly-averaged-by-hour", "reanalysis"]
    if options["mode"] not in modes:
        raise ValueError(f"mode must be one of {format_string_list(modes)}.")

    return options


def format_daterange(year, month, day=None):
    """
    Format the date range string used in ERA5 file names on NCI Gadi,
    https://dx.doi.org/10.25914/5f48874388857.

    Parameters
    ----------
    year : int
        The year.
    month : int
        The month.

    Returns
    -------
    date_range_str : str
        The formatted date range str.
    """
    if day is None:
        date_range_str = f"{year:04}{month:02}"
    else:
        date_range_str = f"{year:04}{month:02}{day:02}"
    return date_range_str


def generate_era5_filepaths(options, start=None, end=None, local=True, daily=True):
    """
    Generate era5 filepaths from dataset options dictionary.

    Parameters
    ----------
    options : dict
        Dictionary containing the input options.

    Returns
    -------
    urls : list
        List of URLs.
    times : list
        Times associated with the URLs.
    """

    if start is None or end is None:
        start = options["start"]
        # Add an hour to the end time to facilitate temporal interpolation
        end = options["end"]

    start = pd.Timestamp(start)
    # Add an hour to the end time to facilitate temporal interpolation
    end = pd.Timestamp(end) + pd.Timedelta(hours=1)

    short_data_format = {"pressure-levels": "pl", "single-levels": "sfc"}

    if local:
        parent = options["parent_local"]
    else:
        parent = options["parent_remote"]

    base_filepath = f"{parent}/era5/{options['data_format']}/{options['mode']}"

    if daily:
        # Note we typically store data locally in daily files
        times = np.arange(
            np.datetime64(f"{start.year:04}-{start.month:02}-{start.day:02}"),
            np.datetime64(f"{end.year:04}-{end.month:02}-{start.day:02}")
            + np.timedelta64(1, "D"),
            np.timedelta64(1, "D"),
        )
    else:
        # On GADI era5 data is stored in monthly files
        times = np.arange(
            np.datetime64(f"{start.year:04}-{start.month:02}"),
            np.datetime64(f"{end.year:04}-{end.month:02}") + np.timedelta64(1, "M"),
            np.timedelta64(1, "M"),
        )

    filepaths = dict(
        zip(options["fields"], [[] for i in range(len(options["fields"]))])
    )

    for field in options["fields"]:
        for time in times:
            time = pd.Timestamp(time)
            if daily:
                daterange_str = format_daterange(time.year, time.month, time.day)
            else:
                daterange_str = format_daterange(time.year, time.month)
            filepath = (
                f"{base_filepath}/{field}/{time.year}/{field}_era5_oper_"
                f"{short_data_format[options['data_format']]}_{daterange_str}.nc"
            )
            filepaths[field].append(filepath)

    for key in filepaths.keys():
        filepaths[key] = sorted(filepaths[key])

    return filepaths


def generate_cdsapi_requests(options, grid_options, daily=False):
    """
    Retrieve ERA5 data using the CDS API.

    Parameters
    ----------
    options : dict
        A dictionary containing the input options.

    Returns
    -------
    cds_name : str
        The name argument for the cdsapi retrieval.
    requests : dict
        A dictionary containing the cdsapi retrieval options.
    local_paths : dict
        A dictionary containing the local file paths.
    """

    short_data_format = {"pressure-levels": "pl", "single-levels": "sfc"}

    requests = dict(zip(options["fields"], [[] for i in range(len(options["fields"]))]))
    local_paths = dict(
        zip(options["fields"], [[] for i in range(len(options["fields"]))])
    )

    cds_name = f"reanalysis-era5-{options['data_format']}"

    start = pd.Timestamp(options["start"])
    # Add an hour to the end time to facilitate temporal interpolation
    end = pd.Timestamp(options["end"]) + pd.Timedelta(hours=1)

    base_path = (
        f"{options['parent_local']}/era5/{options['data_format']}/{options['mode']}"
    )

    # Request a days data at a time fromt the API
    times = np.arange(
        np.datetime64(f"{start.year:04}-{start.month:02}-{start.day:02}"),
        np.datetime64(f"{end.year:04}-{end.month:02}-{end.day:02}")
        + np.timedelta64(1, "D"),
        np.timedelta64(1, "D"),
    )

    for field in options["fields"]:
        for time in times:
            time = pd.Timestamp(time)

            request = {
                "product_type": [options["mode"]],
                "data_format": "netcdf",
                "download_format": "unarchived",
                "variable": [field],
                "pressure_level": options["pressure_levels"],
                "year": [f"{time.year:04}"],
                "month": [f"{time.month:02}"],
                "day": [f"{time.day:02}"],
                "time": [f"{i:02}" for i in range(0, 24)],
            }

            lat = grid_options["latitude"]
            lon = grid_options["longitude"]
            area = [lat[-1], lon[0], lat[0], lon[-1]]
            request["area"] = area

            daterange_str = format_daterange(time.year, time.month, time.day)
            local_path = (
                f"{base_path}/{field}/{time.year}/{field}_era5_oper_"
                f"{short_data_format[options['data_format']]}_{daterange_str}.nc"
            )
            requests[field].append(request)
            local_paths[field].append(local_path)

    return cds_name, requests, local_paths


def issue_cdsapi_requests(cds_name, requests, local_paths):
    """Issue cdsapi requests."""

    def download_data(cds_name, request, local_path):
        c = cdsapi.Client()
        c.retrieve(cds_name, request, local_path)

    for field in requests.keys():
        for i in range(len(local_paths[field])):
            path = Path(local_paths[field][i])
            path.parent.mkdir(parents=True, exist_ok=True)
            # Create a thread to run the download function
            download_thread = threading.Thread(
                target=download_data,
                args=(cds_name, requests[field][i], local_paths[field][i]),
            )
            download_thread.start()

            # Print progress messages while the download is running
            while download_thread.is_alive():
                logger.debug(f"Downloading {path.name}. Please wait.")
                time.sleep(30)  # Adjust the sleep time as needed

            download_thread.join()


def convert_era5():
    """Convert ERA5 data."""
    return


def generate_era5_times():
    """Generate ERA5 times."""
    return


def update_dataset(time, input_record, track_options, dataset_options, grid_options):
    """Update ERA5 dataset."""

    utils.log_dataset_update(logger, dataset_options["name"], time)

    start, end = get_hour_interval(time)
    filepaths = generate_era5_filepaths(dataset_options, start, end, local=True)
    all_files_exist = all(
        Path(filepath).exists() for field in filepaths.values() for filepath in field
    )
    if not all_files_exist and dataset_options["attempt_download"]:
        logger.warning("One or more filepaths do not exist; attempting download.")
        cds_name, requests, local_paths = generate_cdsapi_requests(
            dataset_options, grid_options
        )
        issue_cdsapi_requests(cds_name, requests, local_paths)
        pass

    lat = np.array(grid_options["latitude"])
    lon = np.array(grid_options["longitude"])
    lat_range = (lat.min(), lat.max())
    lon_range = (lon.min(), lon.max())

    with tempfile.TemporaryDirectory() as tmp:
        for field in dataset_options["fields"]:
            for filepath in filepaths[field]:
                logger.debug("Subsetting %s", Path(filepath).name)
                utils.call_ncks(
                    filepath, f"{tmp}/{field}.nc", start, end, lat_range, lon_range
                )
        input_record["dataset"] = xr.open_mfdataset(f"{tmp}/*.nc").load()


def tag_options(
    name=None, dataset="era5_pl", time_method="linear", space_method="linear"
):
    """
    Generate era5 tagging options dictionary.
    """
    if name is None:
        name = dataset

    options = tag.boilerplate_options(name, dataset, time_method, space_method)

    return options


era5_pressure_levels = ["1000", "975", "950", "925", "900", "875", "850", "825", "800"]
era5_pressure_levels += ["775", "750", "700", "650", "600", "550", "500", "450", "400"]
era5_pressure_levels += ["350", "300", "250", "225", "200", "175", "150", "125", "100"]
era5_pressure_levels += ["70", "50", "30", "20", "10", "7", "5", "3", "2", "1"]
