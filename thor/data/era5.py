"""Process ERA5 data."""

import numpy as np
import pandas as pd
from thor.log import setup_logger
from thor.utils import format_string_list
import calendar
from pathlib import Path
import yaml
import inspect


logger = setup_logger(__name__)


def create_options(
    name="era5",
    start="2005-02-01T00:00:00",
    end="2005-03-01T00:00:00",
    mode="reanalysis",
    format="pressure-levels",
    pressure_levels=None,
    parent="/g/data/rt52/era5",
    download_dir="../test/test_data",
    fields=["z", "u", "v"],
    start_latitude=None,
    end_latitude=None,
    start_longitude=None,
    end_longitude=None,
    save=False,
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
    format : str, optional
        The format of the dataset; default is "pressure-levels".
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

    if format == "pressure-levels" and pressure_levels is None:
        pressure_levels = [
            "1000",
            "975",
            "950",
            "925",
            "900",
            "875",
            "850",
            "825",
            "800",
            "775",
            "750",
            "700",
            "650",
            "600",
            "550",
            "500",
            "450",
            "400",
            "350",
            "300",
            "250",
            "225",
            "200",
            "175",
            "150",
            "125",
            "100",
            "70",
            "50",
            "30",
            "20",
            "10",
            "7",
            "5",
            "3",
            "2",
            "1",
        ]

    pressure_levels = [str(level) for level in pressure_levels]

    options = {
        "name": name,
        "start": start,
        "end": end,
        "mode": mode,
        "format": format,
        "parent": parent,
        "parent": parent,
        "fields": fields,
        "pressure_levels": pressure_levels,
        "download_dir": download_dir,
        "start_latitude": start_latitude,
        "start_longitude": start_longitude,
        "end_latitude": end_latitude,
        "end_longitude": end_longitude,
        "save": save,
    }

    for key, value in kwargs.items():
        options[key] = value

    if save:
        filepath = Path(__file__).parent.parent / "option/default/era5.yaml"
        with open(filepath, "w") as outfile:
            yaml.dump(
                options,
                outfile,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )

    return options


def check_options(options):
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

    for key in inspect.getargspec(create_options).args:
        if key not in options.keys():
            raise ValueError(f"Missing required key {key}")

    min_start = np.datetime64("1959-01-01T00:00:00")
    if np.datetime64(options["start"]) < min_start:
        raise ValueError(f"start must be {min_start} or later.")

    formats = ["pressure-levels", "single-levles"]
    if options["format"] not in formats:
        raise ValueError(f"format must be one of {format_string_list(formats)}.")

    if options["format"] == "pressure-levels" and options["pressure_levels"] is None:
        raise ValueError("pressure_levels must be provided for pressure-levels format.")

    modes = ["monthly-averaged", "monthly-averaged-by-hour", "reanalysis"]
    if options["mode"] not in modes:
        raise ValueError(f"mode must be one of {format_string_list(modes)}.")

    return options


def format_daterange(year, month):
    """
    Format the date range string used in ERA5 file names.

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
    last_day = calendar.monthrange(year, month)[1]
    date_range_str = f"{year:04}{month:02}01-{year:04}{month:02}{last_day}"
    return date_range_str


def generate_era5_urls(options):
    """
    Generate cpol URLs from input options dictionary.

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

    start = pd.Timestamp(options["start"])
    end = pd.Timestamp(options["end"])

    short_format = {"pressure-levels": "pl", "single-levels": "sfc"}

    urls = []

    base_url = f"{options['parent']}/{options['format']}/{options['mode']}"

    times = np.arange(
        np.datetime64(f"{start.year:04}-{start.month:02}"),
        np.datetime64(f"{end.year:04}-{end.month:02}") + np.timedelta64(1, "M"),
        np.timedelta64(1, "M"),
    )

    urls = dict(zip(options["fields"], [[] for i in range(len(options["fields"]))]))

    for field in options["fields"]:
        for time in times:
            time = pd.Timestamp(time)
            daterange_str = format_daterange(time.year, time.month)
            url = (
                f"{base_url}/{field}/{time.year}/{field}_era5_oper_"
                f"{short_format[options['format']]}_{daterange_str}.nc"
            )
            urls[field].append(url)

    return urls, times


def generate_cdsapi_requests(options, grid_options):
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

    short_format = {"pressure-levels": "pl", "single-levels": "sfc"}

    requests = dict(zip(options["fields"], [[] for i in range(len(options["fields"]))]))
    local_paths = dict(
        zip(options["fields"], [[] for i in range(len(options["fields"]))])
    )

    cds_name = f"reanalysis-era5-{options['format']}"

    start = pd.Timestamp(options["start"])
    end = pd.Timestamp(options["end"])

    base_path = f"{options['download_dir']}/{options['format']}/{options['mode']}"

    times = np.arange(
        np.datetime64(f"{start.year:04}-{start.month:02}"),
        np.datetime64(f"{end.year:04}-{end.month:02}") + np.timedelta64(1, "M"),
        np.timedelta64(1, "M"),
    )

    for field in options["fields"]:
        for time in times:
            time = pd.Timestamp(time)

            last_day = calendar.monthrange(time.year, time.month)[1]

            request = {
                "product_type": options["mode"],
                "format": "netcdf",
                "variable": field,
                "pressure_level": options["pressure_levels"],
                "year": f"{time.year:04}",
                "month": f"{time.month:02}",
                "day": [f"{i:02}" for i in range(1, last_day + 1)],
                "time": [f"{i:02}" for i in range(0, 24)],
            }

            area = get_cdsapi_area(grid_options)
            if area is not None:
                request["area"] = area

            daterange_str = format_daterange(time.year, time.month)
            local_path = (
                f"{base_path}/{field}/{time.year}/{field}_era5_oper_"
                f"{short_format[options['format']]}_{daterange_str}.nc"
            )
            requests[field].append(request)
            local_paths[field].append(local_path)

    return cds_name, requests, local_paths


def get_cdsapi_area(grid_options):
    if (
        grid_options["start_latitude"] is None
        and grid_options["end_latitude"] is None
        and grid_options["start_longitude"] is None
        and grid_options["end_longitude"] is None
    ):
        return None
    area = []
    keys = ["end_latitude", "start_longitude", "start_latitude", "end_longitude"]
    bounds = [90, 0, -90, 360]
    for key, bound in zip(keys, bounds):
        if grid_options[key] is None:
            area.append(bound)
        else:
            area.append(grid_options[key])
    return area
