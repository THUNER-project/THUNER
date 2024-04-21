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
    parent="/g/data/rt52/era5",
    fields=["z", "u", "v"],
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

    options = {
        "name": name,
        "start": start,
        "end": end,
        "mode": mode,
        "format": format,
        "parent": parent,
        "parent": parent,
        "fields": fields,
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

    for key in options.keys():
        if key not in inspect.getargspec(create_options).args:
            raise ValueError(f"Missing required key {key}")

    min_start = np.datetime64("1959-01-01T00:00:00")
    if np.datetime64(options["start"]) < min_start:
        raise ValueError(f"start must be {min_start} or later.")

    formats = ["pressure-levels", "single-levles"]
    if options["format"] not in formats:
        raise ValueError(f"format must be one of {format_string_list(formats)}.")

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
