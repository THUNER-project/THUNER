"""Process WRF data."""

import numpy as np
import pandas as pd
from thuner.log import setup_logger
from thuner.utils import drop_time
from thuner.config import get_outputs_directory
from pathlib import Path
import yaml
import inspect


logger = setup_logger(__name__)


def create_options(
    name="MCASClimate",
    start="2010-02-01T00:00:00",
    end="2010-02-05T00:00:00",
    version="v1-0",
    parent="https://dapds00.nci.org.au/thredds/dodsC/ks32/ARCCSS_Data/MCASClimate",
    fields=["U", "V"],
    save=False,
    **kwargs,
):
    """
    Generate input options dictionary.

    Parameters
    ----------
    name : str, optional
        Name of the dataset; default is "MCASClimate".
    start : str, optional
        Start time of the dataset in ISO format; default is "2010-02-01T00:00:00".
    end : str, optional
        End time of the dataset in ISO format; default is "2010-02-05T00:00:00".
    version : str, optional
        Version of the dataset; default is "v1-0".
    parent : str, optional
        Parent URL for the dataset; default is "https://dapds00.nci.org.au/thredds/dodsC/ks32/ARCCSS_Data/MCASClimate/".
    fields : list of str, optional
        List of fields to be included in the dataset; default is ["U", "V"].
    save : bool, optional
        Whether to save the dataset; default is True.
    **kwargs
        Additional keyword arguments.
    """

    options = {
        "name": name,
        "start": start,
        "end": end,
        "version": version,
        "parent": parent,
        "fields": fields,
    }

    for key, value in kwargs.items():
        options[key] = value

    if save:
        filepath = get_outputs_directory() / "option/default/wrf.yml"
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
        if key not in inspect.getfullargspec(create_options).args:
            raise ValueError(f"Missing required key {key}")

    return options


def generate_MCASClimate_urls(options):
    """
    Generate ACCESS URLs.

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

    start = drop_time(np.datetime64(options["start"]))
    end = drop_time(np.datetime64(options["end"]))

    base_url = f"{options['parent']}"

    urls = dict(zip(options.fields, [[] for i in range(len(options.fields))]))

    times = np.arange(start, end + np.timedelta64(1, "D"), np.timedelta64(1, "D"))
    times = pd.DatetimeIndex(times)
    times = [t for t in times if t.month in [12, 1, 2]]
    for time in times:
        year_str = get_year_str(time)
        for field in options.fields:
            url = (
                f"{base_url}/{options.version}/{year_str}/{field}/"
                f"{field}_WRF_Maritime_Continent_4km_"
                f"{time.year:04}{time.month:02}{time.day:02}.nc"
            )
            urls[field].append(url)
    return urls, times


def get_year_str(time):
    """
    Get the year string.

    Parameters
    ----------
    time : np.datetime64
        The time.

    Returns
    -------
    year_str : str
        The year string.
    """

    time = pd.Timestamp(time)

    if time.month == 12:
        summer_years = f"{time.year:04}{time.year+1:04}"
    else:
        summer_years = f"{time.year-1:04}{time.year:04}"

    return summer_years
