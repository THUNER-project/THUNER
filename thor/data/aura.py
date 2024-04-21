"""Process AURA data."""

import numpy as np
import pandas as pd
from thor.log import setup_logger
from thor.data.odim import convert_odim
from thor.data.utils import download_file, unzip_file, consolidate_netcdf
from thor.utils import format_string_list, drop_time
from pathlib import Path
import yaml
import inspect
from urllib.parse import urlparse


logger = setup_logger(__name__)


def create_options(
    name="operational",
    start="2005-02-01T00:00:00",
    end="2005-02-02T00:00:00",
    level="1",
    radar="63",
    format="ODIM",
    parent="https://dapds00.nci.org.au/thredds/fileServer/rq0",
    fields=["reflectivity", "reflectivity_horizontal"],
    weighting_function="Barnes2",
    save=False,
    **kwargs,
):
    """
    Generate input options dictionary.

    Parameters
    ----------
    name : str, optional
        The name of the dataset; default is "operational".
    level : str, optional
        The level of the dataset; default is "1".
    radar : str, optional
        The radar number; default is "63".
    format : str, optional
        The format of the dataset; default is "ODIM".
    parent : str, optional
        The parent URL; default is "https://dapds00.nci.org.au/thredds/fileServer/rq0".
    fields : list, optional
        The fields to include in the dataset; default is ["reflectivity", "reflectivity_horizontal"].
    weighting_function : str, optional
        The weighting function for pyart gridding; default is "Barnes2".
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
        "level": level,
        "radar": radar,
        "format": format,
        "parent": parent,
        "parent": parent,
        "fields": fields,
        "weighting_function": weighting_function,
        "save": save,
    }

    for key, value in kwargs.items():
        options[key] = value

    if save:
        filepath = Path(__file__).parent.parent / "option/default/aura.yaml"
        logger.debug(f"Saving options to {filepath}")
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

    names = ["cpol", "operational"]
    if options["name"] == "cpol":

        min_start = np.datetime64("1998-13-06T00:00:00")
        max_end = np.datetime64("2017-05-02T00:00:00")
        if np.datetime64(options["start"]) < min_start:
            raise ValueError(f"start must be {min_start} or later.")
        if np.datetime64(options["end"]) > max_end:
            raise ValueError(f"end must be {max_end} or earlier.")

        formats = ["grid_150km_2500m", "grid_70km_1000m"]
        if options["format"] not in formats:
            raise ValueError(f"format must be one of {format_string_list(formats)}.")
        levels = ["1", "1b", "2"]
        if options["level"] not in levels:
            raise ValueError(f"level must be one of {format_string_list(levels)}.")
    elif options["name"] == "operational":
        min_start = np.datetime64("1993-09-07")
        if np.datetime64(options["start"]) < min_start:
            raise ValueError(f"start must be {min_start} or later.")
        levels = ["1", "1b"]
        if options["level"] not in levels:
            raise ValueError(f"level must be one of {format_string_list(levels)}.")
    else:
        raise ValueError(f"name must be one of {format_string_list(names)}.")


def generate_cpol_urls(options):
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

    start = drop_time(options["start"])
    end = drop_time(options["end"])

    urls = []

    base_url = f"{options['parent']}/cpol"

    if options["level"] == "1b":

        times = np.arange(start, end + np.timedelta64(10, "m"), np.timedelta64(10, "m"))
        times = pd.DatetimeIndex(times)

        base_url += f"/cpol_level_1b/v{options['version']}/"
        if "grid" in options["format"]:
            base_url += f"gridded/{options['format']}/"
            if "150" in options["format"]:
                format_string = "grid150"
            else:
                format_string = "grid75"
        elif options["format"] == "ppi":
            base_url += "ppi/"
        for time in times:
            url = (
                f"{base_url}{time.year}/{time.year}{time.month:02}{time.day:02}/"
                f"twp10cpol{format_string}.b2.{time.year}{time.month:02}{time.day:02}."
                f"{time.hour:02}{time.minute:02}{time.second:02}.nc"
            )
            urls.append(url)
    elif options["level"] == "2":

        times = np.arange(
            start.astype("datetime64[D]"),
            end.astype("datetime64[D]") + np.timedelta64(1, "D"),
            np.timedelta64(1, "D"),
        )
        times = pd.DatetimeIndex(times)

        base_url += f"/cpol_level_2/v{options['version']}/{options['format']}"
        try:
            variable = options["variable"]
            if variable == "equivalent_reflectivity_factor":
                variable_short = "reflectivity"
        except KeyError:
            variable = "equivalent_reflectivity_factor"
            variable_short = "reflectivity"

        base_url += f"/{variable}"

        for time in times:
            url = f"{base_url}/twp1440cpol.{variable_short}.c1.{time.year}{time.month:02}{time.day:02}.nc"
            urls.append(url)

    return urls, times


def generate_operational_urls(options):
    """
    Generate operational radar URLs from input options dictionary. Note level 1 are
    zipped ODIM files, level 1b are zipped netcdf files.

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

    start = np.datetime64(options["start"])
    end = np.datetime64(options["end"])

    urls = []
    base_url = f"{options['parent']}"

    times = np.arange(start, end + np.timedelta64(1, "D"), np.timedelta64(1, "D"))
    times = pd.DatetimeIndex(times)

    if options["level"] == "1":

        base_url += f"/{options['radar']}"

        for time in times:
            url = (
                f"{base_url}/{time.year:04}/vol/{options['radar']}"
                f"_{time.year}{time.month:02}{time.day:02}.pvol.zip"
            )
            urls.append(url)
    elif options["level"] == "1b":

        base_url += f"/level_1b/{options['radar']}/grid"

        for time in times:
            url = (
                f"{base_url}/{time.year:04}/{options['radar']}"
                f"_{time.year}{time.month:02}{time.day:02}_grid.zip"
            )
            urls.append(url)

    return urls, times


def setup_operational(data_options, grid_options, url, directory):
    """
    Setup operational radar data for a given date.

    Parameters
    ----------
    options : dict
        Dictionary containing the input options.
    url : str
        The URL where the radar data can be found.
    directory : str
        Where to extract the zip file and save the netCDF.

    Returns
    -------
    dataset : object
        The processed radar data.
    """

    if "http" in urlparse(url).scheme:
        filepath = download_file(url, directory)
    else:
        filepath = url
    extracted_filepaths = unzip_file(filepath)[0]
    if data_options["level"] == "1":
        dataset = convert_odim(
            extracted_filepaths,
            data_options,
            grid_options,
            out_dir=directory,
        )
    elif data_options["level"] == "1b":
        dataset = consolidate_netcdf(
            extracted_filepaths, fields=data_options["fields"], concat_dim="time"
        )

    return dataset
