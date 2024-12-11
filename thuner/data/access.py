"""Process ACCESS data."""

import numpy as np
import pandas as pd
from thuner.log import setup_logger
from thuner.utils import format_string_list, drop_time
from thuner.config import get_outputs_directory
import yaml


logger = setup_logger(__name__)


def create_options(
    name="access",
    start="2022-02-01T00:00:00",
    end="2022-02-02T00:00:00",
    model="c",
    domain="dn",
    mode="fcmm",
    level="sfc",
    init_time="0000",
    parent_local="https://dapds00.nci.org.au/thredds/dodsC/wr45/ops_aps3",
    fields=["radar_refl_1km", "maxcol_refl", "uwnd10m", "vwnd10m"],
    save=False,
    **kwargs,
):
    """
    Generate input options dictionary.

    Parameters
    ----------
    name : str, optional
        Name of the dataset; default is "access".
    start : str, optional
        Start time of the dataset in ISO format; default is "2022-02-01T00:00:00".
    end : str, optional
        End time of the dataset in ISO format; default is "2022-02-02T00:00:00".
    model : str, optional
        Model type; default is "c".
    domain : str, optional
        Domain type; default is "dn".
    mode : str, optional
        Mode type; default is "fcmm".
    level : str, optional
        Level type; default is "sfc".
    init_time : str, optional
        Initialization time; default is "0000".
    parent : str, optional
        Parent URL for the dataset; default is "https://dapds00.nci.org.au/thredds/dodsC/wr45/ops_aps3".
    fields : list of str, optional
        List of fields to be included in the dataset; default is ["radar_refl_1km", "maxcol_refl", "uwnd10m", "vwnd10m"].

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
        "model": model,
        "domain": domain,
        "mode": mode,
        "level": level,
        "init_time": init_time,
        "parent_local": parent_local,
        "fields": fields,
    }

    for key, value in kwargs.items():
        options[key] = value

    if save:
        filepath = str(get_outputs_directory() / "option/access.yml")
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

    models = ["g", "ge", "c"]
    domains = ["bn", "ph", "ad", "vt", "sy", "dn"]
    modes = ["fc", "an", "fcmm"]
    if options.mode not in modes:
        raise ValueError(f"mode must be one of {format_string_list(modes)}.")

    levels = ["sfc", "ml"]
    if options["level"] not in levels:
        raise ValueError(f"level must be one of {format_string_list(levels)}.")

    start = np.datetime64(options["start"])
    min_start = np.datetime64("2019-07-23T01:00:00")
    if start < min_start:
        raise ValueError(f"start must be {min_start} or later.")

    if options["model"] not in models:
        raise ValueError(f"model must be one of {format_string_list(models)}.")

    if options["model"] == "C":
        if options["domain"] not in domains:
            raise ValueError(f"domain must be one of {format_string_list(domains)}.")
        init_times = ["0000", "0600", "1200", "1800"]
        if options["init_time"] not in init_times:
            raise ValueError(
                f"init_time must be one of {format_string_list(init_times)}."
            )


def generate_access_urls(options):
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

    domains = ["bn", "ph", "ad", "vt", "sy", "dn"]

    urls = dict(zip(options.fields, [[] for i in range(len(options.fields))]))

    if options["model"] == "g":
        base_url += f"/access-{options['model']}/1"
    elif options["model"] == "ge":
        raise ValueError("No reflectivity data for global ensemble model.")
    elif options["model"] == "c":
        if options["domain"] not in domains:
            raise ValueError(f"domain must be one of {format_string_list(domains)}.")
        base_url += f"/access-{options['domain']}/1"
    times = np.arange(start, end + np.timedelta64(1, "D"), np.timedelta64(1, "D"))
    times = pd.DatetimeIndex(times)
    for time in times:
        for field in options.fields:
            url = (
                f"{base_url}/{time.year:04}{time.month:02}{time.day:02}/"
                f"{options['init_time']}/{options.mode}"
                f"/{options['level']}/{field}.nc"
            )
            urls[field].append(url)
    return urls, times
