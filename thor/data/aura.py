"""Process AURA data."""

import numpy as np
import pandas as pd
from thor.log import setup_logger

logger = setup_logger(__name__)


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
    """

    start = np.datetime64(options["start"])
    end = np.datetime64(options["end"])

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

    return urls


def generate_operational_urls(options):
    """
    Generate operational radar URLs from input options dictionary.

    Parameters
    ----------
    options : dict
        Dictionary containing the input options.

    Returns
    -------
    urls : list
        List of URLs.
    """

    return
