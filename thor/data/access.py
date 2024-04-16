"""Process ACCESS data."""

import numpy as np
import pandas as pd
from thor.log import setup_logger
from thor.data.utils import download_file, unzip_file, consolidate_netcdf


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
    """

    start = np.datetime64(options["start"])
    end = np.datetime64(options["end"])

    urls = []

    # base_url = f"{options['parent']}/cpol"

    # if options["level"] == "1b":
