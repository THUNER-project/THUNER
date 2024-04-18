"""Module for generating input configuration files."""

from pathlib import Path
import yaml


def input_options(
    name="cpol",
    start="2005-11-13T12:00:00",
    end="2005-11-13T22:00:00",
    format="grid_150km_2500m",
    timestep=None,
    start_latitude=None,
    end_latitude=None,
    start_longitude=None,
    end_longitude=None,
    save=True,
    parent="https://dapds00.nci.org.au/thredds/dodsC/hj10",
    **kwargs,
):
    """
    Generate input options dictionary.

    Parameters
    ----------
    name : str, optional
        Name of the dataset; see input.py. Default is "CPOL".
    start : np.datetime64, optional
        Start time of the dataset. Default is np.datetime64("2005-11-13T12:00:00").
    end : np.datetime64, optional
        End time of the dataset. Default is np.datetime64("2005-11-13T22:00:00").
    level : str, optional
        Level of the dataset. Default is "1b".
    version : str, optional
        Version of the dataset. Default is "2020".
    format : str, optional
        Format of the dataset. Default is "gridded".
    timestep : np.timedelta64, optional
        Time interval between each time step. Default is None.
    start_latitude : float, optional
        Latitude of the starting point. Default is None.
    end_latitude : float, optional
        Latitude of the ending point. Default is None.
    start_longitude : float, optional
        Longitude of the starting point. Default is None.
    end_longitude : float, optional
        Longitude of the ending point. Default is None.
    parent : str, optional
        Parent directory of the dataset. If None, will default

    Returns
    -------
    options : dict
        Dictionary containing the input options.
    """

    options = {
        "name": name,
        "start": start,
        "end": end,
        "format": format,
        "timestep": timestep,
        "start_latitude": start_latitude,
        "end_latitude": end_latitude,
        "start_longitude": start_longitude,
        "end_longitude": end_longitude,
        "parent": parent,
    }

    for key, value in kwargs.items():
        options[key] = value

    if save:
        filepath = Path(__file__).parent / "default/input.yaml"
        with open(filepath, "w") as outfile:
            yaml.dump(
                options,
                outfile,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )

    return options
