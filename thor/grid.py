"""Deal with thor grid objects"""

from pathlib import Path
import yaml
import inspect


def create_options(
    name="cartesian",
    timestep=None,
    start_latitude=None,
    end_latitude=None,
    start_longitude=None,
    end_longitude=None,
    central_latitude=None,
    central_longitude=None,
    projection=None,
    start_x=-150e3,
    end_x=-150e3,
    start_y=-150e3,
    end_y=150e3,
    start_z=0,
    end_z=25e3,
    grid_size=(500, 2500, 2500),
    save=True,
    **kwargs,
):
    """
    Generate grid options dictionary.

    Parameters
    ----------
    timestep : int, optional
        Time step for the dataset; default is None.
    start_latitude : float, optional
        Starting latitude for the dataset; default is None.
    end_latitude : float, optional
        Ending latitude for the dataset; default is None.
    start_longitude : float, optional
        Starting longitude for the dataset; default is None.
    end_longitude : float, optional
        Ending longitude for the dataset; default is None.

    Returns
    -------
    options : dict
        Dictionary containing the grid options.
    """

    options = {
        "timestep": timestep,
        "start_latitude": start_latitude,
        "end_latitude": end_latitude,
        "start_longitude": start_longitude,
        "end_longitude": end_longitude,
        "central_latitude": central_latitude,
        "central_longitude": central_longitude,
        "projection": projection,
        "start_x": start_x,
        "end_x": end_x,
        "start_y": start_y,
        "end_y": end_y,
        "start_z": start_z,
        "end_z": end_z,
        "grid_size": grid_size,
    }

    for key, value in kwargs.items():
        options[key] = value

    if save:
        filepath = Path(__file__).parent / "default/grid.yaml"
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

    return options
