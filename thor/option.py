"""Methods for creating and modifying default tracking configurations."""

import yaml
from pathlib import Path
import numpy as np
from thor.utils import now_str, check_component_options
from thor.config import get_outputs_directory
from thor.log import setup_logger


logger = setup_logger(__name__)


# Tracking scheme configurations.
def tint_options(
    search_margin=10,  # km
    local_flow_margin=10,  # km
    global_flow_margin=150,  # km
    unique_global_flow=True,
    max_cost=1e5,
    max_velocity_mag=60,  # m/s
    max_velocity_diff=60,  # m/s
    global_shift_altitude=1500,
):
    """
    Set options for the TINT tracking algorithm.

    Parameters
    ----------
    search_margin : int, optional
        Margin for object matching. Does not affect flow vectors.
    local_flow_margin : int, optional
        Margin around object for phase correlation.
    max_cost : int, optional
        Maximum allowable matching disparity score. Units of km.
    max_velocity_mag : int, optional
        Maximum allowable global shift magnitude.
    max_shift_disp : int, optional
        Maximum magnitude of shift difference.
    global_shift_altitude : int, optional
        Altitude in m for calculating global shift.
    altitudes : list, optional
        Altitudes over which to detect objects. Range defined by two element list [a,b],
        with included altitudes then a <= z < b. If None, use all altitudes.

    Returns
    -------
    dict
        A dictionary of TINT options.
    """

    options = {
        "search_margin": search_margin,
        "local_flow_margin": local_flow_margin,
        "global_flow_margin": global_flow_margin,
        "unique_global_flow": unique_global_flow,
        "max_cost": max_cost,
        "max_velocity_mag": max_velocity_mag,
        "max_velocity_diff": max_velocity_diff,
        "global_shift_altitude": global_shift_altitude,
    }
    return options


def mint_options(
    search_margin=25,  # km
    local_flow_margin=35,  # km
    global_flow_margin=150,  # km
    unique_global_flow=True,
    max_velocity_mag=60,
    max_velocity_diff=60,  # m/s
    max_velocity_diff_alt=25,  # m/s
):
    """
    Set options for the MINT tracking algorithm.

    Parameters
    ----------
    search_margin : int, optional
        Margin for object matching, does not affect flow vectors. Defaults to 50000.
    local_flow_margin : int, optional
        Margin around object for phase correlation. Defaults to 40000.
    max_velocity_mag : int, optional
        Maximum allowable global shift magnitude. Defaults to 60.
    max_disparity : int, optional
        Maximum allowable disparity value. Defaults to 999.
    max_shift_disp : int, optional
        Maximum magnitude of shift difference. Defaults to 60.
    alt_max_shift_disp : int, optional
        Alternative maximum magnitude of shift difference. Defaults to 25.
    global_shift_altitude : int, optional
        Altitude in m for calculating global shift. Defaults to 2000.

    Returns
    -------
    dict
        A dictionary of MINT options.
    """
    options = {
        **tint_options(
            search_margin,
            local_flow_margin,
            global_flow_margin,
            unique_global_flow,
            max_velocity_mag,
            max_velocity_diff,
        ),
        "max_velocity_diff_alt": max_velocity_diff_alt,
    }
    return options


def boilerplate_object(
    name,
    hierarchy_level,
    method="detect",
    mask_options=None,
    deque_length=2,
    tags=None,
):
    """THOR object boilerplate.

    Parameters
    ----------
    name : str
        Name of object to track.
    input_method : str
        Method used to input object data.
    hierarchy_level : int
        Hierarchy level of object.
    deque_length : int, optional
        How many previous scans to store when tracking.

    Returns
    -------
    options : dict
        Dictionary of boilerplate configuration options.
    """

    if mask_options is None:
        mask_options = {"save": True, "load": False}
    else:
        check_component_options(mask_options)

    options = {
        "name": name,
        "hierarchy_level": hierarchy_level,
        "method": method,
        "deque_length": deque_length,
        "mask_options": mask_options,
        "tags": tags,
    }
    return options


def detected_object(
    name,
    dataset,
    variable,
    hierarchy_level,
    detection_method,
    tracking_method,
    flatten_method="vertical_max",
    altitudes=None,
    min_area=50,
    tags=None,
):
    """Initialize THOR object configuration for detected objects, i.e.
    objects at the lowest hierarchy level.

    Parameters
    ----------
    name : str
        Name of object to track.
    hierarchy_level : int
        Hierarchy level of object.
    detection_method : str or None
        Method used to detect object.
    tracking_method : str or None
        Method used to track object.
    altitudes : list, optional
        Altitudes over which to detect objects.
    min_area : int, optional
        Minimum area of object in km squared.


    Returns
    -------
    options : dict
        Dictionary of global configuration options.
    """

    options = {
        **boilerplate_object(name, hierarchy_level, tags=tags),
        "dataset": dataset,
        "variable": variable,
        "detection": {
            "method": detection_method,
            "altitudes": altitudes,
            "flatten_method": flatten_method,
            "min_area": min_area,
        },
        "tracking": {"method": tracking_method},
    }

    return options


def grouped_object(
    name,
    dataset,
    member_objects,
    member_levels,
    member_min_areas,
    hierarchy_level,
    grouping_method,
    tracking_method,
    tags=None,
    **kwargs,
):
    """Initialize THOR object configuration for grouped objects, i.e.
    objects at higher hierarchy levels.

    Parameters
    ----------
    name : str
        Name of object to track.
    dataset : str
        Name of dataset to use for plotting.
    member_objects : list
        List of objects to group. Order matters. E.g. list objects by
        increasing altitude.
    member_levels : list
        List of hierarchy levels of the objects to group.
    member_min_areas: list
        List of minimum area (after grouping) for each member object
    hierarchy_level : int
        Hierarchy level of object.
    grouping_method : str or None
        Method used to group objects.
    tracking_method : str or None
        Method used to track object.


    Returns
    -------
    options : dict
        Dictionary of global configuration options.
    """
    if not all([member_level < hierarchy_level for member_level in member_levels]):
        raise ValueError(
            "Member object hierarchy levels must be less than grouped object level."
        )

    mask_options = {"save": True, "load": False}

    options = {
        **boilerplate_object(
            name, hierarchy_level, method="group", tags=tags, mask_options=mask_options
        ),
        "dataset": dataset,
        "grouping": {
            "method": grouping_method,
            "member_objects": member_objects,
            "member_levels": member_levels,
            "member_min_areas": member_min_areas,
        },
        "tracking": {"method": tracking_method, "options": mint_options(**kwargs)},
    }

    options["tracking"]["options"]["matched_object"] = "cell"

    return options


# Hierarchy level 0 object configurations.
def cell_object(
    name="cell",
    dataset="cpol",
    variable="reflectivity",
    hierarchy_level=0,
    detection_method="steiner",
    flatten_method="vertical_max",
    threshold=None,
    tracking_method="tint",
    altitudes=[500, 3e3],
    min_area=10,
    tags=None,
    **kwargs,
):
    """Creates default THOR configuration for tracking cells.

    Parameters
    ----------
    name : str
        Name of object to track.
    dataset : str
        Name of dataset to use for plotting.
    variable : str
        Variable to use for detection.
    filename : str
        Name of file to write configuration to.

    Returns
    -------
    options : dict
        Dictionary of default configuration options.
    """

    options = detected_object(
        name,
        dataset,
        variable,
        hierarchy_level,
        detection_method,
        tracking_method,
        altitudes=altitudes,
        flatten_method=flatten_method,
        tags=tags,
        min_area=min_area,
    )
    if threshold:
        options["detection"]["threshold"] = threshold
    if tracking_method == "tint":
        options["tracking"]["options"] = tint_options(**kwargs)
    elif tracking_method == "mint":
        options["tracking"]["options"] = mint_options(**kwargs)

    return options


def anvil_object(
    name="anvil",
    dataset="cpol",
    variable="reflectivity",
    hierarchy_level=0,
    detection_method="threshold",
    threshold=15,
    tracking_method="tint",
    altitudes=None,
    min_area=200,
    tags=None,
    **kwargs,
):
    """Creates default THOR configuration for tracking anvils.

    Parameters
    ----------
    filename : str
        Name of file to write configuration to.

    Returns
    -------
    options : dict
        Dictionary of default configuration options.
    """

    options = detected_object(
        name,
        dataset,
        variable,
        hierarchy_level,
        detection_method,
        tracking_method,
        min_area=min_area,
        tags=tags,
    )
    if threshold:
        options["detection"]["threshold"] = threshold
    options["detection"]["altitudes"] = altitudes
    if tracking_method == "tint":
        options["tracking"]["options"] = tint_options(**kwargs)
    elif tracking_method == "mint":
        options["tracking"]["options"] = mint_options(**kwargs)

    return options


# Hierarchy level 1 object configurations.
def mcs_object(
    dataset,
    name="mcs",
    member_objects=["cell", "middle_cloud", "anvil"],
    member_levels=[0, 0, 0],
    member_min_areas=[80, 400, 800],  # km^2
    hierarchy_level=1,
    grouping_method="graph",
    tracking_method="mint",
    tags=None,
    **kwargs,
):
    """Creates default THOR configuration for tracking MCSs.

    Parameters
    ----------
    filename : str
        Name of file to write configuration to.

    Returns
    -------
    options : dict
        Dictionary of default configuration options.
    """

    options = grouped_object(
        name,
        dataset,
        member_objects,
        member_levels,
        member_min_areas,
        hierarchy_level,
        grouping_method,
        tracking_method,
        tags=tags,
        **kwargs,
    )
    return options


# Consolidated configurations.
def cell(dataset, tags=None, **kwargs):
    """Creates default THOR configuration for tracking convective cells.

    Parameters
    ----------
    filename : str
        Name of file to write configuration to.

    Returns
    -------
    options : dict
        Dictionary of default configuration options.
    """

    options = [{"cell": cell_object(dataset=dataset, tags=tags, **kwargs)}]

    return options


def anvil(dataset, tags=None, **kwargs):
    """Creates default THOR configuration for tracking stratiform anvils.

    Parameters
    ----------
    filename : str
        Name of file to write configuration to.

    Returns
    -------
    options : dict
        Dictionary of default configuration options.
    """

    options = [{"anvil": anvil_object(dataset=dataset, tags=tags, **kwargs)}]

    return options


def mcs(dataset, tags=None, **kwargs):
    """Creates default THOR configuration for tracking MCSs.

    Parameters
    ----------
    filename : str
        Name of file to write configuration to.

    Returns
    -------
    options : dict
        Dictionary of default configuration options.
    """

    options = [
        {
            "cell": cell_object(
                altitudes=[500, 3000],
                dataset=dataset,
                flatten_method="vertical_max",
                threshold=40,
                tracking_method="mint",
                **kwargs,
            ),
            "middle_cloud": cell_object(
                name="middle_cloud",
                dataset=dataset,
                threshold=20,
                tracking_method=None,
                detection_method="threshold",
                flatten_method="vertical_max",
                altitudes=[3500, 7000],
                **kwargs,
            ),
            "anvil": anvil_object(
                altitudes=[7500, 10000],
                dataset=dataset,
                tracking_method="mint",
                **kwargs,
            ),
        },
        {"mcs": mcs_object(tags=tags, dataset=dataset, **kwargs)},
    ]

    return options


def check_options(options):
    """
    Check the tracking options.

    Parameters
    ----------
    options : dict
        Dictionary containing the input options.

    Returns
    -------
    options : dict
        Dictionary containing the input options.
    """

    object_names = []
    for level_options in options:
        for object_options in level_options.values():
            object_names.append(object_options["name"])
            if "global_flow_margin" in object_options.keys():
                if options["global_flow_margin"] > 5e3:
                    raise ValueError("Global flow radius must be less than 5000 km.")
    if len(object_names) != len(list(set(object_names))):
        raise ValueError("Object names must be unique.")

    return options


def save_track_options(
    options, filename=None, options_directory=None, append_time=False
):

    if options_directory is None:
        options_directory = get_outputs_directory() / "options/track_options"
    if filename is None:
        filename = "track_options"
        append_time = True
    save_options(options, filename, options_directory, append_time)


def save_options(options, filename=None, options_directory=None, append_time=False):

    if filename is None:
        filename = now_str()
        append_time = False
    else:
        filename = Path(filename).stem
    if append_time:
        filename += f"_{now_str()}"
    filename += ".yaml"
    if options_directory is None:
        options_directory = get_outputs_directory() / "options"
    if not options_directory.exists():
        options_directory.mkdir(parents=True)
    filepath = options_directory / filename
    logger.debug("Saving options to %s", options_directory / filename)
    with open(filepath, "w") as outfile:
        yaml.dump(
            options,
            outfile,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )


def consolidate_options(options_list):
    """Consolidate the options into a dictionary."""
    consolidated_options = {}
    for options in options_list:
        consolidated_options[options["name"]] = options
    return consolidated_options
