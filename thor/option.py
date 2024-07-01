"""Methods for creating and modifying default tracking configurations."""

import yaml
from pathlib import Path
from thor.utils import now_str, check_component_options
from thor.config import get_outputs_directory
from thor.log import setup_logger


logger = setup_logger(__name__)


# Tracking scheme configurations.
def tint_options(
    search_margin=4000,
    flow_margin=40000,
    max_disparity=999,
    max_flow_mag=50,
    max_shift_disparity=15,
    global_shift_altitude=1500,
):
    """
    Set options for the TINT tracking algorithm.

    Parameters
    ----------
    search_margin : int, optional
        Margin for object matching. Does not affect flow vectors.
    flow_margin : int, optional
        Margin around object for phase correlation.
    max_disparity : int, optional
        Maximum allowable matching disparity score.
    max_flow_mag : int, optional
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
        "flow_margin": flow_margin,
        "max_disparity": max_disparity,
        "max_flow_mag": max_flow_mag,
        "max_shift_disparity": max_shift_disparity,
        "global_shift_altitude": global_shift_altitude,
    }
    return options


def mint_options(
    search_margin=50000,
    flow_margin=40000,
    max_flow_mag=60,
    max_disparity=999,
    max_shift_disparity=60,
    alt_max_shift_disparity=25,
    global_shift_altitude=2000,
):
    """
    Set options for the MINT tracking algorithm.

    Parameters
    ----------
    search_margin : int, optional
        Margin for object matching, does not affect flow vectors. Defaults to 50000.
    flow_margin : int, optional
        Margin around object for phase correlation. Defaults to 40000.
    max_flow_mag : int, optional
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
            flow_margin,
            max_flow_mag,
            max_disparity,
            max_shift_disparity,
            global_shift_altitude,
        ),
        "max_shift_disp_alt": alt_max_shift_disparity,
    }
    return options


def boilerplate_object(
    name,
    hierarchy_level,
    method="detect",
    mask_options=None,
    deque_length=2,
    tags=None,
    display=False,
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
        "display": display,
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
    min_area=10,
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
    hierarchy_level,
    grouping_method,
    tracking_method,
    tags=None,
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

    mask_options = {"save": False, "load": False}

    options = {
        **boilerplate_object(
            name, hierarchy_level, method="group", tags=tags, mask_options=mask_options
        ),
        "dataset": dataset,
        "grouping": {
            "method": grouping_method,
            "member_objects": member_objects,
            "member_levels": member_levels,
        },
        "tracking": {"method": tracking_method, "options": mint_options()},
    }

    return options


# Hierarchy level 0 object configurations.
def cell_object(
    name="cell",
    dataset="cpol",
    variable="reflectivity",
    hierarchy_level=0,
    detection_method="steiner",
    flatten_method="cross_section",
    threshold=None,
    tracking_method="tint",
    global_shift_altitude=2000,
    altitudes=[3e3],
    min_area=20,
    tags=None,
):
    """Creates default THOR configuration for tracking cells.

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
        altitudes=altitudes,
        flatten_method=flatten_method,
        tags=tags,
        min_area=min_area,
    )
    if threshold:
        options["detection"]["threshold"] = threshold
    if tracking_method == "tint":
        options["tracking"]["options"] = tint_options(
            global_shift_altitude=global_shift_altitude
        )

    return options


def anvil_object(
    name="anvil",
    dataset="cpol",
    variable="reflectivity",
    hierarchy_level=0,
    detection_method="threshold",
    threshold=15,
    tracking_method="tint",
    global_shift_altitude=8000,
    altitudes=None,
    min_area=50,
    tags=None,
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
        options["tracking"]["options"] = tint_options(
            global_shift_altitude=global_shift_altitude
        )

    return options


# Hierarchy level 1 object configurations.
def mcs_object(
    dataset,
    name="mcs",
    member_objects=["cell", "middle_cloud", "anvil"],
    member_levels=[0, 0, 0],
    hierarchy_level=1,
    grouping_method="graph",
    tracking_method="mint",
    tags=None,
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
        hierarchy_level,
        grouping_method,
        tracking_method,
        tags=tags,
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
                altitudes=[3000], dataset=dataset, flatten_method="cross_section"
            ),
            "middle_cloud": cell_object(
                name="middle_cloud",
                dataset=dataset,
                threshold=20,
                tracking_method=None,
                detection_method="threshold",
                flatten_method="vertical_max",
                altitudes=[3500, 7500],
            ),
            "anvil": anvil_object(altitudes=[7500, 10000], dataset=dataset),
        },
        {"mcs": mcs_object(tags=tags, dataset=dataset)},
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

    # check options of component objects etc

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
