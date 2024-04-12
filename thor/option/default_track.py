"""Methods for creating and modifying default tracking configurations."""

import yaml
from pathlib import Path


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


# Trackable object default configurations.
def boilerplate_object(name, hierarchy_level):
    """THOR object boilerplate.

    Parameters
    ----------
    name : str
        Name of object to track.
    input_method : str
        Method used to input object data.
    hierarchy_level : int
        Hierarchy level of object.

    Returns
    -------
    options : dict
        Dictionary of boilerplate configuration options.
    """

    options = {
        "name": name,
        "hierarchy_level": hierarchy_level,
    }
    return options


def detected_object(
    name,
    input_method,
    hierarchy_level,
    detection_method,
    tracking_method,
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

    Returns
    -------
    options : dict
        Dictionary of global configuration options.
    """

    options = {
        **boilerplate_object(name, hierarchy_level),
        "input": {"method": input_method},
        "detection": {"method": detection_method, "altitudes": None},
        "tracking": {"method": tracking_method},
    }

    return options


def grouped_object(
    name,
    member_objects,
    hierarchy_level,
    grouping_method,
    tracking_method,
):
    """Initialize THOR object configuration for grouped objects, i.e.
    objects at higher hierarchy levels.

    Parameters
    ----------
    name : str
        Name of object to track.
    member_objects : list
        List of objects to group.
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

    options = {
        **boilerplate_object(name, hierarchy_level),
        "grouping": {"method": grouping_method, "member_objects": member_objects},
        "tracking": {"method": tracking_method, "options": mint_options()},
    }

    return options


# Hierarchy level 0 object configurations.
def cell_object(
    name="cell",
    input_method="filenames",
    hierarchy_level=0,
    detection_method="steiner",
    tracking_method="tint",
    global_shift_altitude=2000,
    altitudes=None,
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
        name, input_method, hierarchy_level, detection_method, tracking_method
    )
    options["detection"]["altitudes"] = altitudes
    if tracking_method == "tint":
        options["tracking"]["options"] = tint_options(
            global_shift_altitude=global_shift_altitude
        )

    return options


def anvil_object(
    name="anvil",
    input_method="filenames",
    hierarchy_level=0,
    detection_method="threshold",
    threshold=15,
    tracking_method="tint",
    global_shift_altitude=8000,
    altitudes=None,
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
        name, input_method, hierarchy_level, detection_method, tracking_method
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
    name="mcs",
    member_objects=["cell", "anvil"],
    hierarchy_level=1,
    grouping_method="graph",
    tracking_method="mint",
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
        name, member_objects, hierarchy_level, grouping_method, tracking_method
    )
    return options


# Consolidated configurations.
def cell(save=True, **kwargs):
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

    options = [{"cell": cell_object(**kwargs)}]

    if save:
        filepath = Path(__file__).parent / "default/cell.yaml"
        with open(filepath, "w") as outfile:
            yaml.dump(
                options,
                outfile,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )

    return options


def anvil(save=True, **kwargs):
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

    options = [{"anvil": anvil_object(**kwargs)}]

    if save:
        filepath = Path(__file__).parent / "default/anvil.yaml"
        with open(filepath, "w") as outfile:
            yaml.dump(
                options,
                outfile,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )

    return options


def mcs(save=True, **kwargs):
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
            "cell": cell_object(altitudes=[500, 3500]),
            "middle_cloud": cell_object(
                name="middle_cloud", tracking_method=None, altitudes=[3500, 7500]
            ),
            "anvil": anvil_object(altitudes=[7500, 10000]),
        },
        {"mcs": mcs_object()},
    ]

    if save:
        filepath = Path(__file__).parent / "default/mcs.yaml"
        with open(filepath, "w") as outfile:
            yaml.safe_dump(
                options,
                outfile,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )

    return options
