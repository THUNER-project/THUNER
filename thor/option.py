"""Functions for creating and modifying default tracking configurations."""

import yaml
import copy
from pathlib import Path
import numpy as np
from thor.utils import now_str, check_component_options
from thor.config import get_outputs_directory
from thor.log import setup_logger
import thor.attribute as attribute


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
    search_margin=35,  # km
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
    write_interval=1,
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
    tags : list, optional
        List of tags to apply to object.
    mask_options : dict, optional
        Dictionary of mask options.
    write_interval : int, optional
        Interval at which to write data to disk in units of hours.

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
        "write_interval": write_interval,
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
    attribute_options=None,
    **kwargs,
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

    profile_dataset = kwargs.get("profile_dataset", "era5_pl")
    tag_dataset = kwargs.get("tag_dataset", "era5_sl")

    if attribute_options is None:
        attribute_options = {"core": attribute.core.default()}
        # attribute_options.update(
        #     {"profile": attribute.profile.default([profile_dataset])}
        # )
        # attribute_options.update({"tag": attribute.tag.default([tag_dataset])})
        attribute_options.update({"quality": attribute.quality.default()})

    options = {
        **boilerplate_object(name, hierarchy_level),
        "dataset": dataset,
        "variable": variable,
        "detection": {
            "method": detection_method,
            "altitudes": altitudes,
            "flatten_method": flatten_method,
            "min_area": min_area,
        },
        "tracking": {"method": tracking_method},
        "attributes": attribute_options,
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
    matched_object=None,
    attribute_options=None,
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
        message = "Member hierarchy levels must be less than grouped object level."
        raise ValueError(message)

    mask_options = {"save": True, "load": False}

    # Let "core" be an attribute option for "group" attributes, with
    # the "core" attribute then being a dictionary containing the core attributes of
    # each member object.
    if attribute_options is None:
        core_tracked = attribute.core.default(tracked=True, matched=True)
        core_untracked = attribute.core.default(tracked=False, matched=True)
        # Note attributes for grouped objects are specified slightly differently
        # than for detected objects; the dictionary has an extra layer of nesting to
        # separate the attributes for member objects and for the grouped object.
        attribute_options = {"member_objects": {}, name: {}}
        member_options = attribute_options["member_objects"]
        # By default assume that the first member object is the matched/tracked object.
        member_options[member_objects[0]] = {}
        member_options[member_objects[0]]["core"] = core_tracked
        member_options[member_objects[0]]["quality"] = attribute.quality.default()
        member_options[member_objects[0]]["ellipse"] = attribute.ellipse.default()
        for i in range(1, len(member_objects)):
            member_options[member_objects[i]] = {}
            member_options[member_objects[i]]["core"] = core_untracked
            member_options[member_objects[i]]["quality"] = attribute.quality.default()
            # member_options[member_objects[0]]["ellipse"] = attribute.ellipse.default()
        # Define the attributes for the grouped object.
        attribute_options[name]["core"] = core_tracked
        attribute_options[name]["group"] = attribute.group.default()
        profile_dataset = kwargs.get("profile_dataset", "era5_pl")
        tag_dataset = kwargs.get("tag_dataset", "era5_sl")
        attribute_options[name]["profile"] = attribute.profile.default(
            [profile_dataset]
        )
        attribute_options[name]["tag"] = attribute.tag.default([tag_dataset])

    options = {
        **boilerplate_object(
            name, hierarchy_level, method="group", mask_options=mask_options
        ),
        "dataset": dataset,
        "grouping": {
            "method": grouping_method,
            "member_objects": member_objects,
            "member_levels": member_levels,
            "member_min_areas": member_min_areas,
        },
        "tracking": {"method": tracking_method, "options": mint_options(**kwargs)},
        "attributes": attribute_options,
    }

    # If no matched object specified, assume first member object used for matching.
    if matched_object is None:
        matched_object = member_objects[0]
    options["tracking"]["options"]["matched_object"] = matched_object

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
    attribute_options=None,
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
        min_area=min_area,
        attribute_options=attribute_options,
        **kwargs,
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
    attribute_options=None,
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
        attribute_options=attribute_options,
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
    member_objects=["cell", "middle_echo", "anvil"],
    member_levels=[0, 0, 0],
    member_min_areas=[80, 400, 800],  # km^2
    hierarchy_level=1,
    grouping_method="graph",
    tracking_method="mint",
    attribute_options=None,
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
        attribute_options=attribute_options,
        **kwargs,
    )
    return options


# Consolidated configurations.
def cell(dataset, **kwargs):
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

    options = [{"cell": cell_object(dataset=dataset, **kwargs)}]

    return options


def anvil(dataset, **kwargs):
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

    options = [{"anvil": anvil_object(dataset=dataset, **kwargs)}]

    return options


def mcs(dataset, **kwargs):
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

    # Create the attribute dictionary for the unmatched/untracked middle_echo objects.
    # For the cell and anvil objects, attributes are obtained from matching.
    untracked_attr_options = {"core": attribute.core.default(tracked=False)}
    untracked_attr_options.update({"quality": attribute.quality.default(matched=False)})

    options = [
        {
            "cell": cell_object(
                altitudes=[500, 3000],
                dataset=dataset,
                flatten_method="vertical_max",
                threshold=40,
                tracking_method=None,
                attribute_options=untracked_attr_options,
                **kwargs,
            ),
            "middle_echo": cell_object(
                name="middle_echo",
                dataset=dataset,
                threshold=20,
                tracking_method=None,
                detection_method="threshold",
                flatten_method="vertical_max",
                altitudes=[3500, 7000],
                attribute_options=untracked_attr_options,
                **kwargs,
            ),
            "anvil": anvil_object(
                altitudes=[7500, 10000],
                dataset=dataset,
                tracking_method=None,
                attribute_options=untracked_attr_options,
                **kwargs,
            ),
        },
        {"mcs": mcs_object(dataset=dataset, **kwargs)},
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
    options, options_directory=None, filename="track", append_time=False
):
    """Save the tracking options to a yml file."""
    # Create a copy so we can drop the attributes info - this is stored in the attributes
    # metadata instead.
    options = copy.deepcopy(options)
    for level_options in options:
        for object_options in level_options.values():
            object_options.pop("attributes", None)

    if options_directory is None:
        options_directory = get_outputs_directory() / "options/track"
    save_options(options, filename, options_directory, append_time)


def save_options(options, filename=None, options_directory=None, append_time=False):
    """Save the options to a yml file."""
    if filename is None:
        filename = now_str()
        append_time = False
    else:
        filename = Path(filename).stem
    if append_time:
        filename += f"_{now_str()}"
    filename += ".yml"
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
