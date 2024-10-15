"""Functions for creating and modifying default tracking configurations."""

import yaml
import copy
from pathlib import Path
import numpy as np
from typing import Any, Dict, Optional, ClassVar, List, Annotated, Type
from pydantic import Field, BaseModel, field_validator, ValidationInfo, model_validator
from thor.utils import now_str, check_component_options
from thor.config import get_outputs_directory
from thor.log import setup_logger
import thor.attribute as attribute


logger = setup_logger(__name__)


def convert_value(value: Any) -> Any:
    """
    Convenience function to convert options attributes to types serializable as yaml.
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [convert_value(v) for v in value.tolist()]
    if isinstance(value, BaseOptions):
        fields = value.model_fields.keys()
        return {field: convert_value(getattr(value, field)) for field in fields}
    if isinstance(value, np.datetime64):
        return str(value)
    if isinstance(value, dict):
        return {convert_value(k): convert_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [convert_value(v) for v in value]
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, type):
        return value.__name__
    return value


class BaseOptions(BaseModel):
    """
    The base class for all options classes. This class is built on the pydantic
    BaseModel class, which is similar to python dataclasses but with type checking.
    """

    # Allow arbitrary types in the options classes.
    class Config:
        arbitrary_types_allowed = True

    # Ensure that floats in all options classes are np.float32
    @field_validator("*")
    def convert_floats(cls, v):
        if isinstance(v, float):
            return np.float32(v)
        return v

    def to_dict(self) -> Dict[str, Any]:
        fields = self.model_fields.keys()
        return {field: convert_value(getattr(self, field)) for field in fields}

    def from_dict(self, options_dict: Dict[str, Any]):
        for key, value in options_dict.items():
            setattr(self, key, convert_value(key, value))

    def to_yaml(self, filepath: str):
        with open(filepath, "w") as f:
            args_dict = {"default_flow_style": False, "allow_unicode": True}
            args_dict = {"sort_keys": False}
            yaml.dump(self.to_dict(), f, **args_dict)

    def from_yaml(self, filepath: str):
        with open(filepath, "r") as file:
            self.from_dict(yaml.safe_load(file))


class TintOptions(BaseOptions):
    """
    Options for the TINT tracking algorithm. See the following publications
    """

    name: str = Field("mint", description="Name of the tracking algorithm.")
    # Define convenience variable _description just to tidy up the code.
    # Specifying type ClassVar tells pydantic this variable isn't an instance variable.
    _description: ClassVar = "Margin in km for object matching. "
    _description += "Does not affect flow vectors."
    search_margin: float = Field(10, description=_description, gt=0)
    _description = "Margin in km around object for phase correlation."
    local_flow_margin: float = Field(10, description=_description, gt=0)
    _description = "Margin in km around object for global flow vectors."
    global_flow_margin: float = Field(150, description=_description, gt=0)
    _description = "If True, create unique global flow vectors for each object."
    unique_global_flow: bool = Field(True, description=_description)
    _description = "Maximum allowable matching cost. Units of km."
    max_cost: float = Field(1e3, description=_description, gt=0)
    _description = "Maximum allowable shift magnitude."
    max_velocity_mag: float = Field(60, description=_description, gt=0)
    _description = "Maximum allowable shift difference."
    max_velocity_diff: float = Field(60, description=_description, gt=0)


class MintOptions(TintOptions):
    """
    Options for the MINT tracking algorithm.
    """

    name: str = Field("mint", description="Name of the tracking algorithm.")
    _description: ClassVar = "Margin for object matching. Does not affect flow vectors."
    search_margin: int = Field(25, description=_description, gt=0)
    _description = "Margin around object for phase correlation."
    local_flow_margin: int = Field(35, description=_description, gt=0)
    _description = "Margin around object for global flow vectors."
    max_velocity_diff_alt: int = Field(25, description=_description, gt=0)


# Tracking scheme configurations.
def tint_options(
    search_margin=10,  # km
    local_flow_margin=10,  # km
    global_flow_margin=150,  # km
    unique_global_flow=True,
    max_cost=1e3,
    max_velocity_mag=60,  # m/s
    max_velocity_diff=60,  # m/s
    global_shift_altitude=1500,
):
    """
    Set options for the TINT tracking algorithm.

    Parameters
    ----------
    search_margin : int, optional
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
            search_margin=search_margin,
            local_flow_margin=local_flow_margin,
            global_flow_margin=global_flow_margin,
            unique_global_flow=unique_global_flow,
            max_velocity_mag=max_velocity_mag,
            max_velocity_diff=max_velocity_diff,
        ),
        "max_velocity_diff_alt": max_velocity_diff_alt,
    }
    return options


class MaskOptions(BaseOptions):
    """Options for saving and loading masks."""

    _description: ClassVar = "If True, save masks as .nc files."
    save: bool = Field(True, description=_description)
    _description = "If True, load masks from .nc files."
    load: bool = Field(False, description=_description)


class BaseObjectOptions(BaseOptions):
    """Base class for object options."""

    name: str = Field(..., description="Name of the object.")
    _description: ClassVar = "Level of the object in the hierachy. Higher level "
    _description += "objects depend on lower level objects."
    hierarchy_level: int = Field(..., description=_description, ge=0)
    _description = "Method used to obtain the object, e.g. detect or group."
    method: str = Field("detect", description=_description)
    _description = "Number of previous grids to store when tracking."
    deque_length: int = Field(2, description=_description, gt=0, lt=10)
    _description = "Options for saving and loading object masks."
    mask_options: MaskOptions = Field(MaskOptions(), description=_description)
    _description = "Interval at which to write data to disk in hours."
    write_interval: int = Field(1, description=_description, gt=0, lt=24 * 60)

    # Check method is either detect or group.
    @field_validator("method")
    def _check_method(cls, value):
        if value not in ["detect", "group"]:
            raise ValueError("Method must be detect or group.")


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


class TrackingOptions(BaseOptions):

    _description: ClassVar = "Method used to track the object."
    _description = "Name of object used for matching. Should be the name of the given "
    _description += "detected object, or the name of a member object comprisng a "
    _description += "grouped object."
    matched_object: str = Field(None, description=_description)


class AttributeTypeOptions(BaseOptions):
    pass


class DetectionOptions(BaseOptions):
    """Options for object detection."""

    _description: ClassVar = "Method used to detect the object."
    method: str = Field(..., description=_description)
    _description = "Altitudes over which to detect objects."
    altitudes: List[int] = Field([], description=_description)
    _description = "Method used to flatten the object."
    flatten_method: str = Field("vertical_max", description=_description)
    _description = "Minimum area of object in km squared."
    min_area: int = Field(10, description=_description)
    _description = "Threshold used for detection if required."
    threshold: Optional[int] = Field(None, description=_description)

    @field_validator("method")
    def _check_method(cls, value):
        if value not in ["steiner", "threshold"]:
            raise ValueError("Detection method must be detect or group.")
        return value

    @model_validator(mode="after")
    def _check_threshold(cls, values):
        if values.method == "detect" and values.threshold is None:
            raise ValueError("Threshold not provided for detection method.")
        return values


class DetectedObjectOptions(BaseObjectOptions):
    """Options for detected objects."""

    _description: ClassVar = "Name of the dataset used for detection."
    dataset: str = Field(..., description=_description, examples=["cpol", "gridrad"])
    _description = "Variable to use for detection."
    variable: str = Field("reflectivity", description=_description)
    _description = "Method used to detect the object."
    _args_dict: ClassVar = {"description": _description}
    _args: ClassVar = [DetectionOptions(method="steiner")]
    detection: DetectionOptions = Field(*_args, **_args_dict)
    tracking: BaseOptions = Field(TintOptions(), description="Tracking options.")
    attributes: Dict = Field({}, description="Options for object attributes.")


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
        attribute_options.update(
            {"profile": attribute.profile.default([profile_dataset])}
        )
        attribute_options.update({"tag": attribute.tag.default([tag_dataset])})
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


# Define a custom type with constraints
PositiveFloat = Annotated[float, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]


class GroupingOptions(BaseOptions):
    """Options class for grouping lower level objects into higher level objects."""

    method: str = Field("graph", description="Method used to group objects.")
    _description: ClassVar = "Names of objects to group."
    _args_dict: ClassVar = {"description": _description}
    member_objects: List[str] = Field([], **_args_dict)
    _description = "Hierachy levels of objects to group"
    _args_dict = {"description": _description}
    member_levels: List[NonNegativeInt] = Field([], **_args_dict)
    _description = "Minimum area of each member object in km squared."
    _args_dict = {"description": _description}
    member_min_areas: List[PositiveFloat] = Field([], **_args_dict)

    # Check lists are the same length.
    @model_validator(mode="after")
    def _check_list_length(cls, values):
        member_objects = values.member_objects
        member_levels = values.member_levels
        member_min_areas = values.member_min_areas
        lengths = [len(member_objects), len(member_levels), len(member_min_areas)]
        if len(set(lengths)) != 1:
            message = "Member objects, levels, and areas must have the same length."
            raise ValueError(message)
        return values


class GroupedObjectOptions(BaseObjectOptions):
    """Options for grouped objects."""

    _description: ClassVar = "Options for grouping objects."
    grouping: GroupingOptions = Field(GroupingOptions(), description=_description)
    tracking: Type[BaseOptions] = Field(MintOptions(), description="Tracking options.")
    attributes: Dict = Field({}, description="Options for object attributes.")


class LevelOptions(BaseOptions):
    """
    Options for a tracking hierachy level. Objects identified at lower levels are
    used to define objects at higher levels.
    """

    objects: Dict[str, BaseObjectOptions] = Field({}, description="Hierachy levels.")


class TrackOptions(BaseOptions):
    """
    Options for the levels of a tracking hierarchy.
    """

    levels: List[LevelOptions] = Field([], description="Hierachy levels.")


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


# def default_convective(name, dataset):
#     """Get the default object options for convective objects."""
#     args_dict = {"name": name, "dataset": dataset, "variable": "reflectivity"}
#     args_dict.update({"hierarchy_level": 0, "detection_method": "steiner"})
#     tracking_options = {"method": "tint"}
#     convective_object_options = DetectedObjectOptions(**args_dict)


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
    filtered_kwargs = {k: v for k, v in kwargs.items() if k != "attribute_options"}

    # Test removal of attributes for component objects, instead get attributes for the
    # "member objects" comprising the grouped object

    options = [
        {
            "cell": cell_object(
                altitudes=[500, 3000],
                dataset=dataset,
                flatten_method="vertical_max",
                threshold=40,
                tracking_method=None,
                attribute_options={},
                **filtered_kwargs,
            ),
            "middle_echo": cell_object(
                name="middle_echo",
                dataset=dataset,
                threshold=20,
                tracking_method=None,
                detection_method="threshold",
                flatten_method="vertical_max",
                altitudes=[3500, 7000],
                attribute_options={},
                **filtered_kwargs,
            ),
            "anvil": anvil_object(
                altitudes=[7500, 10000],
                dataset=dataset,
                tracking_method=None,
                attribute_options={},
                **filtered_kwargs,
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


def default_convective(dataset="cpol"):
    """Build default options for convective objects."""
    return DetectedObjectOptions(
        name="convective",
        hierarchy_level=0,
        dataset="cpol",
        variable="reflectivity",
        detection={
            "method": "steiner",
            "altitudes": [500, 3e3],
            "threshold": 40,
        },
    )


def default_middle(dataset="cpol"):
    """Build default options for mid-level echo objects."""
    return DetectedObjectOptions(
        name="middle_echo",
        hierarchy_level=0,
        dataset="cpol",
        variable="reflectivity",
        detection={
            "method": "threshold",
            "altitudes": [3.5e3, 7e3],
            "threshold": 20,
        },
    )


def default_anvil(dataset="cpol"):
    """Build default options for anvil objects."""
    return DetectedObjectOptions(
        name="anvil",
        hierarchy_level=0,
        dataset="cpol",
        variable="reflectivity",
        detection={
            "method": "threshold",
            "altitudes": [7500, 10000],
            "threshold": 15,
        },
    )


def default_mcs(dataset="cpol"):
    """Build default options for MCS objects."""

    grouping = GroupingOptions(
        member_objects=["convective", "middle_echo", "anvil"],
        member_levels=[0, 0, 0],
        member_min_areas=[80, 400, 800],
    )
    mcs_options = GroupedObjectOptions(name="mcs", hierarchy_level=1, grouping=grouping)
    kwargs = {}
    core_tracked = attribute.core.default(tracked=True, matched=True)
    core_untracked = attribute.core.default(tracked=False, matched=True)
    # Note attributes for grouped objects are specified slightly differently
    # than for detected objects; the dictionary has an extra layer of nesting to
    # separate the attributes for member objects and for the grouped object.

    name = mcs_options.name
    member_objects = mcs_options.grouping.member_objects

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
    attribute_options[name]["profile"] = attribute.profile.default([profile_dataset])
    attribute_options[name]["tag"] = attribute.tag.default([tag_dataset])
    mcs_options.attributes = attribute_options
    return mcs_options


def default_track_options(dataset="cpol"):
    convective_options = default_convective(dataset)
    middle_options = default_middle(dataset)
    anvil_options = default_anvil(dataset)
    mcs_options = default_mcs(dataset)
    obj_dict = {
        obj_options.name: obj_options
        for obj_options in [convective_options, middle_options, anvil_options]
    }
    level_0 = LevelOptions(objects=obj_dict)
    obj_dict = {obj_options.name: obj_options for obj_options in [mcs_options]}
    level_1 = LevelOptions(objects=obj_dict)
    levels = [level_0, level_1]
    track_options = TrackOptions(levels=levels)
    return track_options
