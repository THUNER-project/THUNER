"""Functions for creating and modifying default tracking configurations."""

import numpy as np
from typing import Dict, List, Annotated
from pydantic import Field, field_validator, model_validator
from thuner.log import setup_logger
from thuner.option.attribute import Attributes
from thuner.option.utils import BaseOptions


logger = setup_logger(__name__)


_summary = {
    "name": "Name of the tracking algorithm.",
    "search_margin": "Margin in km for object matching. Does not affect flow vectors.",
    "local_flow_margin": "Margin in km around object for phase correlation.",
    "global_flow_margin": "Margin in km around object for global flow vectors.",
    "unique_global_flow": "If True, create unique global flow vectors for each object.",
    "max_cost": "Maximum allowable matching cost. Units of km.",
    "max_velocity_mag": "Maximum allowable shift magnitude.",
    "max_velocity_diff": "Maximum allowable shift difference.",
    "matched_object": "Name of object used for matching.",
}


class TintOptions(BaseOptions):
    """
    Options for the TINT tracking algorithm. See the following publications
    """

    name: str = Field("mint", description=_summary["name"])
    search_margin: float = Field(10, description=_summary["search_margin"], gt=0)
    local_flow_margin: float = Field(
        10, description=_summary["local_flow_margin"], gt=0
    )
    global_flow_margin: float = Field(
        150, description=_summary["global_flow_margin"], gt=0
    )
    unique_global_flow: bool = Field(True, description=_summary["unique_global_flow"])
    max_cost: float = Field(2e2, description=_summary["max_cost"], gt=0, lt=1e3)
    max_velocity_mag: float = Field(60, description=_summary["max_velocity_mag"], gt=0)
    max_velocity_diff: float = Field(
        60, description=_summary["max_velocity_diff"], gt=0
    )
    matched_object: str | None = Field(None, description=_summary["matched_object"])


_summary["max_velocity_diff_alt"] = "Alternative max shift difference used by MINT."


class MintOptions(TintOptions):
    """
    Options for the MINT tracking algorithm.
    """

    name: str = Field("mint", description=_summary["name"])
    search_margin: int = Field(25, description=_summary["search_margin"], gt=0)
    local_flow_margin: int = Field(35, description=_summary["local_flow_margin"], gt=0)
    max_velocity_diff_alt: int = Field(
        25, description=_summary["max_velocity_diff_alt"], gt=0
    )


class MaskOptions(BaseOptions):
    """
    Options for saving and loading masks. Note thuner uses .zarr format for saving
    masks, which is great for sparse, chunked arrays.
    """

    save: bool = Field(True, description="If True, save masks as .zarr files.")
    load: bool = Field(False, description="If True, load masks from .zarr files.")


_summary.update(
    {
        "matched_object": """Name of object used for matching. Should be the name 
    of the given detected object, or the name of a member object comprising a grouped 
    object.""",
        "hierarchy_level": """Level of the object in the hierachy. Higher level objects 
    depend on lower level objects.""",
        "method": "Method used to obtain the object, e.g. detect or group.",
        "dataset": """Name of the dataset used for detection. This field will likely "
    be moved elsewhere for grouped objects in future.""",
        "deque_length": "Length of the deque used for tracking.",
        "mask_options": "Options for saving and loading masks.",
        "write_interval": "Interval in hours for writing objects to disk.",
        "allowed_gap": "Allowed gap in minutes between consecutive times when tracking.",
        "grouping": "Options for grouping objects.",
        "detect_method": "Method used to detect the object.",
        "altitudes": "Altitudes over which to detect objects.",
        "flatten_method": "Method used to flatten the object.",
    }
)


class BaseObjectOptions(BaseOptions):
    """Base class for object options."""

    name: str = Field(..., description="Name of the object.")
    hierarchy_level: int = Field(0, description=_summary["hierarchy_level"], ge=0)
    method: str = Field("detect", description=_summary["method"])
    dataset: str = Field(
        ..., description=_summary["dataset"], examples=["cpol", "gridrad"]
    )
    deque_length: int = Field(2, description=_summary["deque_length"], gt=0, lt=10)
    mask_options: MaskOptions = Field(
        MaskOptions(), description=_summary["mask_options"]
    )
    write_interval: int = Field(
        1, description=_summary["write_interval"], gt=0, lt=24 * 60
    )
    allowed_gap: int = Field(30, description=_summary["allowed_gap"], gt=0, lt=6 * 60)

    # Check method is either detect or group.
    @field_validator("method")
    def _check_method(cls, value):
        if value not in ["detect", "group"]:
            raise ValueError("Method must be detect or group.")
        return value


_summary["min_area"] = "Minimum area of the object in km squared."
_summary["threshold"] = "Threshold used for detection if required."


class DetectionOptions(BaseOptions):
    """Options for object detection."""

    method: str = Field(..., description=_summary["detect_method"])
    altitudes: List[int] = Field([], description=_summary["altitudes"])

    flatten_method: str = Field("vertical_max", description=_summary["flatten_method"])
    min_area: int = Field(10, description=_summary["min_area"])
    threshold: int | None = Field(None, description=_summary["threshold"])

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


_summary["variable"] = "Variable to use for detection."
_summary["detection"] = "Method used to detect the object."
_summary["attributes"] = "Options for object attributes."


class DetectedObjectOptions(BaseObjectOptions):
    """Options for detected objects."""

    variable: str = Field("reflectivity", description=_summary["variable"])
    detection: DetectionOptions = Field(
        DetectionOptions(method="steiner"), description=_summary["detection"]
    )
    tracking: BaseOptions | None = Field(TintOptions(), description="Tracking options.")
    attributes: Attributes | None = Field(None, description=_summary["attributes"])


# Define a custom type with constraints
PositiveFloat = Annotated[float, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]


_summary["member_levels"] = "Hierachy levels of objects to group"
_summary["member_min_areas"] = "Minimum area of each member object in km squared."


class GroupingOptions(BaseOptions):
    """Options class for grouping lower level objects into higher level objects."""

    method: str = Field("graph", description="Method used to group objects.")
    member_objects: List[str] = Field([], description="Names of objects to group")
    member_levels: List[NonNegativeInt] = Field(
        [], description=_summary["member_levels"]
    )
    member_min_areas: List[PositiveFloat] = Field(
        [], description=_summary["member_min_areas"]
    )

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


AnyTrackingOptions = TintOptions | MintOptions


class GroupedObjectOptions(BaseObjectOptions):
    """Options for grouped objects."""

    grouping: GroupingOptions = Field(
        GroupingOptions(), description=_summary["grouping"]
    )
    tracking: AnyTrackingOptions = Field(MintOptions(), description="Tracking options.")
    attributes: Attributes | None = Field(None, description=_summary["attributes"])


AnyObjectOptions = DetectedObjectOptions | GroupedObjectOptions

_summary["objects"] = "Options for each object in the level."


class LevelOptions(BaseOptions):
    """
    Options for a tracking hierachy level. Objects identified at lower levels are
    used to define objects at higher levels.
    """

    objects: List[AnyObjectOptions] = Field([], description=_summary["objects"])
    _object_lookup: Dict[str, BaseObjectOptions] = {}

    @model_validator(mode="after")
    def initialize_object_lookup(cls, values):
        values._object_lookup = {obj.name: obj for obj in values.objects}
        return values

    def options_by_name(self, obj_name: str) -> BaseObjectOptions:
        return self._object_lookup.get(obj_name)


class TrackOptions(BaseOptions):
    """
    Options for the levels of a tracking hierarchy.
    """

    levels: List[LevelOptions] = Field([], description="Hierachy levels.")
    _object_lookup: Dict[str, BaseObjectOptions] = {}

    @model_validator(mode="after")
    def initialize_object_lookup(cls, values):
        object_names = []
        lookup_dicts = []
        for level in values.levels:
            lookup_dicts.append(level._object_lookup)
            object_names += level._object_lookup.keys()
        if len(object_names) != len(set(object_names)):
            message = "Object names must be unique to facilitate name based lookup."
            raise ValueError(message)
        new_lookup_dict = {}
        for lookup_dict in lookup_dicts:
            new_lookup_dict.update(lookup_dict)
        values._object_lookup = new_lookup_dict
        return values

    def options_by_name(self, obj_name: str) -> BaseObjectOptions:
        return self._object_lookup.get(obj_name)


def consolidate_options(options_list):
    """Consolidate the options into a dictionary."""
    consolidated_options = {}
    for options in options_list:
        consolidated_options[options.name] = options
    return consolidated_options
