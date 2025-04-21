"""Classes for managing tracking related options."""

from typing import List, Annotated, Literal, Union
from pydantic import Field, model_validator
from thuner.log import setup_logger
from thuner.option.attribute import Attributes
from thuner.utils import BaseOptions

__all__ = [
    "TintOptions",
    "MintOptions",
    "MaskOptions",
    "BaseObjectOptions",
    "DetectionOptions",
    "DetectedObjectOptions",
    "GroupingOptions",
    "GroupedObjectOptions",
    "LevelOptions",
    "TrackOptions",
]


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

    name: str = "tint"
    _desc = "Margin in km for object matching. Does not affect flow vectors."
    search_margin: float = Field(10, description=_desc, gt=0)
    _desc = "Margin in km around object for phase correlation."
    local_flow_margin: float = Field(10, description=_desc, gt=0)
    _desc = "Margin in km around object for global flow vectors."
    global_flow_margin: float = Field(150, description=_desc, gt=0)
    _desc = "If True, create unique global flow vectors for each object."
    unique_global_flow: bool = Field(True, description=_desc)
    _desc = "Maximum allowable matching cost. Units of km."
    max_cost: float = Field(2e2, description=_desc, gt=0, lt=1e3)
    _desc = "Maximum allowable shift velocity magnitude. Units of m/s."
    max_velocity_mag: float = Field(60, description=_desc, gt=0)
    _desc = "Maximum allowable shift difference. Units of m/s."
    max_velocity_diff: float = Field(60, description=_desc, gt=0)
    _desc = "Name of object used for matching/tracking."
    matched_object: str | None = Field(None, description=_desc)


class MintOptions(TintOptions):
    """
    Options for the MINT tracking algorithm.
    """

    name: str = "mint"
    _desc = "Margin in km for object matching. Does not affect flow vectors."
    search_margin: int = Field(25, description=_desc, gt=0)
    _desc = "Margin in km around object for phase correlation."
    local_flow_margin: int = Field(35, description=_desc, gt=0)
    _desc = "Alternative max shift difference used by MINT."
    max_velocity_diff_alt: int = Field(25, description=_desc, gt=0)


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
        "dataset": "Name of the dataset used for detection.",
        "deque_length": "Length of the deque used for tracking.",
        "mask_options": "Options for saving and loading masks.",
        "write_interval": "Interval in hours for writing objects to disk.",
        "allowed_gap": "Allowed gap in minutes between consecutive times when tracking.",
        "grouping": "Options for grouping objects.",
        "detect_method": "Method used to detect the object.",
        "altitudes": "Altitudes over which to detect objects.",
        "flatten_method": "Method used to flatten the object.",
        "attributes": "Options for object attributes.",
    }
)


class BaseObjectOptions(BaseOptions):
    """Base class for object options."""

    name: str = Field(..., description="Name of the object.")
    hierarchy_level: int = Field(0, description=_summary["hierarchy_level"], ge=0)
    method: Literal["detect", "group"] = Field("detect", description=_summary["method"])
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
    attributes: Attributes | None = Field(None, description=_summary["attributes"])


_summary["min_area"] = "Minimum area of the object in km squared."
_summary["threshold"] = "Threshold used for detection if required."


class DetectionOptions(BaseOptions):
    """Options for object detection."""

    _desc = "Method used to detect the object."
    method: Literal["steiner", "threshold"] = Field(..., description=_desc)
    altitudes: List[int] = Field([], description=_summary["altitudes"])

    flatten_method: str = Field("vertical_max", description=_summary["flatten_method"])
    min_area: int = Field(10, description=_summary["min_area"])
    threshold: int | None = Field(None, description=_summary["threshold"])

    @model_validator(mode="after")
    def _check_threshold(cls, values):
        """Check threshold value is provided if applicable."""
        if values.method == "detect" and values.threshold is None:
            raise ValueError("Threshold not provided for detection method.")
        return values


_summary["variable"] = "Variable to use for detection."
_summary["detection"] = "Method used to detect the object."


class DetectedObjectOptions(BaseObjectOptions):
    """Options for detected objects."""

    object_type: Literal["detected"] = Field("detected", description="Type of object.")
    variable: str = Field("reflectivity", description=_summary["variable"])
    detection: DetectionOptions = Field(
        DetectionOptions(method="steiner"), description=_summary["detection"]
    )
    tracking: BaseOptions | None = Field(TintOptions(), description="Tracking options.")

    @model_validator(mode="after")
    def _check_mask(cls, values):
        """Check if masks saved if tracking options provided."""
        if values.tracking is not None and not values.mask_options.save:
            message = "Masks must be saved when objects are being tracked."
            raise ValueError(message)
        return values


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
        """Check list lengths are consistent."""
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

    object_type: Literal["grouped"] = Field("grouped", description="Type of object.")
    grouping: GroupingOptions = Field(
        GroupingOptions(), description=_summary["grouping"]
    )
    tracking: AnyTrackingOptions = Field(MintOptions(), description="Tracking options.")


# Unclear why an additional discriminator is needed here. Perhaps due to the list.
AnyObjectOptions = Annotated[
    DetectedObjectOptions | GroupedObjectOptions, Field(discriminator="object_type")
]


_summary["objects"] = "Options for each object in the level."


class LevelOptions(BaseOptions):
    """
    Options for a tracking hierachy level. Objects identified at lower levels are
    used to define objects at higher levels.
    """

    objects: List[AnyObjectOptions] = Field([], description=_summary["objects"])
    _description = "Dictionary for looking up ObjectOptions by object name."
    _object_lookup = {}
    _desc = "Names of the objects comprising this tracking level."
    object_names: List[str] = Field([], description=_desc)

    @model_validator(mode="after")
    def initialize_object_lookup(cls, values):
        """Initialize object lookup dictionary."""
        values._object_lookup = {obj.name: obj for obj in values.objects}
        values.object_names = [obj.name for obj in values.objects]
        if len(values.object_names) != len(set(values.object_names)):
            message = "Object names must be unique to facilitate name based lookup."
            raise ValueError(message)
        return values

    def object_by_name(self, obj_name: str) -> BaseObjectOptions:
        """Return object options by name."""
        return self._object_lookup.get(obj_name)


class TrackOptions(BaseOptions):
    """
    Options for the levels of a tracking hierarchy.
    """

    levels: List[LevelOptions] = Field([], description="Hierachy levels.")
    _object_lookup = {}
    object_names: List[str] = Field([], description="Names of the objects.")

    @model_validator(mode="after")
    def initialize_object_lookup(cls, values):
        """Initialize object lookup dictionary."""
        object_names = []
        lookup_dicts = []
        for level in values.levels:
            lookup_dicts.append(level._object_lookup)
            object_names += level._object_lookup.keys()
        if len(object_names) != len(set(object_names)):
            message = "Object names must be unique to facilitate name based lookup."
            raise ValueError(message)
        for lookup_dict in lookup_dicts:
            values._object_lookup.update(lookup_dict)
        values.object_names = object_names
        return values

    def object_by_name(self, obj_name: str) -> BaseObjectOptions:
        """Return object options by name."""
        try:
            return self._object_lookup.get(obj_name)
        except KeyError:
            message = f"Object {obj_name} not found in object lookup."
            raise KeyError(message)
