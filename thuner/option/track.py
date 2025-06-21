"""Classes for managing tracking related options."""

from typing import List, Annotated, Literal
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


class TintOptions(BaseOptions):
    """
    Options for the TINT tracking algorithm. See the following publications
    """

    name: str = "tint"
    _desc = "Margin in km for object matching. Does not affect flow vectors."
    search_margin: float = Field(10.0, description=_desc, gt=0)
    _desc = "Margin in km around object for phase correlation."
    local_flow_margin: float = Field(10.0, description=_desc, gt=0)
    _desc = "Margin in km around object for global flow vectors."
    global_flow_margin: float = Field(150.0, description=_desc, gt=0)
    _desc = "If True, create unique global flow vectors for each object."
    unique_global_flow: bool = Field(True, description=_desc)
    _desc = "Maximum allowable matching cost. Units of km."
    max_cost: float = Field(2e2, description=_desc, gt=0, lt=1e3)
    _desc = "Maximum allowable shift velocity magnitude. Units of m/s."
    max_velocity_mag: float = Field(60.0, description=_desc, gt=0)
    _desc = "Maximum allowable shift difference. Units of m/s."
    max_velocity_diff: float = Field(60.0, description=_desc, gt=0)
    _desc = "Name of object used for matching/tracking."
    matched_object: str | None = Field(None, description=_desc)


class MintOptions(TintOptions):
    """
    Options for the MINT tracking algorithm.
    """

    name: str = "mint"
    _desc = "Margin in km for object matching. Does not affect flow vectors."
    search_margin: float = Field(25.0, description=_desc, gt=0)
    _desc = "Margin in km around object for phase correlation."
    local_flow_margin: float = Field(35.0, description=_desc, gt=0)
    _desc = "Alternative max shift difference used by MINT."
    max_velocity_diff_alt: float = Field(25.0, description=_desc, gt=0)


class MaskOptions(BaseOptions):
    """
    Options for saving and loading masks. Note thuner uses .zarr format for saving
    masks, which is great for sparse, chunked arrays.
    """

    save: bool = Field(True, description="If True, save masks as .zarr files.")
    load: bool = Field(False, description="If True, load masks from .zarr files.")


class BaseObjectOptions(BaseOptions):
    """Base class for object options."""

    name: str = Field(..., description="Name of the object.")
    _desc = "Level of the object in the hierachy. Higher level objects may depend on "
    _desc += "lower level objects."
    hierarchy_level: int = Field(0, description=_desc, ge=0)
    _desc = "Method used to obtain the object, i.e. detect or group."
    method: Literal["detect", "group"] = Field("detect", description=_desc)
    # TODO: Create validator to check datasets here are present in DataOptions instance.
    _desc = "Name of the dataset used for detection if applicable."
    dataset: str = Field(..., description=_desc, examples=["cpol", "gridrad"])
    _desc = "Length of the deque used for tracking."
    deque_length: int = Field(2, description=_desc, gt=0, lt=10)
    _desc = "Options for saving and loading masks."
    mask_options: MaskOptions = Field(MaskOptions(), description=_desc)
    _desc = "Interval in hours for writing objects to disk."
    write_interval: int = Field(1, description=_desc, gt=0, lt=24 * 60)
    _desc = "Allowed gap in minutes between consecutive times when tracking."
    allowed_gap: int = Field(30, description=_desc, gt=0, lt=6 * 60)
    _desc = "Options for object attributes."
    attributes: Attributes | None = Field(None, description=_desc)


class DetectionOptions(BaseOptions):
    """Options for object detection."""

    _desc = "Method used to detect the object."
    method: Literal["steiner", "threshold"] = Field(..., description=_desc)
    _desc = "Altitudes over which to detect objects."
    altitudes: List[int] = Field([], description=_desc)
    _desc = "Method used to flatten the grid before detection if relevant."
    flatten_method: str = Field("vertical_max", description=_desc)
    _desc = "Minimum area of the object in km squared."
    min_area: int = Field(10, description=_desc)
    _desc = "Threshold used for detection if required."
    threshold: int | None = Field(None, description=_desc)

    @model_validator(mode="after")
    def _check_threshold(cls, values):
        """Check threshold value is provided if applicable."""
        if values.method == "detect" and values.threshold is None:
            raise ValueError("Threshold not provided for detection method.")
        return values


def _check_mask_values(values):
    """Check if masks saved if tracking options provided."""
    if values.tracking is not None and not values.mask_options.save:
        message = "Masks must be saved when objects are being tracked."
        raise ValueError(message)
    return values


class DetectedObjectOptions(BaseObjectOptions):
    """Options for detected objects."""

    object_type: Literal["detected"] = Field("detected", description="Type of object.")
    _desc = "Variable to use for detection."
    variable: str = Field("reflectivity", description=_desc)
    _desc = "Method used to detect the object."
    detection: DetectionOptions = Field(
        DetectionOptions(method="steiner"), description=_desc
    )
    tracking: BaseOptions | None = Field(TintOptions(), description="Tracking options.")

    @model_validator(mode="after")
    def _check_mask(cls, values):
        """Check if masks saved if tracking options provided."""
        return _check_mask_values(values)


# Define a custom type with constraints
PositiveFloat = Annotated[float, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]


class GroupingOptions(BaseOptions):
    """Options class for grouping lower level objects into higher level objects."""

    method: str = Field("graph", description="Method used to group objects.")
    member_objects: List[str] = Field([], description="Names of objects to group")
    _desc = "Hierarchy levels of objects to group."
    member_levels: List[NonNegativeInt] = Field([], description=_desc)
    _desc = "Minimum area of each member object in km squared."
    member_min_areas: List[PositiveFloat] = Field([], description=_desc)

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
    _desc = "Options for grouping objects."
    grouping: GroupingOptions = Field(GroupingOptions(), description=_desc)
    tracking: AnyTrackingOptions = Field(MintOptions(), description="Tracking options.")

    @model_validator(mode="after")
    def _check_mask(cls, values):
        """Check if masks saved if tracking options provided."""
        return _check_mask_values(values)


# Unclear why an additional discriminator is needed here. Perhaps due to the list.
AnyObjectOptions = Annotated[
    DetectedObjectOptions | GroupedObjectOptions, Field(discriminator="object_type")
]


class LevelOptions(BaseOptions):
    """
    Options for a tracking hierachy level. Objects identified at lower levels are
    used to define objects at higher levels.
    """

    _desc = "Options for each object in the level."
    objects: List[AnyObjectOptions] = Field([], description=_desc)
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
