"""Classes for managing tracking related options."""

from typing import List, Annotated, Literal, Callable
from pydantic import Field, model_validator, ValidationError
from thuner.log import setup_logger
from thuner.option.attribute import Attributes
from thuner.utils import BaseOptions, Retrieval
from thuner.detect.preprocess import vertical_max

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
    search_margin: float = Field(
        10.0,
        description="Margin in km for object matching. Does not affect flow vectors.",
        gt=0,
    )
    local_flow_margin: float = Field(
        10.0,
        description="Margin in km around object for phase correlation.",
        gt=0,
    )
    global_flow_margin: float = Field(
        150.0,
        description="Margin in km around object for global flow vectors.",
        gt=0,
    )
    unique_global_flow: bool = Field(
        True,
        description="If True, create unique global flow vectors for each object.",
    )
    max_cost: float = Field(
        2e2,
        description="Maximum allowable matching cost. Units of km.",
        gt=0,
        lt=1e3,
    )
    max_velocity_mag: float = Field(
        60.0,
        description="Maximum allowable shift velocity magnitude. Units of m/s.",
        gt=0,
    )
    max_velocity_diff: float = Field(
        60.0,
        description="Maximum allowable shift difference. Units of m/s.",
        gt=0,
    )
    matched_object: str | None = Field(
        None,
        description="Name of object used for matching/tracking.",
    )


class MintOptions(TintOptions):
    """
    Options for the MINT tracking algorithm.
    """

    name: str = "mint"
    search_margin: float = Field(
        25.0,
        description="Margin in km for object matching. Does not affect flow vectors.",
        gt=0,
    )
    local_flow_margin: float = Field(
        35.0,
        description="Margin in km around object for phase correlation.",
        gt=0,
    )
    max_velocity_diff_alt: float = Field(
        25.0,
        description="Alternative max shift difference used by MINT.",
        gt=0,
    )


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
    hierarchy_level: int = Field(
        0,
        description=(
            "Level of the object in the hierachy. Higher level objects may depend on "
            "lower level objects."
        ),
        ge=0,
    )
    method: Literal["detect", "group"] = Field(
        "detect",
        description="Method used to obtain the object, i.e. detect or group.",
    )
    dataset: str = Field(
        ...,
        description="Name of the dataset used for detection if applicable.",
        examples=["cpol", "gridrad"],
    )
    deque_length: int = Field(
        2,
        description="Length of the deque used for tracking.",
        gt=0,
        lt=10,
    )
    mask_options: MaskOptions = Field(
        MaskOptions(),
        description="Options for saving and loading masks.",
    )
    write_interval: int = Field(
        1,
        description="Interval in hours for writing objects to disk.",
        gt=0,
        lt=24 * 60,
    )
    allowed_gap: int = Field(
        30,
        description="Allowed gap in minutes between consecutive times when tracking.",
        gt=0,
        lt=6 * 60,
    )
    attributes: Attributes | None = Field(
        None,
        description="Options for object attributes.",
    )


class DetectionOptions(BaseOptions):
    """Options for object detection."""

    method: Literal["steiner", "threshold"] = Field(
        ...,
        description="Method used to detect the object.",
    )
    altitudes: List[int] = Field(
        [],
        description="Altitudes over which to detect objects.",
    )
    flatten_method: Retrieval | None = Field(
        Retrieval(function=vertical_max),
        description="Method used to flatten the grid before detection if relevant.",
    )
    min_area: int = Field(10, description="Minimum area of the object in km squared.")
    threshold: int | None = Field(
        None,
        description="Threshold used for detection if required.",
    )
    threshold_type: Literal["minima", "maxima"] = Field(
        "minima",
        description="Threshold type, i.e. a minima or maxima threshold.",
    )

    @model_validator(mode="after")
    def _check_threshold(self):
        """Check threshold value is provided if applicable."""
        if self.method == "detect" and self.threshold is None:
            raise ValueError("Threshold not provided for detection method.")
        return self


def _check_mask_values(values):
    """Check if masks saved if tracking options provided."""
    if values.tracking is not None and not values.mask_options.save:
        message = "Masks must be saved when objects are being tracked."
        raise ValueError(message)
    return values


AnyTrackingOptions = TintOptions | MintOptions


class DetectedObjectOptions(BaseObjectOptions):
    """Options for detected objects."""

    object_type: Literal["detected"] = Field("detected", description="Type of object.")
    variable: str = Field("reflectivity", description="Variable to use for detection.")
    detection: DetectionOptions = Field(
        DetectionOptions(method="steiner"),
        description="Method used to detect the object.",
    )
    tracking: AnyTrackingOptions | None = Field(
        TintOptions(),
        description="Options for tracking the object.",
    )

    @model_validator(mode="after")
    def _check_mask(self):
        """Check if masks saved if tracking options provided."""
        return _check_mask_values(self)


# Define a custom type with constraints
PositiveFloat = Annotated[float, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]


class GroupingOptions(BaseOptions):
    """Options class for grouping lower level objects into higher level objects."""

    method: str = Field("graph", description="Method used to group objects.")
    member_objects: List[str] = Field([], description="Names of objects to group")
    member_levels: List[NonNegativeInt] = Field(
        [],
        description="Hierarchy levels of objects to group.",
    )
    member_min_areas: List[PositiveFloat] = Field(
        [],
        description="Minimum area of each member object in km squared.",
    )

    # Check lists are the same length.
    @model_validator(mode="after")
    def _check_list_length(self):
        """Check list lengths are consistent."""
        member_objects = self.member_objects
        member_levels = self.member_levels
        member_min_areas = self.member_min_areas
        lengths = [len(member_objects), len(member_levels), len(member_min_areas)]
        if len(set(lengths)) != 1:
            message = "Member objects, levels, and areas must have the same length."
            raise ValueError(message)
        return self


class GroupedObjectOptions(BaseObjectOptions):
    """Options for grouped objects."""

    object_type: Literal["grouped"] = Field("grouped", description="Type of object.")
    grouping: GroupingOptions = Field(
        GroupingOptions(),
        description="Options for grouping objects.",
    )
    tracking: AnyTrackingOptions | None = Field(
        MintOptions(),
        description="Options for tracking the object.",
    )

    @model_validator(mode="after")
    def _check_mask(self):
        """Check if masks saved if tracking options provided."""
        return _check_mask_values(self)


# Unclear why an additional discriminator is needed here. Perhaps due to the list.
AnyObjectOptions = Annotated[
    DetectedObjectOptions | GroupedObjectOptions, Field(discriminator="object_type")
]


class LevelOptions(BaseOptions):
    """
    Options for a tracking hierachy level. Objects identified at lower levels are
    used to define objects at higher levels.
    """

    objects: List[AnyObjectOptions] = Field(
        [],
        description="Options for each object in the level.",
    )
    _object_lookup = {}
    object_names: List[str] = Field(
        [],
        description="Names of the objects comprising this tracking level.",
    )

    @model_validator(mode="after")
    def initialize_object_lookup(self):
        """Initialize object lookup dictionary."""
        self._object_lookup = {obj.name: obj for obj in self.objects}
        self.object_names = [obj.name for obj in self.objects]
        if len(self.object_names) != len(set(self.object_names)):
            message = "Object names must be unique to facilitate name based lookup."
            raise ValueError(message)
        return self

    def object_by_name(self, obj_name: str) -> BaseObjectOptions:
        """Return object options by name."""
        return self._object_lookup.get(obj_name)


def _check_grouped_object(object_options, object_levels):
    """
    Helper function to check a grouped object lists member object heierachy level
    correctly.
    """
    for i, member_name in enumerate(object_options.grouping.member_objects):
        if member_name not in object_levels:
            message = f"Grouped object '{object_options.name}' references member "
            message += f"object '{member_name}' which doesn't exist in track options."
            raise ValidationError(message)

        member_level = object_levels[member_name]
        expected_level = object_options.grouping.member_levels[i]

        if member_level != expected_level:
            message = f"Grouped object '{object_options.name}' expects member "
            message += f"'{member_name}' at level {expected_level}, "
            message += f"but it's actually at level {member_level}."
            raise ValidationError(message)

        if member_level >= object_options.hierarchy_level:
            message = f"Grouped object '{object_options.name}' at level "
            message += f"{object_options.hierarchy_level} cannot reference member "
            message += f"'{member_name}' at level {member_level}. "
            message += f"Member objects must be at lower hierarchy levels."
            raise ValidationError(message)


class TrackOptions(BaseOptions):
    """
    Options for the levels of a tracking hierarchy.
    """

    levels: List[LevelOptions] = Field([], description="Hierachy levels.")
    _object_lookup = {}
    object_names: List[str] = Field([], description="Names of the objects.")

    @model_validator(mode="after")
    def initialize_object_lookup(self):
        """Initialize object lookup dictionary."""
        object_names = []
        lookup_dicts = []
        for level in self.levels:
            lookup_dicts.append(level._object_lookup)
            object_names += level._object_lookup.keys()
        if len(object_names) != len(set(object_names)):
            message = "Object names must be unique to facilitate name based lookup."
            raise ValueError(message)
        for lookup_dict in lookup_dicts:
            self._object_lookup.update(lookup_dict)
        self.object_names = object_names
        return self

    def object_by_name(self, obj_name: str) -> BaseObjectOptions:
        """Return object options by name."""
        try:
            return self._object_lookup.get(obj_name)
        except KeyError:
            message = f"Object {obj_name} not found in object lookup."
            raise KeyError(message)

    @model_validator(mode="after")
    def _validate_grouped_objects(self):
        """
        Validate that the tracking hierarchy is consistent - grouped objects should
        only reference objects from lower hierarchy levels.
        """
        # Build a mapping of object names to their hierarchy levels
        object_levels = {}
        for level in self.levels:
            for obj in level.objects:
                object_levels[obj.name] = obj.hierarchy_level

        # Check grouped objects reference appropriate member objects
        for level in self.levels:
            for obj in level.objects:
                if hasattr(obj, "grouping") and obj.grouping:
                    _check_grouped_object(obj, object_levels)
        return self
