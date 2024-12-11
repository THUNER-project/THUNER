"""Functions for creating and modifying default tracking configurations."""

import yaml
from pathlib import Path
import numpy as np
from typing import Any, Dict, List, Annotated, Union
from pydantic import Field, BaseModel, field_validator, model_validator
from thuner.log import setup_logger
import thuner.attribute as attribute


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
    if type(value) is np.float32:
        return float(value)
    return value


class BaseOptions(BaseModel):
    """
    The base class for all options classes. This class is built on the pydantic
    BaseModel class, which is similar to python dataclasses but with type checking.
    """

    type: str = Field(None, description="Type of the options class.")

    # Allow arbitrary types in the options classes.
    class Config:
        arbitrary_types_allowed = True

    # Ensure that floats in all options classes are np.float32
    @model_validator(mode="after")
    def convert_floats(cls, values):
        for field in values.model_fields:
            if type(getattr(values, field)) is float:
                setattr(values, field, np.float32(getattr(values, field)))
        return values

    @model_validator(mode="after")
    def _set_type(cls, values):
        values.type = cls.__name__
        return values

    def to_dict(self) -> Dict[str, Any]:
        fields = self.model_fields.keys()
        return {field: convert_value(getattr(self, field)) for field in fields}

    def to_yaml(self, filepath: str):
        Path(filepath).parent.mkdir(exist_ok=True, parents=True)
        with open(filepath, "w") as f:
            kwargs = {"default_flow_style": False, "allow_unicode": True}
            kwargs = {"sort_keys": False}
            yaml.dump(self.to_dict(), f, **kwargs)


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
    Options for saving and loading masks. Note thuner uses .zarr format saving masks,
    which is great for sparse, chunked arrays.
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
        "write_interval": "Interval in minutes for writing objects to disk.",
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
    hierarchy_level: int = Field(..., description=_summary["hierarchy_level"], ge=0)
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


class AttributeTypeOptions(BaseOptions):
    pass


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


class DetectedObjectOptions(BaseObjectOptions):
    """Options for detected objects."""

    variable: str = Field("reflectivity", description=_summary["variable"])
    detection: DetectionOptions = Field(
        DetectionOptions(method="steiner"), description=_summary["detection"]
    )
    tracking: BaseOptions | None = Field(TintOptions(), description="Tracking options.")
    attributes: Dict = Field({}, description="Options for object attributes.")


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
    attributes: None | Dict = Field(None, description="Options for object attributes.")


AnyObjectOptions = DetectedObjectOptions | GroupedObjectOptions


class LevelOptions(BaseOptions):
    """
    Options for a tracking hierachy level. Objects identified at lower levels are
    used to define objects at higher levels.
    """

    objects: List[AnyObjectOptions] = Field([], description="Hierachy levels.")
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


def default_convective(dataset="cpol"):
    """Build default options for convective objects."""
    return DetectedObjectOptions(
        name="convective",
        hierarchy_level=0,
        dataset=dataset,
        variable="reflectivity",
        detection={
            "method": "steiner",
            "altitudes": [500, 3e3],
            "threshold": 40,
        },
        tracking=None,
    )


def default_middle(dataset="cpol"):
    """Build default options for mid-level echo objects."""
    return DetectedObjectOptions(
        name="middle",
        hierarchy_level=0,
        dataset=dataset,
        variable="reflectivity",
        detection={
            "method": "threshold",
            "altitudes": [3.5e3, 7e3],
            "threshold": 20,
        },
        tracking=None,
    )


def default_anvil(dataset="cpol"):
    """Build default options for anvil objects."""
    return DetectedObjectOptions(
        name="anvil",
        hierarchy_level=0,
        dataset=dataset,
        variable="reflectivity",
        detection={
            "method": "threshold",
            "altitudes": [7500, 10000],
            "threshold": 15,
        },
        tracking=None,
    )


def default_mcs(dataset="cpol"):
    """Build default options for MCS objects."""

    grouping = GroupingOptions(
        member_objects=["convective", "middle", "anvil"],
        member_levels=[0, 0, 0],
        member_min_areas=[80, 400, 800],
    )
    tracking = MintOptions(matched_object="convective")
    mcs_options = GroupedObjectOptions(
        name="mcs",
        dataset=dataset,
        hierarchy_level=1,
        grouping=grouping,
        tracking=tracking,
    )
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
    """Build default options for tracking MCS."""

    mask_options = MaskOptions(save=False, load=False)
    convective_options = default_convective(dataset)
    convective_options.mask_options = mask_options
    middle_options = default_middle(dataset)
    middle_options.mask_options = mask_options
    anvil_options = default_anvil(dataset)
    anvil_options.mask_options = mask_options
    mcs_options = default_mcs(dataset)
    level_0 = LevelOptions(objects=[convective_options, middle_options, anvil_options])
    level_1 = LevelOptions(objects=[mcs_options])
    levels = [level_0, level_1]
    track_options = TrackOptions(levels=levels)
    return track_options


def synthetic_track_options():
    convective_options = default_convective(dataset="synthetic")
    convective_options.tracking = MintOptions(
        global_flow_margin=70, unique_global_flow=False
    )
    return TrackOptions(levels=[LevelOptions(objects=[convective_options])])
