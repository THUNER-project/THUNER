"""
Tracking utilities. 

"""

from collections import deque
from pydantic import BaseModel, Field, model_validator
import numpy as np
import xarray as xr
from typing import Dict
from thuner.attribute.utils import AttributesRecord
from thuner.option.data import DataOptions
from thuner.option.track import TrackOptions, BaseObjectOptions, LevelOptions


class BaseInputRecord(BaseModel):
    """
    Base input record class. An input record will be defined for each dataset, and store
    the appropriate grids and files during tracking or tagging.
    """

    # Allow arbitrary types in the input record classes.
    class Config:
        arbitrary_types_allowed = True

    name: str
    filepaths: list[str] | dict | None = None
    write_interval: np.timedelta64 = np.timedelta64(1, "h")
    _desc = "Dataset from which to draw grids. This is updated periodically."
    dataset: xr.Dataset | xr.DataArray | None = Field(None, description=_desc)

    # Initialize attributes not to be set during object creation.
    # In pydantic these attributes begin with an underscore.
    _current_file_index: int = -1
    _last_write_time: np.datetime64 | None = None
    _time_list: list = []
    _filepath_list: list = []


def _init_deques(values, names):
    """Convenience function for initializing deques from names."""
    for name in names:
        new_deque = deque([None] * values.deque_length, values.deque_length)
        setattr(values, name, new_deque)
    return values


class TrackInputRecord(BaseInputRecord):
    """
    input record class for datasets used for tracking.
    """

    deque_length: int = Field(2, description="Number of grids/masks to keep in memory.")

    _desc = "Next grid to carry out detection/matching."
    next_grid: xr.DataArray | xr.Dataset | None = Field(None, description=_desc)
    grids: deque | None = None
    next_domain_mask: xr.DataArray | xr.Dataset | None = None
    domain_masks: deque | None = None
    next_boundary_mask: xr.DataArray | xr.Dataset | None = None
    boundary_masks: deque | None = None
    next_boundary_coordinates: xr.DataArray | xr.Dataset | None = None
    boundary_coodinates: deque | None = None
    _desc = "Dictionaries descibing synthetic objects. See thuner.data.synthetic."
    synthetic_objects: list[dict] | None = Field(None, description=_desc)
    _desc = "Synthetic base dataset. See thuner.data.synthetic."
    synthetic_base_dataset: xr.DataArray | xr.Dataset | None = Field(
        None, description=_desc
    )

    @model_validator(mode="after")
    def _initialize_deques(cls, values):
        names = ["grids", "domain_masks"]
        names += ["boundary_masks", "boundary_coodinates"]
        return _init_deques(values, names)


class InputRecords(BaseModel):
    """
    Class for managing the input records for all the datasets of a given run.
    """

    # Allow arbitrary types in the input records class.
    class Config:
        arbitrary_types_allowed = True

    data_options: DataOptions

    track: Dict[str, TrackInputRecord] = {}
    tag: Dict[str, BaseInputRecord] = {}

    @model_validator(mode="after")
    def _initialize_input_records(cls, values):
        data_options = values.data_options
        for name in data_options._dataset_lookup.keys():
            dataset_options = data_options.dataset_by_name(name)
            kwargs = {"name": name, "filepaths": dataset_options.filepaths}
            if dataset_options.use == "track":
                kwargs["deque_length"] = dataset_options.deque_length
                values.track[name] = TrackInputRecord(**kwargs)
            elif dataset_options.use == "tag":
                values.tag[name] = BaseInputRecord(**kwargs)
            else:
                raise ValueError(f"Use must be 'tag' or 'track'.")
        return values


class ObjectTracks(BaseModel):
    """
    Class for recording the attributes and grids etc for tracking a particular object.
    """

    # Allow arbitrary types in the class.
    class Config:
        arbitrary_types_allowed = True

    _desc = "Options for the object to be tracked."
    object_options: BaseObjectOptions = Field(..., description=_desc)
    _desc = "Name of the object to be tracked."
    name: str | None = Field(None, description=_desc)
    _desc = "Number of current/previous objects to keep in memory."
    deque_length: int = Field(2, description=_desc)
    _desc = "Running count of the number of objects tracked."
    object_count: int = Field(0, description=_desc)

    _desc = "Next grid for tracking."
    next_grid: xr.DataArray | xr.Dataset | None = Field(None, description=_desc)
    _desc = "Deque of current/previous grids."
    grids: deque | None = Field(None, description=_desc)

    _desc = "Interval between current and next grids."
    next_time_interval: np.timedelta64 | None = Field(None, description=_desc)
    _desc = "Interval between current and previous grids."
    previous_time_interval: deque | None = Field(None, description=_desc)

    _desc = "Next time for tracking."
    next_time: np.datetime64 | None = Field(None, description=_desc)
    _desc = "Deque of current/previous times."
    times: deque | None = Field(None, description=_desc)

    _desc = "Next mask for tracking."
    next_mask: xr.DataArray | xr.Dataset | None = Field(None, description=_desc)
    _desc = "Deque of current/previous masks."
    masks: deque | None = Field(None, description=_desc)

    _desc = "Next matched mask for tracking."
    next_matched_mask: xr.DataArray | xr.Dataset | None = Field(None, description=_desc)
    _desc = "Deque of current/previous matched masks."
    matched_masks: deque | None = Field(None, description=_desc)

    _desc = "Current match record."
    match_record: dict | None = Field(None, description=_desc)
    _desc = "Deque of previous match records."
    previous_match_records: deque | None = Field(None, description=_desc)

    _desc = "Attributes for the object."
    attributes: AttributesRecord | None = Field(None, description=_desc)
    _desc = "Attributes for the object collected during current iteration."
    current_attributes: AttributesRecord | None = Field(None, description=_desc)

    _desc = "Area of each grid cell in km^2."
    gridcell_area: xr.DataArray | xr.Dataset | None = Field(None, description=_desc)

    _last_write_time: np.datetime64 | None = None

    @model_validator(mode="after")
    def _initialize_deques(cls, values):
        names = ["grids", "previous_time_interval", "times"]
        names += ["masks", "matched_masks", "previous_match_records"]
        return _init_deques(values, names)

    @model_validator(mode="after")
    def _check_name(cls, values):
        if values.name is None:
            values.name = values.object_options.name
        elif values.name != values.object_options.name:
            raise ValueError("Name must match object_options name.")
        return values

    @model_validator(mode="after")
    def _initialize_attributes(cls, values):
        options = values.object_options.attributes
        if options is not None:
            values.attributes = AttributesRecord(attribute_options=options)
            values.current_attributes = AttributesRecord(attribute_options=options)
        return values


class LevelTracks(BaseModel):
    """
    Class for recording the attributes and grids etc for tracking a particular hierachy
    level.
    """

    # Allow arbitrary types in the class.
    class Config:
        arbitrary_types_allowed = True

    _desc = "Options for the given level of the hierachy."
    level_options: LevelOptions = Field(..., description=_desc)
    objects: dict[str, ObjectTracks] = Field({}, description="Objects to be tracked.")

    @model_validator(mode="after")
    def _initialize_objects(cls, values):
        for obj_options in values.level_options.objects:
            values.objects[obj_options.name] = ObjectTracks(object_options=obj_options)
        return values


class Tracks(BaseModel):
    """
    Class for recording tracks of all hierachy levels.
    """

    # Allow arbitrary types in the class.
    class Config:
        arbitrary_types_allowed = True

    levels: list[LevelTracks] = Field([], description="Tracks for each hierachy level.")
    track_options: TrackOptions = Field(..., description="Options for tracking.")

    @model_validator(mode="after")
    def _initialize_levels(cls, values):
        for level_options in values.track_options.levels:
            values.levels.append(LevelTracks(level_options=level_options))
        return values
