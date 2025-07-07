"""Tracking utilities."""

from collections import deque
from pydantic import BaseModel, Field, model_validator, ConfigDict
import numpy as np
import xarray as xr
from typing import Dict, Callable
from thuner.attribute.utils import AttributesRecord
from thuner.option.data import DataOptions
from thuner.option.track import TrackOptions, BaseObjectOptions, LevelOptions
from thuner.utils import DataObject

__all__ = []


class BaseInputRecord(BaseModel):
    """
    Base input record class. An input record will be defined for each dataset, and store
    the appropriate grids and files during tracking or tagging.
    """

    # Allow arbitrary types in the input record classes.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(..., description="Name of the input dataset being recorded.")
    _desc = "The relevant dataset filepaths used for the run."
    filepaths: list[str] | dict | None = Field(None, description=_desc)
    _desc = "How often to move attribute data from working memory to hard disk."
    write_interval: np.timedelta64 = Field(np.timedelta64(1, "h"), description=_desc)
    _desc = "Dataset from which to draw grids, which is updated as needed as the run "
    _desc += "progresses. In this context, a 'dataset' is an xarray.DataArray or "
    _desc += "xarray.Dataset corresponding to a single file. A `grid` is a single time "
    _desc += "step extracted from a dataset."
    dataset: DataObject | None = Field(None, description=_desc)
    _desc = "The regridder function for this dataset. This should be left as None and"
    _desc += " inferred during tracking."
    regridder: Callable | None = Field(None, description=_desc)

    # Index of the file corresponding to the currently stored dataset.
    # Initially set to -1 to indicate no file has been read yet.
    _current_file_index: int = -1
    # The last time data was written to disk; Used to assess if write_interval
    # has been reached.
    _last_write_time: np.datetime64 | None = None
    # List of times considered during the tracking run.
    _time_list: list = []
    # List of filepaths considered during the tracking run corresponding to
    # _time_list. Note multiple times can correspond to the same file.
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

    _desc = "Next grid to carry out detection/matching. "
    _desc += "A 'grid' in thuner is a single time step."
    next_grid: DataObject | None = Field(None, description=_desc)
    _desc = "Deque of current/previous grids."
    grids: deque[DataObject] | None = Field(None, description=_desc)
    _desc = "The domain mask, i.e. region of valid values, for the next grid."
    next_domain_mask: DataObject | None = Field(None, description=_desc)
    _desc = "Deque of current/previous domain masks."
    domain_masks: deque[DataObject] | None = Field(None, description=_desc)
    _desc = "The next grid's boundary mask, i.e. mask of boundary pixels."
    next_boundary_mask: DataObject | None = Field(None, description=_desc)
    _desc = "Deque of current/previous boundary masks."
    boundary_masks: deque[DataObject] | None = Field(None, description=_desc)
    _desc = "The next grid's boundary coordinates."
    next_boundary_coordinates: Dict | None = Field(None, description=_desc)
    _desc = "Deque of current/previous boundary coordinates."
    boundary_coodinates: deque | None = Field(None, description=_desc)
    _desc = "Dictionaries descibing synthetic objects. See thuner.data.synthetic."
    synthetic_objects: list[dict] | None = Field(None, description=_desc)
    _desc = "Synthetic base dataset. See thuner.data.synthetic."
    synthetic_base_dataset: DataObject | None = Field(None, description=_desc)

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
    model_config = ConfigDict(arbitrary_types_allowed=True)

    data_options: DataOptions = Field(..., description="Options for the datasets.")

    _desc = "Dictionary containing the input records for tracking datasets."
    track: Dict[str, TrackInputRecord] = Field({}, description=_desc)
    _desc = "Dictionary containing the input records for tagging datasets."
    tag: Dict[str, BaseInputRecord] = Field({}, description=_desc)

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
    model_config = ConfigDict(arbitrary_types_allowed=True)

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
    grids: deque[DataObject] | None = Field(None, description=_desc)

    _desc = "Interval between current and next grids."
    next_time_interval: np.timedelta64 | None = Field(None, description=_desc)
    _desc = "Interval between current and previous grids."
    previous_time_interval: deque | None = Field(None, description=_desc)

    _desc = "Next time for tracking."
    next_time: np.datetime64 | None = Field(None, description=_desc)
    _desc = "Deque of current/previous times."
    times: deque[np.datetime64] | None = Field(None, description=_desc)

    _desc = "Next mask for tracking."
    next_mask: xr.DataArray | xr.Dataset | None = Field(None, description=_desc)
    _desc = "Deque of current/previous masks."
    masks: deque[DataObject] | None = Field(None, description=_desc)

    _desc = "Next matched mask for tracking."
    next_matched_mask: xr.DataArray | xr.Dataset | None = Field(None, description=_desc)
    _desc = "Deque of current/previous matched masks."
    matched_masks: deque[DataObject] | None = Field(None, description=_desc)

    _desc = "Current match record."
    match_record: Dict | None = Field(None, description=_desc)
    _desc = "Deque of previous match records."
    previous_match_records: deque[Dict] | None = Field(None, description=_desc)

    _desc = "Attributes for the object."
    attributes: AttributesRecord | None = Field(None, description=_desc)
    _desc = "Attributes for the object collected during current iteration."
    current_attributes: AttributesRecord | None = Field(None, description=_desc)

    _desc = "Area of each grid cell in km^2."
    gridcell_area: DataObject | None = Field(None, description=_desc)

    _last_write_time: np.datetime64 | None = None

    @model_validator(mode="after")
    def _initialize_deques(cls, values):
        """Initialize the deques for the object."""
        names = ["grids", "previous_time_interval", "times"]
        names += ["masks", "matched_masks", "previous_match_records"]
        return _init_deques(values, names)

    @model_validator(mode="after")
    def _check_name(cls, values):
        """Check the name of the object matches the object_options name."""
        if values.name is None:
            values.name = values.object_options.name
        elif values.name != values.object_options.name:
            raise ValueError("Name must match object_options name.")
        return values

    @model_validator(mode="after")
    def _initialize_attributes(cls, values):
        """Initialize the attributes for the object."""
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
    model_config = ConfigDict(arbitrary_types_allowed=True)

    _desc = "Options for the given level of the hierachy."
    level_options: LevelOptions = Field(..., description=_desc)
    objects: dict[str, ObjectTracks] = Field({}, description="Objects to be tracked.")

    @model_validator(mode="after")
    def _initialize_objects(cls, values):
        """Initialize the objects of the given level."""
        for obj_options in values.level_options.objects:
            values.objects[obj_options.name] = ObjectTracks(object_options=obj_options)
        return values


class Tracks(BaseModel):
    """
    Class for recording tracks of all hierachy levels.
    """

    # Allow arbitrary types in the class.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    levels: list[LevelTracks] = Field([], description="Tracks for each hierachy level.")
    track_options: TrackOptions = Field(..., description="Options for tracking.")

    @model_validator(mode="after")
    def _initialize_levels(cls, values):
        """Initialize the levels of the tracking hierarchy."""
        for level_options in values.track_options.levels:
            values.levels.append(LevelTracks(level_options=level_options))
        return values
