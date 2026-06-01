"""Tracking utilities."""

from collections import deque
from pydantic import BaseModel, Field, model_validator, ConfigDict
import numpy as np
import xarray as xr
from typing import Dict, Callable
from thuner.attribute.utils import AttributesRecord
from thuner.match.utils import MatchRecord
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
    filepaths: list[str] | dict | None = Field(
        None,
        description="The relevant dataset filepaths used for the run.",
    )
    write_interval: np.timedelta64 = Field(
        np.timedelta64(1, "h"),
        description=(
            "How often to move attribute data from working memory to hard disk."
        ),
    )
    dataset: DataObject | None = Field(
        None,
        description=(
            "Dataset from which to draw grids, which is updated as needed as the run "
            "progresses. In this context, a 'dataset' is an xarray.DataArray or "
            "xarray.Dataset corresponding to a single file. A `grid` is a single time "
            "step extracted from a dataset."
        ),
    )
    weights_filepath: Callable | None = Field(
        None,
        description=(
            "The regridder function for this dataset. This should be left as None and"
            " inferred during tracking."
        ),
    )

    # Index of the file corresponding to the currently stored dataset.
    # Initially set to -1 to indicate no file has been read yet.
    _current_file_index: int = -1
    # The last time data was written to disk; Used to assess if write_interval
    # has been reached.
    last_write_time: np.datetime64 | None = None
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

    next_grid: DataObject | None = Field(
        None,
        description=(
            "Next grid to carry out detection/matching. "
            "A 'grid' in thuner is a single time step."
        ),
    )
    grids: deque[DataObject] | None = Field(
        None,
        description="Deque of current/previous grids.",
    )
    next_domain_mask: DataObject | None = Field(
        None,
        description="The domain mask, i.e. region of valid values, for the next grid.",
    )
    domain_masks: deque[DataObject] | None = Field(
        None,
        description="Deque of current/previous domain masks.",
    )
    next_boundary_mask: DataObject | None = Field(
        None,
        description="The next grid's boundary mask, i.e. mask of boundary pixels.",
    )
    boundary_masks: deque[DataObject] | None = Field(
        None,
        description="Deque of current/previous boundary masks.",
    )
    next_boundary_coordinates: Dict | None = Field(
        None,
        description="The next grid's boundary coordinates.",
    )
    boundary_coordinates: deque | None = Field(
        None,
        description="Deque of current/previous boundary coordinates.",
    )
    @model_validator(mode="after")
    def _initialize_deques(self):
        names = ["grids", "domain_masks"]
        names += ["boundary_masks", "boundary_coordinates"]
        return _init_deques(self, names)


class InputRecords(BaseModel):
    """
    Class for managing the input records for all the datasets of a given run.
    """

    # Allow arbitrary types in the input records class.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    data_options: DataOptions = Field(..., description="Options for the datasets.")

    track: Dict[str, TrackInputRecord] = Field(
        {},
        description="Dictionary containing the input records for tracking datasets.",
    )
    tag: Dict[str, BaseInputRecord] = Field(
        {},
        description="Dictionary containing the input records for tagging datasets.",
    )

    @model_validator(mode="after")
    def _initialize_input_records(self):
        data_options = self.data_options
        for name in data_options._dataset_lookup.keys():
            dataset_options = data_options.dataset_by_name(name)
            kwargs = {"name": name, "filepaths": dataset_options.filepaths}
            if dataset_options.use == "track":
                kwargs["deque_length"] = dataset_options.deque_length
                self.track[name] = TrackInputRecord(**kwargs)
            elif dataset_options.use == "tag":
                self.tag[name] = BaseInputRecord(**kwargs)
            else:
                raise ValueError(f"Use must be 'tag' or 'track'.")
        return self


class ObjectTracks(BaseModel):
    """
    Class for recording the attributes and grids etc for tracking a particular object.
    """

    # Allow arbitrary types in the class.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    object_options: BaseObjectOptions = Field(
        ...,
        description="Options for the object to be tracked.",
    )
    name: str | None = Field(None, description="Name of the object to be tracked.")
    deque_length: int = Field(
        2,
        description="Number of current/previous objects to keep in memory.",
    )
    object_count: int = Field(
        0,
        description="Running count of the number of objects tracked.",
    )

    next_grid: xr.DataArray | xr.Dataset | None = Field(
        None,
        description="Next grid for tracking.",
    )
    grids: deque[DataObject] | None = Field(
        None,
        description="Deque of current/previous grids.",
    )

    next_time_interval: np.timedelta64 | None = Field(
        None,
        description="Interval between current and next grids.",
    )
    previous_time_interval: deque | None = Field(
        None,
        description="Interval between current and previous grids.",
    )

    next_time: np.datetime64 | None = Field(None, description="Next time for tracking.")
    times: deque[np.datetime64] | None = Field(
        None,
        description="Deque of current/previous times.",
    )

    next_mask: xr.DataArray | xr.Dataset | None = Field(
        None,
        description="Next mask for tracking.",
    )
    masks: deque[DataObject] | None = Field(
        None,
        description="Deque of current/previous masks.",
    )

    next_matched_mask: xr.DataArray | xr.Dataset | None = Field(
        None,
        description="Next matched mask for tracking.",
    )
    matched_masks: deque[DataObject] | None = Field(
        None,
        description="Deque of current/previous matched masks.",
    )

    match_record: MatchRecord | None = Field(
        None, description="Current match record."
    )

    attributes: AttributesRecord | None = Field(
        None,
        description="Attributes for the object.",
    )
    current_attributes: AttributesRecord | None = Field(
        None,
        description="Attributes for the object collected during current iteration.",
    )

    gridcell_area: DataObject | None = Field(
        None,
        description="Area of each grid cell in km^2.",
    )

    last_write_time: np.datetime64 | None = Field(
        None,
        description=(
            "The last time data was written to disk; Used to assess if write_interval "
            "has been reached."
        ),
    )

    @model_validator(mode="after")
    def _initialize_deques(self):
        """Initialize the deques for the object."""
        names = ["grids", "previous_time_interval", "times"]
        names += ["masks", "matched_masks"]
        return _init_deques(self, names)

    @model_validator(mode="after")
    def _check_name(self):
        """Check the name of the object matches the object_options name."""
        if self.name is None:
            self.name = self.object_options.name
        elif self.name != self.object_options.name:
            raise ValueError("Name must match object_options name.")
        return self

    @model_validator(mode="after")
    def _initialize_attributes(self):
        """Initialize the attributes for the object."""
        options = self.object_options.attributes
        if options is not None:
            self.attributes = AttributesRecord(attribute_options=options)
            self.current_attributes = AttributesRecord(attribute_options=options)
        return self


class LevelTracks(BaseModel):
    """
    Class for recording the attributes and grids etc for tracking a particular hierarchy
    level.
    """

    # Allow arbitrary types in the class.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    level_options: LevelOptions = Field(
        ...,
        description="Options for the given level of the hierarchy.",
    )
    objects: dict[str, ObjectTracks] = Field({}, description="Objects to be tracked.")

    @model_validator(mode="after")
    def _initialize_objects(self):
        """Initialize the objects of the given level."""
        for obj_options in self.level_options.objects:
            self.objects[obj_options.name] = ObjectTracks(object_options=obj_options)
        return self


class Tracks(BaseModel):
    """
    Class for recording tracks of all hierarchy levels.
    """

    # Allow arbitrary types in the class.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    levels: list[LevelTracks] = Field([], description="Tracks for each hierarchy level.")
    track_options: TrackOptions = Field(..., description="Options for tracking.")

    @model_validator(mode="after")
    def _initialize_levels(self):
        """Initialize the levels of the tracking hierarchy."""
        for level_options in self.track_options.levels:
            self.levels.append(LevelTracks(level_options=level_options))
        return self
