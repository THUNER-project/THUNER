"""Track storm objects in a dataset."""

import shutil
from collections import deque
import copy
from pydantic import BaseModel, Field, model_validator
import numpy as np
import xarray as xr
from typing import Dict
from thuner.log import setup_logger
import thuner.data.dispatch as dispatch
import thuner.detect.detect as detect
import thuner.group.group as group
import thuner.visualize as visualize
import thuner.match.match as match
from thuner.config import get_outputs_directory
import thuner.utils as utils
import thuner.write as write
import thuner.attribute as attribute
from thuner.attribute.utils import AttributesRecord
from thuner.option.data import DataOptions
from thuner.option.grid import GridOptions
from thuner.option.track import TrackOptions, BaseObjectOptions, LevelOptions

logger = setup_logger(__name__)


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


def consolidate_options(data_options, grid_options, track_options, visualize_options):
    """Consolidate the options for a given run."""
    options = {"data_options": data_options, "grid_options": grid_options}
    options.update({"track_options": track_options})
    options.update({"visualize_options": visualize_options})
    return options


def track(
    times,
    data_options: DataOptions,
    grid_options: GridOptions,
    track_options: TrackOptions,
    visualize_options=None,
    output_directory=None,
):
    """
    Track objects across the hierachy simultaneously.

    Parameters
    ----------
    filenames : list of str
        List of filepaths to the netCDF files that need to be consolidated.
    data_options : dict
        Dictionary containing the data options.
    grid_options : dict
        Dictionary containing the grid options.
    track_options : dict
        Dictionary containing the track options.

    Returns
    -------
    pandas.DataFrame
        The pandas dataframe containing the object tracks.
    xarray.Dataset
        The xarray dataset containing the object masks.
    """
    logger.info("Beginning thuner tracking. Saving output to %s.", output_directory)
    tracks = Tracks(track_options=track_options)
    input_records = InputRecords(data_options=data_options)

    consolidated_options = consolidate_options(
        track_options, data_options, grid_options, visualize_options
    )

    # Clear masks directory to prevent overwriting
    if (output_directory / "masks").exists():
        shutil.rmtree(output_directory / "masks")
    if (output_directory / "attributes").exists():
        shutil.rmtree(output_directory / "attributes")

    current_time = None
    for next_time in times:

        if output_directory is None:
            consolidated_options["start_time"] = str(next_time)
            hash_str = utils.hash_dictionary(consolidated_options)
            output_directory = (
                get_outputs_directory() / f"runs/{utils.now_str()}_{hash_str[:8]}"
            )

        logger.info(f"Processing {utils.format_time(next_time, filename_safe=False)}.")
        args = [next_time, input_records.track, track_options, data_options]
        args += [grid_options, output_directory]
        dispatch.update_track_input_records(*args)
        args = [current_time, input_records.tag, track_options, data_options]
        args += [grid_options]
        dispatch.update_tag_input_records(*args)
        # loop over levels
        for level_index in range(len(track_options.levels)):
            logger.info("Processing hierarchy level %s.", level_index)
            track_level_args = [next_time, level_index, tracks, input_records]
            track_level_args += [data_options, grid_options, track_options]
            track_level_args += [visualize_options, output_directory]
            track_level(*track_level_args)

        current_time = next_time

    # Write final data to file
    # write.mask.write_final(tracks, track_options, output_directory)
    write.attribute.write_final(tracks, track_options, output_directory)
    write.filepath.write_final(input_records.track, output_directory)
    # Aggregate files previously written to file
    # write.mask.aggregate(track_options, output_directory)
    write.attribute.aggregate(track_options, output_directory)
    write.filepath.aggregate(input_records.track, output_directory)
    # Animate the relevant figures
    visualize.visualize.animate_all(visualize_options, output_directory)


def track_level(
    next_time,
    level_index,
    tracks,
    input_records,
    data_options: DataOptions,
    grid_options,
    track_options: TrackOptions,
    visualize_options,
    output_directory,
):
    """Track a hierarchy level."""
    level_tracks = tracks.levels[level_index]
    level_options = track_options.levels[level_index]

    def get_track_object_args(obj, level_options):
        logger.info("Tracking %s.", obj)
        object_options = level_options.options_by_name(obj)
        if "dataset" not in object_options.model_fields:
            dataset_options = None
        else:
            dataset_options = data_options.dataset_by_name(object_options.dataset)
        track_object_args = [next_time, level_index, obj, tracks, input_records]
        track_object_args += [dataset_options, grid_options, track_options]
        track_object_args += [visualize_options, output_directory]
        return track_object_args

    for obj in level_tracks.objects.keys():
        track_object_args = get_track_object_args(obj, level_options)
        track_object(*track_object_args)

    return level_tracks


def track_object(
    next_time,
    level_index,
    obj,
    tracks,
    input_records,
    dataset_options,
    grid_options,
    track_options,
    visualize_options,
    output_directory,
):
    """Track the given object."""
    # Get the object options
    object_options = track_options.levels[level_index].options_by_name(obj)
    object_tracks = tracks.levels[level_index].objects[obj]
    track_input_records = input_records.track

    # Update current and previous next_time
    if object_tracks.next_time is not None:
        current_time = copy.deepcopy(object_tracks.next_time)
        object_tracks.times.append(current_time)
    object_tracks.next_time = next_time

    if object_options.mask_options.save:
        # Write masks to zarr file
        write.mask.write(object_tracks, object_options, output_directory)
    # Write existing data to file if necessary
    if write.utils.write_interval_reached(next_time, object_tracks, object_options):
        write.attribute.write(object_tracks, object_options, output_directory)
        object_tracks._last_write_time = next_time

    # Detect objects at next_time
    if "grouping" in object_options.model_fields:
        get_objects = group.group
    elif "detection" in object_options.model_fields:
        get_objects = detect.detect
    else:
        raise ValueError("No known method for obtaining objects provided.")
    get_objects_args = [track_input_records, tracks, level_index, obj, dataset_options]
    get_objects_args += [object_options, grid_options]
    get_objects(*get_objects_args)

    match.match(object_tracks, object_options, grid_options)

    # Visualize the operation of the algorithm
    visualize_args = [track_input_records, tracks, level_index, obj, track_options]
    visualize_args += [grid_options, visualize_options, output_directory]
    visualize.runtime.visualize(*visualize_args)
    # Update the lists used to periodically write data to file
    if object_tracks.times[-1] is not None:
        args = [input_records, object_tracks, object_options, grid_options]
        attribute.attribute.record(*args)


get_objects_dispatcher = {
    "detect": detect.detect,
    "group": group.group,
}
