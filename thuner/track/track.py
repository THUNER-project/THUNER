"""Track storm objects in a dataset."""

import shutil
import copy
from typing import Iterable
import numpy as np
from pathlib import Path
from thuner.log import setup_logger
import thuner.data.dispatch as dispatch
import thuner.detect.detect as detect
import thuner.group.group as group
import thuner.visualize.runtime as runtime
import thuner.visualize.visualize as visualize
import thuner.match.match as match
from thuner.config import get_outputs_directory
import thuner.utils as utils
import thuner.write as write
import thuner.attribute.attribute as attribute
from thuner.option.data import DataOptions
from thuner.option.grid import GridOptions
from thuner.option.track import TrackOptions
from thuner.option.visualize import VisualizeOptions
from thuner.track.utils import InputRecords, Tracks

logger = setup_logger(__name__)


__all__ = ["track"]


def consolidate_options(data_options, grid_options, track_options, visualize_options):
    """Consolidate the options for a given run."""
    options = {"data_options": data_options, "grid_options": grid_options}
    options.update({"track_options": track_options})
    options.update({"visualize_options": visualize_options})
    return options


def track(
    times: Iterable[np.datetime64],
    data_options: DataOptions,
    grid_options: GridOptions,
    track_options: TrackOptions,
    visualize_options: VisualizeOptions = None,
    output_directory: str | Path = None,
):
    """
    Track objects described in track_options, in the datasets described in
    data_options, using the grid described in grid_options.

    Parameters
    ----------
    times : Iterable[np.datetime64]
        The times to track the objects.
    data_options : DataOptions
        The data options.
    grid_options : GridOptions
        The grid options.
    track_options : TrackOptions
        The track options.
    visualize_options : VisualizeOptions, optional
        The runtime visualization options for visualizing the tracking process.
        Defaults to None.
    output_directory : str | Path, optional
        The directory in which to save the output. If None, use the output directory
        specified in the THUNER config file. See thuner.config.get_outputs_directory.
        Defaults to None.
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
    visualize.animate_all(visualize_options, output_directory)


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
    runtime.visualize(*visualize_args)
    # Update the lists used to periodically write data to file
    if object_tracks.times[-1] is not None:
        args = [input_records, object_tracks, object_options, grid_options]
        attribute.record(*args)


get_objects_dispatcher = {
    "detect": detect.detect,
    "group": group.group,
}
