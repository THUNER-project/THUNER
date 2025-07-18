"""Parallel processing utilities."""

import shutil
import gc
import os
import multiprocessing as mp
import time
import glob
from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr
from thuner.log import setup_logger, logging_listener
import thuner.attribute as attribute
import thuner.write as write
import thuner.analyze as analyze
import thuner.data as data
import thuner.track.track as thuner_track
import thuner.option as option
import thuner.utils as utils

logger = setup_logger(__name__)


__all__ = ["track"]


def track(
    times,
    data_options,
    grid_options,
    track_options,
    visualize_options=None,
    output_directory=None,
    num_processes=4,
    cleanup=True,
    dataset_name="gridrad",
    debug_mode=False,
):
    """
    Perform tracking in parallel using multiprocessing by splitting the time domain
    into intervals, tracking each interval in parallel, and then stitching the results
    back together.

    Parameters
    ----------
    times : Iterable[np.datetime64]
        The times to track the objects.
    data_options : :class:`thuner.option.data.DataOptions`
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

    if dataset_name not in data_options.dataset_names:
        raise ValueError(f"Dataset name {dataset_name} not in data options.")

    if num_processes > os.cpu_count():
        raise ValueError("Number of processes cannot exceed number of cpus.")
    elif num_processes > 3 / 4 * os.cpu_count():
        logger.warning("Number of processes over 3/4 of available CPUs.")

    if visualize_options is not None and num_processes > 1:
        message = "Runtime visualizations require that num_processes be set to 1."
        raise ValueError(message)

    times = sorted(list(times))
    intervals, num_processes = get_time_intervals(times, num_processes)
    logger.info(f"Beginning parallel tracking with {num_processes} processes.")

    if num_processes == 1:
        args = [times, data_options, grid_options, track_options, visualize_options]
        args += [output_directory]
        thuner_track.track(*args)
        return
    if visualize_options is not None:
        message = "Runtime visualizations are not supported during parallel tracking."
        message += " Setting visualize_options to None."
        visualize_options = None
        logger.warning(message)

    if debug_mode:
        for i, time_interval in enumerate(intervals):
            args = [i, time_interval, data_options.model_copy(deep=True)]
            args += [grid_options.model_copy(deep=True)]
            args += [track_options.model_copy(deep=True)]
            args += [None, output_directory, dataset_name]
            track_interval(*args)
    else:
        kwargs = {"initializer": utils.initialize_process, "processes": num_processes}
        with logging_listener(), mp.get_context("spawn").Pool(**kwargs) as pool:
            results = []
            for i, time_interval in enumerate(intervals):
                time.sleep(1)
                args = [i, time_interval, data_options.model_copy(deep=True)]
                args += [grid_options.model_copy(deep=True)]
                args += [track_options.model_copy(deep=True)]
                args += [None, output_directory, dataset_name]
                args = tuple(args)
                results.append(pool.apply_async(track_interval, args))
            pool.close()
            pool.join()
            utils.check_results(results)

    stitch_run(output_directory, intervals, cleanup=cleanup)


def track_interval(
    i,
    time_interval,
    data_options,
    grid_options,
    track_options,
    visualize_options,
    output_parent,
    dataset_name,
):

    # Silence the welcome message
    os.environ["THUNER_QUIET"] = "1"

    output_directory = output_parent / f"interval_{i}"
    output_directory.mkdir(parents=True, exist_ok=True)
    options_directory = output_directory / "options"
    options_directory.mkdir(parents=True, exist_ok=True)
    data_options = data_options.model_copy(deep=True)
    grid_options = grid_options.model_copy(deep=True)
    track_options = track_options.model_copy(deep=True)
    if visualize_options is not None:
        visualize_options = None
    interval_data_options = get_interval_data_options(data_options, time_interval)
    interval_data_options.to_yaml(options_directory / "data.yml")
    grid_options.to_yaml(options_directory / "grid.yml")
    track_options.to_yaml(options_directory / "track.yml")
    filepaths = interval_data_options.dataset_by_name(dataset_name).filepaths
    times = utils.generate_times(filepaths)
    args = [times, interval_data_options, grid_options, track_options]
    args += [visualize_options, output_directory]
    thuner_track.track(*args)
    gc.collect()


def get_interval_data_options(data_options: option.data.DataOptions, interval):
    """Get the data options for a given interval."""
    interval_data_options = data_options.model_copy(deep=True)
    for i, dataset_options in enumerate(interval_data_options.datasets):
        name = dataset_options.name
        dataset_options.start = interval[0]
        dataset_options.end = interval[1]
        new_filepaths = dataset_options.get_filepaths()
        dataset_options.filepaths = new_filepaths
        interval_data_options.datasets[i] = dataset_options
    # Revalidate the model to rebuild the dataset lookup dict
    interval_data_options = interval_data_options.model_validate(interval_data_options)
    return interval_data_options


def get_time_intervals(times, num_processes):
    """
    Split the times, which have been recovered from the filenames, into intervals.
    If the intervals are too small, set num_processes to 1.
    """
    # If less than 6 times, use one process
    if len(times) < 6:
        start_time = str(pd.Timestamp(times[0]))
        end_time = str(pd.Timestamp(times[-1]))
        intervals = [(start_time, end_time)]
        logger.info("Less than 6 times, using one process.")
        num_processes = 1
        return intervals, num_processes

    interval_size = int(np.ceil(len(times) / num_processes))
    if interval_size < 6:
        # If less than 6 times per interval, recalculate num processes
        message = f"Less than 6 times per interval with {num_processes} processes."
        logger.info(message)
        num_processes = int(np.ceil(len(times) / 6))
        interval_size = int(np.ceil(len(times) / num_processes))
        message = f"Instead using {num_processes} processes, with {interval_size} "
        message += "times per interval."
        logger.info(message)

    previous, next = 0, interval_size
    end = len(times) - 1
    intervals = []
    while next <= end:
        start_time = str(pd.Timestamp(times[previous]))
        end_time = str(pd.Timestamp(times[next]))
        intervals.append((start_time, end_time))
        previous = next - 1
        next = previous + interval_size
    if next > end:
        start_time = str(pd.Timestamp(times[previous]))
        end_time = str(pd.Timestamp(times[-1]))
        intervals.append((start_time, end_time))
    return intervals, num_processes


def get_filepath_dicts(output_parent, intervals):
    """Get the filepaths for all csv and mask files."""
    csv_file_dict, mask_file_dict, record_file_dict, weights_file_dict = {}, {}, {}, {}
    for i in range(len(intervals)):
        csv_filepath = output_parent / f"interval_{i}/attributes/**/*.csv"
        csv_file_dict[i] = sorted(glob.glob(str(csv_filepath), recursive=True))
        mask_filepath = output_parent / f"interval_{i}/**/*.zarr"
        mask_file_dict[i] = sorted(glob.glob(str(mask_filepath), recursive=True))
        record_filepath = output_parent / f"interval_{i}/records/**/*.csv"
        record_file_dict[i] = sorted(glob.glob(str(record_filepath), recursive=True))
    if len(np.unique([len(l) for l in csv_file_dict.values()])) != 1:
        raise ValueError("Different number of csv files output for each interval")
    if len(np.unique([len(l) for l in mask_file_dict.values()])) != 1:
        raise ValueError("Different number of mask files output for each interval")
    return csv_file_dict, mask_file_dict, record_file_dict


def match_dataarray(da_1, da_2):
    """Match the objects of two mask DataArrays."""
    matching_ids = {}
    # Check if binary regions of masks are the same
    if not ((da_1 > 0) == (da_2 > 0)).all().values:
        return matching_ids

    # Get unique values of datasets, excluding 0
    ids_1 = np.unique(da_1.values)
    ids_1 = ids_1[ids_1 != 0]
    ids_2 = np.unique(da_2.values)
    ids_2 = ids_2[ids_2 != 0]

    # Match ids in ds_1 to those of ds_2
    flat_dim = list(da_1.dims)
    for id in ids_1:
        da_2_flat = da_2.stack(flat_dim=flat_dim)
        da_1_flat = da_1.stack(flat_dim=flat_dim)
        matches = np.unique(da_2_flat.where(da_1_flat == id, 1, drop=True).values)
        if 0 in matches or len(matches) > 1:
            raise ValueError(f"Masks do not match.")
        matching_ids[int(id)] = int(matches[0])
    return matching_ids


def match_dataset(ds_1, ds_2):
    # Check if times are the same
    if ds_1["time"].values != ds_2["time"].values:
        raise ValueError("Times are not the same")

    # Check if the mask names are the same
    if list(ds_1.data_vars) != list(ds_2.data_vars):
        raise ValueError("Mask names are not the same")

    matching_ids = {}
    for mask_name in ds_1.data_vars:
        da_1, da_2 = ds_1[mask_name].squeeze(), ds_2[mask_name].squeeze()
        matching_ids.update(match_dataarray(da_1, da_2))
    return matching_ids


def get_tracked_objects(track_options):
    """Get the names of objects which are tracked."""
    tracked_objects = []
    all_objects = []
    for level_options in track_options.levels:
        for object_options in level_options.objects:
            all_objects.append(object_options.name)
            if object_options.tracking is not None:
                tracked_objects.append(object_options.name)
    return tracked_objects, all_objects


def get_match_dicts(intervals, mask_file_dict, tracked_objects):
    """Get the match dictionaries for each interval."""
    match_dicts = {}
    time_dicts = {}
    for i in range(len(intervals) - 1):
        filepaths_1 = mask_file_dict[i]
        filepaths_2 = mask_file_dict[i + 1]
        objects_1 = [Path(filepath).stem for filepath in filepaths_1]
        objects_2 = [Path(filepath).stem for filepath in filepaths_2]

        if objects_1 != objects_2:
            raise ValueError("Different objects in each filepath list.")
        interval_match_dicts = {}
        interval_time_dicts = {}

        for j, obj in enumerate(objects_1):
            ds_2 = xr.open_mfdataset(filepaths_2[j], chunks={}, engine="zarr")
            ds_2 = ds_2.isel(time=0)
            ds_2 = ds_2.load()
            time = ds_2["time"].values
            interval_time_dicts[obj] = time
            ds_1 = xr.open_mfdataset(filepaths_1[j], chunks={}, engine="zarr")
            if time not in ds_1.time:
                if obj not in tracked_objects:
                    interval_match_dicts[obj] = None
                else:
                    # Set the interval match dict to empty dict
                    interval_match_dicts[obj] = {}
                continue
            ds_1 = ds_1.sel(time=time)
            ds_1 = ds_1.load()
            time = ds_1["time"].values
            if obj not in tracked_objects:
                interval_match_dicts[obj] = None
            else:
                interval_match_dicts[obj] = match_dataset(ds_1, ds_2)

        match_dicts[i] = interval_match_dicts
        time_dicts[i] = interval_time_dicts
    return match_dicts, time_dicts


def stitch_records(record_file_dict, intervals):
    """Stitch together all record files."""
    logger.info("Stitching record files.")
    for i in range(len(record_file_dict[0])):
        filepaths = [record_file_dict[j][i] for j in range(len(intervals))]
        dfs = [attribute.utils.read_attribute_csv(filepath) for filepath in filepaths]
        metadata_path = Path(filepaths[0]).with_suffix(".yml")
        attribute_dict = attribute.utils.read_metadata_yml(metadata_path)
        filepath = Path(filepaths[0])
        filepath = Path(*[part for part in filepath.parts if part != "interval_0"])
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df = pd.concat(dfs).sort_index().drop_duplicates()
        write.attribute.write_csv(filepath, df, attribute_dict)


def stitch_run(output_parent, intervals, cleanup=True):
    """Stitch together all attribute files for a given run."""
    logger.info("Stitching all attribute, mask and record files.")
    options = analyze.utils.read_options(output_parent / "interval_0")
    track_options = options["track"]
    tracked_objects = get_tracked_objects(track_options)[0]
    all_file_dicts = get_filepath_dicts(output_parent, intervals)
    csv_file_dict, mask_file_dict, record_file_dict = all_file_dicts
    args = [intervals, mask_file_dict, tracked_objects]
    match_dicts, time_dicts = get_match_dicts(*args)
    number_attributes = len(csv_file_dict[0])
    stitch_records(record_file_dict, intervals)

    # Copy the regridder weights folder from interval_0 to the output parent
    weights_path_0 = output_parent / "interval_0" / "records" / "regridder_weights"
    weights_path = output_parent / "records" / "regridder_weights"
    if weights_path_0.exists():
        shutil.copytree(weights_path_0, weights_path, dirs_exist_ok=True)

    id_dicts = {}
    logger.info("Stitching attribute files.")
    for i in range(number_attributes):
        filepaths = [csv_file_dict[j][i] for j in range(len(intervals))]
        dfs = [attribute.utils.read_attribute_csv(filepath) for filepath in filepaths]
        metadata_path = Path(filepaths[0]).with_suffix(".yml")
        attribute_dict = attribute.utils.read_metadata_yml(metadata_path)
        example_filepath = Path(filepaths[0])
        attributes_index = example_filepath.parts.index("attributes")
        obj = example_filepath.parts[attributes_index + 1]
        attribute_type = example_filepath.stem
        if Path(example_filepath.parts[attributes_index + 2]).stem == attribute_type:
            member_object = False
        else:
            member_object = True
        args = [dfs, obj, filepaths, attribute_dict, match_dicts, time_dicts, id_dicts]
        args += [intervals, tracked_objects]
        id_dict = stitch_attribute(*args)
        if not member_object and obj in tracked_objects:
            id_dicts[obj] = id_dict
    stitch_masks(mask_file_dict, intervals, id_dicts)
    # Remove all interval directories
    if cleanup:
        for i in range(len(intervals)):
            shutil.rmtree(Path(output_parent / f"interval_{i}"))


def apply_mapping(mapping, mask):
    """Apply mapping to mask."""
    new_mask = mask.copy()
    for key in mapping.keys():
        for var in mask.data_vars:
            new_mask[var] = xr.where(mask[var] == key, mapping[key], new_mask[var])
    return new_mask


def get_mapping(id_dicts, obj, interval):
    """Get mapping for a given object and interval number."""
    try:
        mapping = id_dicts[obj].xs(interval, level="interval")
        id_type = list(mapping.columns)[0]
        mapping = mapping[id_type].to_dict()
    except KeyError:
        mapping = {}
    return mapping


def stitch_mask(intervals, masks, id_dicts, filepaths, obj):
    """Stitch together mask files for a given object."""
    new_masks = []
    for i in range(len(intervals)):
        mask = masks[i]
        mapping = get_mapping(id_dicts, obj, i)
        new_mask = apply_mapping(mapping, mask)
        if i > 0:
            time = masks[i - 1].time[-1].values
            if time not in np.array(masks[i].time.values):
                message = "Time intervals have produced non-overlapping time domains "
                message += "for masks. This can occur due to missing files at the "
                message += " overlap time."
                logger.warning(message)
            else:
                # Slice new mask, exluding times contained in the previous interval
                # Note the actual "slice" function doesn't work with high precision
                # datetime indexes! Use boolean indexing on time dimension instead
                condition = new_mask.time.values > time
                new_mask = new_mask.sel(time=condition)
        new_masks.append(new_mask)
    mask = xr.concat(new_masks, dim="time")
    mask = mask.astype(np.uint32)
    coords = [c for c in mask.coords if c in ["x", "y", "latitude", "longitude"]]
    for coord in coords:
        mask.coords[coord] = mask.coords[coord].astype(np.float32)
    filepath = Path(filepaths[0])
    filepath = Path(*[part for part in filepath.parts if part != "interval_0"])
    filepath.parent.mkdir(parents=True, exist_ok=True)
    mask.to_zarr(filepath, mode="w")


def stitch_masks(mask_file_dict, intervals, id_dicts):
    """Stitch together all mask files."""
    logger.info("Stitching mask files.")
    # Loop over all objects
    for k in range(len(mask_file_dict[0])):
        filepaths = [mask_file_dict[j][k] for j in range(len(intervals))]
        example_filepath = filepaths[0]
        kwargs = {"chunks": {"time": 1}, "engine": "zarr"}
        masks = [xr.open_dataset(filepath, **kwargs) for filepath in filepaths]
        obj = Path(example_filepath).stem
        # Stitch together masks for that object
        stitch_mask(intervals, masks, id_dicts, filepaths, obj)


def relabel_id_string(i, df, column_name, id_dicts, mapping=None, object_name=None):
    """Relabel the ids in a space seperated string."""
    row = df.iloc[i]
    if str(row[column_name]) == "nan":
        return
    if mapping is None:
        mapping = get_mapping(id_dicts, object_name, row["interval"])
    obj_ids = row[column_name].split(" ")
    new_obj_ids = []
    for obj_id in obj_ids:
        obj_id = int(obj_id)
        new_obj_id = mapping[obj_id]
        new_obj_ids.append(str(new_obj_id))
    new_obj_ids = " ".join(new_obj_ids)
    df.at[i, column_name] = new_obj_ids


def stitch_attribute(
    dfs,
    obj,
    filepaths,
    attribute_dict,
    match_dicts,
    time_dicts,
    id_dicts,
    intervals,
    tracked_objects,
):
    """Stitch together attribute files."""
    new_dfs = []
    current_max_id = 0

    if obj in tracked_objects:
        id_type = "universal_id"
    else:
        id_type = "id"

    # First ensure object ids increase sequentially over all intervals
    for i, df in enumerate(dfs):
        index_columns = list(df.index.names)
        df["interval"] = i
        df = df.reset_index()
        df["time"] = df["time"].astype("datetime64[s]")
        df["original_id"] = df[id_type]
        unique_ids = df[id_type].unique()
        if len(unique_ids) > 0:
            max_id = df[id_type].unique().max()
        else:
            max_id = 0
        df[id_type] = df[id_type] + current_max_id
        current_max_id += max_id
        if i > 0:
            start_time = time_dicts[i - 1][obj]
            df = df[df["time"] > start_time]
        df = df.set_index(index_columns)
        new_dfs.append(df)
    df = pd.concat(new_dfs)
    index_columns = list(df.index.names)
    df = df.reset_index()

    # Next relabel the ids based on the match_dicts if the object is matched/tracked
    if obj in tracked_objects:
        df = relabel_tracked(intervals, match_dicts, obj, df)

    # Finally, relabel the ids based to ensure no id is skipped, which can occur
    # after the relabelling step
    unique_ids = df[id_type].unique()
    mapping = {old_id: new_id + 1 for new_id, old_id in enumerate(sorted(unique_ids))}
    df[id_type] = df[id_type].map(mapping)

    # Relabel parents. Note we can use the mapping dict defined above as parents were
    # relabelled in the same way as the ids in the relabel_tracked function.
    if "parents" in df.columns:
        for i in range(len(df)):
            relabel_id_string(i, df, "parents", id_dicts, mapping)

    # Relabel the member objects. Here we use the mapping dict specific to the
    # given interval, which uses the original id as key, as the member_objects were
    # not changed by the relabel_tracked function.
    attribute_names = list(attribute_dict._attribute_lookup.keys())
    if "member_objects" in attribute_names:
        attribute_group = attribute_dict.attribute_by_name("member_objects")
        members_matched = attribute_group.retrieval.keyword_arguments["members_matched"]
        for i, obj_attr in enumerate(attribute_group.attributes):
            member_obj = obj_attr.name.replace("_ids", "")
            if members_matched[i]:
                for i in range(len(df)):
                    args = [i, df, f"{member_obj}_ids", id_dicts]
                    relabel_id_string(*args, object_name=member_obj)

    id_dict = df[[id_type, "original_id", "interval"]].drop_duplicates()
    id_dict = id_dict.set_index(["interval", "original_id"]).sort_index()

    df = df.set_index(index_columns).sort_index()
    df = df.drop(["original_id", "interval"], axis=1)

    filepath = Path(filepaths[0])
    filepath = Path(*[part for part in filepath.parts if part != "interval_0"])
    filepath.parent.mkdir(parents=True, exist_ok=True)
    write.attribute.write_csv(filepath, df, attribute_dict)
    return id_dict


def relabel_tracked(intervals, match_dicts, obj, df):
    # Relabel universal ids in interval i
    for i in range(len(intervals) - 1):
        match_dict = match_dicts[i][obj]
        reversed_match_dict = {v: k for k, v in match_dict.items()}
        current_interval = df["interval"] == i
        next_interval = df["interval"] == i + 1
        # relabel universal ids based on match_dict
        for next_key in reversed_match_dict.keys():
            current_key = reversed_match_dict[next_key]
            condition = current_interval & (df["original_id"] == current_key)
            # Get the universal id of the object in the current interval with current_key
            universal_ids = df.loc[condition]["universal_id"].unique()
            # Confirm that the universal id is unique
            # Note we do nothing if universal_ids is empty, which can occur if the object
            # was only detected in the very last scan of the current interval
            if len(universal_ids) > 1:
                raise ValueError(f"Non unique universal id.")
            elif len(universal_ids) == 1:
                universal_id = int(universal_ids[0])
                # Relabel the universal id of the corresponding object in the next interval
                condition = next_interval & (df["original_id"] == next_key)
                df.loc[condition, "universal_id"] = universal_id
                # Relabel parents objects in the next interval
        if "parents" in df.columns:
            args = [df, next_interval, current_interval, reversed_match_dict]
            df = relabel_parents(*args)

    return df


def relabel_parents(df, next_interval, current_interval, reversed_match_dict):
    """
    Relabel parents based on reversed_match_dict.
    """
    parents = df.loc[next_interval, "parents"]
    new_parents = []
    for object_parents in parents:
        if str(object_parents) == "nan":
            new_parents.append("nan")
            continue
        new_object_parents = []
        for p in object_parents.split(" "):
            p = int(p)
            if p in reversed_match_dict:
                # If parent p in the match dict, get the universal id of the parent
                # from the current interval
                current_key = reversed_match_dict[p]
                condition = current_interval & (df["original_id"] == current_key)
                # Get the universal id of the object in the current interval with current_key
                universal_ids = df.loc[condition]["universal_id"].unique()
                universal_id = int(universal_ids[0])
                new_object_parents.append(str(universal_id))
            else:
                # If parent p is not in the match dict, use the universal id of the parent
                # from the next interval
                condition = next_interval & (df["original_id"] == p)
                universal_ids = df.loc[condition, "universal_id"].unique()
                universal_id = int(universal_ids[0])
                new_object_parents.append(str(universal_id))

        new_parents.append(" ".join(new_object_parents))
    df.loc[next_interval, "parents"] = new_parents
    return df
