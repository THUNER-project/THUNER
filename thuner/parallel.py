"""Parallel processing utilities."""

import shutil
import gc
import multiprocessing
import os
import glob
from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr
from thuner.log import setup_logger
import thuner.attribute as attribute
import thuner.write as write
import thuner.analyze as analyze
import thuner.data as data
import thuner.grid as grid
import thuner.track as track
import thuner.data.option as option


logger = setup_logger(__name__)


get_filepaths_dispatcher = {
    "cpol": data.aura.get_cpol_filepaths,
    "era5_pl": data.era5.get_era5_filepaths,
    "era5_sl": data.era5.get_era5_filepaths,
    "gridrad": data.gridrad.get_gridrad_filepaths,
}


def initialize_process():
    """
    Use to set the initializer argument when creating a multiprocessing.Pool object.
    This will ensure that all processes in the pool are non-daemonic, and avoid the
    associated errors.
    """
    multiprocessing.current_process().daemon = False


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
    output_directory = output_parent / f"interval_{i}"
    output_directory.mkdir(parents=True, exist_ok=True)
    options_directory = output_directory / "options"
    options_directory.mkdir(parents=True, exist_ok=True)
    data_options = data_options.model_copy(deep=True)
    grid_options = grid_options.copy()
    track_options = track_options.model_copy(deep=True)
    if visualize_options is not None:
        visualize_options = visualize_options.copy()
    interval_data_options = get_interval_data_options(data_options, time_interval)
    interval_data_options.to_yaml(options_directory / "data.yml")
    grid_options.to_yaml(options_directory / "grid.yml")
    track_options.to_yaml(options_directory / "track.yml")
    times = data.utils.generate_times(
        interval_data_options.dataset_by_name(dataset_name)
    )
    args = [times, interval_data_options, grid_options, track_options]
    args += [visualize_options, output_directory]
    track.simultaneous_track(*args)
    gc.collect()


def get_interval_data_options(data_options: option.DataOptions, interval):
    """Get the data options for a given interval."""
    interval_data_options = data_options.model_copy(deep=True)
    for i, dataset_options in enumerate(interval_data_options.datasets):
        name = dataset_options.name
        dataset_options.start = interval[0]
        dataset_options.end = interval[1]
        get_filepaths = get_filepaths_dispatcher[name]
        new_filepaths = get_filepaths(dataset_options)
        dataset_options.filepaths = new_filepaths
        interval_data_options.datasets[i] = dataset_options
    # Revalidate the model to rebuild the dataset lookup dict
    interval_data_options = interval_data_options.model_validate(interval_data_options)
    return interval_data_options


def check_results(results):
    """Check pool results for exceptions."""
    for result in results:
        try:
            result.get(timeout=5 * 60)
        except Exception as exc:
            print(f"Generated an exception: {exc}")


def check_futures(futures):
    """Check futures for exceptions."""
    for future in futures:
        try:
            future.result(timeout=20 * 60)
        except Exception as exc:
            print(f"Generated an exception: {exc}")


def get_period(start, end, minimum_period=pd.Timedelta(1, "h")):
    """
    Get a suitable interval period for parallel processing based on number of cpus.
    """
    total_interval = pd.Timestamp(end) - pd.Timestamp(start)
    period = total_interval / 16
    # Get the interval in units of minimum_period
    period = period.ceil(minimum_period)
    period = max(period, minimum_period)
    return period


def get_time_intervals(
    start, end, period=pd.Timedelta(1, "h"), overlap=pd.Timedelta(10, "m")
):
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    intervals = []
    previous, next = start, start + period + overlap
    while next <= end:
        intervals.append((str(previous), str(next)))
        previous = next - overlap
        next = previous + period + overlap
    if next > end:
        intervals.append((str(previous), str(end)))
    return intervals


def get_filepath_dicts(output_parent, intervals):
    """Get the filepaths for all csv and mask files."""
    csv_file_dict, mask_file_dict, record_file_dict = {}, {}, {}
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
            ds_1 = xr.open_mfdataset(filepaths_1[j], chunks={}, engine="zarr")
            ds_1 = ds_1.sel(time=time)
            ds_1 = ds_1.load()
            time = ds_1["time"].values
            if obj not in tracked_objects:
                interval_match_dicts[obj] = None
            else:
                interval_match_dicts[obj] = match_dataset(ds_1, ds_2)
            interval_time_dicts[obj] = time
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
        args = [dfs, obj, filepaths, attribute_dict, match_dicts, time_dicts]
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
                message += "for masks"
                raise ValueError(message)
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


# @profile
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


# @profile
def stitch_attribute(
    dfs,
    obj,
    filepaths,
    attribute_dict,
    match_dicts,
    time_dicts,
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

    for i, df in enumerate(dfs):
        index_columns = list(df.index.names)
        df["interval"] = i
        df = df.reset_index()
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

    if obj in tracked_objects:
        df = relabel_tracked(intervals, match_dicts, obj, df)

    unique_ids = df[id_type].unique()
    mapping = {old_id: new_id + 1 for new_id, old_id in enumerate(sorted(unique_ids))}
    df[id_type] = df[id_type].map(mapping)
    # Fix parents...
    if "parents" in df.columns:
        for i in range(len(df)):
            row = df.iloc[i]
            if str(row["parents"]) == "nan":
                continue
            parents = row["parents"].split(" ")
            new_parents = []
            for p in parents:
                p = int(p)
                new_parent = mapping[p]
                new_parents.append(str(new_parent))
            new_parents = " ".join(new_parents)
            df.at[i, "parents"] = new_parents

    id_dict = df[[id_type, "original_id", "interval"]].drop_duplicates()
    id_dict = id_dict.set_index(["interval", "original_id"]).sort_index()

    # if "parents" in df.columns:
    #     df = relabel_parents(df, id_dict)

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
    """Relabel parents based on reversed_match_dict. This is fiddly."""
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
