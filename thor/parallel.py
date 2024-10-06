"""Parallel processing utilities."""

import shutil
import glob
from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr
import concurrent.futures
from thor.log import setup_logger
import thor.attribute as attribute
import thor.write as write
import thor.analyze as analyze

logger = setup_logger(__name__)


def check_results(results):
    """Check pool results for exceptions."""
    for result in results:
        try:
            result.get()  # Wait for the result and handle exceptions
        except Exception as exc:
            print(f"Generated an exception: {exc}")


def check_futures(futures):
    """Check the status of the futures."""
    for future in concurrent.futures.as_completed(futures):
        try:
            future.result()
        except Exception as exc:
            logger.error("Generated an exception: %s", exc)


def generate_time_intervals(start, end, period="h"):
    start = pd.Timestamp(start).floor(period)
    end = pd.Timestamp(end).ceil(period)
    intervals = []
    previous, next = start, start + pd.Timedelta(1, period)
    while next <= end:
        intervals.append((str(previous), str(next)))
        previous = next
        next = previous + pd.Timedelta(1, period)
    return intervals


def get_filepath_dicts(output_parent, intervals):
    """Get the filepaths for all csv and mask files."""
    csv_file_dict, mask_file_dict, record_file_dict = {}, {}, {}
    for i in range(len(intervals)):
        csv_filepath = output_parent / f"interval_{i}/attributes/**/*.csv"
        csv_file_dict[i] = sorted(glob.glob(str(csv_filepath), recursive=True))
        mask_filepath = output_parent / f"interval_{i}/**/*.nc"
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
    for level_options in track_options:
        for obj in level_options:
            all_objects.append(obj)
            if level_options[obj]["tracking"]["method"] is not None:
                tracked_objects.append(obj)
    return tracked_objects, all_objects


def get_match_dicts(intervals, mask_file_dict, tracked_objects):
    """Get the match dictionaries for each interval."""
    match_dicts = {}
    for i in range(len(intervals) - 1):
        filepaths_1 = mask_file_dict[i]
        filepaths_2 = mask_file_dict[i + 1]
        objects_1 = [Path(filepath).stem for filepath in filepaths_1]
        objects_2 = [Path(filepath).stem for filepath in filepaths_2]

        if objects_1 != objects_2:
            raise ValueError("Different objects in each filepath list.")
        interval_dicts = {}
        for j, obj in enumerate(objects_1):
            if obj not in tracked_objects:
                interval_dicts[obj] = None
                continue
            ds_1 = xr.open_mfdataset(filepaths_1[j], chunks={"time": 1})
            ds_1 = ds_1.isel(time=-1).load()
            ds_2 = xr.open_mfdataset(filepaths_2[j], chunks={"time": 1})
            ds_2 = ds_2.isel(time=0).load()
            interval_dicts[obj] = match_dataset(ds_1, ds_2)
        match_dicts[i] = interval_dicts
    return match_dicts


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


def stitch_run(output_parent, intervals):
    """Stitch together all attribute files for a given run."""
    logger.info("Stitching all attribute, mask and record files.")
    options = analyze.utils.read_options(output_parent / "interval_0")
    track_options = options["track"]
    tracked_objects = get_tracked_objects(track_options)[0]
    all_file_dicts = get_filepath_dicts(output_parent, intervals)
    csv_file_dict, mask_file_dict, record_file_dict = all_file_dicts
    match_dicts = get_match_dicts(intervals, mask_file_dict, tracked_objects)
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
        args = [dfs, obj, filepaths, attribute_dict, match_dicts]
        args += [intervals, tracked_objects]
        id_dict = stitch_attribute(*args)
        if not member_object and obj in tracked_objects:
            id_dicts[obj] = id_dict
    stitch_masks(mask_file_dict, intervals, id_dicts)
    # Remove all interval directories
    for i in range(len(intervals)):
        shutil.rmtree(Path(output_parent / f"interval_{i}"))


def apply_mapping(mapping, mask):
    """Apply mapping to mask."""
    new_mask = mask.copy()
    for key in mapping.keys():
        for var in mask.data_vars:
            new_mask[var].values[mask[var].values == key] = mapping[key]
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
        if i < len(intervals) - 1:
            if masks[i + 1].time[0] != masks[i].time[-1]:
                message = "Time intervals have produced non-continuous masks"
                raise ValueError(message)
            new_mask = new_mask.isel(time=slice(0, -1))
        new_masks.append(new_mask)
    mask = xr.concat(new_masks, dim="time")
    mask = mask.astype(np.uint32)
    filepath = Path(filepaths[0])
    filepath = Path(*[part for part in filepath.parts if part != "interval_0"])
    filepath.parent.mkdir(parents=True, exist_ok=True)
    mask.to_netcdf(filepath)


def stitch_masks(mask_file_dict, intervals, id_dicts):
    """Stitch together all mask files."""
    logger.info("Stitching mask files.")
    # Loop over all objects
    for k in range(len(mask_file_dict[0])):
        filepaths = [mask_file_dict[j][k] for j in range(len(intervals))]
        example_filepath = filepaths[0]
        masks = [xr.open_dataset(filepath) for filepath in filepaths]
        obj = Path(example_filepath).stem
        # Stitch together masks for that object
        stitch_mask(intervals, masks, id_dicts, filepaths, obj)


def stitch_attribute(
    dfs,
    obj,
    filepaths,
    attribute_dict,
    match_dicts,
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
    # Relabel universal ids based on match_dicts
    for i in range(len(intervals) - 1):
        match_dict = match_dicts[i][obj]
        reversed_match_dict = {v: k for k, v in match_dict.items()}
        current_interval = df["interval"] == i
        next_interval = df["interval"] == i + 1
        for next_key in reversed_match_dict.keys():
            current_key = reversed_match_dict[next_key]
            condition = current_interval & (df["original_id"] == current_key)
            universal_ids = df.loc[condition]["universal_id"].unique()
            if len(universal_ids) > 1:
                raise ValueError(f"Non unique universal id.")
            elif len(universal_ids) == 1:
                universal_id = int(universal_ids[0])
                condition = next_interval & (df["original_id"] == next_key)
                df.loc[condition, "universal_id"] = universal_id
            # Note we do nothing if universal_ids is empty, which can occur if the object
            # was only detected in the very last scan of the current interval
    return df
