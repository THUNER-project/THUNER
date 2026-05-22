"""Parallel processing utilities."""

import shutil
import gc
import os
import multiprocessing as mp
import time
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
from thuner.config import get_zarr_store_name

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
    # times = utils.generate_times(filepaths)
    dataset_options = interval_data_options.dataset_by_name(dataset_name)
    times = utils.generate_dataset_times(dataset_options)
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


def _relabel_attribute_dfs(
    dfs,
    obj,
    attribute_dict,
    match_dicts,
    time_dicts,
    id_dicts,
    intervals,
    tracked_objects,
):
    """Relabel ids across per-interval attribute dfs and concatenate.

    Pure DataFrame manipulation — does not touch disk. Used by both the CSV
    and zarr stitching paths.

    Returns
    -------
    (df, id_dict): concatenated/relabeled DataFrame and the per-interval
        old-id → new-id lookup table for this object.
    """
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
    return df, id_dict


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


def _interval_store(output_parent, i):
    """Return the path to the zarr store for parallel interval ``i``."""
    return Path(output_parent) / f"interval_{i}" / get_zarr_store_name()


def _run_store(output_parent):
    """Return the path to the unified zarr store at the run root."""
    return Path(output_parent) / get_zarr_store_name()


def _is_leaf_zarr_group(path: Path) -> bool:
    """A leaf zarr group has children that are arrays (.zarray), not groups."""
    if not path.is_dir():
        return False
    for child in path.iterdir():
        if child.is_dir() and (child / ".zarray").exists():
            return True
    return False


def list_leaf_groups(store_path: Path, parent_relpath: str):
    """Yield posix-style relpaths of leaf groups under parent_relpath."""
    root = Path(store_path) / parent_relpath
    if not root.exists():
        return
    for path in sorted(root.rglob("*")):
        if path.is_dir() and _is_leaf_zarr_group(path):
            yield path.relative_to(store_path).as_posix()


def get_group_dicts(output_parent, intervals):
    """Discover zarr groups in each interval's unified zarr store.

    Returns three dicts (attribute, mask, record) keyed by interval index.
    Each value is a sorted list of group paths (relative to that interval's
    store) so they line up across intervals.
    """
    attr_dict, mask_dict, record_dict = {}, {}, {}
    for i in range(len(intervals)):
        store = _interval_store(output_parent, i)
        attr_dict[i] = sorted(list_leaf_groups(store, "attributes"))
        mask_dict[i] = sorted(list_leaf_groups(store, "masks"))
        record_dict[i] = sorted(list_leaf_groups(store, "records"))
    if len(np.unique([len(l) for l in attr_dict.values()])) != 1:
        raise ValueError("Different number of attribute groups for each interval")
    if len(np.unique([len(l) for l in mask_dict.values()])) != 1:
        raise ValueError("Different number of mask groups for each interval")
    return attr_dict, mask_dict, record_dict


def get_match_dicts(output_parent, intervals, mask_group_dict, tracked_objects):
    """Build per-interval id-match dicts from per-interval zarr stores."""
    match_dicts, time_dicts = {}, {}
    for i in range(len(intervals) - 1):
        groups_1 = mask_group_dict[i]
        groups_2 = mask_group_dict[i + 1]
        # Group is "masks/<obj>"; extract obj from the last component
        objects_1 = [g.rsplit("/", 1)[-1] for g in groups_1]
        objects_2 = [g.rsplit("/", 1)[-1] for g in groups_2]
        if objects_1 != objects_2:
            raise ValueError("Different objects in each interval's mask groups.")
        store_1 = _interval_store(output_parent, i)
        store_2 = _interval_store(output_parent, i + 1)

        interval_match_dicts, interval_time_dicts = {}, {}
        for j, obj in enumerate(objects_1):
            kwargs = {"engine": "zarr", "chunks": {}}
            ds_2 = xr.open_dataset(store_2, group=groups_2[j], **kwargs)
            ds_2 = ds_2.isel(time=0).load()
            time = ds_2["time"].values
            interval_time_dicts[obj] = time
            ds_1 = xr.open_dataset(store_1, group=groups_1[j], **kwargs)
            if time not in ds_1.time:
                if obj not in tracked_objects:
                    interval_match_dicts[obj] = None
                else:
                    interval_match_dicts[obj] = {}
                continue
            ds_1 = ds_1.sel(time=time).load()
            if obj not in tracked_objects:
                interval_match_dicts[obj] = None
            else:
                interval_match_dicts[obj] = match_dataset(ds_1, ds_2)

        match_dicts[i] = interval_match_dicts
        time_dicts[i] = interval_time_dicts
    return match_dicts, time_dicts


def _build_attribute_type_lookup(track_options):
    """Build a ``{group_path: AttributeType}`` lookup from track options.

    Covers both top-level object attribute types and member-object attribute
    types. Filepath-records groups are handled separately because they aren't
    in ``track_options``.
    """
    lookup: dict[str, "attribute.utils.AttributeType"] = {}
    for level in track_options.levels:
        for object_options in level.objects:
            if object_options.attributes is None:
                continue
            obj_name = object_options.name
            for attr_type in object_options.attributes.attribute_types:
                lookup[f"attributes/{obj_name}/{attr_type.name}"] = attr_type
            member_attrs = object_options.attributes.member_attributes
            if member_attrs is None:
                continue
            for member, attrs in member_attrs.items():
                for attr_type in attrs.attribute_types:
                    key = f"attributes/{obj_name}/{member}/{attr_type.name}"
                    lookup[key] = attr_type
    return lookup


def stitch_records(output_parent, intervals, record_group_dict):
    """Stitch per-interval filepath records into the unified zarr store."""
    logger.info("Stitching record groups.")
    out_store = _run_store(output_parent)
    n_groups = len(record_group_dict[0])
    for k in range(n_groups):
        group = record_group_dict[0][k]
        # Filepath record groups have path ``records/filepaths/<dataset_name>``;
        # rebuild the AttributeType deterministically from that name.
        dataset_name = group.rsplit("/", 1)[-1]
        attribute_type = write.filepath._filepath_attribute_type(dataset_name)
        dfs = [
            attribute.utils.read_attribute_zarr(
                _interval_store(output_parent, i), group
            )
            for i in range(len(intervals))
        ]
        df = pd.concat(dfs).sort_index()
        df = df.reset_index().drop_duplicates().set_index("time")
        write.attribute.write_attributes(out_store, group, df, attribute_type)


def stitch_masks(output_parent, intervals, mask_group_dict, id_dicts):
    """Stitch per-interval mask groups into the unified zarr store."""
    logger.info("Stitching mask groups.")
    out_store = _run_store(output_parent)
    n_groups = len(mask_group_dict[0])
    for k in range(n_groups):
        group = mask_group_dict[0][k]
        obj = group.rsplit("/", 1)[-1]
        masks = []
        for i in range(len(intervals)):
            store = _interval_store(output_parent, i)
            kwargs = {"engine": "zarr", "chunks": {"time": 1}}
            masks.append(xr.open_dataset(store, group=group, **kwargs))
        new_masks = []
        for i in range(len(intervals)):
            mask = masks[i]
            mapping = get_mapping(id_dicts, obj, i)
            new_mask = apply_mapping(mapping, mask)
            if i > 0:
                time = masks[i - 1].time[-1].values
                if time not in np.array(masks[i].time.values):
                    message = (
                        "Time intervals have produced non-overlapping time domains "
                    )
                    message += "for masks. This can occur due to missing files at the "
                    message += " overlap time."
                    logger.warning(message)
                else:
                    condition = new_mask.time.values > time
                    new_mask = new_mask.sel(time=condition)
            new_masks.append(new_mask)
        mask = xr.concat(new_masks, dim="time")
        mask = mask.astype(np.uint32)
        coords = [c for c in mask.coords if c in ["x", "y", "latitude", "longitude"]]
        for coord in coords:
            mask.coords[coord] = mask.coords[coord].astype(np.float32)
        out_store.parent.mkdir(parents=True, exist_ok=True)
        # Overwrite the group inside the unified store
        mask.to_zarr(out_store, group=group, mode="w")


def stitch_run(output_parent, intervals, cleanup=True):
    """Stitch per-interval zarr stores into a single unified ``<configured zarr store>``.

    Reads each interval's unified zarr store store, relabels universal_ids, and writes
    the unified result to ``<output_parent>/<configured zarr store>/``.
    """
    logger.info("Stitching all attribute, mask and record groups.")
    options = analyze.utils.read_options(output_parent / "interval_0")
    track_options = options["track"]
    tracked_objects = get_tracked_objects(track_options)[0]
    attr_group_dict, mask_group_dict, record_group_dict = get_group_dicts(
        output_parent, intervals
    )
    args = [output_parent, intervals, mask_group_dict, tracked_objects]
    match_dicts, time_dicts = get_match_dicts(*args)
    stitch_records(output_parent, intervals, record_group_dict)

    # Copy regridder weights folder if it exists.
    weights_path_0 = output_parent / "interval_0" / "regridder_weights"
    weights_path = output_parent / "regridder_weights"
    if weights_path_0.exists():
        shutil.copytree(weights_path_0, weights_path, dirs_exist_ok=True)

    out_store = _run_store(output_parent)
    id_dicts = {}
    logger.info("Stitching attribute groups.")
    attribute_type_lookup = _build_attribute_type_lookup(track_options)
    for k in range(len(attr_group_dict[0])):
        group = attr_group_dict[0][k]  # e.g. "attributes/mcs/conv/core"
        # Group structure: attributes/<obj>/(<member>/)?<at_name>
        parts = group.split("/")
        obj = parts[1]
        member_object = len(parts) != 3
        attribute_type = attribute_type_lookup.get(group)
        if attribute_type is None:
            raise KeyError(
                f"No AttributeType found in track options for group {group!r}."
            )
        dfs = [
            attribute.utils.read_attribute_zarr(
                _interval_store(output_parent, i), group
            )
            for i in range(len(intervals))
        ]
        df, id_dict = _relabel_attribute_dfs(
            dfs,
            obj,
            attribute_type,
            match_dicts,
            time_dicts,
            id_dicts,
            intervals,
            tracked_objects,
        )
        write.attribute.write_attributes(out_store, group, df, attribute_type)
        if not member_object and obj in tracked_objects:
            id_dicts[obj] = id_dict
    stitch_masks(output_parent, intervals, mask_group_dict, id_dicts)

    if cleanup:
        for i in range(len(intervals)):
            shutil.rmtree(Path(output_parent / f"interval_{i}"))


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
