"""
Ground-truth tables for synthetic datasets.

The state of every synthetic object is known exactly at every time, so a ground-truth
table can be derived directly from the options — no tracking run required. This lets
tracking output (object velocities, sizes, orientations) be checked against the known
truth. The table is written to the unified zarr store as a ``truth/<dataset_name>``
group, alongside the tracked ``attributes``, ``masks`` and ``records`` groups.
"""

import numpy as np
import pandas as pd
import xarray as xr
from thuner.log import setup_logger
from thuner.utils import store_path
from thuner.data.synthetic.options import SyntheticOptions

logger = setup_logger(__name__)

# Sentinel for "the truth object's centre fell in no detected mask" (object ids are >= 1).
NO_MATCH = 0


def synthetic_ground_truth(synthetic_options, times):
    """Build a per-object, per-time ground-truth table for one synthetic dataset.

    Each object is advanced from its starting state to each time in ``times``; the
    known attributes (id, position, velocity, geometry) form one row per object per
    time. Deterministic — derived from the options alone, no tracking run required.
    """
    rows = []
    for index, obj in enumerate(synthetic_options.objects):
        object_id = obj.id if obj.id is not None else index
        for time in times:
            truth = obj.advance(time).ground_truth()
            truth["id"] = object_id
            rows.append(truth)
    return pd.DataFrame(rows).set_index(["time", "id"]).sort_index()


def write_ground_truth(output_directory, data_options, times):
    """Write ground-truth tables for all synthetic datasets to the zarr store.

    Each synthetic dataset's truth lands in a ``truth/<dataset_name>`` group. Returns
    a ``{dataset_name: DataFrame}`` mapping of what was written.
    """
    # Lazy import: data is imported before write during package init.
    from thuner.write.attribute import write_attribute

    written = {}
    for dataset_options in data_options.datasets:
        if not isinstance(dataset_options, SyntheticOptions):
            continue
        df = synthetic_ground_truth(dataset_options, times)
        write_attribute(output_directory, "truth", dataset_options.name, df=df)
        logger.info("Wrote ground truth for %s.", dataset_options.name)
        written[dataset_options.name] = df
    return written


def _mask_cell_value(mask_at_time, latitude, longitude, grid_options):
    """Return the mask value at the cell nearest ``(latitude, longitude)``."""
    if grid_options.name == "geographic":
        value = mask_at_time.sel(
            latitude=latitude, longitude=longitude, method="nearest"
        )
        return float(value)
    # Cartesian: latitude/longitude are 2D curvilinear coords over the spatial dims.
    squared = (mask_at_time["latitude"] - latitude) ** 2
    squared = squared + (mask_at_time["longitude"] - longitude) ** 2
    indices = np.unravel_index(int(squared.values.argmin()), squared.shape)
    selector = dict(zip(squared.dims, indices))
    return float(mask_at_time.isel(**selector))


def _matched_uid(mask_das, time, latitude, longitude, grid_options, column):
    """The single detected uid whose mask contains ``(latitude, longitude)`` at ``time``.

    ``mask_das`` is the list of masks to search (one for a plain/member target, several
    for a grouped object's members). Returns ``NO_MATCH`` if the centre is in none.
    """
    uids = set()
    for mask_da in mask_das:
        mask_at_time = mask_da.sel(time=time, method="nearest")
        value = _mask_cell_value(mask_at_time, latitude, longitude, grid_options)
        if value and not np.isnan(value):
            uids.add(int(value))
    if len(uids) > 1:
        message = f"Truth centre matched multiple uids {sorted(uids)} for {column!r}; "
        message += "interleaved member masks are not supported."
        raise ValueError(message)
    return uids.pop() if uids else NO_MATCH


def match_truth_to_masks(truth, mask_sources, grid_options):
    """Append detected-uid columns to a ground-truth table by centre containment.

    Each truth row's object centre is looked up in the detected masks; the uid of the
    object whose mask contains it is recorded (``NO_MATCH`` if none). ``mask_sources``
    maps each output column name to the list of mask DataArrays to search.
    """
    matched = truth.copy()
    for column, mask_das in mask_sources.items():
        mask_das = [mask_da.load() for mask_da in mask_das]
        uids = [
            _matched_uid(
                mask_das, row.time, row.latitude, row.longitude, grid_options, column
            )
            for row in matched.itertuples()
        ]
        matched[column] = np.array(uids, dtype=np.int64)
    return matched


def _resolve_mask_sources(target_objects, track_options, store):
    """Resolve ``target_objects`` to ``{column_name: [mask DataArray, ...]}``.

    A plain object name resolves to its own ``{name}_mask``; a ``(group, member)`` tuple
    to that member's mask within the group; a grouped object name to the union of its
    member masks (all carrying the group uid). Column names follow
    ``{name}_{idtype}`` / ``{group}_{member}_{idtype}`` / ``{group}_{idtype}``.
    """

    def open_group(name):
        group = f"masks/{name}"
        return xr.open_dataset(store, group=group, engine="zarr", decode_timedelta=True)

    def id_type(obj):
        return "universal_id" if obj.tracking is not None else "id"

    sources = {}
    for target in target_objects:
        if isinstance(target, str):
            obj = track_options.object_by_name(target)
            grouping = getattr(obj, "grouping", None)
            dataset = open_group(target)
            column = f"{target}_{id_type(obj)}"
            if grouping is None:
                sources[column] = [dataset[f"{target}_mask"]]
            else:
                sources[column] = [
                    dataset[f"{member}_mask"] for member in grouping.member_objects
                ]
        else:
            group, member = target
            group_options = track_options.object_by_name(group)
            dataset = open_group(group)
            column = f"{group}_{member}_{id_type(group_options)}"
            sources[column] = [dataset[f"{member}_mask"]]
    return sources


def match_ground_truth(output_directory):
    """Augment each synthetic dataset's ground-truth table with detected-uid columns.

    For every synthetic dataset with ``target_objects`` set, records which detected
    object's mask contains each truth object's centre (per target), writing the new
    columns back to ``truth/<dataset_name>``. ``target_objects`` is assumed already
    validated by :class:`thuner.option.option.Options`.
    """
    from thuner.write.attribute import write_attribute
    from thuner.attribute.utils import read_attribute
    from thuner.analyze.utils import read_options

    options = read_options(output_directory)
    track_options, grid_options = options["track"], options["grid"]
    store = store_path(output_directory)

    matched_tables = {}
    for dataset_options in options["data"].datasets:
        if not isinstance(dataset_options, SyntheticOptions):
            continue
        if not dataset_options.target_objects:
            continue
        truth = read_attribute(output_directory, "truth", dataset_options.name)
        sources = _resolve_mask_sources(
            dataset_options.target_objects, track_options, store
        )
        matched = match_truth_to_masks(truth, sources, grid_options)
        matched = matched.set_index(["time", "id"]).sort_index()
        write_attribute(
            output_directory, "truth", dataset_options.name, df=matched, overwrite=True
        )
        logger.info("Matched ground truth for %s.", dataset_options.name)
        matched_tables[dataset_options.name] = matched
    return matched_tables
