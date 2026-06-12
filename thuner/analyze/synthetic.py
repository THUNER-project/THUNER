"""
Match synthetic ground-truth objects to detected objects.

After a tracking run, each synthetic dataset's ground-truth table can be augmented with
the uids of the detected objects whose masks overlap each truth object's expected
footprint, letting tracking output be scored against the known truth -- a single
overlapping uid marks an unambiguous match (e.g. for comparing true vs detected
velocities). The pure ground-truth table itself (derived from the options alone) lives in
:mod:`thuner.data.synthetic.truth`.

These functions read tracking output (options, masks) and write attributes, so they sit
in the ``analyze`` layer rather than in ``data`` — keeping the dependency direction
pointing downward and avoiding the circular import a ``data``-level home would force.
"""

import numpy as np
import pandas as pd
import xarray as xr
from thuner.log import setup_logger
from thuner.utils import store_path
from thuner.write.attribute import write_attribute
from thuner.analyze.utils import read_options
from thuner.data.synthetic.options import SyntheticOptions

logger = setup_logger(__name__)

# Recorded when a truth object's mask overlaps no detected object. An empty string,
# following the space-separated-id convention of the ``parents`` core attribute.
NO_MATCH = ""


def write_ground_truth(output_directory, times):
    """Write ground-truth tables for all synthetic datasets to the zarr store.

    Each synthetic dataset's generator is replayed *once* over ``times`` on ``grid_options``
    to recover both the objects and their ground-truth rows (so the truth matches what was
    rendered). When the dataset declares ``target_objects``, the recovered objects' truth
    masks are overlapped with the detected masks in the same pass, appending detected-uid
    columns without a second replay. Each table lands in a ``truth/<dataset_name>`` group;
    returns a ``{dataset_name: DataFrame}`` mapping of what was written.
    """
    options = read_options(output_directory)
    data_options, grid_options = options["data"], options["grid"]
    track_options = options["track"]
    store = store_path(output_directory)

    written = {}
    for dataset_options in data_options.datasets:
        if not isinstance(dataset_options, SyntheticOptions):
            continue
        # One replay recovers the objects, used both for the truth rows and -- when the
        # dataset is matched -- for the truth masks. This collapses the two passes (write
        # then match) that previously each replayed the whole scene into a single pass.
        objects = dataset_options.generator.replay(times, grid_options)
        truth = pd.DataFrame(obj.ground_truth() for obj in objects)
        truth = _match_targets(
            truth, objects, dataset_options, track_options, store, grid_options
        )
        truth = truth.set_index(["time", "id"]).sort_index()
        write_attribute(output_directory, "truth", dataset_options.name, df=truth)
        logger.info("Wrote ground truth for %s.", dataset_options.name)
        written[dataset_options.name] = truth
    return written


def _match_targets(truth, objects, dataset_options, track_options, store, grid_options):
    """Append detected-uid columns for ``dataset_options.target_objects``, if any.

    Returns ``truth`` unchanged when the dataset declares no targets (``None`` -> warn,
    since matching was presumably intended; empty -> nothing to match by design); otherwise
    overlaps each truth object's mask with the detected masks resolved from the store.
    """
    target_objects = dataset_options.target_objects
    if target_objects is None:
        logger.warning("Dataset %r has no target_objects.", dataset_options.name)
        return truth
    if not target_objects:
        return truth
    sources = _resolve_mask_sources(target_objects, track_options, store)
    return match_truth_to_masks(truth, sources, objects, grid_options)


def _matched_uids(obj, mask_das, grid_options):
    """The set of detected uids whose mask overlaps ``obj``'s truth mask.

    ``mask_das`` is the list of masks to search (one for a plain/member target, several
    for a grouped object's members, all carrying the group uid). The object's truth mask
    (:meth:`~thuner.data.synthetic.objects.SyntheticObject.to_mask`) is overlapped with
    each at ``obj``'s time; an empty set means it overlaps no detected object. More than
    one uid means the true object spans several detected objects -- recorded, not an error.
    """
    truth_mask = obj.to_mask(grid_options)
    uids = set()
    for mask_da in mask_das:
        mask_at_time = mask_da.sel(time=np.datetime64(obj.time), method="nearest")
        for cell_value in np.unique(mask_at_time.values[truth_mask]):
            cell_value = float(cell_value)
            if cell_value and not np.isnan(cell_value):
                uids.add(int(cell_value))
    return uids


def match_truth_to_masks(truth, mask_sources, objects, grid_options):
    """Append detected-uid columns to a ground-truth table by mask overlap."""
    matched = truth.copy()
    for column, mask_das in mask_sources.items():
        logger.info(f"Matching {column} ground truth to detected masks.")
        mask_das = [mask_da.load() for mask_da in mask_das]
        uid_strings = []
        for obj in objects:
            uids = _matched_uids(obj, mask_das, grid_options)
            uid_strings.append(" ".join(str(uid) for uid in sorted(uids)))
        matched[column] = uid_strings
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


def count_matches(matched, source):
    """Number of detected objects each truth object's mask overlaps.

    Reads the space-separated uid strings produced by :func:`match_truth_to_masks`: 0 means
    unmatched, 1 an unambiguous match (the case usable for true-vs-detected comparison) and
    >1 a true object spanning several detected objects.
    """
    return matched[source].apply(
        lambda value: 0 if pd.isna(value) else len(value.split())
    )
