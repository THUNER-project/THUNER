"""
Ground-truth tables for synthetic datasets.

The state of every synthetic object is known exactly at every time, so a ground-truth
table can be derived directly from the options — no tracking run required. This lets
tracking output (object velocities, sizes, orientations) be checked against the known
truth. The table is written to the unified zarr store as a ``truth/<dataset_name>``
group, alongside the tracked ``attributes``, ``masks`` and ``records`` groups.
"""

import pandas as pd
from thuner.log import setup_logger
from thuner.data.synthetic.options import SyntheticOptions

logger = setup_logger(__name__)


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
