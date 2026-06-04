"""
Ground-truth tables for synthetic datasets.

The state of every synthetic object is known exactly at every time, so a ground-truth
table can be derived directly from the options — no tracking run required. This lets
tracking output (object velocities, sizes, orientations) be checked against the known
truth. See :mod:`thuner.analyze.synthetic` for writing this truth to the zarr store and
matching it to detected objects.
"""

import pandas as pd


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
