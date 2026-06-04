"""
Ground-truth tables for synthetic datasets.

The state of every synthetic object is known exactly at every time, so a ground-truth
table can be derived by replaying the dataset's generator over the same times and grid
used for tracking. This lets tracking output (object velocities, sizes, orientations) be
checked against the known truth. See :mod:`thuner.analyze.synthetic` for writing this
truth to the zarr store and matching it to detected objects.
"""


def synthetic_ground_truth(synthetic_options, times, grid_options):
    """Build a per-object, per-time ground-truth table for one synthetic dataset.

    Replays the dataset's generator over ``times`` on ``grid_options`` (the same stepping
    used to render the data), so the table is consistent with the rendered field by
    construction. Returns a DataFrame indexed by ``(time, id)``.
    """
    return synthetic_options.generator.ground_truth(times, grid_options)
