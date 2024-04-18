"""
Import modules.
from data.thor import check_cache
from detect.detect import detect
from group import group


def track(**kwargs):

    What should user keyword arguments include?
    Some method for creating new grids?

    Parse user arguments.
    Initialize.
    Create object analogous to record in TINT - simple dictionary?

    For each time i:
        If THOR grid for time i exists in cache:
            Load.
        Else:
            Load raw data.
            Convert to THOR grid.
            Save to cache if specified.

        For each level j in hierarchy:
            process_hierarchy_level(current_data, previous_data, i, j)
        conclude_iteration(current_data, previous_data, **kwargs):

    Export pandas dataframe to CSV and xarray datasets to netCDF.
    Return pandas and netCDF objects.


def track_level(track_options, input_options):

    Detect, match, and write object types at level level_ind.
    For each object type k at level_ind:
        If level_ind == 0:
            Detect objects in THOR grid at time i.
            objects = detect(user_args, current_data, **kwargs)
        Else:
            Form objects by grouping those at lower levels of
            the hierarchy.
            objects = group(user_args, current_data, **kwargs)
        If objects at time i-1 exist:
            Match objects between times i and i-1.
            Determine merge/split etc.
        Write objects to pandas.
        Write object masks to xarray.


def conclude_iteration(current_data, previous_data, **kwargs):
    Set previous_data to current_data.
"""
