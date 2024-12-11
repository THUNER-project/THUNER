"""Utility functions for analyzing thuner output."""

from pathlib import Path
import yaml
import glob
import numpy as np
import thuner.option as option
from thuner.data.option import DataOptions
import thuner.grid as grid


def get_angle(u1, v1, u2, v2):
    """
    Get the angle between two vectors. Angle calculated as second vector direction minus
    first vector direction.
    """

    angle_1 = np.arctan2(v1, u1)
    angle_2 = np.arctan2(v2, u2)
    # Get angle between vectors, but signed so that in range -np.pi to np.pi
    return np.mod(angle_2 - angle_1 + np.pi, 2 * np.pi) - np.pi


def read_options(output_directory):
    """Read run options from yml files."""
    options_directory = Path(output_directory) / "options"
    options_filepaths = glob.glob(str(options_directory / "*.yml"))
    all_options = {}
    for filepath in options_filepaths:
        with open(filepath, "r") as file:
            options = yaml.safe_load(file)
            name = Path(filepath).stem
            if name == "track":
                options = option.TrackOptions(**options)
            if name == "data":
                options = DataOptions(**options)
            if name == "grid":
                options = grid.GridOptions(**options)
            all_options[name] = options
    return all_options


def temporal_smooth(df, window_size=6):
    """
    Apply a temporal smoother to each object.
    """

    def smooth_group(group):
        smoothed = group.rolling(window=window_size, min_periods=1, center=True).mean()
        return smoothed

    # Group over all indexes except time, i.e. only smooth over time index
    indexes_to_group = [idx for idx in df.index.names if idx != "time"]
    smoothed_df = df.groupby(indexes_to_group, group_keys=False)
    smoothed_df = smoothed_df.apply(smooth_group)
    return smoothed_df
