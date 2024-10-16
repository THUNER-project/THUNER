"""Utility functions for analyzing thor output."""

from pathlib import Path
import yaml
import glob
import thor.option as option


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
