"""Utility functions for analyzing thuner output."""

from pathlib import Path
import yaml
import glob
import numpy as np
import thuner.option as option
import thuner.attribute.core as core
from thuner.attribute.utils import read_attribute_csv
from thuner.option.attribute import Attribute, AttributeType
import thuner.write as write
import pandas as pd


__all__ = ["read_options"]


def quality_control(
    object_name,
    output_directory,
    analysis_options,
    analysis_directory=None,
):
    """
    Perform quality control on MCSs based on the provided options.

    Parameters
    ----------
    output_directory : str
        Path to the thuner run output directory.
    analysis_options : AnalysisOptions
        Options for analysis and quality control checks.

    Returns
    -------
    pd.DataFrame
        DataFrame describing quality control checks.
    """

    output_directory = Path(output_directory)
    if analysis_directory is None:
        analysis_directory = output_directory / "analysis"

    # Determine if the system is sufficiently contained within the domain
    filepath = output_directory / f"attributes/{object_name}/quality.csv"
    quality = read_attribute_csv(filepath)

    max_boundary_overlap = analysis_options.max_boundary_overlap
    quality = quality.rename(columns={"boundary_overlap": "contained"})
    overlap_check = quality["contained"] <= max_boundary_overlap

    # Check if velocity/shear vectors are sufficiently large
    filepath = analysis_directory / "velocities.csv"
    velocities = read_attribute_csv(filepath)
    velocity_magnitude = velocities[["u", "v"]].pow(2).sum(axis=1).pow(0.5)
    velocity_check = velocity_magnitude >= analysis_options.min_velocity
    velocity_check.name = "velocity"

    # Check system area is of appropriate size, treating the system area as the maximum
    # area of the member objects
    filepath = output_directory / f"attributes/{object_name}/core.csv"
    parents = read_attribute_csv(filepath, columns=["parents"])
    area = read_attribute_csv(filepath, columns=["area"])
    area = area.rename(columns={"area": f"{object_name}_area"})

    min_area, max_area = analysis_options.min_area, analysis_options.max_area
    area_check = (area >= min_area) & (area <= max_area)
    area_check.name = "area"

    # Check the duration of the system is sufficiently long
    # First get the duration of each object from the velocity dataframe
    id_group = velocities.reset_index().groupby("universal_id")["time"]
    duration = id_group.agg(lambda x: x.max() - x.min())
    duration_check = duration >= np.timedelta64(analysis_options.min_duration, "m")
    duration_check.name = "duration"
    dummy_df = velocities[[]].reset_index()
    merge_kwargs = {"on": "universal_id", "how": "left"}
    duration_check = dummy_df.merge(duration_check, **merge_kwargs)
    duration_check = duration_check.set_index(velocities.index.names)

    # Check if the object fails boundary overlap checks when first detected
    id_group = overlap_check.reset_index().groupby("universal_id")
    initially_contained = id_group.agg(lambda x: x.iloc[0])
    initially_contained = initially_contained.drop(columns="time")
    new_name = {"contained": "initially_contained"}
    initially_contained = initially_contained.rename(columns=new_name)
    dummy_df = velocities[[]].reset_index()
    initially_contained = dummy_df.merge(initially_contained, **merge_kwargs)
    initially_contained = initially_contained.set_index(velocities.index.names)

    # Check whether the object has parents. When plotting we may only wish to filter out
    # short duration objects if they are not part of a larger system
    parents_check = parents.reset_index().groupby("universal_id")["parents"]
    parents_check = parents_check.agg(lambda x: x.notna().any())
    parents_check = dummy_df.merge(parents_check, on="universal_id", how="left")
    parents_check = parents_check.set_index(velocities.index.names)

    # Record whether the given object has children, using the parents column
    has_parents = parents["parents"].dropna()
    children_check = pd.Series(False, index=velocities.index, name="children")
    children_check = children_check.reset_index()
    for i in range(len(has_parents)):
        parents = [int(p) for p in has_parents.iloc[i].split(" ")]
        for parent in parents:
            row_cond = children_check["universal_id"] == parent
            children_check.loc[row_cond, "children"] = True
    children_check = children_check.set_index(velocities.index.names)

    # Check the linearity of the system
    filepath = output_directory / f"attributes/{object_name}/ellipse.csv"
    ellipse = read_attribute_csv(filepath, columns=["major", "minor"])
    major_check = ellipse["major"] >= analysis_options.min_major_axis_length
    major_check.name = "major_axis"
    axis_ratio = ellipse["major"] / ellipse["minor"]
    axis_ratio_check = axis_ratio >= analysis_options.min_axis_ratio
    axis_ratio_check.name = "axis_ratio"

    names = ["contained", "initially_contained", "velocity", "area"]
    names += ["major_axis", "axis_ratio", "duration", "parents", "children"]
    descriptions = [
        "Is the object sufficiently contained within the domain?",
        "Is the object contained within the domain when first detected?",
        "Is the object velocity sufficiently large?",
        "Is the object area sufficiently large?",
        "Is the object major axis length sufficiently large?",
        "Is the object axis ratio sufficiently large?",
        "Is the object duration sufficiently long?",
        "Does the object have parents?",
        "Does the object have children?",
    ]

    data_type, precision, units, retrieval = bool, None, None, None
    attributes = []
    for name, description in zip(names, descriptions):
        kwargs = {"name": name, "retrieval": retrieval, "data_type": data_type}
        kwargs.update({"precision": precision, "description": description})
        kwargs.update({"units": units})
        attributes.append(Attribute(**kwargs))

    attributes.append(core.time())
    attributes.append(core.record_universal_id())
    attribute_type = AttributeType(name="quality", attributes=attributes)
    filepath = analysis_directory / "quality.csv"
    quality = [overlap_check, initially_contained, velocity_check, area_check]
    quality += [major_check, axis_ratio_check, duration_check]
    quality += [parents_check, children_check]
    quality = pd.concat(quality, axis=1)
    quality = write.attribute.write_csv(filepath, quality, attribute_type)


def smooth_flow_velocities(filepath, output_directory, window_size=6):
    """Smooth the flow velocities."""
    velocities = read_attribute_csv(filepath, columns=["u_flow", "v_flow"])

    velocities = temporal_smooth(velocities, window_size=window_size)
    velocities = velocities.rename(columns={"u_flow": "u", "v_flow": "v"})

    analysis_directory = output_directory / "analysis"

    # Create metadata for the attributes
    names = ["u", "v"]
    descriptions = [
        "System ground relative zonal velocity.",
        "System ground relative meridional velocity.",
    ]

    data_type, precision, units, retrieval = float, 1, "m/s", None
    attributes = []
    for name, description in zip(names, descriptions):
        kwargs = {"name": name, "retrieval": retrieval, "data_type": data_type}
        kwargs.update({"precision": precision, "description": description})
        kwargs.update({"units": units})
        attributes.append(Attribute(**kwargs))
    attributes.append(core.time())
    attributes.append(core.record_universal_id())
    attribute_type = AttributeType(name="velocities", attributes=attributes)
    filepath = analysis_directory / "velocities.csv"
    write.attribute.write_csv(filepath, velocities, attribute_type)


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
                options = option.track.TrackOptions(**options)
            if name == "data":
                options = option.data.DataOptions(**options)
            if name == "grid":
                options = option.grid.GridOptions(**options)
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
