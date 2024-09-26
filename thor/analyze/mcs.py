"""
Methods for analyzing MCSs. In particular, for implementing the methodologies 
presented in the following papers:

Short et al. (2023), Objectively diagnosing characteristics of mesoscale organization 
from radar reflectivity and ambient winds. https://dx.doi.org/10.1175/MWR-D-22-0146.1

Note the general efficiency strategy is to use dask arrays. Note care must be taken to
deal with multi-indexes created after grouping etc.  
"""

import pandas as pd
from thor.attribute.utils import read_attribute_csv
import thor.analyze.utils as utils


def temporal_smooth(df, window_size=6, dask=True):
    """
    Apply a temporal smoother to each object.
    """
    # columns = [col for col in df.columns if col != "universal_id"]

    def smooth_group(group):
        group_no_id = group.drop(columns=["universal_id"])
        smoothed = group_no_id.rolling(window=window_size, min_periods=1, center=True)
        smoothed = smoothed.mean()
        smoothed["universal_id"] = group["universal_id"]
        # Reorder columns so meta works
        smoothed = smoothed[list(df.columns)]
        return smoothed

    smoothed_df = df.groupby("universal_id", group_keys=False)
    if dask:
        smoothed_df = smoothed_df.apply(smooth_group, meta=smoothed_df.head(0))
    else:
        smoothed_df = smoothed_df.apply(smooth_group)
    return smoothed_df


def simple_attribute_metadata(names, precision=1):
    all_attributes = {}
    attribute_dict = {"data_type": "float", "precision": precision, "method": None}
    for name in names:
        all_attributes[name] = attribute_dict
        all_attributes[name].update({"name": name})
    return all_attributes


def process_cell(output_directory, convective_label="cell", window_size=6):
    """Process convective objects ready for analysis."""

    dask = True
    multi_index = False
    options = utils.read_options(output_directory)
    cell_options = options["track"][0][convective_label]
    cell_altitudes = cell_options["detection"]["altitudes"]
    profile_filepath = output_directory / "attributes/mcs/profile.csv"
    winds = read_attribute_csv(profile_filepath, dask=dask, multi_index=multi_index)
    winds = winds[["universal_id", "altitude", "u", "v"]]
    winds = winds
    core_filepath = output_directory / "attributes/mcs/core.csv"
    velocities = read_attribute_csv(core_filepath, dask=dask, multi_index=multi_index)
    velocities = velocities[["universal_id", "u_flow", "v_flow"]]
    velocities = temporal_smooth(velocities, window_size=window_size, dask=dask)
    velocities = velocities.rename(columns={"u_flow": "u", "v_flow": "v"})

    cell_winds = winds[winds["altitude"].between(*cell_altitudes)]
    cell_mean_winds = cell_winds.groupby(["time", "universal_id"]).mean()[["u", "v"]]
    # Reset the index to remove the multi-index created by grouping
    cell_mean_winds = cell_mean_winds.reset_index().set_index(["time"])
    cell_top = winds[winds["altitude"] == cell_altitudes[1]][["u", "v"]]
    cell_bottom = winds[winds["altitude"] == cell_altitudes[0]][["u", "v"]]
    cell_shear = cell_top - cell_bottom
    relative_velocities = velocities[["u", "v"]] - cell_mean_winds[["u", "v"]]
    cell_mean_winds = cell_mean_winds.rename(
        columns={"u": "u_ambient_mean", "v": "v_ambient_mean"}
    )
    cell_shear = cell_shear.rename(
        columns={"u": "u_ambient_shear", "v": "v_ambient_shear"}
    )
    relative_velocities = relative_velocities.rename(
        columns={"u": "u_relative", "v": "v_relative"}
    )
    names = ["u", "v", "u_ambient_shear", "v_ambient_shear"]
    names += ["u_ambient_mean", "v_ambient_mean"]
    names += ["u_relative", "v_relative"]
    attributes = simple_attribute_metadata(names)
    return (
        velocities,
        cell_mean_winds,
        cell_shear,
        relative_velocities,
        attributes,
    )
