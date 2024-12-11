"""
Functions for analyzing MCSs. In particular, for implementing the methodologies 
presented in the following papers:

Short et al. (2023), Objectively diagnosing characteristics of mesoscale organization 
from radar reflectivity and ambient winds. https://dx.doi.org/10.1175/MWR-D-22-0146.1
"""

import numpy as np
import pandas as pd
from pathlib import Path
from pydantic import Field
from thuner.attribute.utils import read_attribute_csv, get_attribute_dict
import thuner.analyze.utils as utils
import thuner.write as write
import thuner.attribute as attribute
import thuner.log as log
import thuner.option as option

logger = log.setup_logger(__name__)


def process_velocities(
    output_directory, window_size=6, analysis_directory=None, profile_dataset="era5_pl"
):
    """
    Process winds and velocities for analysis.

    Parameters
    ----------
    output_directory : str
        Path to the thuner run output directory.
    """

    if analysis_directory is None:
        analysis_directory = output_directory / "analysis"

    options = utils.read_options(output_directory)
    mcs_options = options["track"].levels[1].options_by_name("mcs")
    member_objects = mcs_options.grouping.member_objects
    convective_label = member_objects[0]

    convective_options = options["track"].levels[0].options_by_name(convective_label)
    altitudes = convective_options.detection.altitudes

    filepath = output_directory / "attributes/mcs/core.csv"
    velocities = read_attribute_csv(filepath, columns=["u_flow", "v_flow"])
    velocities = utils.temporal_smooth(velocities, window_size=window_size)
    velocities = velocities.rename(columns={"u_flow": "u", "v_flow": "v"})

    if profile_dataset is not None:
        filepath = output_directory / f"attributes/mcs/{profile_dataset}/profile.csv"
        winds = read_attribute_csv(filepath, columns=["u", "v"])

        indexer = pd.IndexSlice[:, :, altitudes[0] : altitudes[1]]
        mean_winds = winds.loc[indexer].groupby(["time", "universal_id"]).mean()
        new_names = {"u": "u_ambient", "v": "v_ambient"}
        mean_winds = mean_winds.rename(columns=new_names)

        # Calculate a shear vector as the difference between the winds at the top and
        # bottom of layer used to detect convective echoes
        top = winds.xs(altitudes[1], level="altitude")
        bottom = winds.xs(altitudes[0], level="altitude")
        shear = top - bottom
        new_names = {"u": "u_shear", "v": "v_shear"}
        shear = shear.rename(columns=new_names)

        # Calculate system wind-relative velocities
        new_names_vel = {"u": "u_relative", "v": "v_relative"}
        renamed_velocities = velocities.rename(columns=new_names_vel)
        new_names_mean = {"u_ambient": "u_relative", "v_ambient": "v_relative"}
        renamed_mean_winds = mean_winds.rename(columns=new_names_mean)
        relative_velocities = renamed_velocities - renamed_mean_winds

        velocities_list = [velocities, mean_winds, shear, relative_velocities]
    else:
        velocities_list = [velocities]

    # Check if dataframes aligned
    if profile_dataset is not None and not velocities.index.equals(mean_winds.index):
        raise ValueError("Dataframes are not aligned. Perhaps enforce alignment first?")

    all_velocities = pd.concat(velocities_list, axis=1)

    # Create metadata for the attributes
    names = ["u", "v", "u_shear", "v_shear", "u_ambient", "v_ambient"]
    names += ["u_relative", "v_relative"]
    descriptions = [
        "System ground relative zonal velocity.",
        "System ground relative meridional velocity.",
        f"Ambient zonal shear between {altitudes[0]} and {altitudes[1]} m.",
        f"Ambient meridional between {altitudes[0]} and {altitudes[1]} m.",
        f"Mean ambient zonal winds from {altitudes[0]} and {altitudes[1]} m.",
        f"Mean ambient meridional winds from {altitudes[0]} and {altitudes[1]} m.",
        "System wind relative zonal velocity.",
        "System wind relative meridional velocity.",
    ]
    if "u_shear" not in all_velocities.columns:
        names = names[:2]
        descriptions = descriptions[:2]

    data_type, precision, units, method = float, 1, "m/s", None
    attributes = {}
    for name, description in zip(names, descriptions):
        args = [name, method, data_type, precision, description, units]
        attributes[name] = get_attribute_dict(*args)
    attributes["time"] = attribute.core.time()
    attributes["universal_id"] = attribute.core.identity("universal_id")
    filepath = analysis_directory / "velocities.csv"
    all_velocities = write.attribute.write_csv(filepath, all_velocities, attributes)
    return all_velocities


_summary = {
    "window_size": "Window size for temporal smoothing of velocities.",
    "min_area": "Minimum area of MCS in km^2.",
    "max_area": "Maximum area of MCS in km^2.",
    "max_boundary_overlap": "Maximum fraction of system member object pixels touching boundary.",
    "min_major_axis_length": "Minimum major axis length of MCS in km.",
    "min_axis_ratio": "Minimum axis ratio of MCS.",
    "min_duration": "Minimum duration of MCS in minutes.",
    "min_offset": "Minimum stratiform offset in km.",
    "min_shear": "Minimum shear in m/s.",
    "min_velocity": "Minimum velocity in m/s.",
    "min_relative_velocity": "Minimum relative velocity in m/s.",
    "quadrant_buffer_angle": "Buffer angle in degrees for quadrant based classification.",
}


class AnalysisOptions(option.BaseOptions):
    """Options for convective system analysis."""

    window_size: int = Field(6, description=_summary["window_size"], ge=1)
    min_area: float = Field(1e2, description=_summary["min_area"], ge=0)
    max_area: float = Field(np.inf, description=_summary["max_area"], gt=0)
    max_boundary_overlap: float = Field(
        1e-3, description=_summary["max_boundary_overlap"], gt=0
    )
    min_major_axis_length: float = Field(
        25, description=_summary["min_major_axis_length"], ge=0
    )
    min_axis_ratio: float = Field(2, description=_summary["min_axis_ratio"], ge=0)
    min_duration: float = Field(30, description=_summary["min_duration"], ge=0)
    min_offset: float = Field(10, description=_summary["min_offset"], ge=0)
    min_shear: float = Field(2, description=_summary["min_shear"], ge=0)
    min_velocity: float = Field(5, description=_summary["min_velocity"], ge=0)
    min_relative_velocity: float = Field(
        2, description=_summary["min_relative_velocity"], ge=0
    )
    quadrant_buffer_angle: float = Field(
        10, description=_summary["quadrant_buffer_angle"], ge=0
    )


def quality_control(
    output_directory, analysis_options: AnalysisOptions, analysis_directory=None
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

    options = utils.read_options(output_directory)
    mcs_options = options["track"].levels[1].options_by_name("mcs")
    member_objects = mcs_options.grouping.member_objects
    convective_label = member_objects[0]
    anvil_label = member_objects[1]

    # Determine if the system is sufficiently contained within the domain
    filepath = output_directory / f"attributes/mcs/{convective_label}/quality.csv"
    convective = read_attribute_csv(filepath)
    filepath = output_directory / f"attributes/mcs/{anvil_label}/quality.csv"
    anvil = read_attribute_csv(filepath)
    filepath = output_directory / "attributes/mcs/core.csv"
    mcs = read_attribute_csv(filepath, columns=["parents"])
    max_boundary_overlap = analysis_options.max_boundary_overlap
    convective = convective.rename(columns={"boundary_overlap": "convective_contained"})
    anvil = anvil.rename(columns={"boundary_overlap": "anvil_contained"})
    convective_check = convective["convective_contained"] <= max_boundary_overlap
    anvil_check = anvil["anvil_contained"] < max_boundary_overlap

    # Check if velocity/shear vectors are sufficiently large
    filepath = analysis_directory / "velocities.csv"
    velocities = read_attribute_csv(filepath)
    velocity_magnitude = velocities[["u", "v"]].pow(2).sum(axis=1).pow(0.5)
    velocity_check = velocity_magnitude >= analysis_options.min_velocity
    velocity_check.name = "velocity"
    if "u_shear" in velocities.columns:
        shear_magnitude = velocities[["u_shear", "v_shear"]].pow(2).sum(axis=1).pow(0.5)
        shear_check = shear_magnitude >= analysis_options.min_shear
        shear_check.name = "shear"
        relative_velocity = velocities[["u_relative", "v_relative"]]
        relative_velocity_magnitude = relative_velocity.pow(2).sum(axis=1).pow(0.5)
        min_relative_velocity = analysis_options.min_relative_velocity
        relative_velocity_check = relative_velocity_magnitude >= min_relative_velocity
        relative_velocity_check.name = "relative_velocity"

    # Check system area is of appropriate size, treating the system area as the maximum
    # area of the member objects
    all_areas = []
    for obj in member_objects:
        filepath = output_directory / f"attributes/mcs/{obj}/core.csv"
        area = read_attribute_csv(filepath, columns=["area"])
        area = area.rename(columns={"area": f"{obj}_area"})
        all_areas.append(area)
    area = pd.concat(all_areas, axis=1).max(axis=1)
    min_area, max_area = analysis_options.min_area, analysis_options.max_area
    area_check = (area >= min_area) & (area <= max_area)
    area_check.name = "area"

    # Check the stratiform offset is sufficiently large
    filepath = output_directory / f"attributes/mcs/group.csv"
    offset = read_attribute_csv(filepath, columns=["x_offset", "y_offset"])
    offset_magnitude = offset.pow(2).sum(axis=1).pow(0.5)
    offset_check = offset_magnitude >= analysis_options.min_offset
    offset_check.name = "offset"

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
    both_contained = pd.concat([convective_check, anvil_check], axis=1).all(axis=1)
    id_group = both_contained.reset_index().groupby("universal_id")
    initial_check = id_group.agg(lambda x: x.iloc[0])
    initial_check = initial_check.drop(columns="time")
    new_name = {0: "initially_contained"}
    initial_check = initial_check.rename(columns=new_name)
    dummy_df = velocities[[]].reset_index()
    initial_check = dummy_df.merge(initial_check, **merge_kwargs)
    initial_check = initial_check.set_index(velocities.index.names)

    # Check whether the object has parents. When plotting we may only wish to filter out
    # short duration objects if they are not part of a larger system
    parents_check = mcs.reset_index().groupby("universal_id")["parents"]
    parents_check = parents_check.agg(lambda x: x.notna().any())
    parents_check = dummy_df.merge(parents_check, on="universal_id", how="left")
    parents_check = parents_check.set_index(velocities.index.names)

    # Record whether the given object has children, using the parents column
    has_parents = mcs["parents"].dropna()
    children_check = pd.Series(False, index=velocities.index, name="children")
    children_check = children_check.reset_index()
    for i in range(len(has_parents)):
        parents = [int(p) for p in has_parents.iloc[i].split(" ")]
        for parent in parents:
            row_cond = children_check["universal_id"] == parent
            children_check.loc[row_cond, "children"] = True
    children_check = children_check.set_index(velocities.index.names)

    # Check the linearity of the system
    filepath = output_directory / f"attributes/mcs/{convective_label}/ellipse.csv"
    ellipse = read_attribute_csv(filepath, columns=["major", "minor"])
    major_check = ellipse["major"] >= analysis_options.min_major_axis_length
    major_check.name = "major_axis"
    axis_ratio = ellipse["major"] / ellipse["minor"]
    axis_ratio_check = axis_ratio >= analysis_options.min_axis_ratio
    axis_ratio_check.name = "axis_ratio"

    names = ["convective_contained", "anvil_contained", "initially_contained"]
    names += ["velocity", "shear", "relative_velocity", "area", "offset"]
    names += ["major_axis", "axis_ratio", "duration", "parents", "children"]
    descriptions = [
        "Is the system convective region sufficiently contained within the domain?",
        "Is the system anvil region sufficiently contained within the domain?",
        "Is the system contained within the domain when first detected?",
        "Is the system velocity sufficiently large?",
        "Is the system shear sufficiently large?",
        "Is the system relative velocity sufficiently large?",
        "Is the system area sufficiently large?",
        "Is the system stratiform offset sufficiently large?",
        "Is the system major axis length sufficiently large?",
        "Is the system axis ratio sufficiently large?",
        "Is the system duration sufficiently long?",
        "Does the system have parent systems?",
        "Does the system have children systems?",
    ]
    if "u_shear" not in velocities.columns:
        names.remove("shear")
        names.remove("relative_velocity")
        descriptions.remove("Is the system shear sufficiently large?")
        descriptions.remove("Is the system relative velocity sufficiently large?")

    data_type, precision, units, method = bool, None, None, None
    attributes = {}
    for name, description in zip(names, descriptions):
        args = [name, method, data_type, precision, description, units]
        attributes[name] = get_attribute_dict(*args)
    attributes["time"] = attribute.core.time()
    attributes["universal_id"] = attribute.core.identity("universal_id")
    filepath = analysis_directory / "quality.csv"
    quality = [convective_check, anvil_check, initial_check, velocity_check, area_check]
    quality += [offset_check, major_check, axis_ratio_check, duration_check]
    quality += [parents_check, children_check]
    if "u_shear" in velocities.columns:
        quality += [shear_check, relative_velocity_check]
    quality = pd.concat(quality, axis=1)
    quality = write.attribute.write_csv(filepath, quality, attributes)
    return quality


ambiguity_quality_dispatcher = {
    "stratiform_offset": ["velocity"],
    "inflow": ["velocity", "relative_velocity"],
    "relative_stratiform_offset": ["relative_velocity"],
    "tilt": ["shear"],
    "propagation": ["shear", "relative_velocity"],
}


def classify_all(
    output_directory,
    analysis_options: AnalysisOptions,
    analysis_directory=None,
    offset_filepath=None,
    velocities_filepath=None,
    quality_filepath=None,
    classify_small_offsets=False,
    classify_ambiguous=False,
):
    """
    Classify MCSs based on quadrants, as described in Short et al. (2023).

    Parameters
    ----------
    output_directory : str
        Path to the thuner run output directory.
    analysis_options : dict
        Dictionary of quality control options.

    Returns
    -------
    pd.DataFrame
        DataFrame describing MCS classifications.
    """
    if analysis_directory is None:
        analysis_directory = output_directory / "analysis"

    if velocities_filepath is None:
        velocities_filepath = analysis_directory / "velocities.csv"
    velocities = read_attribute_csv(velocities_filepath)
    if offset_filepath is None:
        offset_filepath = output_directory / "attributes/mcs/group.csv"
    offset = read_attribute_csv(offset_filepath, columns=["x_offset", "y_offset"])
    if quality_filepath is None:
        quality_filepath = analysis_directory / "quality.csv"
    quality = read_attribute_csv(quality_filepath)

    u, v = velocities["u"], velocities["v"]

    u_shear, v_shear = velocities["u_shear"], velocities["v_shear"]
    u_relative, v_relative = velocities["u_relative"], velocities["v_relative"]
    x_offset, y_offset = offset["x_offset"], offset["y_offset"]

    names = ["stratiform_offset", "inflow", "relative_stratiform_offset"]
    names += ["tilt", "propagation"]
    descriptions = [
        "Stratiform offset classification.",
        "Inflow classification.",
        "Relative stratiform offset classification.",
        "System 'tilt' relative to shear classification.",
        "System propagation relative to shear classification.",
    ]
    data_type, precision, units, method = str, None, None, None
    attributes = {}
    for name, description in zip(names, descriptions):
        args = [name, method, data_type, precision, description, units]
        attributes[name] = get_attribute_dict(*args)
    attributes["time"] = attribute.core.time()
    attributes["universal_id"] = attribute.core.identity("universal_id")
    labels = [["leading", "right", "trailing", "left"]]
    labels += [["front", "right", "rear", "left"]]
    labels += [["leading", "right", "trailing", "left"]]
    labels += [["down-shear", "shear-perpendicular", "up-shear", "shear-perpendicular"]]
    labels += [["down-shear", "shear-perpendicular", "up-shear", "shear-perpendicular"]]

    # Vector 2 defines the center of the first quadrant
    u2_list = [u, u, u_relative, u_shear, u_shear]
    v2_list = [v, v, v_relative, v_shear, v_shear]
    # The quadrant vector 1 falls in determines the classification
    u1_list = [x_offset, u_relative, x_offset, x_offset, u_relative]
    v1_list = [y_offset, v_relative, y_offset, y_offset, v_relative]

    # Offset classifications
    offset_names = ["stratiform_offset", "relative_stratiform_offset", "tilt"]

    all_classifications = []
    for i in range(len(u2_list)):
        angles = utils.get_angle(u1_list[i], v1_list[i], u2_list[i], v2_list[i])
        classified = classify_angles(names[i], angles, labels[i])
        if classify_small_offsets and names[i] in offset_names:
            args = [classified, x_offset, y_offset, analysis_options.min_offset]
            classified = classify_small_offsets(*args)
        if classify_ambiguous:
            unambiguous = quality[ambiguity_quality_dispatcher[names[i]]].all(axis=1)
            classified = classify_ambiguous(classified, unambiguous)
        all_classifications.append(classified)

    classifications = pd.concat(all_classifications, axis=1)
    filepath = analysis_directory / "classification.csv"
    classifications = write.attribute.write_csv(filepath, classifications, attributes)
    return classifications


def classify_angles(name, angles, category_labels):
    """
    Classify the quadrants based on the angles between the vectors.

    Parameters
    ----------
    name : str
        Name of the classification
    angles : pd.DataFrame
        DataFrame of angles between vectors.
    category_labels : list
        List of category labels, [front_label, right_label, rear_label, left_label].

    Returns
    -------
    pd.Series
        Series of classifications.
    """

    classification = pd.Series(pd.NA, index=angles.index, name=name, dtype=str)
    front_cond = (-np.pi / 4 < angles) & (angles <= np.pi / 4)
    classification[front_cond] = category_labels[0]
    right_cond = (np.pi / 4 < angles) & (angles <= 3 * np.pi / 4)
    classification[right_cond] = category_labels[1]
    rear_cond = (3 * np.pi / 4 < angles) | (angles <= -3 * np.pi / 4)
    classification[rear_cond] = category_labels[2]
    left_cond = (-3 * np.pi / 4 < angles) & (angles <= -np.pi / 4)
    classification[left_cond] = category_labels[3]
    return classification


def classify_small_offsets(classified, x_offset, y_offset, min_offset):
    """
    Classify small offsets as 'centered' if the offset is less than the minimum offset.
    """

    offset_magnitude = np.sqrt(x_offset**2 + y_offset**2)
    classified[offset_magnitude < min_offset] = "centered"
    return classified


def classify_ambiguous(classified, unambiguous):
    """
    Classify ambiguous classifications as 'ambiguous'.
    """

    classified[~unambiguous] = "ambiguous"
    return classified
