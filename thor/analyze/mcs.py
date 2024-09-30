"""
Methods for analyzing MCSs. In particular, for implementing the methodologies 
presented in the following papers:

Short et al. (2023), Objectively diagnosing characteristics of mesoscale organization 
from radar reflectivity and ambient winds. https://dx.doi.org/10.1175/MWR-D-22-0146.1 
"""

import numpy as np
import pandas as pd
from thor.attribute.utils import read_attribute_csv, get_attribute_dict
import thor.analyze.utils as utils
import thor.write as write
import thor.attribute as attribute


def process_velocities(
    output_directory, window_size=6, analysis_directory=None, profile_dataset="era5_pl"
):
    """
    Process winds and velocities for analysis.

    Parameters
    ----------
    output_directory : str
        Path to the thor run output directory.
    """

    if analysis_directory is None:
        analysis_directory = output_directory / "analysis"

    options = utils.read_options(output_directory)
    member_objects = options["track"][1]["mcs"]["grouping"]["member_objects"]
    convective_label = member_objects[0]

    options = utils.read_options(output_directory)
    # Get the options for the convective objects
    convective_options = options["track"][0][convective_label]
    altitudes = convective_options["detection"]["altitudes"]
    filepath = output_directory / f"attributes/mcs/{profile_dataset}/profile.csv"
    winds = read_attribute_csv(filepath, columns=["u", "v"])
    filepath = output_directory / "attributes/mcs/core.csv"
    velocities = read_attribute_csv(filepath, columns=["u_flow", "v_flow"])
    velocities = utils.temporal_smooth(velocities, window_size=window_size)
    velocities = velocities.rename(columns={"u_flow": "u", "v_flow": "v"})

    # Take mean of ambient winds over altitudes used to detect convective echoes
    indexer = pd.IndexSlice[:, :, altitudes[0] : altitudes[1]]
    mean_winds = winds.loc[indexer].groupby(["time", "universal_id"]).mean()
    new_names = {"u": "u_ambient", "v": "v_ambient"}
    mean_winds = mean_winds.rename(columns=new_names)

    # Check if dataframes aligned
    if not velocities.index.equals(mean_winds.index):
        raise ValueError("Dataframes are not aligned. Perhaps enforce alignment first?")

    # Calculate a shear vector as the difference between the winds at the top and
    # bottom of layer used to detect convective echoes
    top = winds.xs(altitudes[1], level="altitude")
    bottom = winds.xs(altitudes[0], level="altitude")
    shear = top - bottom
    new_names = {"u": "u_shear", "v": "v_shear"}
    shear = shear.rename(columns=new_names)

    # Calculate system wind-relative velocities
    new_names_vel = {"u": "u_relative", "v": "v_relative"}
    new_names_mean = {"u_ambient": "u_relative", "v_ambient": "v_relative"}
    renamed_velocities = velocities.rename(columns=new_names_vel)
    renamed_mean_winds = mean_winds.rename(columns=new_names_mean)
    relative_velocities = renamed_velocities - renamed_mean_winds

    all_velocities = pd.concat(
        [velocities, mean_winds, shear, relative_velocities], axis=1
    )

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


def analysis_options(
    window_size=6,
    min_area=1e2,
    max_area=np.inf,
    max_boundary_overlap=1e-3,
    duration=30,
    min_major_axis_length=0,
    min_axis_ratio=0,
    min_offset=10,
    min_shear=2,
    min_velocity=5,
    min_relative_velocity=2,
    quadrant_buffer_angle=10,
):
    """
    Get quality control options for MCS analysis. These criteria are used to filter
    system observations before classifications of various kinds are performed.

    Parameters
    ----------
    window_size : int, optional
        Window size for temporal smoothing of velocities. The default is 6.
    min_area : float, optional
        Minimum area of MCS in km^2. The default is 1e2.
    max_area : float, optional
        Maximum area of MCS in km^2. The default is 1e6.
    max_boundary_overlap : float, optional
        Maximum fraction of system member object pixels touching boundary.
        The default is 1e-3.
    min_major_axis_length : float, optional
        Minimum major axis length of MCS in km. The default is None.
    min_axis_ratio : float, optional
        Minimum axis ratio of MCS. The default is None.
    duration : float, optional
        Minimum duration of MCS in minutes. The default is 30.
    min_offset : float, optional
        Minimum stratiform offset in km. The default is 10.
    min_shear : float, optional
        Minimum shear in m/s. The default is 2.
    min_velocity : float, optional
        Minimum velocity in m/s. The default is 5.
    min_relative_velocity : float, optional
        Minimum relative velocity in m/s. The default is 2.
    quadrant_buffer_angle : float, optional
        Buffer angle in degrees for quadrant based classification. The default is 10.

    Returns
    -------
    dict
        Dictionary of quality control options.
    """
    options = {
        "window_size": window_size,
        "min_area": min_area,
        "max_area": max_area,
        "max_boundary_overlap": max_boundary_overlap,
        "min_major_axis_length": min_major_axis_length,
        "min_axis_ratio": min_axis_ratio,
        "duration": duration,
        "min_offset": min_offset,
        "min_shear": min_shear,
        "min_velocity": min_velocity,
        "min_relative_velocity": min_relative_velocity,
        "quadrant_buffer_angle": quadrant_buffer_angle,
    }
    return options


def quality_control(output_directory, analysis_options, analysis_directory=None):
    """
    Perform quality control on MCSs based on the provided options.

    Parameters
    ----------
    output_directory : str
        Path to the thor run output directory.
    analysis_options : dict
        Dictionary of quality control options.

    Returns
    -------
    pd.DataFrame
        DataFrame describing quality control checks.
    """
    if analysis_directory is None:
        analysis_directory = output_directory / "analysis"

    options = utils.read_options(output_directory)
    member_objects = options["track"][1]["mcs"]["grouping"]["member_objects"]
    convective_label = member_objects[0]
    anvil_label = member_objects[1]

    # Determine if the system is sufficiently contained within the domain
    filepath = output_directory / f"attributes/mcs/{convective_label}/quality.csv"
    convective = read_attribute_csv(filepath)
    filepath = output_directory / f"attributes/mcs/{anvil_label}/quality.csv"
    anvil = read_attribute_csv(filepath)
    max_boundary_overlap = analysis_options["max_boundary_overlap"]
    convective = convective.rename(columns={"boundary_overlap": "convective_contained"})
    anvil = anvil.rename(columns={"boundary_overlap": "anvil_contained"})
    convective_check = convective["convective_contained"] <= max_boundary_overlap
    anvil_check = anvil["anvil_contained"] < max_boundary_overlap

    # Check if velocity/shear vectors are sufficiently large
    filepath = analysis_directory / "velocities.csv"
    velocities = read_attribute_csv(filepath)
    velocity_magnitude = velocities[["u", "v"]].pow(2).sum(axis=1).pow(0.5)
    velocity_check = velocity_magnitude >= analysis_options["min_velocity"]
    velocity_check.name = "velocity"
    shear_magnitude = velocities[["u_shear", "v_shear"]].pow(2).sum(axis=1).pow(0.5)
    shear_check = shear_magnitude >= analysis_options["min_shear"]
    shear_check.name = "shear"
    relative_velocity = velocities[["u_relative", "v_relative"]]
    relative_velocity_magnitude = relative_velocity.pow(2).sum(axis=1).pow(0.5)
    min_relative_velocity = analysis_options["min_relative_velocity"]
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
    min_area, max_area = analysis_options["min_area"], analysis_options["max_area"]
    area_check = (area >= min_area) & (area <= max_area)
    area_check.name = "area"

    # Check the stratiform offset is sufficiently large
    filepath = output_directory / f"attributes/mcs/group.csv"
    offset = read_attribute_csv(filepath, columns=["x_offset", "y_offset"])
    offset_magnitude = offset.pow(2).sum(axis=1).pow(0.5)
    offset_check = offset_magnitude >= analysis_options["min_offset"]
    offset_check.name = "offset"

    # Check the duration of the system is sufficiently long
    # First get the duration of each object from the velocity dataframe
    dummy_df = pd.DataFrame(index=velocities.index)
    dummy_df.index.names = velocities.index.names
    time_group = velocities.reset_index().groupby("universal_id")["time"]
    duration = time_group.agg(lambda x: x.max() - x.min())
    duration_check = duration >= np.timedelta64(analysis_options["duration"], "m")
    duration_check.name = "duration"
    dummy_df = velocities[[]].reset_index()
    duration_check = dummy_df.merge(duration_check, on="universal_id", how="left")
    duration_check = duration_check.set_index(velocities.index.names)

    # Check the linearity of the system
    filepath = output_directory / f"attributes/mcs/{convective_label}/ellipse.csv"
    ellipse = read_attribute_csv(filepath, columns=["major", "minor"])
    major_check = ellipse["major"] >= analysis_options["min_major_axis_length"]
    major_check.name = "major_axis"
    axis_ratio = ellipse["major"] / ellipse["minor"]
    axis_ratio_check = axis_ratio >= analysis_options["min_axis_ratio"]
    axis_ratio_check.name = "axis_ratio"

    names = ["convective_contained", "anvil_contained", "velocity", "shear"]
    names += ["relative_velocity", "area", "offset", "major_axis", "axis_ratio"]
    names += ["duration"]
    descriptions = [
        "Is the system convective region sufficiently contained within the domain?",
        "Is the system anvil region sufficiently contained within the domain?",
        "Is the system velocity sufficiently large?",
        "Is the system shear sufficiently large?",
        "Is the system relative velocity sufficiently large?",
        "Is the system area sufficiently large?",
        "Is the system stratiform offset sufficiently large?",
        "Is the system major axis length sufficiently large?",
        "Is the system axis ratio sufficiently large?",
        "Is the system duration sufficiently long?",
    ]

    data_type, precision, units, method = bool, None, None, None
    attributes = {}
    for name, description in zip(names, descriptions):
        args = [name, method, data_type, precision, description, units]
        attributes[name] = get_attribute_dict(*args)

    filepath = analysis_directory / "quality.csv"
    quality = [convective_check, anvil_check, velocity_check, shear_check]
    quality += [relative_velocity_check, area_check, offset_check, major_check]
    quality += [axis_ratio_check, duration_check]
    quality = pd.concat(quality, axis=1)
    quality = write.attribute.write_csv(filepath, quality, attributes)
    return quality


def classify_all(output_directory, analysis_directory=None):
    """
    Classify MCSs based on quadrants, as described in Short et al. (2023).

    Parameters
    ----------
    output_directory : str
        Path to the thor run output directory.
    analysis_options : dict
        Dictionary of quality control options.

    Returns
    -------
    pd.DataFrame
        DataFrame describing MCS classifications.
    """
    if analysis_directory is None:
        analysis_directory = output_directory / "analysis"

    filepath = analysis_directory / "velocities.csv"
    velocities = read_attribute_csv(filepath)
    filepath = output_directory / "attributes/mcs/group.csv"
    offset = read_attribute_csv(filepath, columns=["x_offset", "y_offset"])

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
    labels = [["leading", "right", "trailing", "left"]]
    labels += [["front", "right", "back", "left"]]
    labels += [["leading", "right", "trailing", "left"]]
    labels += [["down-shear", "shear-perpendicular", "up-shear", "shear-perpendicular"]]
    labels += [["down-shear", "shear-perpendicular", "up-shear", "shear-perpendicular"]]

    # Vector 2 defines the center of the first quadrant
    u2_list = [u, u, u_relative, u_shear, u_shear]
    v2_list = [v, v, v_relative, v_shear, v_shear]
    # The quadrant vector 2 falls in determines the classification
    u1_list = [x_offset, u_relative, x_offset, x_offset, u_relative]
    v1_list = [y_offset, v_relative, y_offset, y_offset, v_relative]

    all_classifications = []
    for i in range(len(u2_list)):
        angles = get_angle(u1_list[i], v1_list[i], u2_list[i], v2_list[i])
        all_classifications.append(classify(names[i], angles, labels[i]))

    classifications = pd.concat(all_classifications, axis=1)
    filepath = analysis_directory / "classification.csv"
    classifications = write.attribute.write_csv(filepath, classifications, attributes)
    return classifications


def get_angle(u1, v1, u2, v2):
    """
    Get the angle between two vectors. Angle calculated as second vector direction minus
    first vector direction.
    """

    angle_1 = np.arctan2(v1, u1)
    angle_2 = np.arctan2(v2, u2)
    # Get angle between vectors, but signed so that in range -np.pi to np.pi
    return np.mod(angle_2 - angle_1 + np.pi, 2 * np.pi) - np.pi


def classify(name, angles, category_labels):
    """
    Classify the quadrants based on the angles between the vectors.

    Parameters
    ----------
    name : str
        Name of the classification
    angles : pd.DataFrame
        DataFrame of angles between vectors.
    category_labels : list
        List of category labels, [front_label, right_label, back_label, left_label].

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
    back_cond = (3 * np.pi / 4 < angles) | (angles <= -3 * np.pi / 4)
    classification[back_cond] = category_labels[2]
    left_cond = (-3 * np.pi / 4 < angles) & (angles <= -np.pi / 4)
    classification[left_cond] = category_labels[3]
    return classification
