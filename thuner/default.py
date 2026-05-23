"""Default options configurations."""

import thuner.option.track as track_option
import thuner.option.visualize as visualize_option
import thuner.option.attribute as attribute_option
import thuner.attribute.core as core
import thuner.attribute.group as group
import thuner.attribute.tag as tag
import thuner.attribute.profile as profile
import thuner.attribute.ellipse as ellipse
import thuner.attribute.quality as quality
import thuner.visualize.runtime as vis_runtime
from thuner.utils import Retrieval, AttributeHandler
from thuner.config import get_zarr_store_name

__all__ = [
    "convective",
    "middle",
    "anvil",
    "mcs",
    "track",
    "satellite_anvil",
    "satellite_track",
]


def convective(dataset="cpol"):
    """Build default options for convective objects."""
    detection = {"method": "steiner", "altitudes": [500, 3e3], "threshold": 40}
    return track_option.DetectedObjectOptions(
        name="convective",
        dataset=dataset,
        variable="reflectivity",
        detection=detection,
        tracking=None,
    )


def middle(dataset="cpol"):
    """Build default options for mid-level echo objects."""
    detection = {"method": "threshold", "altitudes": [3.5e3, 7e3], "threshold": 20}
    return track_option.DetectedObjectOptions(
        name="middle",
        dataset=dataset,
        variable="reflectivity",
        detection=detection,
        tracking=None,
    )


def anvil(dataset="cpol"):
    """Build default options for anvil objects."""
    detection = {"method": "threshold", "altitudes": [7.5e3, 10e3], "threshold": 15}
    return track_option.DetectedObjectOptions(
        name="anvil",
        dataset=dataset,
        variable="reflectivity",
        detection=detection,
        tracking=None,
    )


def satellite_anvil(dataset="himawari"):
    """Build default options for anvil objects."""
    det_kwargs = {"method": "threshold", "threshold": 235, "threshold_type": "maxima"}
    det_kwargs.update({"flatten_method": None, "min_area": 500})
    attribute_types = [core.default_tracked()]
    attribute_types += [quality.default()]
    attribute_types += [ellipse.default()]
    tracking = track_option.MintOptions(
        global_flow_margin=70,
        unique_global_flow=False,
    )
    attributes = attribute_option.Attributes(
        name="anvil",
        attribute_types=attribute_types,
    )
    return track_option.DetectedObjectOptions(
        name="anvil",
        dataset=dataset,
        variable="brightness_temperature",
        detection=det_kwargs,
        tracking=tracking,
        attributes=attributes,
    )


def mcs(tracking_dataset="cpol", profile_dataset="era5_pl", tag_dataset="era5_sl"):
    """Build default options for MCS objects."""

    name = "mcs"
    member_objects = ["convective", "middle", "anvil"]

    grouping = track_option.GroupingOptions(
        name=name,
        member_objects=member_objects,
        member_levels=[0, 0, 0],
        member_min_areas=[80, 400, 800],
    )
    tracking = track_option.MintOptions(matched_object="convective")

    # Assume the first member object is used for tracking.
    obj = member_objects[0]
    attribute_types = [core.default_tracked()]
    attribute_types += [quality.default(member_object=obj)]
    attribute_types += [ellipse.default()]
    attributes = track_option.Attributes(
        name=member_objects[0],
        attribute_types=attribute_types,
    )
    member_attributes = {obj: attributes}
    for obj in member_objects[1:]:
        attribute_types = [core.default_member()]
        attribute_types += [quality.default(member_object=obj)]
        member_attributes[obj] = track_option.Attributes(
            name=obj,
            attribute_types=attribute_types,
        )

    mcs_core = core.default_tracked()
    # Add echo top height attribute to the mcs core attributes
    echo_top_height = core.echo_top_height()
    mcs_core.attributes += [echo_top_height]

    attribute_types = [mcs_core, group.default()]
    attribute_types += [profile.default(profile_dataset)]
    attribute_types += [tag.default(tag_dataset)]
    attributes = attribute_option.Attributes(
        name="mcs",
        attribute_types=attribute_types,
        member_attributes=member_attributes,
    )

    mcs_options = track_option.GroupedObjectOptions(
        name=name,
        dataset=tracking_dataset,
        grouping=grouping,
        tracking=tracking,
        attributes=attributes,
        hierarchy_level=1,
        method="group",
    )

    return mcs_options


def access_c_mcs(tracking_dataset="access_1km"):
    """
    Build options for ACCESS-C MCS objects. Note we use 1km reflectivity to detect the
    convective objects, and the colmax reflectivity to detect the anvil objects. See
    https://doi.org/10.1175/MWR-D-23-0033.1
    """

    name = "mcs"
    member_objects = ["convective", "anvil"]

    grouping = track_option.GroupingOptions(
        name=name,
        member_objects=member_objects,
        member_levels=[0, 0],
        member_min_areas=[80, 800],
    )
    tracking = track_option.MintOptions(
        matched_object="convective", global_flow_margin=70, unique_global_flow=False
    )

    # Assume the first member object is used for tracking.
    obj = member_objects[0]
    attribute_types = [core.default_tracked()]
    attribute_types += [quality.default(member_object=obj)]
    attribute_types += [ellipse.default()]
    attributes = track_option.Attributes(
        name=member_objects[0],
        attribute_types=attribute_types,
    )
    member_attributes = {obj: attributes}
    for obj in member_objects[1:]:
        attribute_types = [core.default_member()]
        attribute_types += [quality.default(member_object=obj)]
        member_attributes[obj] = track_option.Attributes(
            name=obj,
            attribute_types=attribute_types,
        )

    mcs_core = core.default_tracked()
    attribute_types = [mcs_core, group.default()]
    # Leave out the profile and tag attributes for now
    # attribute_types += [profile.default(profile_dataset)]
    # attribute_types += [tag.default(tag_dataset)]
    attributes = attribute_option.Attributes(
        name="mcs",
        attribute_types=attribute_types,
        member_attributes=member_attributes,
    )

    mcs_options = track_option.GroupedObjectOptions(
        name=name,
        dataset=tracking_dataset,
        grouping=grouping,
        tracking=tracking,
        attributes=attributes,
        hierarchy_level=1,
        method="group",
    )

    return mcs_options


def track(dataset_name: str = "cpol"):
    """Build default options for tracking MCS."""

    mask_options = track_option.MaskOptions(save=False, load=False)
    convective_options = convective(dataset_name)
    convective_options.mask_options = mask_options
    middle_options = middle(dataset_name)
    middle_options.mask_options = mask_options
    anvil_options = anvil(dataset_name)
    anvil_options.mask_options = mask_options
    mcs_options = mcs(dataset_name)
    objects = [convective_options, middle_options, anvil_options]
    level_0 = track_option.LevelOptions(objects=objects)
    level_1 = track_option.LevelOptions(objects=[mcs_options])
    levels = [level_0, level_1]
    return track_option.TrackOptions(levels=levels)


def access_c_track(
    convective_dataset: str = "access_1km", anvil_dataset: str = "access_maxcol"
):
    """Build default options for tracking MCS."""

    mask_options = track_option.MaskOptions(save=False, load=False)
    convective_options = convective(convective_dataset)
    convective_options.mask_options = mask_options
    anvil_options = anvil(anvil_dataset)
    anvil_options.mask_options = mask_options
    # Assume the convective and anvil datasets are 2D, so set flatten method to None
    convective_options.detection.flatten_method = None
    anvil_options.detection.flatten_method = None
    mcs_options = access_c_mcs(convective_dataset)
    objects = [convective_options, anvil_options]
    level_0 = track_option.LevelOptions(objects=objects)
    level_1 = track_option.LevelOptions(objects=[mcs_options])
    levels = [level_0, level_1]
    return track_option.TrackOptions(levels=levels)


def satellite_track(dataset_name: str = "himawari"):
    """Build default options for tracking anvils in satellite data."""
    anvil_options = satellite_anvil(dataset_name)
    level = track_option.LevelOptions(objects=[anvil_options])
    return track_option.TrackOptions(levels=[level])


def runtime(visualize_directory, objects=["mcs"]):
    """Build default options for runtime visualization."""

    objects_dict = {}
    for obj in objects:
        match_figure = visualize_option.FigureOptions(
            name="tint_match",
            function=vis_runtime.visualize_tint_match,
        )
        figures = visualize_option.ObjectRuntimeOptions(
            name=obj,
            parent_local=visualize_directory,
            figures=[match_figure],
        )
        objects_dict[figures.name] = figures
    visualize_options = visualize_option.RuntimeOptions(objects=objects_dict)
    return visualize_options


def synthetic_track():
    """Build default options for tracking synthetic MCS."""

    convective_options = convective(dataset="synthetic")
    attribute_types = [core.default_tracked()]
    attributes = track_option.Attributes(
        name="convective",
        attribute_types=attribute_types,
    )
    convective_options.attributes = attributes
    convective_options.tracking = track_option.MintOptions(
        global_flow_margin=70,
        unique_global_flow=False,
    )
    levels = [track_option.LevelOptions(objects=[convective_options])]
    return track_option.TrackOptions(levels=levels)


def synthetic_runtime(visualize_directory):
    """Build default options for runtime visualization."""

    match_figure = visualize_option.FigureOptions(
        name="match",
        function=vis_runtime.visualize_tint_match,
    )
    convective_figures = visualize_option.ObjectRuntimeOptions(
        name="convective",
        parent_local=visualize_directory,
        figures=[match_figure],
    )

    objects_dict = {convective_figures.name: convective_figures}
    visualize_options = visualize_option.RuntimeOptions(objects=objects_dict)
    return visualize_options


def build_velocity_handler(
    output_parent,
    attributes,
    quality_variables,
    name="velocity",
    color="tab:purple",
    label="Object Velocity",
    reverse=False,
):
    """Convenience function to build a velocity attribute handler."""
    velocity_filepath = str(
        output_parent / get_zarr_store_name() / "analysis/velocities"
    )
    quality_filepath = str(output_parent / get_zarr_store_name() / "analysis/quality")
    vis_func = "thuner.visualize.attribute.velocity_horizontal"
    vis_kwargs = {"color": color, "reverse": reverse}
    method = Retrieval(function=vis_func, keyword_arguments=vis_kwargs)
    leg_func = "thuner.visualize.horizontal.displacement_legend_artist"
    leg_kwargs = {"color": color, "label": label}
    legend_method = Retrieval(function=leg_func, keyword_arguments=leg_kwargs)
    return AttributeHandler(
        name=name,
        attributes=attributes,
        filepath=velocity_filepath,
        method=method,
        label=label,
        legend_method=legend_method,
        quality_filepath=quality_filepath,
        quality_variables=quality_variables,
    )


def build_horizontal_text_handler(
    output_parent, attributes, quality_variables, name="universal_id"
):
    """Convenience function to build a horizontal text attribute handler."""
    velocity_filepath = str(
        output_parent / get_zarr_store_name() / "analysis/velocities"
    )
    quality_filepath = str(output_parent / get_zarr_store_name() / "analysis/quality")
    vis_func = "thuner.visualize.attribute.text_horizontal"
    vis_kwargs = {"labelled_attribute": "universal_id"}
    method = Retrieval(function=vis_func, keyword_arguments=vis_kwargs)
    return AttributeHandler(
        name=name,
        attributes=attributes,
        filepath=velocity_filepath,
        method=method,
        label="Object ID",
        quality_filepath=quality_filepath,
        quality_variables=quality_variables,
    )


def build_displacement_handler(
    output_parent,
    attributes,
    quality_variables,
    name="offset",
    color="tab:blue",
    label="Stratiform Offset",
    reverse=False,
):
    """Convenience function to build a displacement attribute handler."""
    group_filepath = str(output_parent / get_zarr_store_name() / "attributes/mcs/group")
    quality_filepath = str(output_parent / get_zarr_store_name() / "analysis/quality")
    vis_func = "thuner.visualize.attribute.displacement_horizontal"
    vis_kwargs = {"color": color, "reverse": reverse}
    method = Retrieval(function=vis_func, keyword_arguments=vis_kwargs)
    leg_kwargs = {"color": color, "label": label}
    leg_func = "thuner.visualize.horizontal.displacement_legend_artist"
    legend_method = Retrieval(function=leg_func, keyword_arguments=leg_kwargs)
    return AttributeHandler(
        name=name,
        attributes=attributes,
        method=method,
        filepath=group_filepath,
        label="Stratiform Offset",
        legend_method=legend_method,
        quality_filepath=quality_filepath,
        quality_variables=quality_variables,
    )


def build_orientation_handler(
    output_parent,
    quality_variables,
    attributes=["major", "orientation"],
    name="orientation",
    style="presentation",
    label="Major Axis",
):
    """Convenience function to build an orientation attribute handler."""
    ellipse_filepath = str(
        output_parent / get_zarr_store_name() / "attributes/mcs/convective/ellipse"
    )
    quality_filepath = str(output_parent / get_zarr_store_name() / "analysis/quality")
    vis_func = "thuner.visualize.attribute.orientation_horizontal"
    method = Retrieval(function=vis_func)
    label = "Major Axis"
    leg_func = "thuner.visualize.horizontal.orientation_legend_artist"
    leg_kwargs = {"label": label, "style": style}
    legend_method = Retrieval(function=leg_func, keyword_arguments=leg_kwargs)
    return AttributeHandler(
        name=name,
        attributes=attributes,
        method=method,
        filepath=ellipse_filepath,
        label=label,
        quality_filepath=quality_filepath,
        legend_method=legend_method,
        quality_variables=quality_variables,
    )


def detected_attribute_handlers(
    output_parent, object_name, style="presentation", attributes=None
):
    """Build default attribute handlers for detected objects."""
    if attributes is None:
        attributes = ["universal_id", "velocity"]
    base_qualities = ["contained", "duration"]

    velocity_handler = build_velocity_handler(
        output_parent=output_parent,
        attributes=["u", "v"],
        quality_variables=base_qualities + ["velocity"],
        name="velocity",
        color="tab:purple",
        label="System Velocity",
    )

    id_handler = build_horizontal_text_handler(
        output_parent=output_parent,
        attributes=["universal_id"],
        quality_variables=base_qualities,
    )

    return {object_name: [id_handler, velocity_handler]}


def grouped_attribute_handlers(
    output_parent, style="presentation", member_objects=None, attributes=None
):
    """Build default attribute handlers for grouped objects."""

    if member_objects is None:
        member_objects = ["convective", "anvil"]
    if attributes is None:
        # Initialize dictionary containing the attribute names for each member object.
        attributes = {k: [] for k in member_objects}
        conv_attr = ["universal_id", "velocity", "offset", "orientation"]
        anv_attr = ["universal_id", "offset"]
        attributes[member_objects[0]] = conv_attr
        attributes[member_objects[1]] = anv_attr

    base_qualities = ["convective_contained", "anvil_contained", "duration"]

    velocity_handler = build_velocity_handler(
        output_parent=output_parent,
        attributes=["u", "v"],
        quality_variables=base_qualities + ["velocity"],
        name="velocity",
        color="tab:purple",
        label="System Velocity",
    )

    ambient_handler = build_velocity_handler(
        output_parent=output_parent,
        attributes=["u_ambient", "v_ambient"],
        quality_variables=base_qualities,
        name="ambient",
        color="tab:red",
        label="Ambient Wind",
    )

    shear_handler = build_velocity_handler(
        output_parent=output_parent,
        attributes=["u_shear", "v_shear"],
        quality_variables=base_qualities + ["shear"],
        name="shear",
        color="darkblue",
        label="Ambient Shear",
    )

    relative_handler = build_velocity_handler(
        output_parent=output_parent,
        attributes=["u_relative", "v_relative"],
        quality_variables=base_qualities + ["relative_velocity"],
        color="darkgreen",
        label="Relative System Velocity",
        name="relative",
    )

    inflow_handler = build_velocity_handler(
        output_parent=output_parent,
        attributes=["u_relative", "v_relative"],
        quality_variables=base_qualities + ["relative_velocity"],
        color="darkgreen",
        label="System Relative Inflow",
        name="inflow",
        reverse=True,
    )

    id_handler = build_horizontal_text_handler(
        output_parent=output_parent,
        attributes=["universal_id"],
        quality_variables=base_qualities,
    )

    offset_handler_convective = build_displacement_handler(
        output_parent=output_parent,
        attributes=["x_offset", "y_offset"],
        quality_variables=base_qualities + ["offset"],
    )
    offset_handler_anvil = build_displacement_handler(
        output_parent=output_parent,
        attributes=["x_offset", "y_offset"],
        quality_variables=base_qualities + ["offset"],
        reverse=True,
    )

    orientation_handler = build_orientation_handler(
        output_parent,
        quality_variables=base_qualities + ["axis_ratio"],
        style=style,
    )

    all_conv = [id_handler, velocity_handler, ambient_handler]
    all_conv += [shear_handler, relative_handler, offset_handler_convective]
    all_conv += [inflow_handler, orientation_handler]
    all_anvil = [id_handler, offset_handler_anvil]
    conv_handlers = [h for h in all_conv if h.name in attributes[member_objects[0]]]
    anvil_handlers = [h for h in all_anvil if h.name in attributes[member_objects[1]]]

    return dict(zip(member_objects, [conv_handlers, anvil_handlers]))
