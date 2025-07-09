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
    kwargs = {"name": "convective", "dataset": dataset, "variable": "reflectivity"}
    detection = {"method": "steiner", "altitudes": [500, 3e3], "threshold": 40}
    kwargs.update({"detection": detection, "tracking": None})
    return track_option.DetectedObjectOptions(**kwargs)


def middle(dataset="cpol"):
    """Build default options for mid-level echo objects."""
    kwargs = {"name": "middle", "dataset": dataset, "variable": "reflectivity"}
    detection = {"method": "threshold", "altitudes": [3.5e3, 7e3], "threshold": 20}
    kwargs.update({"detection": detection, "tracking": None})
    return track_option.DetectedObjectOptions(**kwargs)


def anvil(dataset="cpol"):
    """Build default options for anvil objects."""
    kwargs = {"name": "anvil", "dataset": dataset, "variable": "reflectivity"}
    detection = {"method": "threshold", "altitudes": [7.5e3, 10e3], "threshold": 15}
    kwargs.update({"detection": detection, "tracking": None})
    return track_option.DetectedObjectOptions(**kwargs)


def mcs(tracking_dataset="cpol", profile_dataset="era5_pl", tag_dataset="era5_sl"):
    """Build default options for MCS objects."""

    name = "mcs"
    member_objects = ["convective", "middle", "anvil"]
    kwargs = {"name": name, "member_objects": member_objects}
    kwargs.update({"member_levels": [0, 0, 0], "member_min_areas": [80, 400, 800]})

    grouping = track_option.GroupingOptions(**kwargs)
    tracking = track_option.MintOptions(matched_object="convective")

    # Assume the first member object is used for tracking.
    obj = member_objects[0]
    attribute_types = [core.default_tracked()]
    attribute_types += [quality.default(member_object=obj)]
    attribute_types += [ellipse.default()]
    kwargs = {"name": member_objects[0], "attribute_types": attribute_types}
    attributes = track_option.Attributes(**kwargs)
    member_attributes = {obj: attributes}
    for obj in member_objects[1:]:
        attribute_types = [core.default_member()]
        attribute_types += [quality.default(member_object=obj)]
        kwargs = {"name": obj, "attribute_types": attribute_types}
        member_attributes[obj] = track_option.Attributes(**kwargs)

    mcs_core = core.default_tracked()
    # Add echo top height attribute to the mcs core attributes
    echo_top_height = core.echo_top_height()
    mcs_core.attributes += [echo_top_height]

    attribute_types = [mcs_core, group.default()]
    attribute_types += [profile.default(profile_dataset)]
    attribute_types += [tag.default(tag_dataset)]
    kwargs = {"name": "mcs", "attribute_types": attribute_types}
    kwargs.update({"member_attributes": member_attributes})
    attributes = attribute_option.Attributes(**kwargs)

    kwargs = {"name": name, "dataset": tracking_dataset, "grouping": grouping}
    kwargs.update({"tracking": tracking, "attributes": attributes})
    kwargs.update({"hierarchy_level": 1, "method": "group"})
    mcs_options = track_option.GroupedObjectOptions(**kwargs)

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


def satellite_anvil(dataset="himawari"):
    """Build default options for anvil objects."""
    kwargs = {"name": "anvil", "dataset": dataset, "variable": "brightness_temperature"}
    det_kwargs = {"method": "threshold", "threshold": 220, "threshold_type": "maxima"}
    det_kwargs.update({"flatten_method": None, "min_area": 500})
    kwargs.update({"detection": det_kwargs, "tracking": None})
    attribute_types = [core.default_tracked()]
    attribute_types += [quality.default()]
    attribute_types += [ellipse.default()]
    trck_kwargs = {"global_flow_margin": 70, "unique_global_flow": False}
    tracking = track_option.MintOptions(**trck_kwargs)
    attr_kwargs = {"name": "anvil", "attribute_types": attribute_types}
    attributes = attribute_option.Attributes(**attr_kwargs)
    kwargs.update({"tracking": tracking, "attributes": attributes})
    return track_option.DetectedObjectOptions(**kwargs)


def satellite_track(dataset_name: str = "himawari"):
    """Build default options for tracking anvils in satellite data."""
    anvil_options = satellite_anvil()
    level = track_option.LevelOptions(objects=[anvil_options])
    return track_option.TrackOptions(levels=[level])


def runtime(visualize_directory, objects=["mcs"]):
    """Build default options for runtime visualization."""

    objects_dict = {}
    for obj in objects:
        kwargs = {"name": "tint_match", "function": vis_runtime.visualize_tint_match}
        match_figure = visualize_option.FigureOptions(**kwargs)
        kwargs = {"name": obj, "parent_local": visualize_directory}
        kwargs.update({"figures": [match_figure]})
        figures = visualize_option.ObjectRuntimeOptions(**kwargs)
        objects_dict[figures.name] = figures
    visualize_options = visualize_option.RuntimeOptions(objects=objects_dict)
    return visualize_options


def synthetic_track():
    """Build default options for tracking synthetic MCS."""

    convective_options = convective(dataset="synthetic")
    attribute_types = [core.default_tracked()]
    kwargs = {"name": "convective", "attribute_types": attribute_types}
    attributes = track_option.Attributes(**kwargs)
    convective_options.attributes = attributes
    kwargs = {"global_flow_margin": 70, "unique_global_flow": False}
    convective_options.tracking = track_option.MintOptions(**kwargs)
    levels = [track_option.LevelOptions(objects=[convective_options])]
    return track_option.TrackOptions(levels=levels)


def synthetic_runtime(visualize_directory):
    """Build default options for runtime visualization."""

    kwargs = {"name": "match", "function": vis_runtime.visualize_tint_match}
    match_figure = visualize_option.FigureOptions(**kwargs)
    kwargs = {"name": "convective", "parent_local": visualize_directory}
    kwargs.update({"figures": [match_figure]})
    convective_figures = visualize_option.ObjectRuntimeOptions(**kwargs)

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
    velocity_filepath = str(output_parent / "analysis/velocities.csv")
    quality_filepath = str(output_parent / "analysis/quality.csv")
    vis_func = "thuner.visualize.attribute.velocity_horizontal"
    vis_kwargs = {"color": color, "reverse": reverse}
    method = Retrieval(function=vis_func, keyword_arguments=vis_kwargs)
    leg_func = "thuner.visualize.horizontal.displacement_legend_artist"
    leg_kwargs = {"color": color, "label": label}
    legend_method = Retrieval(function=leg_func, keyword_arguments=leg_kwargs)
    kwargs = {"name": name, "attributes": attributes}
    kwargs.update({"filepath": velocity_filepath})
    kwargs.update({"method": method, "label": label, "legend_method": legend_method})
    kwargs.update({"quality_filepath": quality_filepath})
    kwargs.update({"quality_variables": quality_variables})
    return AttributeHandler(**kwargs)


def build_horizontal_text_handler(
    output_parent, attributes, quality_variables, name="universal_id"
):
    """Convenience function to build a horizontal text attribute handler."""
    velocity_filepath = str(output_parent / "analysis/velocities.csv")
    quality_filepath = str(output_parent / "analysis/quality.csv")
    vis_func = "thuner.visualize.attribute.text_horizontal"
    vis_kwargs = {"labelled_attribute": "universal_id"}
    method = Retrieval(function=vis_func, keyword_arguments=vis_kwargs)
    kwargs = {"name": name, "attributes": attributes, "filepath": velocity_filepath}
    kwargs.update({"method": method, "label": "Object ID"})
    kwargs.update({"quality_filepath": quality_filepath})
    kwargs.update({"quality_variables": quality_variables})
    return AttributeHandler(**kwargs)


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
    group_filepath = str(output_parent / "attributes/mcs/group.csv")
    quality_filepath = str(output_parent / "analysis/quality.csv")
    vis_func = "thuner.visualize.attribute.displacement_horizontal"
    vis_kwargs = {"color": color, "reverse": reverse}
    method = Retrieval(function=vis_func, keyword_arguments=vis_kwargs)
    leg_kwargs = {"color": color, "label": label}
    leg_func = "thuner.visualize.horizontal.displacement_legend_artist"
    legend_method = Retrieval(function=leg_func, keyword_arguments=leg_kwargs)
    kwargs = {"name": name, "attributes": attributes}
    kwargs.update({"method": method, "filepath": group_filepath})
    kwargs.update({"label": "Stratiform Offset", "legend_method": legend_method})
    kwargs.update({"quality_filepath": quality_filepath})
    kwargs.update({"quality_variables": quality_variables})
    return AttributeHandler(**kwargs)


def build_orientation_handler(
    output_parent,
    quality_variables,
    attributes=["major", "orientation"],
    name="orientation",
    style="presentation",
    label="Major Axis",
):
    """Convenience function to build an orientation attribute handler."""
    ellipse_filepath = str(output_parent / "attributes/mcs/convective/ellipse.csv")
    quality_filepath = str(output_parent / "analysis/quality.csv")
    vis_func = "thuner.visualize.attribute.orientation_horizontal"
    method = Retrieval(function=vis_func)
    label = "Major Axis"
    leg_func = "thuner.visualize.horizontal.orientation_legend_artist"
    leg_kwargs = {"label": label, "style": style}
    legend_method = Retrieval(function=leg_func, keyword_arguments=leg_kwargs)
    kwargs = {"name": name, "attributes": attributes}
    kwargs.update({"method": method, "filepath": ellipse_filepath, "label": label})
    kwargs.update({"quality_filepath": quality_filepath})
    kwargs.update({"legend_method": legend_method})
    kwargs.update({"quality_variables": quality_variables})
    return AttributeHandler(**kwargs)


def detected_attribute_handlers(
    output_parent, object_name, style="presentation", attributes=None
):
    """Build default attribute handlers for detected objects."""
    if attributes is None:
        attributes = ["universal_id", "velocity"]
    base_qualities = ["contained", "duration"]

    args = [output_parent, ["u", "v"], base_qualities + ["velocity"]]
    kwargs = {"name": "velocity", "color": "tab:purple", "label": "System Velocity"}
    velocity_handler = build_velocity_handler(*args, **kwargs)

    args = [output_parent, ["universal_id"], base_qualities]
    id_handler = build_horizontal_text_handler(*args)

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

    args = [output_parent, ["u", "v"], base_qualities + ["velocity"]]
    kwargs = {"name": "velocity", "color": "tab:purple", "label": "System Velocity"}
    velocity_handler = build_velocity_handler(*args, **kwargs)

    args = [output_parent, ["u_ambient", "v_ambient"], base_qualities]
    kwargs = {"name": "ambient", "color": "tab:red", "label": "Ambient Wind"}
    ambient_handler = build_velocity_handler(*args, **kwargs)

    args = [output_parent, ["u_shear", "v_shear"], base_qualities + ["shear"]]
    kwargs = {"name": "shear", "color": "darkblue", "label": "Ambient Shear"}
    shear_handler = build_velocity_handler(*args, **kwargs)

    args = [output_parent, ["u_relative", "v_relative"]]
    args += [base_qualities + ["relative_velocity"]]
    kwargs = {"color": "darkgreen", "label": "Relative System Velocity"}
    kwargs.update({"name": "relative"})
    relative_handler = build_velocity_handler(*args, **kwargs)

    args = [output_parent, ["u_relative", "v_relative"]]
    args += [base_qualities + ["relative_velocity"]]
    kwargs = {"color": "darkgreen", "label": "System Relative Inflow"}
    kwargs.update({"name": "inflow", "reverse": True})
    inflow_handler = build_velocity_handler(*args, **kwargs)

    args = [output_parent, ["universal_id"], base_qualities]
    id_handler = build_horizontal_text_handler(*args)

    args = [output_parent, ["x_offset", "y_offset"], base_qualities + ["offset"]]
    offset_handler_convective = build_displacement_handler(*args)
    offset_handler_anvil = build_displacement_handler(*args, reverse=True)

    kwargs = {"quality_variables": base_qualities + ["axis_ratio"], "style": style}
    orientation_handler = build_orientation_handler(output_parent, **kwargs)

    all_conv = [id_handler, velocity_handler, ambient_handler]
    all_conv += [shear_handler, relative_handler, offset_handler_convective]
    all_conv += [inflow_handler, orientation_handler]
    all_anvil = [id_handler, offset_handler_anvil]
    conv_handlers = [h for h in all_conv if h.name in attributes[member_objects[0]]]
    anvil_handlers = [h for h in all_anvil if h.name in attributes[member_objects[1]]]

    return dict(zip(member_objects, [conv_handlers, anvil_handlers]))
