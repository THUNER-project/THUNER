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


__all__ = ["convective", "middle", "anvil", "mcs", "track"]


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
    track_options = track_option.TrackOptions(levels=levels)
    return track_options


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
    velocity_filepath = str(output_parent / "analysis/velocities.csv")
    quality_filepath = str(output_parent / "analysis/quality.csv")

    vis_func = "thuner.visualize.attribute.velocity_horizontal"
    color, label = "tab:purple", "System Velocity"
    vis_kwargs = {"color": color}
    method = Retrieval(function=vis_func, keyword_arguments=vis_kwargs)
    leg_func = "thuner.visualize.horizontal.displacement_legend_artist"
    leg_kwargs = {"color": color, "label": label}
    legend_method = Retrieval(function=leg_func, keyword_arguments=leg_kwargs)
    kwargs = {"name": "velocity", "attributes": ["u", "v"]}
    kwargs.update({"filepath": velocity_filepath})
    kwargs.update({"method": method, "label": label, "legend_method": legend_method})
    kwargs.update({"quality_filepath": quality_filepath})
    kwargs.update({"quality_variables": base_qualities + ["velocity", "duration"]})
    velocity_handler = AttributeHandler(**kwargs)

    color, label = "tab:red", "Ambient Wind"
    vis_kwargs = {"color": color}
    method = Retrieval(function=vis_func, keyword_arguments=vis_kwargs)
    leg_kwargs = {"color": color, "label": label}
    legend_method = Retrieval(function=leg_func, keyword_arguments=leg_kwargs)
    kwargs = {"name": "ambient", "attributes": ["u_ambient", "v_ambient"]}
    kwargs.update({"method": method, "filepath": velocity_filepath})
    kwargs.update({"label": label, "legend_method": legend_method})
    kwargs.update({"quality_filepath": quality_filepath})
    kwargs.update({"quality_variables": base_qualities + ["duration"]})
    ambient_handler = AttributeHandler(**kwargs)

    color, label = "darkblue", "Ambient Shear"
    vis_kwargs = {"color": color}
    method = Retrieval(function=vis_func, keyword_arguments=vis_kwargs)
    leg_kwargs = {"color": color, "label": label}
    legend_method = Retrieval(function=leg_func, keyword_arguments=leg_kwargs)
    kwargs = {"name": "shear", "attributes": ["u_shear", "v_shear"]}
    kwargs.update({"method": method, "filepath": velocity_filepath})
    kwargs.update({"label": label, "legend_method": legend_method})
    kwargs.update({"quality_filepath": quality_filepath})
    kwargs.update({"quality_variables": base_qualities + ["shear", "duration"]})
    shear_handler = AttributeHandler(**kwargs)

    color, label = "darkgreen", "Relative System Velocity"
    vis_kwargs = {"color": color}
    method = Retrieval(function=vis_func, keyword_arguments=vis_kwargs)
    leg_kwargs = {"color": color, "label": label}
    legend_method = Retrieval(function=leg_func, keyword_arguments=leg_kwargs)
    kwargs = {"name": "relative", "attributes": ["u_relative", "v_relative"]}
    kwargs.update({"method": method, "filepath": velocity_filepath})
    kwargs.update({"label": label, "legend_method": legend_method})
    kwargs.update({"quality_filepath": quality_filepath})
    quality_vars = base_qualities + ["relative_velocity", "duration"]
    kwargs.update({"quality_variables": quality_vars})
    relative_handler = AttributeHandler(**kwargs)

    vis_func = "thuner.visualize.attribute.text_horizontal"
    vis_kwargs = {"labelled_attribute": "universal_id"}
    method = Retrieval(function=vis_func, keyword_arguments=vis_kwargs)
    kwargs = {"name": "universal_id", "attributes": ["universal_id"]}
    kwargs.update({"filepath": velocity_filepath})
    kwargs.update({"method": method, "label": "Object ID"})
    kwargs.update({"quality_filepath": quality_filepath})
    kwargs.update({"quality_variables": base_qualities})
    id_handler = AttributeHandler(**kwargs)

    group_filepath = str(output_parent / "attributes/mcs/group.csv")
    vis_func = "thuner.visualize.attribute.displacement_horizontal"
    color, label = "tab:blue", "Stratiform Offset"
    vis_kwargs = {"color": color}
    method = Retrieval(function=vis_func, keyword_arguments=vis_kwargs)
    leg_kwargs = {"color": color, "label": label}
    legend_method = Retrieval(function=leg_func, keyword_arguments=leg_kwargs)
    kwargs = {"name": "offset", "attributes": ["x_offset", "y_offset"]}
    kwargs.update({"method": method, "filepath": group_filepath})
    kwargs.update({"label": "Stratiform Offset", "legend_method": legend_method})
    kwargs.update({"quality_filepath": quality_filepath})
    kwargs.update({"quality_variables": base_qualities + ["offset", "duration"]})
    offset_handler_convective = AttributeHandler(**kwargs)
    vis_kwargs["reverse"] = True
    method = Retrieval(function=vis_func, keyword_arguments=vis_kwargs)
    kwargs["method"] = method
    offset_handler_anvil = AttributeHandler(**kwargs)

    ellipse_filepath = str(output_parent / "attributes/mcs/convective/ellipse.csv")
    vis_func = "thuner.visualize.attribute.orientation_horizontal"
    method = Retrieval(function=vis_func)
    label = "Major Axis"
    leg_func = "thuner.visualize.horizontal.orientation_legend_artist"
    leg_kwargs = {"label": label, "style": style}
    legend_method = Retrieval(function=leg_func, keyword_arguments=leg_kwargs)

    kwargs = {"name": "orientation", "attributes": ["major", "orientation"]}
    kwargs.update({"method": method, "filepath": ellipse_filepath, "label": label})
    kwargs.update({"quality_filepath": quality_filepath})
    kwargs.update({"legend_method": legend_method})
    kwargs.update({"quality_variables": base_qualities + ["axis_ratio", "duration"]})
    orientation_handler = AttributeHandler(**kwargs)

    all_conv = [id_handler, velocity_handler, ambient_handler]
    all_conv += [shear_handler, relative_handler]
    all_conv += [offset_handler_convective, orientation_handler]
    all_anvil = [id_handler, offset_handler_anvil]
    conv_handlers = [h for h in all_conv if h.name in attributes[member_objects[0]]]
    anvil_handlers = [h for h in all_anvil if h.name in attributes[member_objects[1]]]

    return dict(zip(member_objects, [conv_handlers, anvil_handlers]))
