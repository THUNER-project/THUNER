"""Default options configurations."""

import thuner.option.track as track_option
import thuner.option.visualize as visualize_option
import thuner.option.attribute as attribute_option
import thuner.attribute as attribute
import thuner.visualize as visualize


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

    core_tracked = attribute.core.default_tracked()
    core_member = attribute.core.default_member()

    # Assume the first member object is used for tracking.
    obj = member_objects[0]
    attribute_types = [core_tracked, attribute.quality.default(member_object=obj)]
    attribute_types += [attribute.ellipse.default()]
    kwargs = {"name": member_objects[0], "attribute_types": attribute_types}
    attributes = track_option.Attributes(**kwargs)
    member_attributes = {obj: attributes}
    for obj in member_objects[1:]:
        attribute_types = [core_member, attribute.quality.default(member_object=obj)]
        kwargs = {"name": obj, "attribute_types": attribute_types}
        member_attributes[obj] = track_option.Attributes(**kwargs)

    attribute_types = [core_tracked, attribute.group.default()]
    attribute_types += [attribute.profile.default(profile_dataset)]
    attribute_types += [attribute.tag.default(tag_dataset)]
    kwargs = {"name": "mcs", "attribute_types": attribute_types}
    kwargs.update({"member_attributes": member_attributes})
    attributes = attribute_option.Attributes(**kwargs)

    kwargs = {"name": name, "dataset": tracking_dataset, "grouping": grouping}
    kwargs.update({"tracking": tracking, "attributes": attributes})
    kwargs.update({"hierarchy_level": 1, "method": "group"})
    mcs_options = track_option.GroupedObjectOptions(**kwargs)

    return mcs_options


def track(dataset="cpol"):
    """Build default options for tracking MCS."""

    mask_options = track_option.MaskOptions(save=False, load=False)
    convective_options = convective(dataset)
    convective_options.mask_options = mask_options
    middle_options = middle(dataset)
    middle_options.mask_options = mask_options
    anvil_options = anvil(dataset)
    anvil_options.mask_options = mask_options
    mcs_options = mcs(dataset)
    objects = [convective_options, middle_options, anvil_options]
    level_0 = track_option.LevelOptions(objects=objects)
    level_1 = track_option.LevelOptions(objects=[mcs_options])
    levels = [level_0, level_1]
    track_options = track_option.TrackOptions(levels=levels)
    return track_options


def runtime(visualize_directory):
    """Build default options for runtime visualization."""

    # kwargs = {"name": "mask", "function": visualize.runtime.visualize_mask}
    # mask_figure = visualize_option.FigureOptions(**kwargs)
    kwargs = {"name": "match", "function": visualize.runtime.visualize_match}
    match_figure = visualize_option.FigureOptions(**kwargs)
    kwargs = {"name": "mcs", "parent_local": visualize_directory}
    kwargs.update({"figures": [match_figure]})
    mcs_figures = visualize_option.ObjectRuntimeOptions(**kwargs)

    # kwargs = {"name": "mask", "function": visualize.runtime.visualize_mask}
    # mask_figure = visualize_option.FigureOptions(**kwargs)
    # kwargs.update({"name": "convective", "figures": [mask_figure]})
    # convective_figures = visualize_option.ObjectRuntimeOptions(**kwargs)
    # objects_dict = {convective_figures.name: convective_figures}
    objects_dict = {mcs_figures.name: mcs_figures}
    visualize_options = visualize_option.RuntimeOptions(objects=objects_dict)
    return visualize_options


def synthetic_track():
    """Build default options for tracking synthetic MCS."""

    convective_options = convective(dataset="synthetic")
    attribute_types = [attribute.core.default_tracked()]
    kwargs = {"name": "convective", "attribute_types": attribute_types}
    attributes = track_option.Attributes(**kwargs)
    convective_options.attributes = attributes
    kwargs = {"global_flow_margin": 70, "unique_global_flow": False}
    convective_options.tracking = track_option.MintOptions(**kwargs)
    levels = [track_option.LevelOptions(objects=[convective_options])]
    return track_option.TrackOptions(levels=levels)


def synthetic_runtime(visualize_directory):
    """Build default options for runtime visualization."""

    # kwargs = {"name": "mask", "function": visualize.runtime.visualize_mask}
    # mask_figure = visualize_option.FigureOptions(**kwargs)
    kwargs = {"name": "match", "function": visualize.runtime.visualize_match}
    match_figure = visualize_option.FigureOptions(**kwargs)
    kwargs = {"name": "convective", "parent_local": visualize_directory}
    kwargs.update({"figures": [match_figure]})
    convective_figures = visualize_option.ObjectRuntimeOptions(**kwargs)

    objects_dict = {convective_figures.name: convective_figures}
    visualize_options = visualize_option.RuntimeOptions(objects=objects_dict)
    return visualize_options
