"""Default options configurations."""

import thuner.option as option
import thuner.attribute as attribute

# from thuner.attribute import core, ellipse, quality, tag, profile, group


def convective(dataset="cpol"):
    """Build default options for convective objects."""
    kwargs = {"name": "convective", "dataset": dataset, "variable": "reflectivity"}
    detection = {"method": "steiner", "altitudes": [500, 3e3], "threshold": 40}
    kwargs.update({"detection": detection, "tracking": None})
    return option.track.DetectedObjectOptions(**kwargs)


def middle(dataset="cpol"):
    """Build default options for mid-level echo objects."""
    kwargs = {"name": "middle", "dataset": dataset, "variable": "reflectivity"}
    detection = {"method": "threshold", "altitudes": [3.5e3, 7e3], "threshold": 20}
    kwargs.update({"detection": detection, "tracking": None})
    return option.track.DetectedObjectOptions(**kwargs)


def anvil(dataset="cpol"):
    """Build default options for anvil objects."""
    kwargs = {"name": "anvil", "dataset": dataset, "variable": "reflectivity"}
    detection = {"method": "threshold", "altitudes": [7.5e3, 10e3], "threshold": 15}
    kwargs.update({"detection": detection, "tracking": None})
    return option.track.DetectedObjectOptions(**kwargs)


def mcs(tracking_dataset="cpol", profile_dataset="era5_pl", tag_dataset="era5_sl"):
    """Build default options for MCS objects."""

    name = "mcs"
    member_objects = ["convective", "middle", "anvil"]
    kwargs = {"name": name, "member_objects": member_objects}
    kwargs.update({"member_levels": [0, 0, 0], "member_min_areas": [80, 400, 800]})

    grouping = option.track.GroupingOptions(**kwargs)
    tracking = option.track.MintOptions(matched_object="convective")

    core_tracked = attribute.core.default_tracked()
    core_member = attribute.core.default_member()

    # Assume the first member object is used for tracking.
    obj = member_objects[0]
    attribute_types = [core_tracked, attribute.quality.default(member_object=obj)]
    attribute_types += [attribute.ellipse.default()]
    kwargs = {"name": member_objects[0], "attribute_types": attribute_types}
    attributes = option.track.Attributes(**kwargs)
    member_attributes = {obj: attributes}
    for obj in member_objects[1:]:
        attribute_types = [core_member, attribute.quality.default(member_object=obj)]
        kwargs = {"name": obj, "attribute_types": attribute_types}
        member_attributes[obj] = option.track.Attributes(**kwargs)

    attribute_types = [core_tracked, attribute.group.default()]
    attribute_types += [attribute.profile.default(profile_dataset)]
    attribute_types += [attribute.tag.default(tag_dataset)]
    kwargs = {"name": "mcs", "attribute_types": attribute_types}
    kwargs.update({"member_attributes": member_attributes})
    attributes = option.attribute.Attributes(**kwargs)

    kwargs = {"name": name, "dataset": tracking_dataset, "grouping": grouping}
    kwargs.update({"tracking": tracking, "attributes": attributes})
    kwargs.update({"hierarchy_level": 1, "method": "group"})
    mcs_options = option.track.GroupedObjectOptions(**kwargs)

    return mcs_options


def track(dataset="cpol"):
    """Build default options for tracking MCS."""

    mask_options = option.track.MaskOptions(save=False, load=False)
    convective_options = convective(dataset)
    convective_options.mask_options = mask_options
    middle_options = middle(dataset)
    middle_options.mask_options = mask_options
    anvil_options = anvil(dataset)
    anvil_options.mask_options = mask_options
    mcs_options = mcs(dataset)
    objects = [convective_options, middle_options, anvil_options]
    level_0 = option.track.LevelOptions(objects=objects)
    level_1 = option.track.LevelOptions(objects=[mcs_options])
    levels = [level_0, level_1]
    track_options = option.track.TrackOptions(levels=levels)
    return track_options


def synthetic_track_options():
    convective_options = convective(dataset="synthetic")
    kwargs = {"global_flow_margin": 70, "unique_global_flow": False}
    convective_options.tracking = option.track.MintOptions(**kwargs)
    levels = [option.track.LevelOptions(objects=[convective_options])]
    return option.track.TrackOptions(levels=levels)
