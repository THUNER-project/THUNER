"""Default tracking-option configurations."""

import thuner.option.track as track_option
import thuner.option.attribute as attribute_option
import thuner.attribute.core as core
import thuner.attribute.group as group
import thuner.attribute.tag as tag
import thuner.attribute.profile as profile
import thuner.attribute.ellipse as ellipse
import thuner.attribute.quality as quality
from thuner.default.utils import member_attributes

__all__ = [
    "convective",
    "middle",
    "anvil",
    "satellite_anvil",
    "mcs",
    "access_c_mcs",
    "track",
    "access_c_track",
    "satellite_track",
    "synthetic_track",
]


def convective(dataset="cpol"):
    """Build default options for convective objects."""
    detection = {"method": "steiner", "altitudes": (500, 3e3), "threshold": 40}
    return track_option.DetectedObjectOptions(
        name="convective",
        dataset=dataset,
        detection=detection,
        tracking=None,
    )


def middle(dataset="cpol"):
    """Build default options for mid-level echo objects."""
    detection = {"method": "threshold", "altitudes": (3.5e3, 7e3), "threshold": 20}
    return track_option.DetectedObjectOptions(
        name="middle",
        dataset=dataset,
        detection=detection,
        tracking=None,
    )


def anvil(dataset="cpol"):
    """Build default options for anvil objects."""
    detection = {"method": "threshold", "altitudes": (7.5e3, 10e3), "threshold": 15}
    return track_option.DetectedObjectOptions(
        name="anvil",
        dataset=dataset,
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
        detection=det_kwargs,
        tracking=tracking,
        attributes=attributes,
    )


def mcs(tracking_dataset="cpol", profile_dataset="era5_pl", tag_dataset="era5_sl"):
    """Build default options for MCS objects."""

    name = "mcs"
    member_objects = ["convective", "middle", "anvil"]

    grouping = track_option.GroupingOptions(
        member_objects=member_objects,
        member_levels=[0, 0, 0],
        member_min_areas=[80, 400, 800],
    )
    tracking = track_option.MintOptions(matched_object="convective")

    member_attrs = member_attributes(member_objects)

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
        member_attributes=member_attrs,
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
        member_objects=member_objects,
        member_levels=[0, 0],
        member_min_areas=[80, 800],
    )
    tracking = track_option.MintOptions(
        matched_object="convective", global_flow_margin=70, unique_global_flow=False
    )

    member_attrs = member_attributes(member_objects)

    mcs_core = core.default_tracked()
    attribute_types = [mcs_core, group.default()]
    # Leave out the profile and tag attributes for now
    # attribute_types += [profile.default(profile_dataset)]
    # attribute_types += [tag.default(tag_dataset)]
    attributes = attribute_option.Attributes(
        name="mcs",
        attribute_types=attribute_types,
        member_attributes=member_attrs,
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


def _two_level_track(detected_options, mcs_options):
    """Assemble a two-level MCS TrackOptions from detected and grouped objects.

    The detected objects form level 0 and the grouped MCS object forms level 1.
    Masks are neither saved nor loaded by default.
    """
    mask_options = track_option.MaskOptions(save=False, load=False)
    for options in detected_options:
        options.mask_options = mask_options
    level_0 = track_option.LevelOptions(objects=detected_options)
    level_1 = track_option.LevelOptions(objects=[mcs_options])
    return track_option.TrackOptions(levels=[level_0, level_1])


def track(dataset_name: str = "cpol"):
    """Build default options for tracking MCS."""

    detected_options = [
        convective(dataset_name),
        middle(dataset_name),
        anvil(dataset_name),
    ]
    return _two_level_track(detected_options, mcs(dataset_name))


def access_c_track(
    convective_dataset: str = "access_1km", anvil_dataset: str = "access_maxcol"
):
    """Build default options for tracking MCS."""

    convective_options = convective(convective_dataset)
    anvil_options = anvil(anvil_dataset)
    # The convective and anvil datasets are 2D, so there is no altitude dimension to
    # flatten over: drop the flatten method and altitude range.
    convective_options.detection.flatten_method = None
    convective_options.detection.altitudes = None
    anvil_options.detection.flatten_method = None
    anvil_options.detection.altitudes = None
    mcs_options = access_c_mcs(convective_dataset)
    return _two_level_track([convective_options, anvil_options], mcs_options)


def satellite_track(dataset_name: str = "himawari"):
    """Build default options for tracking anvils in satellite data."""
    anvil_options = satellite_anvil(dataset_name)
    level = track_option.LevelOptions(objects=[anvil_options])
    return track_option.TrackOptions(levels=[level])


def synthetic_track():
    """Build default options for tracking synthetic MCS."""

    convective_options = convective(dataset="synthetic")
    attribute_types = [core.default_tracked()]
    attributes = attribute_option.Attributes(
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
