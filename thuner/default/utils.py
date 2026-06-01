"""Shared helpers for building default options."""

import thuner.option.attribute as attribute_option
import thuner.attribute.core as core
import thuner.attribute.ellipse as ellipse
import thuner.attribute.quality as quality


def member_attributes(member_objects):
    """Build default per-member attributes for a grouped object.

    The first member object is assumed to be the tracked object, so it gets the
    tracked core, quality and ellipse attributes. The remaining members get the
    member core and quality attributes.
    """
    obj = member_objects[0]
    attribute_types = [core.default_tracked()]
    attribute_types += [quality.default(member_object=obj)]
    attribute_types += [ellipse.default()]
    member_attributes = {
        obj: attribute_option.Attributes(name=obj, attribute_types=attribute_types)
    }
    for obj in member_objects[1:]:
        attribute_types = [core.default_member()]
        attribute_types += [quality.default(member_object=obj)]
        member_attributes[obj] = attribute_option.Attributes(
            name=obj, attribute_types=attribute_types
        )
    return member_attributes
