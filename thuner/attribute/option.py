"""Functions for specifying object attributes."""

from typing import Callable
from pydantic import Field
import thuner.option as option


# def get_attribute_dict(name, method, data_type, precision, description, units):
#     """Create a dictionary of attribute options."""
#     attribute_dict = {}
#     attribute_dict.update({"name": name, "method": method, "data_type": data_type})
#     attribute_dict.update({"precision": precision, "description": description})
#     attribute_dict.update({"units": units})
#     return attribute_dict


# def identity(name="id", method=None, description=None, tracked=True):
#     """
#     Options for id attribute.
#     """
#     data_type = int
#     precision = None
#     units = None
#     if method is None:
#         if tracked:
#             method = {"function": "ids_from_object_record"}
#         else:
#             method = {"function": "ids_from_mask"}
#     if description is None:
#         description = f"{name} taken from object record or object mask. "
#         description += "Unlike uid, id is not necessarily unique across time steps."
#     args = [name, method, data_type, precision, description, units]
#     return utils.get_attribute_dict(*args)


# Convenience functions for creating default core attribute options dictionaries
# def default(names=None, tracked=True, matched=None, grouped=False):
#     """Create a dictionary of default core attributes."""

#     if matched is None:
#         matched = tracked

#     if names is None:
#         names = ["time", "latitude", "longitude"]
#         if not grouped:
#             names += ["area"]
#         if matched:
#             names += ["universal_id"]
#         else:
#             names += ["id"]
#         if tracked:
#             names += ["parents", "u_flow", "v_flow"]
#             names += ["u_displacement", "v_displacement"]

#     attributes_dict = {}
#     for name in names:
#         attributes_dict[name] = attribute_dispatcher[name](name, tracked=tracked)

#     return attributes_dict
