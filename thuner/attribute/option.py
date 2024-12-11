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

_summary = {
    "name": "Name of the attribute.",
    "retrieval_method": "Name of the function/method for obtaining the attribute.",
    "data_type": "Data type of the attribute.",
    "precision": "Number of decimal places for a numerical attribute.",
    "description": "Description of the attribute.",
    "units": "Units of the attribute.",
    "function": "The function/method used to retrieve the attribute.",
}


class BaseAttribute(option.BaseOptions):
    """
    Base attribute description class. An "attribute" will become a column of a pandas
    dataframe or csv file.
    """

    name: str = Field(..., description=_summary["name"])
    retrieval_method: dict | None = Field(
        None, description=_summary["retrieval_method"]
    )
    data_type: type = Field(..., description=_summary["data_type"])
    precision: int | None = Field(None, description=_summary["precision"])
    description: str | None = Field(None, description=_summary["description"])
    units: str | None = Field(None, description=_summary["units"])


class RetrievalMethod(option.BaseOptions):
    """
    Method for retrieving an attribute.
    """

    function: Callable = Field(..., description=_summary["function"])
    kwargs: dict = Field({}, description="Dictionary of key word arguments.")


class AttributeGroup(option.BaseOptions):
    """
    A group of closely related attributes, typically retrieved together,
    e.g. lat/lon or u/v.
    """

    name: str = Field(..., description="Name of the attribute group.")
    attributes: list[BaseAttribute] = Field(
        ..., description="List of attributes comprising the group."
    )
