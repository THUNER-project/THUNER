"""Functions for specifying object attributes."""

import numpy as np
from typing import Callable, Union
from pydantic import Field, model_validator
from thuner.utils import BaseOptions


_summary = {
    "name": "Name of the attribute or attribute group.",
    "retrieval_method": "Name of the function/method for obtaining the attribute.",
    "data_type": "Data type of the attribute.",
    "precision": "Number of decimal places for a numerical attribute.",
    "description": "Description of the attribute.",
    "units": "Units of the attribute.",
    "retrieval": "The function/kwargs used to retrieve the attribute.",
    "function": "The function used to retrieve the attribute.",
    "arguments": "Keyword arguments for the retrieval.",
}


class Retrieval(BaseOptions):
    """
    Class for attribute retrieval methods. Generally a function and a dictionary of
    kwargs.
    """

    function: Callable | None = Field(None, description=_summary["function"])
    arguments: dict | None = Field(None, description=_summary["arguments"])


class Attribute(BaseOptions):
    """
    Base attribute description class. An "attribute" will become a column of a pandas
    dataframe, csv file, sql table, etc.
    """

    name: str = Field(..., description=_summary["name"])
    retrieval: Retrieval | None = Field(None, description=_summary["retrieval"])
    data_type: type = Field(..., description=_summary["data_type"])
    precision: int | None = Field(None, description=_summary["precision"])
    description: str | None = Field(None, description=_summary["description"])
    units: str | None = Field(None, description=_summary["units"])


class AttributeGroup(BaseOptions):
    """
    A group of related attributes retrieved by the same method, e.g. lat/lon or u/v.
    """

    name: str = Field(..., description=_summary["name"])
    attributes: list[Attribute] = Field(
        ..., description="Attributes comprising the group."
    )
    retrieval: Retrieval | None = Field(None, description=_summary["retrieval"])
    description: str | None = Field(None, description=_summary["description"])

    @model_validator(mode="after")
    def check_retrieval(cls, values):
        """
        Check that the retrieval method is the same for all attributes in the group.
        Also check that the shared retrieval method is the same as the group retrieval
        method if one has been provided.
        """
        retrievals = []
        for attribute in values.attributes:
            retrievals.append(attribute.retrieval)
        if np.all(np.array(retrievals) == None):
            # If retrieval for all attributes is None, do nothing
            return values
        if values.retrieval is None and len(set(retrievals)) > 1:
            message = "attributes in group must have the same retrieval method."
            raise ValueError(message)
        elif values.retrieval is None:
            # if retrieval is None, set it to the common retrieval method
            values.retrieval = retrievals[0]
        return values


_summary = {
    "attributes": "List of attributes or attribute groups comprising the type.",
    "attribute_types": "List of the object's attribute types.",
    "dataset": "Dataset for tag attribute types (None if not applicable).",
    "description": "Description of the attribute type.",
}

AttributeList = list[Attribute | AttributeGroup]


class AttributeType(BaseOptions):
    """
    Attribute type options. Each "attribute type" contains attributes and attribute
    groups, and will form a single pandas dataframe, csv file, sql table, etc.
    """

    name: str = Field(..., description="Name of the attribute type.")
    description: str | None = Field(None, description=_summary["description"])
    attributes: AttributeList = Field(..., description=_summary["attributes"])
    # If the attribute type corresponds to a specific tagging dataset, specify it here
    dataset: str | None = Field(None, description=_summary["dataset"])


class DetectedObjectAttributes(BaseOptions):
    """
    Container for the attributes of a given object.
    """

    object_name: str = Field(..., description="Name of the object.")
    attribute_types: list[AttributeType] = Field(
        ..., description=_summary["attribute_types"]
    )


_summary = {
    "member_names": "Names of member objects comprising the grouped object.",
    "attribute_types": "Attribute types of the grouped object.",
    "member_attribute_types": "Dict containing name and attributes of member objects.",
}

AttributesList = list[Union[DetectedObjectAttributes, "GroupedObjectAttributes"]]
AttributesDict = dict[str, AttributesList]


class GroupedObjectAttributes(BaseOptions):
    """
    Container for the attributes of a grouped object.
    """

    name: str = Field(..., description="Name of the grouped object.")
    attribute_types: list[AttributeType] = Field(
        ..., description=_summary["attribute_types"]
    )
    member_attribute_types: AttributesDict | None = Field(
        None, description=_summary["member_attribute_types"]
    )
