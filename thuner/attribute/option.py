"""Functions for specifying object attributes."""

import numpy as np
from typing import Callable
from pydantic import Field, model_validator
from thuner.utils import BaseOptions


_summary = {
    "name": "Name of the attribute or attribute group.",
    "retrieval_method": "Name of the function/method for obtaining the attribute.",
    "data_type": "Data type of the attribute.",
    "precision": "Number of decimal places for a numerical attribute.",
    "description": "Description of the attribute.",
    "units": "Units of the attribute.",
    "retrieval": "The function/method used to retrieve the attribute.",
}


class BaseAttribute(BaseOptions):
    """
    Base attribute description class. An "attribute" will become a column of a pandas
    dataframe or csv file.
    """

    name: str = Field(..., description=_summary["name"])
    retrieval: Callable | str | None = Field(None, description=_summary["retrieval"])
    data_type: type = Field(..., description=_summary["data_type"])
    precision: int | None = Field(None, description=_summary["precision"])
    description: str | None = Field(None, description=_summary["description"])
    units: str | None = Field(None, description=_summary["units"])


class AttributeGroup(BaseOptions):
    """
    A group of closely related attributes retrieved together, e.g. lat/lon or u/v.
    """

    attributes: list[BaseAttribute] = Field(
        ..., description="Attributes comprising the group."
    )
    retrieval: Callable | str | None = Field(None, description=_summary["retrieval"])
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


class AttributeType(BaseOptions):
    """
    Attribute type options. Each "attribute type" will form a pandas dataframe or csv
    file.
    """

    name: str = Field(..., description="Name of the attribute type.")
    attributes: list[BaseAttribute | AttributeGroup] = Field(
        ..., description="List of attributes or attribute groups comprising the type."
    )
