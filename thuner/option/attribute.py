"""Classes for object attribute options."""

import importlib
import numpy as np
from numpy import datetime64
from typing import Callable
from pydantic import Field, model_validator
from thuner.option.utils import BaseOptions


_summary = {
    "name": "Name of the attribute or attribute group.",
    "retrieval_method": "Name of the function/method for obtaining the attribute.",
    "data_type": "Data type of the attribute.",
    "precision": "Number of decimal places for a numerical attribute.",
    "description": "Description of the attribute.",
    "units": "Units of the attribute.",
    "retrieval": "The function/kwargs used to retrieve the attribute.",
    "function": "The function used to retrieve the attribute.",
    "keyword_arguments": "Keyword arguments for the retrieval.",
}


class Retrieval(BaseOptions):
    """
    Class for attribute retrieval methods. Generally a function and a dictionary of
    kwargs.
    """

    function: Callable | str | None = Field(None, description=_summary["function"])
    keyword_arguments: dict = Field({}, description=_summary["keyword_arguments"])

    @model_validator(mode="after")
    def check_function(cls, values):
        if isinstance(values.function, str):
            module_name, function_name = values.function.rsplit(".", 1)
            try:
                module = importlib.import_module(module_name)
                values.function = getattr(module, function_name)
            except ImportError:
                message = f"Could not import function {values.function}."
                raise ImportError(message)
            except AttributeError:
                message = f"Function {values.function} not found in {module_name}."
                raise AttributeError(message)
        return values


class Attribute(BaseOptions):
    """
    Base attribute description class. An "attribute" will become a column of a pandas
    dataframe, csv file, sql table, etc.
    """

    name: str = Field(..., description=_summary["name"])
    retrieval: Retrieval | None = Field(None, description=_summary["retrieval"])
    data_type: type | str = Field(..., description=_summary["data_type"])
    precision: int | None = Field(None, description=_summary["precision"])
    description: str | None = Field(None, description=_summary["description"])
    units: str | None = Field(None, description=_summary["units"])

    @model_validator(mode="after")
    def check_data_type(cls, values):
        """
        Check that the data type is valid.
        """
        if isinstance(values.data_type, str):
            # convert string to type
            if "." in values.data_type:
                module_name, type_name = values.data_type.rsplit(".", 1)
                module = importlib.import_module(module_name)
                values.data_type = getattr(module, type_name)
        return values


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


_summary = {
    "member_names": "Names of member objects comprising the grouped object.",
    "attribute_types": "Attribute types of the grouped object.",
    "member_attributes": "List of object attributes for the member objects.",
}

AttributesDict = dict[str, "Attributes"]


class Attributes(BaseOptions):
    """
    Container for the attributes of a grouped object.
    """

    name: str = Field(..., description="Name of the grouped object.")
    attribute_types: list[AttributeType] = Field(
        ..., description=_summary["attribute_types"]
    )
    member_attributes: AttributesDict | None = Field(
        None, description=_summary["member_attributes"]
    )
