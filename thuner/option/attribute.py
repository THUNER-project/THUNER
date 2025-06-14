"""Classes for object attribute options."""

import importlib
import numpy as np
from pydantic import Field, model_validator
from thuner.utils import BaseOptions, Retrieval

__all__ = [
    "Attribute",
    "AttributeGroup",
    "AttributeType",
    "Attributes",
    "Retrieval",
]


class Attribute(BaseOptions):
    """
    Base attribute description class. An "attribute" will become a column of a pandas
    dataframe, csv file, sql table, etc.
    """

    name: str = Field(..., description="Name of the attribute.")
    _desc = "The function/kwargs used to retrieve the attribute."
    retrieval: Retrieval | None = Field(None, description=_desc)
    data_type: type | str = Field(..., description="Data type of the attribute.")
    _desc = "Number of decimal places for a numerical attribute."
    precision: int | None = Field(None, description=_desc)
    description: str | None = Field(None, description="Description of the attribute.")
    units: str | None = Field(None, description="Units of the attribute.")

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

    name: str = Field(..., description="Name of the attribute group.")
    attributes: list[Attribute] = Field(..., description="Attributes in the group.")
    _desc = "The function/kwargs used to retrieve the attributes in the group."
    retrieval: Retrieval | None = Field(None, description=_desc)
    _desc = "Description of the attribute group."
    description: str | None = Field(None, description=_desc)

    @model_validator(mode="after")
    def check_retrieval(cls, values):
        """
        Check that the retrieval method is the same for all attributes in the group.
        Also check that the shared retrieval method is the same as the group retrieval
        method if one has been provided.
        """
        retrievals = []
        for attribute in values.attributes:
            try:
                retrievals.append(attribute.retrieval)
            except:
                print("none")
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


AttributeList = list[Attribute | AttributeGroup]
AttributeDict = dict[str, Attribute | AttributeGroup]


class AttributeType(BaseOptions):
    """
    Attribute type options. Each "attribute type" contains attributes and attribute
    groups, and will form a single pandas dataframe, csv file, sql table, etc.
    """

    name: str = Field(..., description="Name of the attribute type.")
    _desc = "Description of the attribute type."
    description: str | None = Field(None, description=_desc)
    _desc = "Attributes and attribute groups comprising the attribute type."
    attributes: AttributeList = Field(..., description=_desc)
    # If the attribute type corresponds to a specific tagging dataset, specify it here
    _desc = "Dataset for tag attribute types (None if not applicable)."
    dataset: str | None = Field(None, description=_desc)
    _desc = "Lookup dictionary for attributes."
    _attribute_lookup = {}

    @model_validator(mode="after")
    def initialize_lookup(cls, values):
        """
        Initialize the lookup dictionary for attributes. This is used to quickly access
        attributes by name.
        """
        values._attribute_lookup = {}
        for attribute in values.attributes:
            values._attribute_lookup[attribute.name] = attribute
        return values

    def attribute_by_name(self, name: str) -> Attribute | AttributeGroup:
        """
        Get an attribute by name.
        """
        try:
            return self._attribute_lookup[name]
        except KeyError:
            message = f"Attribute {name} not found in attribute type {self.name}."
            raise KeyError(message)


AttributesDict = dict[str, "Attributes"]


class Attributes(BaseOptions):
    """
    Class for storing all the attribute options for a given object.
    """

    name: str = Field(..., description="Name of the object.", examples=["mcs"])
    _desc = "Attribute types of the object."
    attribute_types: list[AttributeType] = Field(..., description=_desc)
    _desc = "Lookup dictionary for attribute types."
    _attribute_type_lookup = {}
    _desc = "List of object attributes for the member objects."
    member_attributes: AttributesDict | None = Field(None, description=_desc)

    @model_validator(mode="after")
    def initialize_lookup(cls, values):
        """
        Initialize the lookup dictionary for attribute types. This is used to quickly
        access attribute types by name.
        """
        values._attribute_type_lookup = {}
        for attribute_type in values.attribute_types:
            values._attribute_type_lookup[attribute_type.name] = attribute_type
        return values

    def attribute_type_by_name(self, name: str) -> AttributeType:
        """
        Get an attribute type by name.
        """
        try:
            return self._attribute_type_lookup[name]
        except KeyError:
            message = f"Attribute type {name} not found."
            raise KeyError(message)
