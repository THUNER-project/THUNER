import numpy as np
from itertools import chain
from thor.log import setup_logger
import thor.attribute.core as core
import thor.attribute.utils as utils
import xarray as xr

logger = setup_logger(__name__)


def cape(dataset, name, method=None, description=None):
    """
    Specify options for a cape type property.
    """
    attribute_dict = {}
    data_type = float
    precision = 1
    units = "J/kg"
    if method is None:
        method = {"function": "from_centers", "dataset": dataset}
        method["args"] = {"center_type": "area_weighted"}
    if description is None:
        description = f"{name} at the object center."
    attribute_dict.update({"name": name, "method": method, "data_type": data_type})
    attribute_dict.update({"precision": precision, "description": description})
    attribute_dict.update({"units": units})
    return attribute_dict


def default(dataset, names=None, matched=True):
    """Create a dictionary of default attribute options of grouped objects."""

    if names is None:
        names = ["time", "latitude", "longitude", "cape", "cin"]
    if matched:
        id_type = "universal_id"
    else:
        id_type = "id"
    core_method = {"function": "attribute_from_core"}
    attributes = {}
    # Reuse core attributes, just replace the default functions method
    attributes["time"] = core.time(method=core_method)
    attributes["latitude"] = core.coordinate("latitude", method=core_method)
    attributes["longitude"] = core.coordinate("longitude", method=core_method)
    attributes[id_type] = core.identity(id_type, method=core_method)
    if "cape" in names:
        attributes["cape"] = cape(dataset, "cape")
    if "cin" in names:
        attributes["cin"] = cape(dataset, "cin")

    return attributes
