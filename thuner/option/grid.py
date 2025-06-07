"""Classes for grid options."""

import numpy as np
from pydantic import Field, model_validator
from thuner.utils import BaseOptions
from thuner.log import setup_logger

__all__ = ["GridOptions"]

logger = setup_logger(__name__)


class GridOptions(BaseOptions):
    """Class for grid options."""

    name: str = "geographic"
    _desc = "z-coordinates for the dataset."
    altitude: list[float] | None = Field(None, description=_desc)
    _desc = "latitudes for the dataset."
    latitude: list[float] | None = Field(None, description=_desc)
    _desc = "longitudes for the dataset."
    longitude: list[float] | None = Field(None, description=_desc)
    _desc = "Central latitude for the dataset."
    central_latitude: float | None = Field(None, description=_desc)
    _desc = "Central longitude for the dataset."
    central_longitude: float | None = Field(None, description=_desc)
    _desc = "x-coordinates for the dataset in meters."
    x: list[float] | None = Field(None, description=_desc)
    _desc = "y-coordinates for the dataset in meters."
    y: list[float] | None = Field(None, description=_desc)
    _desc = "Projection used if the dataset is cartesian."
    projection: str | None = Field(None, description=_desc)
    _desc = "Spacing for the altitude grid in metres."
    altitude_spacing: float | None = Field(500, description=_desc)
    _desc = "Spacing for the horizontal cartesian grid [y, x] in metres."
    cartesian_spacing: list[float] | None = Field([2500, 2500], description=_desc)
    _desc = "Spacing for the horizontal geographic grid [lat, lon] in degrees."
    geographic_spacing: list[float] | None = Field([0.025, 0.025], description=_desc)
    _desc = "Shape of the dataset."
    shape: tuple[int, int] | None = Field(None, description=_desc)
    _desc = "Whether to attempt to regrid the dataset."
    regrid: bool = Field(True, description=_desc)

    @model_validator(mode="after")
    def _check_altitude(cls, values):
        """Ensure altitudes are initialized."""
        if values.altitude is None and values.altitude_spacing is not None:
            spacing = values.altitude_spacing
            altitude = list(np.arange(0, 20e3 + spacing, spacing))
            altitude = [float(alt) for alt in altitude]
            values.altitude = altitude
            logger.warning("altitude not specified. Using default altitudes.")
        elif values.altitude_spacing is None and values.altitude is None:
            message = "altitude_spacing not specified. Will attempt to infer from "
            message += "input."
            logger.warning(message)
        return values

    @model_validator(mode="after")
    def _check_shape(cls, values):
        """Ensure shape is initialized."""
        latitude, longitude = values.latitude, values.longitude
        if values.shape is None and (latitude is not None and longitude is not None):
            values.shape = (len(latitude), len(longitude))
        if values.shape is None and (values.x is not None and values.y is not None):
            values.shape = (len(values.y), len(values.x))
        else:
            logger.warning("shape not specified. Will attempt to infer from input.")
        return values
