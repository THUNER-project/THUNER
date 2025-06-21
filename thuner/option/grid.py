"""Classes for grid options."""

import numpy as np
from pydantic import Field, model_validator
from typing import Literal
from thuner.utils import BaseOptions
from thuner.log import setup_logger

__all__ = ["GridOptions"]

logger = setup_logger(__name__)


class GridOptions(BaseOptions):
    """Class for grid options."""

    _desc = "Name of the grid."
    name: Literal["geographic", "cartesian"] = Field("geographic", description=_desc)
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

    @model_validator(mode="after")
    def _check_spacing(cls, values):
        """Ensure spacing is consistent with input dimensions."""

        def check_diffs(coord, coord_name, spacing, spacing_name):
            """
            Check if the coordinate is evenly spaced, and if so whether spacing matches
            provided spacing.
            """
            diffs = list(set(np.round(np.diff(coord), 8)))
            if len(diffs) != 1:
                message = f"{spacing_name} and {coord_name} provided, but {coord_name} "
                message += f"not evenly spaced."
                raise ValueError(message)
            if diffs[0] != spacing:
                message = f"{spacing_name} and {coord_name} provided, but actual "
                message += f"{coord_name} spacing {diffs[0]} does not match {spacing}."
                raise ValueError(message)

        if values.cartesian_spacing is not None and values.y is not None:
            args = [values.y, "y", values.cartesian_spacing[0], "cartesian_spacing"]
            check_diffs(*args)
        if values.cartesian_spacing is not None and values.x is not None:
            args = [values.x, "x", values.cartesian_spacing[1], "cartesian_spacing"]
            check_diffs(*args)
        if values.geographic_spacing is not None and values.latitude is not None:
            args = [values.latitude, "latitude", values.geographic_spacing[0]]
            args += ["geographic_spacing"]
            check_diffs(*args)
        if values.geographic_spacing is not None and values.longitude is not None:
            args = [values.longitude, "longitude", values.geographic_spacing[1]]
            args += ["geographic_spacing"]
            check_diffs(*args)
        return values
