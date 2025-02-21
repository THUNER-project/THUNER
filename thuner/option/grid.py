"""Classes for grid options."""

import numpy as np
from pydantic import Field, model_validator
from thuner.utils import BaseOptions
from thuner.log import setup_logger

__all__ = ["GridOptions"]

logger = setup_logger(__name__)


_summary = {
    "timestep": "Time step for the dataset.",
    "start_latitude": "Starting latitude for the dataset.",
    "end_latitude": "Ending latitude for the dataset.",
    "start_longitude": "Starting longitude for the dataset.",
    "end_longitude": "Ending longitude for the dataset.",
    "central_latitude": "Central latitude for the dataset.",
    "central_longitude": "Central longitude for the dataset.",
    "projection": "Projection for the dataset.",
    "start_x": "Starting x-coordinate for the dataset.",
    "end_x": "Ending x-coordinate for the dataset.",
    "start_y": "Starting y-coordinate for the dataset.",
    "end_y": "Ending y-coordinate for the dataset.",
    "start_alt": "Starting z-coordinate for the dataset.",
    "end_alt": "Ending z-coordinate for the dataset.",
    "cartesian_spacing": "Spacing for the cartesian grid [z, y, x] in metres.",
    "geographic_spacing": "Spacing for the geographic grid [z, lat, lon] in metres and degrees.",
    "regrid": "Whether to regrid the dataset.",
    "save": "Whether to save the dataset.",
    "altitude_spacing": "Spacing for the altitude grid in metres.",
    "x": "x-coordinates for the dataset.",
    "y": "y-coordinates for the dataset.",
    "altitude": "z-coordinates for the dataset.",
    "latitude": "latitudes for the dataset.",
    "longitude": "longitudes for the dataset.",
    "shape": "Shape of the dataset.",
}


class GridOptions(BaseOptions):
    """Class for grid options."""

    name: str = "geographic"
    altitude: list[float] | None = Field(None, description=_summary["altitude"])
    latitude: list[float] | None = Field(None, description=_summary["latitude"])
    longitude: list[float] | None = Field(None, description=_summary["longitude"])
    central_latitude: float | None = Field(
        None, description=_summary["central_latitude"]
    )
    central_longitude: float | None = Field(
        None, description=_summary["central_longitude"]
    )
    x: list[float] | None = Field(None, description=_summary["x"])
    y: list[float] | None = Field(None, description=_summary["y"])
    projection: str | None = Field(None, description=_summary["projection"])
    altitude_spacing: float | None = Field(
        500, description=_summary["altitude_spacing"]
    )
    cartesian_spacing: list[float] | None = Field(
        [2500, 2500], description=_summary["cartesian_spacing"]
    )
    geographic_spacing: list[float] | None = Field(
        [0.025, 0.025], description=_summary["geographic_spacing"]
    )
    shape: tuple[int, int] | None = Field(None, description=_summary["shape"])
    regrid: bool = Field(True, description=_summary["regrid"])
    save: bool = Field(False, description=_summary["save"])

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
