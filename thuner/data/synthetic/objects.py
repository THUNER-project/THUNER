"""
Synthetic meteorological object definitions.

Each object is a typed, self-contained description of a synthetic feature: its
kinematics (position, direction, speed), its geometry, and how it renders itself into
a gridded field. New object types are added as :class:`SyntheticObject` subclasses,
each defining its own geometry fields and ``render`` method.
"""

import numpy as np
import xarray as xr
from pydantic import Field
from pyproj import Geod
from thuner.utils import BaseOptions

geod = Geod(ellps="WGS84")


class SyntheticObject(BaseOptions):
    """
    Base class for a synthetic object.

    The base class handles kinematics (constant-velocity geodesic motion) and ground
    truth. Subclasses add geometry fields and implement :meth:`render`.
    """

    id: int | None = Field(
        None, description="Stable identifier; assigned by the generator if None."
    )
    name: str = Field("object", description="Object type label.")
    time: str = Field(..., description="Time at which the object has these properties.")
    center_latitude: float = Field(..., description="Latitude of the object center.")
    center_longitude: float = Field(..., description="Longitude of the object center.")
    direction: float = Field(
        ..., description="Direction of motion in radians clockwise from north."
    )
    speed: float = Field(..., description="Speed of motion in metres per second.")
    field: str = Field(
        "reflectivity", description="Dataset variable this object renders into."
    )

    def advance(self, time):
        """Return a copy of this object moved to ``time`` along its velocity vector."""
        time_diff = np.datetime64(time) - np.datetime64(self.time)
        time_diff = time_diff.astype("timedelta64[s]").astype(float)
        distance = time_diff * self.speed
        new_lon, new_lat = geod.fwd(
            lons=self.center_longitude,
            lats=self.center_latitude,
            az=np.rad2deg(self.direction),
            dist=distance,
        )[0:2]
        update = {
            "center_longitude": new_lon,
            "center_latitude": new_lat,
            "time": str(time),
        }
        return self.model_copy(update=update)

    def render(self, ds, grid_options):
        """Add this object's contribution to ``ds[self.field]``. Implemented by subclasses."""
        raise NotImplementedError

    def velocity(self):
        """Return the ground-truth (u, v) velocity in m/s (eastward, northward)."""
        u = self.speed * np.sin(self.direction)
        v = self.speed * np.cos(self.direction)
        return u, v

    def ground_truth(self):
        """Return the known attributes of this object as a flat dict."""
        u, v = self.velocity()
        return {
            "id": self.id,
            "time": np.datetime64(self.time),
            "latitude": self.center_latitude,
            "longitude": self.center_longitude,
            "u": u,
            "v": v,
        }


class EllipsoidObject(SyntheticObject):
    """
    A rotated 3-D Gaussian ellipsoid, e.g. a convective cell.

    The horizontal cross-section is an ellipse (rotated by ``orientation``, squashed by
    ``eccentricity``); the vertical profile is Gaussian about ``alt_center``.
    """

    name: str = Field("cell", description="Object type label.")
    horizontal_radius: float = Field(
        20, description="Approximate horizontal radius in km."
    )
    eccentricity: float = Field(0.4, description="Minor/major axis ratio.")
    orientation: float = Field(
        np.pi / 4, description="Major-axis orientation in radians."
    )
    alt_center: float = Field(3e3, description="Altitude of the object center in m.")
    alt_radius: float = Field(1e3, description="Vertical radius in m.")
    intensity: float = Field(50, description="Peak field value, e.g. dBZ.")

    def render(self, ds, grid_options):
        """Add an elliptical/Gaussian blob to ``ds[self.field]`` to emulate a cell."""
        LON, LAT, ALT = ds.LON, ds.LAT, ds.ALT

        # Rotate the horizontal coordinates into the object's principal axes.
        lon_rotated = (LON - self.center_longitude) * np.cos(self.orientation)
        lon_rotated += (LAT - self.center_latitude) * np.sin(self.orientation)
        lat_rotated = -(LON - self.center_longitude) * np.sin(self.orientation)
        lat_rotated += (LAT - self.center_latitude) * np.cos(self.orientation)

        # Convert horizontal_radius (km) to an approximate lat/lon radius (degrees).
        horizontal_radius = self.horizontal_radius / 111.32

        distance = np.sqrt(
            (lon_rotated / horizontal_radius) ** 2
            + (lat_rotated / (horizontal_radius * self.eccentricity)) ** 2
            + ((ALT - self.alt_center) / self.alt_radius) ** 2
        )

        values = self.intensity * np.exp(-(distance**2) / 2)
        values = values.where(values >= 0.05 * self.intensity, np.nan)
        values = values.transpose(*ds.dims)
        ds[self.field].values = xr.where(~np.isnan(values), values, ds[self.field])
        return ds

    def ground_truth(self):
        """Augment the base ground truth with the ellipse geometry."""
        truth = super().ground_truth()
        truth.update(
            {
                "horizontal_radius": self.horizontal_radius,
                "eccentricity": self.eccentricity,
                "orientation": self.orientation,
                "intensity": self.intensity,
            }
        )
        return truth
