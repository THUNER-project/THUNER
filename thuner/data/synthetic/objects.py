"""
Synthetic meteorological object definitions.

Each object is a typed, self-contained description of a synthetic feature: its
kinematics (position, direction, speed), its geometry, and how it renders itself into
a gridded field. New object types are added as :class:`SyntheticObject` subclasses,
each defining its own geometry fields and ``render`` method.
"""

from typing import Literal
import numpy as np
import xarray as xr
from pydantic import Field, model_validator
from pyproj import Geod
from thuner.utils import BaseOptions

geod = Geod(ellps="WGS84")

# Finite-difference step (degrees) for the local metres-per-degree scale at an object's
# centre; small enough that the linearised scale is exact to well within rendering needs.
_SCALE_STEP = 0.01


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
            # Use the default precision for coordinates and velocities given in
            # attribute.core
            "latitude": np.round(self.center_latitude, 4),
            "longitude": np.round(self.center_longitude, 4),
            "u": np.round(u, 1),
            "v": np.round(v, 1),
        }


class EllipsoidObject(SyntheticObject):
    """
    A rotated ellipsoid, e.g. a convective cell.

    The horizontal cross-section is an ellipse with full axis lengths ``major`` and
    ``minor`` (km), rotated by ``orientation``. ``style`` selects the intensity profile:
    a 3-D Gaussian about ``center_altitude`` (``'gaussian'``) or a uniform fill of the
    ellipsoid (``'flat'``). ``major``, ``minor`` and ``orientation`` follow the
    ellipse-fit attribute convention (see :mod:`thuner.attribute.ellipse`, where
    ``major``/``minor`` are the full axes returned by ``cv2.fitEllipse``), so ground
    truth is in the same units and convention as tracked output; eccentricity is derived
    from the axes.
    """

    name: str = Field("cell", description="Object type label.")
    major: float = Field(40, description="Major axis (full length) in km.")
    minor: float = Field(16, description="Minor axis (full length) in km.")
    orientation: float = Field(
        np.pi / 4, description="Major-axis orientation in radians."
    )
    center_altitude: float = Field(
        3e3, description="Altitude of the object center in m."
    )
    altitude_radius: float = Field(1e3, description="Vertical radius in m.")
    style: Literal["gaussian", "flat"] = Field(
        "gaussian",
        description=(
            "Intensity profile. 'gaussian': a 3-D Gaussian whose value at the major/minor "
            "ellipse edge is intensity/sqrt(e). 'flat': the ellipsoid filled uniformly "
            "with intensity."
        ),
    )
    intensity: float = Field(
        42 * np.sqrt(np.e),
        description=(
            "Peak field value, e.g. dBZ. The default is chosen so a 'gaussian' object's "
            "value at the major/minor ellipse edge equals the Steiner "
            "definitely-convective threshold (42 dBZ)."
        ),
    )

    @model_validator(mode="after")
    def _check_axes(self):
        """The major axis must be the longer one, and both must be positive."""
        if not 0 < self.minor <= self.major:
            message = "Require 0 < minor <= major for an ellipse "
            message += f"(got major={self.major}, minor={self.minor})."
            raise ValueError(message)
        return self

    def render(self, ds, grid_options):
        """Add an elliptical blob (Gaussian or flat per ``style``) to ``ds[self.field]``."""
        LON, LAT, ALT = ds.LON, ds.LAT, ds.ALT

        # Local east/north distance (km) of each cell from the centre. Metres-per-degree
        # are taken from the WGS84 ellipsoid at the centre, so the ellipse keeps its true
        # shape and size at any latitude (a degree of longitude shrinks polewards).
        clon, clat = self.center_longitude, self.center_latitude
        m_per_deg_lon = geod.inv(clon, clat, clon + _SCALE_STEP, clat)[2] / _SCALE_STEP
        m_per_deg_lat = geod.inv(clon, clat, clon, clat + _SCALE_STEP)[2] / _SCALE_STEP
        east = (LON - clon) * m_per_deg_lon / 1e3
        north = (LAT - clat) * m_per_deg_lat / 1e3

        # Rotate into the ellipse's principal axes and normalise by the semi-axes (km).
        # major/minor are full axis lengths, so the Gaussian scale along each axis (its
        # 1-sigma half-extent) is half of them.
        major_coord = east * np.cos(self.orientation) + north * np.sin(self.orientation)
        minor_coord = -east * np.sin(self.orientation) + north * np.cos(
            self.orientation
        )

        distance = np.sqrt(
            (major_coord / (self.major / 2)) ** 2
            + (minor_coord / (self.minor / 2)) ** 2
            + ((ALT - self.center_altitude) / self.altitude_radius) ** 2
        )

        if self.style == "gaussian":
            values = self.intensity * np.exp(-(distance**2) / 2)
            values = values.where(values >= 0.05 * self.intensity, np.nan)
        else:  # "flat": uniform fill inside the major/minor ellipsoid (distance <= 1).
            values = xr.where(distance <= 1, self.intensity, np.nan)
        values = values.transpose(*ds.dims)
        ds[self.field].values = xr.where(~np.isnan(values), values, ds[self.field])
        return ds

    def ground_truth(self):
        """Augment the base ground truth with the ellipse geometry.

        Eccentricity is inferred from the axes, matching :mod:`thuner.attribute.ellipse`.
        Values are rounded to the precisions declared there (major/minor: 1 dp,
        orientation/eccentricity: 4 dp).
        """
        truth = super().ground_truth()
        eccentricity = np.sqrt(1 - (self.minor / self.major) ** 2)
        truth.update(
            {
                "major": np.round(self.major, 1),
                "minor": np.round(self.minor, 1),
                "orientation": np.round(self.orientation, 4),
                "eccentricity": np.round(eccentricity, 4),
                "intensity": self.intensity,
            }
        )
        return truth
