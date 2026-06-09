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
from thuner.utils import BaseOptions, DatetimeField

geod = Geod(ellps="WGS84")

# Finite-difference step (degrees) for the local metres-per-degree scale at an object's
# centre; small enough that the linearised scale is exact to well within rendering needs.
_SCALE_STEP = 0.01

# Normalised distance at which a Gaussian object falls to 5% of its peak (the render
# cutoff): sqrt(-2 ln 0.05). The horizontal footprint reaches this multiple of the
# semi-axis, used when culling objects that have left the domain.
_GAUSSIAN_EXTENT = float(np.sqrt(-2 * np.log(0.05)))


def _bbox_index_bounds(flags):
    """``(first, last + 1)`` index of the True span in a 1-D boolean array.

    Assumes ``flags`` has at least one True (callers guard with ``.any()``).
    """
    indices = np.flatnonzero(flags)
    return int(indices[0]), int(indices[-1]) + 1


class SyntheticObject(BaseOptions):
    """
    Base class for a synthetic object.

    The base class handles kinematics (constant-velocity geodesic motion), lifecycle
    (``birth_time``/``life_time`` and fade-in/out intensity ramps) and ground truth.
    Subclasses add geometry fields and implement :meth:`render` and
    :meth:`horizontal_extent`.
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
    birth_time: DatetimeField | None = Field(
        None, description="Datetime the object appears. None -> the run start."
    )
    life_time: float | None = Field(
        None, description="Lifetime in minutes. None -> lives until the run ends."
    )
    fade_in_time: float = Field(
        0.0, description="Linear intensity ramp-up in minutes after birth."
    )
    fade_out_time: float = Field(
        0.0, description="Linear intensity ramp-down in minutes before death."
    )

    @model_validator(mode="after")
    def _check_lifecycle(self):
        """Fades must be non-negative and fit within a (positive) lifetime."""
        if self.fade_in_time < 0 or self.fade_out_time < 0:
            raise ValueError("fade_in_time and fade_out_time must be non-negative.")
        if self.life_time is not None:
            if self.life_time <= 0:
                raise ValueError("life_time must be positive.")
            if self.fade_in_time + self.fade_out_time > self.life_time:
                message = "fade_in_time + fade_out_time must not exceed life_time "
                message += f"(got {self.fade_in_time} + {self.fade_out_time} > "
                message += f"{self.life_time})."
                raise ValueError(message)
        return self

    def _age_minutes(self, time):
        """Minutes elapsed since ``birth_time`` at ``time``."""
        elapsed = np.datetime64(time) - np.datetime64(self.birth_time)
        return elapsed / np.timedelta64(1, "m")

    def is_alive(self, time):
        """Whether the object is within its lifetime at ``time``."""
        if self.life_time is None:
            return True
        return self._age_minutes(time) < self.life_time

    def fade_scale(self, time):
        """Linear intensity multiplier in [0, 1] from fade-in/out at ``time``."""
        if self.birth_time is None:
            return 1.0
        age = self._age_minutes(time)
        scale = 1.0
        if self.fade_in_time > 0:
            scale = min(scale, age / self.fade_in_time)
        if self.life_time is not None and self.fade_out_time > 0:
            scale = min(scale, (self.life_time - age) / self.fade_out_time)
        return float(np.clip(scale, 0.0, 1.0))

    def horizontal_extent(self):
        """Maximum horizontal reach (km) of the footprint from the centre."""
        raise NotImplementedError

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
            "definitely-convective threshold of 42 dBZ."
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

    def horizontal_extent(self):
        """Footprint reach (km) along the major axis: the semi-axis times the cutoff."""
        factor = _GAUSSIAN_EXTENT if self.style == "gaussian" else 1.0
        return (self.major / 2) * factor

    def render(self, ds, grid_options):
        """Add an elliptical blob (Gaussian or flat per ``style``) to ``ds[self.field]``.

        Only the object's bounding box is computed and written: the footprint reaches
        ``horizontal_extent`` km horizontally and the matching multiple of
        ``altitude_radius`` vertically, so for a small cell on a large domain the work is
        a tiny fraction of the grid. Everything runs on raw numpy arrays.
        """
        scale = self.fade_scale(self.time)
        if scale <= 0:
            return ds  # not yet faded in (or fully faded out): nothing to render.

        # Metres-per-degree from the WGS84 ellipsoid at the centre, so the ellipse keeps
        # its true shape and size at any latitude (a degree of longitude shrinks
        # polewards).
        clon, clat = self.center_longitude, self.center_latitude
        m_per_deg_lon = geod.inv(clon, clat, clon + _SCALE_STEP, clat)[2] / _SCALE_STEP
        m_per_deg_lat = geod.inv(clon, clat, clon, clat + _SCALE_STEP)[2] / _SCALE_STEP

        # latitude/longitude are 1-D axes on a geographic grid and 2-D curvilinear fields
        # on a cartesian one; handle both by their dimensionality. The resulting arrays
        # are in (dim0, dim1) order, matching the field's two horizontal dimensions.
        lat = ds["latitude"].values
        lon = ds["longitude"].values
        if lat.ndim == 1:
            lat2, lon2 = lat[:, None], lon[None, :]
        else:
            lat2, lon2 = lat, lon

        # Local east/north distance (km), rotated into the ellipse's principal axes and
        # normalised by the semi-axes (major/minor are full axis lengths, so the 1-sigma
        # half-extent is half of them).
        east = (lon2 - clon) * m_per_deg_lon / 1e3
        north = (lat2 - clat) * m_per_deg_lat / 1e3
        cos_o, sin_o = np.cos(self.orientation), np.sin(self.orientation)
        major_coord = east * cos_o + north * sin_o
        minor_coord = -east * sin_o + north * cos_o
        horizontal_dist2 = (major_coord / (self.major / 2)) ** 2 + (
            minor_coord / (self.minor / 2)
        ) ** 2

        # Bounding box: the altitude term only adds to the distance, so any cell already
        # beyond the cutoff horizontally (or in altitude) can never be lit.
        cutoff = _GAUSSIAN_EXTENT if self.style == "gaussian" else 1.0
        horizontal_in = horizontal_dist2 <= cutoff**2
        if not horizontal_in.any():
            return ds  # object footprint lies entirely off the grid.
        i0, i1 = _bbox_index_bounds(np.any(horizontal_in, axis=1))
        j0, j1 = _bbox_index_bounds(np.any(horizontal_in, axis=0))

        alt = ds["altitude"].values
        altitude_in = (
            np.abs(alt - self.center_altitude) <= cutoff * self.altitude_radius
        )
        if not altitude_in.any():
            return ds
        k0, k1 = _bbox_index_bounds(altitude_in)

        # Full (altitude, dim0, dim1) distance, evaluated within the box only.
        hd2 = horizontal_dist2[i0:i1, j0:j1]
        vert2 = ((alt[k0:k1] - self.center_altitude) / self.altitude_radius) ** 2
        distance2 = hd2[None, :, :] + vert2[:, None, None]

        effective = self.intensity * scale  # fade-scaled peak
        if self.style == "gaussian":
            values = effective * np.exp(-distance2 / 2)
            lit = values >= 0.05 * effective
        else:  # "flat": uniform fill inside the major/minor ellipsoid (distance <= 1).
            lit = distance2 <= 1
            values = np.full(distance2.shape, effective)

        # Paint lit cells into the field's box. Field dims are (time, altitude, dim0,
        # dim1) and a synthetic grid carries a single time; later objects paint over
        # earlier ones in any overlap, matching the previous behaviour.
        box = ds[self.field].values[0, k0:k1, i0:i1, j0:j1]
        box[lit] = values[lit]
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
                # Report the fade-scaled peak, matching what was rendered at this time.
                "intensity": np.round(self.intensity * self.fade_scale(self.time), 2),
            }
        )
        return truth
