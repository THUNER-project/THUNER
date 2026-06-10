"""
Synthetic dataset generators.

A generator owns the evolving synthetic scene: step by step it spawns, advances, fades
and culls :class:`~thuner.data.synthetic.objects.SyntheticObject`s, and renders each
time's gridded field. :class:`SyntheticGenerator` is the serialisable base (its run-time
state lives in private attributes, reset per run); :class:`FixedGenerator` replays a
fixed list of objects, and procedural subclasses can spawn objects on the fly.

Ground truth is produced by replaying the *same* stepping on a fresh copy (see
:meth:`SyntheticGenerator.ground_truth`), so the truth table is consistent with the
rendered field by construction, even once stepping becomes stateful (per-step noise,
acceleration, random spawning).
"""

import copy
from typing import Annotated, Literal
import numpy as np
import pandas as pd
import xarray as xr
from pydantic import Field, PrivateAttr, model_validator
from thuner.log import setup_logger
from thuner.utils import BaseOptions, get_mask_boundary
import thuner.grid as grid
from thuner.data.synthetic.objects import EllipsoidObject

logger = setup_logger(__name__)

# Approx km per degree of latitude (longitude scaled by cos(latitude)). Only used for the
# coarse domain-culling margin, so the spherical approximation is plenty.
_KM_PER_DEGREE = 111.32


class SyntheticGenerator(BaseOptions):
    """Base generator: spawn, advance, fade and cull objects, rendering each step.

    Subclasses supply objects via :meth:`initial_objects` (present from the start) and/or
    :meth:`spawn` (created during the run). All stepping, fading, domain-culling,
    rendering and ground-truth machinery lives here.
    """

    domain_buffer: float = Field(
        0.0,
        description=(
            "Extra margin in km added to the grid domain before an object is culled for "
            "having left it (lets objects wander out and return)."
        ),
    )
    aggregation_method: Literal["overwrite", "sum", "mean"] = Field(
        "mean",
        description=(
            "How overlapping objects combine in the rendered field: 'overwrite' (the "
            "last object wins), 'sum' (add contributions) or 'mean' (per-cell average "
            "over the objects covering each cell)."
        ),
    )

    # Transient run state, reset per pass over a grid.
    _live: list = PrivateAttr(default_factory=list)
    _pool: list = PrivateAttr(default_factory=list)
    _grid_options: object = PrivateAttr(default=None)
    _base_dataset: object = PrivateAttr(default=None)
    _next_id: int = PrivateAttr(default=0)
    _start_time: object = PrivateAttr(default=None)

    def initial_objects(self):
        """Objects present (in their birth state) at the start of a run."""
        return []

    def spawn(self, time):
        """New objects created at ``time`` (procedural subclasses override this)."""
        return []

    def reset(self, grid_options, start_time):
        """Initialise run state for a fresh pass over ``grid_options``."""
        self._grid_options = grid_options
        self._ensure_grid_coordinates()
        self._base_dataset = None
        self._next_id = 0
        self._start_time = start_time
        self._live = []
        self._pool = []
        for obj in self.initial_objects():
            self._admit(obj, start_time)

    def _admit(self, obj, start_time):
        """Assign a stable id and resolve a missing ``birth_time``, then pool the object."""
        if obj.id is None:
            obj.id = self._next_id
            self._next_id += 1
        if obj.birth_time is None:
            obj.birth_time = str(start_time)
        self._pool.append(obj)

    def _evolve(self, time):
        """Advance the scene to ``time``: spawn, birth, advance, then cull dead/off-domain."""
        for obj in self.spawn(time):
            self._admit(obj, time)
        t = np.datetime64(time)
        born = [obj for obj in self._pool if np.datetime64(obj.birth_time) <= t]
        self._pool = [obj for obj in self._pool if np.datetime64(obj.birth_time) > t]
        self._live.extend(born)
        self._live = [obj.advance(time) for obj in self._live]
        self._live = [
            obj for obj in self._live if obj.is_alive(time) and self._in_domain(obj)
        ]

    def step(self, time, grid_options):
        """Advance to ``time`` and return the rendered dataset."""
        if self._grid_options is not grid_options:
            self.reset(grid_options, time)
        self._evolve(time)
        return self._render(time)

    def ground_truth(self, times, grid_options):
        """Replay the same stepping on a fresh copy, collecting per-time object truth.

        Returns a DataFrame indexed by ``(time, id)``. Because it re-runs the identical
        step sequence, the positions/intensities match the rendered field exactly.
        """
        clone = self.model_validate(self.model_dump())
        clone.reset(grid_options, times[0])
        rows = []
        for time in times:
            clone._evolve(time)
            rows.extend(obj.ground_truth() for obj in clone._live)
        return pd.DataFrame(rows).set_index(["time", "id"]).sort_index()

    def _in_domain(self, obj):
        """Whether the object's footprint still overlaps the buffered grid domain."""
        grid_options = self._grid_options
        lats = np.asarray(grid_options.latitude)
        lons = np.asarray(grid_options.longitude)
        margin_km = self.domain_buffer + obj.horizontal_extent()
        margin_lat = margin_km / _KM_PER_DEGREE
        cos_lat = max(np.cos(np.deg2rad(obj.center_latitude)), 0.01)
        margin_lon = margin_km / (_KM_PER_DEGREE * cos_lat)
        in_lat = (
            lats.min() - margin_lat <= obj.center_latitude <= lats.max() + margin_lat
        )
        in_lon = (
            lons.min() - margin_lon <= obj.center_longitude <= lons.max() + margin_lon
        )
        return bool(in_lat and in_lon)

    def _render(self, time):
        """Render the live objects for ``time`` into a fresh copy of the base dataset.

        Overlapping objects are combined per ``aggregation_method``: ``"overwrite"`` lets
        the last object win (cheapest), while ``"sum"`` and ``"mean"`` accumulate each
        object's contribution -- the mean then dividing by the per-cell count of objects
        covering each cell. Each object only touches its own bounding box, so the cost
        stays proportional to the footprints rather than the grid.
        """
        if self._base_dataset is None:
            self._base_dataset = self._create_base_dataset(time)
        ds = copy.deepcopy(self._base_dataset)
        ds["time"] = np.array([np.datetime64(time)])

        if self.aggregation_method == "overwrite":
            for obj in self._live:
                ds = obj.render(ds, self._grid_options)
            return ds

        # "sum" / "mean": accumulate each object's contribution into the rendered field,
        # tracking how many objects covered each cell. The base field is all-NaN, so
        # cells no object touches stay NaN (empty), not 0.
        field = ds["reflectivity"].values
        total = np.zeros_like(field)
        contributing_count = np.zeros_like(field)
        for obj in self._live:
            footprint = obj._footprint(ds)
            if footprint is None:
                continue
            box_slice, footprint_mask, rendered_values = footprint
            total[box_slice][footprint_mask] += rendered_values[footprint_mask]
            contributing_count[box_slice][footprint_mask] += 1
        covered = contributing_count > 0
        field[:] = np.nan
        if self.aggregation_method == "mean":
            field[covered] = total[covered] / contributing_count[covered]
        else:
            field[covered] = total[covered]
        return ds

    def _ensure_grid_coordinates(self):
        """Fill in the grid's geographic/Cartesian coordinates if not already set."""
        grid_options = self._grid_options
        missing_geographic = (grid_options.latitude is None) or (
            grid_options.longitude is None
        )
        missing_cartesian = (grid_options.x is None) or (grid_options.y is None)
        if grid_options.name == "cartesian" and missing_geographic:
            X, Y = np.meshgrid(grid_options.x, grid_options.y)
            LON, LAT = grid.cartesian_to_geographic_lcc(grid_options, X, Y)
            grid_options.latitude = LAT
            grid_options.longitude = LON
        if grid_options.name == "geographic" and missing_cartesian:
            LON, LAT = np.meshgrid(grid_options.longitude, grid_options.latitude)
            X, Y = grid.geographic_to_cartesian_lcc(grid_options, LAT, LON)
            grid_options.x = X
            grid_options.y = Y

    def _create_base_dataset(self, time):
        """Build the empty base dataset (NaN field + coordinates) for the grid."""
        grid_options = self._grid_options
        dims = grid.get_coordinate_names(grid_options)
        if dims == ["latitude", "longitude"]:
            alternative_dims = ["y", "x"]
        elif dims == ["y", "x"]:
            alternative_dims = ["latitude", "longitude"]
        else:
            raise ValueError("Invalid grid options")

        time = np.array([np.datetime64(time)]).astype("datetime64[ns]")
        meridional_dim = np.array(getattr(grid_options, dims[0]))
        zonal_dim = np.array(getattr(grid_options, dims[1]))
        alt = np.array(grid_options.altitude)

        ds_values = np.ones((1, len(alt), len(meridional_dim), len(zonal_dim))) * np.nan
        coords = {"time": time, "altitude": alt}
        coords.update({dims[0]: meridional_dim, dims[1]: zonal_dim})
        variables_dict = {
            "reflectivity": (["time", "altitude", dims[0], dims[1]], ds_values),
            alternative_dims[0]: (
                [dims[0], dims[1]],
                getattr(grid_options, alternative_dims[0]),
            ),
            alternative_dims[1]: (
                [dims[0], dims[1]],
                getattr(grid_options, alternative_dims[1]),
            ),
        }
        ds = xr.Dataset(variables_dict, coords=coords)
        ds["reflectivity"].attrs.update({"long_name": "reflectivity", "units": "dBZ"})

        cell_areas = grid.get_cell_areas(grid_options)
        ds["gridcell_area"] = (dims, cell_areas)
        ds["gridcell_area"].attrs.update(
            {"units": "km^2", "standard_name": "area", "valid_min": 0}
        )

        # Synthetic data fills the whole grid, so the domain mask is all True and its
        # boundary traces the grid edge. This gives synthetic datasets the same
        # (domain_mask, boundary_mask, gridcell_area) fields the real converted datasets
        # carry, so they save, load and visualize through the identical path.
        domain_mask = xr.DataArray(
            np.ones((len(meridional_dim), len(zonal_dim)), dtype=bool),
            dims=dims,
            coords={dims[0]: meridional_dim, dims[1]: zonal_dim},
        )
        ds["domain_mask"] = domain_mask
        _, _, boundary_mask = get_mask_boundary(domain_mask, grid_options)
        ds["boundary_mask"] = boundary_mask
        return ds


class FixedGenerator(SyntheticGenerator):
    """Generator that replays a fixed list of objects."""

    objects: list[EllipsoidObject] = Field(
        ..., description="Synthetic objects to generate. See thuner.data.synthetic."
    )

    def initial_objects(self):
        """Seed the run with deep copies of the configured objects."""
        return copy.deepcopy(self.objects)


class RandomEllipseGenerator(SyntheticGenerator):
    """Spawn random ellipse cells over time, deterministically given ``seed``.

    ``initial_count`` cells are present at the start; further cells appear as a Poisson
    process at ``spawn_rate`` per hour. Each cell's centre, geometry, motion and lifetime
    are drawn uniformly from the configured ranges. Identical ``seed`` (with the same
    times and grid) yields an identical scene, so the rendered field and the replayed
    ground truth always agree.
    """

    seed: int = Field(0, description="Seed for the random number generator.")
    spawn_rate: float = Field(
        10.0, description="Expected number of new cells per hour (Poisson)."
    )
    initial_count: int = Field(1, description="Number of cells present at the start.")
    major_range: tuple[float, float] = Field(
        (20.0, 40.0), description="Min/max full major axis in km."
    )
    aspect_range: tuple[float, float] = Field(
        (0.4, 0.9), description="Min/max minor/major axis ratio (in (0, 1])."
    )
    speed_range: tuple[float, float] = Field(
        (0.0, 20.0), description="Min/max speed in m/s."
    )
    life_time_range: tuple[float, float] = Field(
        (30.0, 120.0), description="Min/max lifetime in minutes."
    )
    fade_fraction: float = Field(
        0.2, description="Fade-in/out duration as a fraction of lifetime (in [0, 0.5])."
    )
    intensity: float = Field(
        42 * np.sqrt(np.e), description="Peak field value for drawn cells, e.g. dBZ."
    )
    style: Literal["gaussian", "flat"] = Field(
        "gaussian", description="Intensity profile for drawn cells."
    )

    _rng: object = PrivateAttr(default=None)
    _prev_time: object = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _check_ranges(self):
        """Ranges must be ordered and the aspect/fade fractions sensibly bounded."""
        for name in ["major_range", "aspect_range", "speed_range", "life_time_range"]:
            low, high = getattr(self, name)
            if low > high:
                raise ValueError(f"{name} must have min <= max (got {low} > {high}).")
        if not (self.aspect_range[0] > 0 and self.aspect_range[1] <= 1):
            raise ValueError("aspect_range must lie within (0, 1].")
        if not 0 <= self.fade_fraction <= 0.5:
            raise ValueError("fade_fraction must be in [0, 0.5].")
        if self.spawn_rate < 0 or self.initial_count < 0:
            raise ValueError("spawn_rate and initial_count must be non-negative.")
        return self

    def reset(self, grid_options, start_time):
        """Re-seed the RNG so each pass over the same times reproduces the same scene."""
        self._rng = np.random.default_rng(self.seed)
        self._prev_time = None
        super().reset(grid_options, start_time)

    def initial_objects(self):
        """The cells present at the start of the run."""
        return [self._draw_object(self._start_time) for _ in range(self.initial_count)]

    def spawn(self, time):
        """Poisson-draw new cells for the interval since the previous step."""
        t = np.datetime64(time)
        if self._prev_time is None:
            self._prev_time = t
            return []  # first step: the start population is handled by initial_objects.
        dt_hours = (t - self._prev_time) / np.timedelta64(1, "h")
        self._prev_time = t
        count = int(self._rng.poisson(self.spawn_rate * dt_hours))
        return [self._draw_object(time) for _ in range(count)]

    def _domain_bounds(self):
        """(lat_min, lat_max, lon_min, lon_max) of the current grid."""
        lats = np.asarray(self._grid_options.latitude)
        lons = np.asarray(self._grid_options.longitude)
        return (
            float(lats.min()),
            float(lats.max()),
            float(lons.min()),
            float(lons.max()),
        )

    def _draw_object(self, time):
        """Draw one random ellipse cell, born at ``time`` somewhere in the domain."""
        rng = self._rng
        lat_min, lat_max, lon_min, lon_max = self._domain_bounds()
        major = rng.uniform(*self.major_range)
        minor = major * rng.uniform(*self.aspect_range)
        life = rng.uniform(*self.life_time_range)
        fade = self.fade_fraction * life
        return EllipsoidObject(
            time=str(time),
            center_latitude=rng.uniform(lat_min, lat_max),
            center_longitude=rng.uniform(lon_min, lon_max),
            direction=rng.uniform(0.0, 2 * np.pi),
            speed=rng.uniform(*self.speed_range),
            major=major,
            minor=minor,
            orientation=rng.uniform(0.0, np.pi),
            life_time=life,
            fade_in_time=fade,
            fade_out_time=fade,
            intensity=self.intensity,
            style=self.style,
        )


AnyGenerator = Annotated[
    FixedGenerator | RandomEllipseGenerator, Field(discriminator="type")
]
