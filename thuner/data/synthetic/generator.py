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
import numpy as np
import pandas as pd
import xarray as xr
from pydantic import Field, PrivateAttr
from thuner.log import setup_logger
from thuner.utils import BaseOptions
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

    # Transient run state, reset per pass over a grid.
    _live: list = PrivateAttr(default_factory=list)
    _pool: list = PrivateAttr(default_factory=list)
    _grid_options: object = PrivateAttr(default=None)
    _base_dataset: object = PrivateAttr(default=None)
    _next_id: int = PrivateAttr(default=0)

    # --- hooks for subclasses ------------------------------------------------
    def initial_objects(self):
        """Objects present (in their birth state) at the start of a run."""
        return []

    def spawn(self, time):
        """New objects created at ``time`` (procedural subclasses override this)."""
        return []

    # --- run lifecycle -------------------------------------------------------
    def reset(self, grid_options, start_time):
        """Initialise run state for a fresh pass over ``grid_options``."""
        self._grid_options = grid_options
        self._ensure_grid_coordinates()
        self._base_dataset = None
        self._next_id = 0
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

    # --- domain culling ------------------------------------------------------
    def _in_domain(self, obj):
        """Whether the object's footprint still overlaps the buffered grid domain."""
        grid_options = self._grid_options
        lats = np.asarray(grid_options.latitude)
        lons = np.asarray(grid_options.longitude)
        margin_km = self.domain_buffer + obj.horizontal_extent()
        margin_lat = margin_km / _KM_PER_DEGREE
        cos_lat = max(np.cos(np.deg2rad(obj.center_latitude)), 0.01)
        margin_lon = margin_km / (_KM_PER_DEGREE * cos_lat)
        in_lat = lats.min() - margin_lat <= obj.center_latitude <= lats.max() + margin_lat
        in_lon = lons.min() - margin_lon <= obj.center_longitude <= lons.max() + margin_lon
        return bool(in_lat and in_lon)

    # --- rendering (grid plumbing) -------------------------------------------
    def _render(self, time):
        """Render the live objects for ``time`` into a fresh copy of the base dataset."""
        if self._base_dataset is None:
            self._base_dataset = self._create_base_dataset(time)
        ds = copy.deepcopy(self._base_dataset)
        ds["time"] = np.array([np.datetime64(time)])
        for obj in self._live:
            ds = obj.render(ds, self._grid_options)
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
        LON, LAT, ALT = xr.broadcast(ds.time, ds.longitude, ds.latitude, ds.altitude)[
            1:
        ]
        ds["LON"], ds["LAT"], ds["ALT"] = LON, LAT, ALT
        return ds


class FixedGenerator(SyntheticGenerator):
    """Generator that replays a fixed list of objects."""

    objects: list[EllipsoidObject] = Field(
        ..., description="Synthetic objects to generate. See thuner.data.synthetic."
    )

    def initial_objects(self):
        """Seed the run with deep copies of the configured objects."""
        return copy.deepcopy(self.objects)


# Becomes Annotated[FixedGenerator | RandomEllipseGenerator, Field(discriminator="type")]
# once procedural generators are added (Phase 2).
AnyGenerator = FixedGenerator
