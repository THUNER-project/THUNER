"""
Synthetic dataset generator.

Turns a set of :class:`~thuner.data.synthetic.objects.SyntheticObject` into per-time
gridded data. The generator owns the evolving synthetic state — the current objects
(advanced in time on each step) and a lazily built empty base dataset — so that no
synthetic-specific state needs to live on the tracking input records.
"""

import copy
import numpy as np
import xarray as xr
from thuner.log import setup_logger
import thuner.grid as grid

logger = setup_logger(__name__)


class SyntheticGenerator:
    """Produces per-time-step synthetic datasets from a set of objects."""

    def __init__(self, objects, grid_options):
        self.objects = copy.deepcopy(objects)
        # Assign stable ids to any objects the user left unlabelled.
        for i, obj in enumerate(self.objects):
            if obj.id is None:
                obj.id = i
        self.grid_options = grid_options
        self.base_dataset = None

    def step(self, time):
        """Advance objects to ``time`` and return the rendered dataset for that time."""
        self._ensure_grid_coordinates()
        self.objects = [obj.advance(time) for obj in self.objects]
        if self.base_dataset is None:
            self.base_dataset = self._create_base_dataset(time)
        ds = copy.deepcopy(self.base_dataset)
        ds["time"] = np.array([np.datetime64(time)])
        for obj in self.objects:
            ds = obj.render(ds, self.grid_options)
        return ds

    def _ensure_grid_coordinates(self):
        """Fill in the grid's geographic/Cartesian coordinates if not already set."""
        grid_options = self.grid_options
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
        grid_options = self.grid_options
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
