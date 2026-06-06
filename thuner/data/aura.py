"""Process AURA data."""

import os

# Check if system is unix-like, as xESMF is not supported on Windows
if os.name == "posix":
    import xesmf as xe
else:
    message = "Warning: Windows systems cannot run xESMF for regridding."
    message += "If you need regridding, consider using a Linux or MacOS system."
    print(message)

import zipfile
import io
import os
from typing import Literal
from contextlib import ExitStack
import fnmatch

import xarray as xr
import numpy as np
import pandas as pd
from pydantic import Field, model_validator
from thuner.log import setup_logger
import thuner.data._utils as _utils
import thuner.grid as grid
import thuner.utils as utils

__all__ = [
    "AuraOptions",
    "CpolOptions",
    "get_cpol_filepaths",
]

logger = setup_logger(__name__)


class AuraOptions(utils.BaseDatasetOptions):
    """Base options class for AURA datasets."""

    def model_post_init(self, __context):
        """
        If unset by user, change default values inherited from the base class.
        """
        super().model_post_init(__context)
        self._change_defaults(fields=["reflectivity"], reuse_regridder=True)

    # Define additional fields for AURA
    range: float = Field(142.5, description="Range of the radar in km.")
    range_units: str = Field("km", description="Units of the range.")


class CpolOptions(AuraOptions):
    """Options for CPOL datasets."""

    def model_post_init(self, __context):
        """Use model_post_init to change default inherited values."""
        super().model_post_init(__context)
        url = "https://dapds00.nci.org.au/thredds/fileServer/hj10"
        self._change_defaults(name="cpol", parent_remote=url)

    # Define additional fields for CPOL
    level: Literal["1", "1b", "2"] = Field("1b", description="Processing level.")
    _FormatChoices = Literal["grid_150km_2500m", "grid_70km_1000m", "ppi"]
    data_format: _FormatChoices = Field("grid_150km_2500m", description="Data format.")
    version: str = Field("v2020", description="Data version.")

    # Override get_filepaths and grid_from_dataset with CPOL specific versions.
    def get_filepaths(self):
        """Get CPOL fielpaths."""
        return get_cpol_filepaths(self)

    def convert_dataset(self, time, filepath, track_options, grid_options):
        """Convert CPOL dataset."""
        return convert_cpol(
            time=time,
            filepath=filepath,
            track_options=track_options,
            dataset_options=self,
            grid_options=grid_options,
        )

    @model_validator(mode="after")
    def _check_times(self):
        if np.datetime64(self.start) < np.datetime64("1998-12-06T00:00:00"):
            raise ValueError("start must be 1998-12-06 or later.")
        if np.datetime64(self.end) > np.datetime64("2017-05-02T00:00:00"):
            raise ValueError("end must be 2017-05-02 or earlier.")
        return self

    @model_validator(mode="after")
    def _check_filepaths(self):
        if self.filepaths is None:
            logger.info("Generating CPOL filepaths.")
            self.filepaths = get_cpol_filepaths(self)
        if self.filepaths is None:
            raise ValueError("filepaths not provided or badly formed.")
        return self


def get_cpol_filepaths(options: CpolOptions):
    """
    Get CPOL filepaths assuming same filenames and directory structure as remote location.
    """

    start = np.datetime64(options.start).astype("datetime64[m]")
    end = np.datetime64(options.end).astype("datetime64[m]")

    filepaths = []

    base_url = utils.get_parent(options)
    base_url += "/cpol"

    if options.level == "1b":

        times = np.arange(start, end + np.timedelta64(10, "m"), np.timedelta64(10, "m"))
        times = pd.DatetimeIndex(times)

        base_url += f"/cpol_level_1b/{options.version}/"
        if "grid" in options.data_format:
            base_url += f"gridded/{options.data_format}/"
            if "150" in options.data_format:
                data_format_string = "grid150"
            else:
                data_format_string = "grid75"
        elif options.data_format == "ppi":
            base_url += "ppi/"
        for time in times:
            filepath = (
                f"{base_url}{time.year}/{time.year}{time.month:02}{time.day:02}/"
                f"twp10cpol{data_format_string}.b2."
                f"{time.year}{time.month:02}{time.day:02}."
                f"{time.hour:02}{time.minute:02}{time.second:02}.nc"
            )
            filepaths.append(filepath)
    else:
        raise NotImplementedError(
            "Only level 1b CPOL data is currently supported. Convert manually first."
        )

    return sorted(filepaths)


def convert_cpol(time, filepath, track_options, dataset_options, grid_options):
    """Convert CPOL data to a standard format. Retrieve the boundary data."""

    time_str = utils.format_time(time, filename_safe=False)
    logger.info(f"Updating {dataset_options.name} dataset for {time_str}.")

    cpol = xr.open_dataset(filepath, decode_timedelta=True)

    if time not in cpol.time.values:
        raise ValueError(f"{time} not in {filepath}")

    point_coords = ["point_latitude", "point_longitude", "point_altitude"]
    cpol = cpol[dataset_options.fields + point_coords]
    new_names = {"point_latitude": "latitude", "point_longitude": "longitude"}
    new_names.update({"point_altitude": "altitude"})
    cpol = cpol.rename(new_names)
    cpol["altitude"] = cpol["altitude"].isel(x=0, y=0)
    cpol = cpol.swap_dims({"z": "altitude"})
    cpol = cpol.drop_vars("z")

    for var in ["latitude", "longitude"]:
        cpol[var] = cpol[var].isel(altitude=0)

    if grid_options.name == "geographic":
        dims = ["latitude", "longitude"]
        latitude, longitude = grid.infer_geographic_grid(grid_options, cpol)
        regridder = _utils.get_geographic_regridder(
            cpol, grid_options, dataset_options, latitude=latitude, longitude=longitude
        )
        ds = regridder(cpol)
        ds = _utils.copy_attributes(ds, cpol)
    elif grid_options.name == "cartesian":
        dims = ["y", "x"]
        # Implement cartesian regridding here.
        # Interpolate vertically
        ds = cpol.interp(altitude=grid_options.altitude, method="linear")

    # THUNER convention uses longitude in the range [0, 360]
    ds["longitude"] = ds["longitude"] % 360
    # Update grid_options if necessary
    utils.infer_grid_options(ds, grid_options)
    cell_areas = grid.get_cell_areas(grid_options)
    ds["gridcell_area"] = (dims, cell_areas)
    new_entries = {"units": "km^2", "standard_name": "area", "valid_min": 0}
    ds["gridcell_area"].attrs.update(new_entries)
    if grid_options.altitude is None:
        grid_options.altitude = ds["altitude"].values
    else:
        ds = ds.interp(altitude=grid_options.altitude, method="linear")

    # Get the domain mask and domain boundary. Note this is the region where data
    # exists, not the detected object masks from the detect module.
    domain_mask = _utils.mask_from_range(ds, dataset_options, grid_options)
    all_coords = utils.get_mask_boundary(domain_mask, grid_options)
    boundary_coords, simple_boundary_coords, boundary_mask = all_coords
    ds["domain_mask"] = domain_mask
    ds["boundary_mask"] = boundary_mask

    ds = _utils.apply_mask(ds, grid_options)

    return ds, boundary_coords, simple_boundary_coords


class OperationalOptions(AuraOptions):
    """Options for an individual operational radar dataset."""

    def model_post_init(self, __context):
        """Use model_post_init to change default inherited values."""
        super().model_post_init(__context)
        url = "https://dapds00.nci.org.au/thredds/fileServer/rq0"
        self._change_defaults(name="operational", parent_remote=url)

    # Define additional fields for the operational radar
    level: Literal["1", "1b", "2"] = Field(
        "1", description="Radar data processing level."
    )
    radar: int = Field(3, description="Radar ID number.")
    weighting_function: Literal["Barnes2", "Barnes", "Cressman", "Nearest"] = Field(
        "Barnes2",
        description=(
            "Weighting function used by pyart to reconstruct the grid from ODIM."
        ),
    )
    timestep: Literal[5, 10] = Field(
        10, description="Timestep in minutes for operational radar data."
    )

    def get_filepaths(self):
        """Get operational radar filepaths."""
        return get_operational_filepaths(self)


def get_operational_filepaths(options: OperationalOptions):
    """
    Generate operational radar URLs from input options dictionary. Note level 1 are
    zipped ODIM files, level 1b are zipped netcdf files.
    """

    start = np.datetime64(options.start)
    end = np.datetime64(options.end)
    step = np.timedelta64(options.timestep, "m")
    base_filepath = utils.get_parent(options)
    times = np.arange(start, end + step, step)
    times = pd.DatetimeIndex(times)
    radar = options.radar
    filepaths = []

    if not options.level == "1":
        raise NotImplementedError("Only level 1 data currently implemented.")
    base_filepath += f"/rq0/"
    for time in times:
        date_str = f"{time.year:04}{time.month:02}{time.day:02}"
        zip_filepath = f"{base_filepath}/{radar}/{time.year:04}/vol/"
        zip_filepath += f"{radar}_{date_str}.pvol.zip"
        time_str = f"{time.hour:02}{time.minute:02}00"
        h5_filename = f"{radar}_{date_str}_{time_str}.pvol.h5"
        filepaths.append((zip_filepath, h5_filename))

    return filepaths


OPERATIONAL_NAMES = {
    "reflectivity_horizontal": "reflectivity",
    "lat": "latitude",
    "lon": "longitude",
    "z": "altitude",
}


def convert_operational_level_1(
    time,
    filepath,
    track_options,
    dataset_options: OperationalOptions,
    grid_options,
):
    """
    Convert level 1 operational radar data for a given date. Here, filepath is a tuple
    where the first element is the zip path, and the second is the filename inside.
    """
    with zipfile.ZipFile(filepath[0]) as zip:
        try:
            operational = _utils.read_odim(
                io.BytesIO(zip.read(filepath[1])),
                weighting_function=dataset_options.weighting_function,
            )
        except (ValueError, OSError):
            logger.warning(f"Failed to read {filepath[1]} from {filepath[0]}.")
            # build empty dataset
            operational = None

    operational = operational.rename(OPERATIONAL_NAMES)
    # Move latitude and longitude from coordinates to data variables
    dims = ["latitude", "longitude"]
    operational = operational.reset_coords(dims)
    kept_coords = {
        "time",
        "altitude",
        "y",
        "x",
        "origin_longitude",
        "origin_latitude",
    }
    dropped_coords = set(operational.coords) - kept_coords
    operational = operational.drop_vars(dropped_coords)
    operational = operational[dataset_options.fields + dims]

    if grid_options.name == "geographic":
        dims = ["latitude", "longitude"]
        latitude, longitude = grid.infer_geographic_grid(grid_options, operational)
        regridder = _utils.get_geographic_regridder(
            operational,
            grid_options,
            dataset_options,
            latitude=latitude,
            longitude=longitude,
        )
        ds = regridder(operational)
        ds = _utils.copy_attributes(ds, operational)
    elif grid_options.name == "cartesian":
        dims = ["y", "x"]
        ds = operational
    ds = ds.interp(altitude=grid_options.altitude, method="linear")

    # THUNER convention uses longitude in the range [0, 360]
    ds["longitude"] = ds["longitude"] % 360
    # Update grid_options if necessary
    utils.infer_grid_options(ds, grid_options)
    cell_areas = grid.get_cell_areas(grid_options)
    ds["gridcell_area"] = (dims, cell_areas)
    new_entries = {"units": "km^2", "standard_name": "area", "valid_min": 0}
    ds["gridcell_area"].attrs.update(new_entries)
    if grid_options.altitude is None:
        grid_options.altitude = ds["altitude"].values
    else:
        ds = ds.interp(altitude=grid_options.altitude, method="linear")

    # Get the domain mask and domain boundary. Note this is the region where data
    # exists, not the detected object masks from the detect module.
    domain_mask = _utils.mask_from_range(ds, dataset_options, grid_options)
    all_coords = utils.get_mask_boundary(domain_mask, grid_options)
    boundary_coords, simple_boundary_coords, boundary_mask = all_coords
    ds["domain_mask"] = domain_mask
    ds["boundary_mask"] = boundary_mask

    ds = _utils.apply_mask(ds, grid_options)

    return ds, boundary_coords, simple_boundary_coords
