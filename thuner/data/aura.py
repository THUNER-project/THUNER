"""Process AURA data."""

import os

# Check if system is unix-like, as xESMF is not supported on Windows
if os.name == "posix":
    import xesmf as xe
else:
    message = "Warning: Windows systems cannot run xESMF for regridding."
    message += "If you need regridding, consider using a Linux or MacOS system."
    print(message)


from urllib.parse import urlparse
import xarray as xr
import numpy as np
import pandas as pd
from typing import Literal
from pydantic import Field, model_validator
from thuner.log import setup_logger
import thuner.data._utils as _utils
import thuner.grid as grid
import thuner.utils as utils

__all__ = [
    "AuraOptions",
    "CpolOptions",
    "OperationalOptions",
    "get_cpol_filepaths",
    "get_operational_filepaths",
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
        args = [time, filepath, track_options, self, grid_options]
        return convert_cpol(*args)

    @model_validator(mode="after")
    def _check_times(cls, values):
        if np.datetime64(values.start) < np.datetime64("1998-12-06T00:00:00"):
            raise ValueError("start must be 1998-12-06 or later.")
        if np.datetime64(values.end) > np.datetime64("2017-05-02T00:00:00"):
            raise ValueError("end must be 2017-05-02 or earlier.")
        return values

    @model_validator(mode="after")
    def _check_filepaths(cls, values):
        if values.filepaths is None:
            logger.info("Generating CPOL filepaths.")
            values.filepaths = get_cpol_filepaths(values)
        if values.filepaths is None:
            raise ValueError("filepaths not provided or badly formed.")
        return values


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


class OperationalOptions(AuraOptions):
    """Options for operational AURA datasets."""

    # Overwrite the default values from the base class. Note these objects are still
    # pydantic Fields. See https://github.com/pydantic/pydantic/issues/1141

    def model_post_init(self, __context):
        """
        If unset by user, change default values inherited from the base class.
        """
        super().model_post_init(__context)
        url = "https://dapds00.nci.org.au/thredds/fileServer/rq0"
        kwargs = {"name": "operational", "parent_remote": url}
        kwargs.update({"fields": ["reflectivity"], "reuse_regridder": False})
        self._change_defaults(**kwargs)

    # Define additional fields for the operational radar
    level: str = Field("1")
    data_format: str = Field("ODIM")
    radar: int = Field(63, description="Radar ID number.")
    weighting_function: str = Field(
        "Barnes2",
        description=(
            "Weighting function used by pyart to reconstruct the grid from ODIM."
        ),
    )


def get_operational_filepaths(options: OperationalOptions):
    """
    Generate operational radar URLs from input options dictionary. Note level 1 are
    zipped ODIM files, level 1b are zipped netcdf files.
    """

    start = np.datetime64(options["start"])
    end = np.datetime64(options["end"])

    urls = []
    base_url = f"{utils.get_parent(options)}"

    times = np.arange(start, end + np.timedelta64(1, "D"), np.timedelta64(1, "D"))
    times = pd.DatetimeIndex(times)

    if options["level"] == "1":
        base_url += f"/{options['radar']}"
        for time in times:
            url = f"{base_url}/{time.year:04}/vol/{options['radar']}"
            url += f"_{time.year}{time.month:02}{time.day:02}.pvol.zip"
            urls.append(url)
    elif options["level"] == "1b":
        base_url += f"/level_1b/{options['radar']}/grid"
        for time in times:
            url = f"{base_url}/{time.year:04}/{options['radar']}"
            url += f"_{time.year}{time.month:02}{time.day:02}_grid.zip"
            urls.append(url)

    return sorted(urls)


def convert_cpol(time, filepath, track_options, dataset_options, grid_options):
    """Convert CPOL data to a standard format. Retrieve the boundary data."""

    time_str = utils.format_time(time, filename_safe=False)
    logger.info(f"Updating {dataset_options.name} dataset for {time_str}.")

    cpol = xr.open_dataset(filepath)

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


def convert_operational():
    """Convert operational AURA data."""
    pass


def update_operational_input_record(
    time, input_record, track_options, dataset_options, grid_options
):
    """Update an operational AURA dataset."""
    pass
