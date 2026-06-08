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
from pathlib import Path
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

    def convert_dataset(self, time, unit, track_options, grid_options):
        """Convert CPOL dataset."""
        return convert_cpol(
            time=time,
            filepath=unit.sources[0],
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
            time_str = str(np.datetime64(time))
            unit = utils.InputUnit(
                sources=[filepath], start_time=time_str, end_time=time_str
            )
            filepaths.append(unit)
    else:
        raise NotImplementedError(
            "Only level 1b CPOL data is currently supported. Convert manually first."
        )

    return sorted(filepaths, key=lambda u: u.time_bounds()[0])


def regrid_aura(dataset, grid_options, dataset_options, weights_filepath=None):
    """Regrid an AURA dataset using xesmf for the given grid options."""
    if grid_options.name == "geographic":
        latitude, longitude = grid.infer_geographic_grid(grid_options, dataset)
        regridder = _utils.get_geographic_regridder(
            dataset,
            grid_options,
            dataset_options,
            latitude=latitude,
            longitude=longitude,
            weights_filepath=weights_filepath,
        )
        ds = regridder(dataset)
        ds = _utils.copy_attributes(ds, dataset)
    elif grid_options.name == "cartesian":
        # Implement cartesian regridding here.
        # Interpolate vertically
        ds = dataset.interp(altitude=grid_options.altitude, method="linear")

    # THUNER convention uses longitude in the range [0, 360]
    ds["longitude"] = ds["longitude"] % 360
    # Update grid_options if necessary
    utils.infer_grid_options(ds, grid_options)
    if grid_options.altitude is None:
        grid_options.altitude = ds["altitude"].values
    else:
        ds = ds.interp(altitude=grid_options.altitude, method="linear")

    # Get the domain mask and domain boundary. Note this is the region where data
    # exists, not the detected object masks from the detect module.
    domain_mask = _utils.mask_from_range(ds, dataset_options, grid_options)
    ds["domain_mask"] = domain_mask
    return ds


def get_gridcell_areas(grid_options, ds):
    """Get gridcell areas for the given grid options."""
    if grid_options.name == "cartesian":
        dims = ["y", "x"]
    else:
        dims = ["latitude", "longitude"]
    cell_areas = grid.get_cell_areas(grid_options)
    ds["gridcell_area"] = (dims, cell_areas)
    new_entries = {"units": "km^2", "standard_name": "area", "valid_min": 0}
    ds["gridcell_area"].attrs.update(new_entries)


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

    ds = regrid_aura(cpol, grid_options, dataset_options)
    get_gridcell_areas(grid_options, ds)

    all_coords = utils.get_mask_boundary(ds["domain_mask"], grid_options)
    boundary_coords, simple_boundary_coords, boundary_mask = all_coords
    ds["boundary_mask"] = boundary_mask
    ds = _utils.apply_mask(ds, grid_options)

    return ds, boundary_coords, simple_boundary_coords


class OperationalOptions(AuraOptions):
    """Options for an individual operational radar dataset."""

    def model_post_init(self, __context):
        """Use model_post_init to change default inherited values."""
        super().model_post_init(__context)
        url = "https://dapds00.nci.org.au/thredds/fileServer/rq0"
        self._change_defaults(name="operational", parent_remote=url, range=200)

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

    def convert_dataset(self, time, unit, track_options, grid_options):
        """Convert operational radar dataset."""
        return convert_operational(
            time=time,
            filepath=unit.sources[0],
            track_options=track_options,
            dataset_options=self,
            grid_options=grid_options,
        )

    @model_validator(mode="after")
    def _check_filepaths(self):
        if self.filepaths is None:
            logger.info("Generating operational filepaths.")
            self.filepaths = get_operational_filepaths(self)
        if self.filepaths is None:
            raise ValueError("filepaths not provided or badly formed.")
        return self


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
    base_filepath += f"/rq0"
    for time in times:
        date_str = f"{time.year:04}{time.month:02}{time.day:02}"
        zip_filepath = f"{base_filepath}/{radar}/{time.year:04}/vol/"
        zip_filepath += f"{radar}_{date_str}.pvol.zip"
        time_str = f"{time.hour:02}{time.minute:02}00"
        h5_filename = f"{radar}_{date_str}_{time_str}.pvol.h5"
        nominal_time = str(np.datetime64(time))
        unit = utils.InputUnit(
            sources=[[zip_filepath, h5_filename]],
            start_time=nominal_time,
            end_time=nominal_time,
        )
        filepaths.append(unit)

    return filepaths


OPERATIONAL_NAMES = {
    "reflectivity_horizontal": "reflectivity",
    "lat": "latitude",
    "lon": "longitude",
    "z": "altitude",
}

REFLECTIVITY_ATTRS = {
    "units": "dBZ",
    "standard_name": "equivalent_reflectivity_factor",
    "long_name": "reflectivity",
}


def regrid_operational(filepath, dataset_options, grid_options, weights_filepath=None):
    """Regrid operational radar data using xesmf for the given grid options."""
    if dataset_options.level != "1":
        raise NotImplementedError("Only level 1 data currently implemented.")

    if not Path(filepath[0]).exists():
        logger.warning(f"Filepath {filepath[0]} does not exist. Skipping.")
        return None

    with zipfile.ZipFile(filepath[0]) as zip:
        try:
            if not filepath[1] in zip.namelist():
                logger.warning(f"{filepath[1]} not found in {filepath[0]}. Skipping.")
                return None
            operational = _utils.read_odim(
                io.BytesIO(zip.read(filepath[1])),
                weighting_function=dataset_options.weighting_function,
            )
            step = dataset_options.timestep
            operational = operational.assign_coords(
                time=operational["time"].dt.round(f"{step}min")
            )
            # pyart decodes times to cftime (object dtype); THUNER works in datetime64.
            time_index = operational.indexes["time"]
            if isinstance(time_index, xr.CFTimeIndex):
                operational = operational.assign_coords(
                    time=time_index.to_datetimeindex()
                )
        except (ValueError, OSError):
            logger.warning(
                f"Failed to read {filepath[1]} from {filepath[0]}. Skipping."
            )
            # build empty dataset
            return None

    operational = operational.rename(OPERATIONAL_NAMES)
    # Move latitude and longitude from coordinates to data variables
    dims = ["latitude", "longitude"]
    operational = operational.reset_coords(dims)
    origin_names = {"origin_longitude", "origin_latitude", "origin_altitude"}
    # present = [n for n in origin_names if n in operational.variables]
    operational.attrs.update({n: operational[n].item() for n in origin_names})
    kept_coords = {"time", "altitude", "y", "x"}
    dropped_coords = set(operational.coords) - kept_coords
    operational = operational.drop_vars(dropped_coords)
    operational = operational[dataset_options.fields + dims]
    ds = regrid_aura(
        operational, grid_options, dataset_options, weights_filepath=weights_filepath
    )
    if "reflectivity" in ds.data_vars:
        ds["reflectivity"] = ds["reflectivity"].where(ds["reflectivity"] >= -10)
        ds["reflectivity"].attrs.update(REFLECTIVITY_ATTRS)
    return ds


def convert_operational(
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
    ds = regrid_operational(filepath, dataset_options, grid_options)
    if ds is None:
        return _utils.empty_dataset_and_coordinates(grid_options)

    get_gridcell_areas(grid_options, ds)
    all_coords = utils.get_mask_boundary(ds["domain_mask"], grid_options)
    boundary_coords, simple_boundary_coords, boundary_mask = all_coords
    ds["boundary_mask"] = boundary_mask
    ds = _utils.apply_mask(ds, grid_options)

    return ds, boundary_coords, simple_boundary_coords


class OperationalEnsembleOptions(AuraOptions):
    """
    Options for an operational radar ensemble dataset. Currently this is just a quick
    and dirty ensemble using max reflectivities.
    """

    def model_post_init(self, __context):
        """Use model_post_init to change default inherited values."""
        super().model_post_init(__context)
        url = "https://dapds00.nci.org.au/thredds/fileServer/rq0"
        self._change_defaults(name="operational", parent_remote=url, range=200)

    # Define additional fields for the operational radar
    level: Literal["1", "1b", "2"] = Field(
        "1", description="Radar data processing level."
    )
    radars: list[int] = Field([3, 4, 40], description="Radar ID numbers.")
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
        return get_operational_ensemble_filepaths(self)

    def convert_dataset(self, time, unit, track_options, grid_options):
        """Convert operational radar dataset."""
        return convert_operational_ensemble(
            time=time,
            filepath=unit.sources,
            track_options=track_options,
            dataset_options=self,
            grid_options=grid_options,
        )

    def converted_filepath(self, unit):
        """Converted path named by the ensemble's radars, e.g. ``3_4_40_<date>.nc``."""
        if unit.converted_filepath is not None:
            return unit.converted_filepath
        parent_local = str(self.parent_local)
        parent_converted = self.converted_options.parent_converted
        if parent_converted is None:
            raise ValueError("parent_converted must be set to derive converted paths.")
        parent_converted = str(parent_converted)
        archive, member = unit.sources[0][0], unit.sources[0][1]
        radar_str = "_".join(str(radar) for radar in self.radars)
        # Mirror the raw archive directory under parent_converted, but rename the
        # single-radar directory (e.g. '3') to the ensemble's radar string so the
        # directory tree matches the converted filenames, e.g. '3_4_40_...'.
        # Raw structure: rq0/<radar>/<year>/vol, so the radar directory sits two
        # levels above the 'vol' directory that holds the archive.
        out_dir = Path(
            str(Path(archive).parent).replace(parent_local, parent_converted)
        )
        out_parts = list(out_dir.parts)
        out_parts[-3] = radar_str
        out_dir = Path(*out_parts)
        # member like '3_20211014_160000.pvol.h5' -> date and time fields.
        parts = Path(member).name.split("_")
        date_str, time_str = parts[1], parts[2].split(".")[0]
        converted = str(out_dir / f"{radar_str}_{date_str}_{time_str}.nc")
        unit.converted_filepath = converted
        return converted

    @model_validator(mode="after")
    def _check_filepaths(self):
        if self.filepaths is None:
            logger.info("Generating operational ensemble filepaths.")
            self.filepaths = get_operational_ensemble_filepaths(self)
        if self.filepaths is None:
            raise ValueError("filepaths not provided or badly formed.")
        return self


def get_operational_ensemble_filepaths(options: OperationalEnsembleOptions):
    """
    Get filepaths for an operational radar ensemble. This is just a list of filepaths
    for each radar in the ensemble, which can be converted separately and then combined.
    """
    per_radar = []
    for radar in options.radars:
        options_dict = options.model_dump()
        options_dict["radar"] = radar
        options_dict.pop("radars")
        options_dict.pop("type")
        options_dict.pop("filepaths", None)
        per_radar.append(get_operational_filepaths(OperationalOptions(**options_dict)))
    # Combine the per-radar units for each time into a single multi-source ensemble
    # unit, whose sources are the [archive, member] reference for each radar.
    units = []
    for radar_units in zip(*per_radar):
        sources = [radar_unit.sources[0] for radar_unit in radar_units]
        units.append(
            utils.InputUnit(
                sources=sources,
                start_time=radar_units[0].start_time,
                end_time=radar_units[0].end_time,
            )
        )
    return units


def convert_operational_ensemble(
    time,
    filepath,
    track_options,
    dataset_options: OperationalEnsembleOptions,
    grid_options,
):
    """
    Convert an operational radar ensemble by converting each member separately and then
    taking the max reflectivity across the ensemble. Note this is a quick and dirty
    ensemble, and does not account for differences in radar coverage or other factors.
    """
    datasets = []
    for ds_filepath, radar in zip(filepath, dataset_options.radars):
        # Reset latitude and longitude so they are re-inferred for each grid. Then we
        # will patch the grids together. This economizes the ODIM regridding step.
        grid_options.latitude = None
        grid_options.longitude = None
        weights_filepath = dataset_options.weights_filepath
        radar_weights_filepath = Path(weights_filepath).parent
        radar_weights_filepath /= f"{Path(weights_filepath).stem}_{radar}.nc"
        if not Path(ds_filepath[0]).exists():
            logger.warning(
                f"Filepath {ds_filepath[0]} does not exist. Skipping radar {radar}."
            )
            continue
        ds = regrid_operational(
            ds_filepath,
            dataset_options,
            grid_options,
            weights_filepath=radar_weights_filepath,
        )
        if ds is None:
            logger.warning(f"Failed to regrid radar {radar}. Skipping.")
            continue
        datasets.append(ds)

    if len(datasets) == 0:
        logger.warning("No operational radar datasets found. Returning empty dataset.")
        return _utils.empty_dataset_and_coordinates(grid_options)

    stacked = xr.concat(datasets, dim="radar", join="outer")
    # keep_attrs so the per-radar variable metadata (e.g. reflectivity units) survives
    # the reduction; xarray drops attrs on reductions by default.
    dataset = stacked.max(dim="radar", skipna=True, keep_attrs=True)

    dataset["domain_mask"] = dataset["domain_mask"].fillna(0)
    utils.infer_grid_options(dataset, grid_options)
    get_gridcell_areas(grid_options, dataset)
    all_coords = utils.get_mask_boundary(dataset["domain_mask"], grid_options)
    boundary_coords, simple_boundary_coords, boundary_mask = all_coords
    dataset["boundary_mask"] = boundary_mask
    dataset = _utils.apply_mask(dataset, grid_options)

    return dataset, boundary_coords, simple_boundary_coords
