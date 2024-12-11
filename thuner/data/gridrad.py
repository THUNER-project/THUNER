import copy
from typing import Union
from pathlib import Path
from functools import reduce
import numpy as np
import pandas as pd
import xarray as xr
from skimage.morphology import remove_small_objects
import thuner.data.utils as utils
from thuner.log import setup_logger
import thuner.grid as grid


logger = setup_logger(__name__)

gridrad_variables = [
    "Reflectivity",
    "SpectrumWidth",
    "AzShear",
    "Divergence",
    "DifferentialReflectivity",
    "DifferentialPhase",
    "CorrelationCoefficient",
]


gridrad_names_dict = {
    "reflectivity": "Reflectivity",
    "spectrum_width": "SpectrumWidth",
    "azimuthal_shear": "AzShear",
    "divergence": "Divergence",
    "differential_reflectivity": "DifferentialReflectivity",
    "differential_phase": "DifferentialPhase",
    "correlation_coefficient": "CorrelationCoefficient",
}


def get_event_directories(year, base_local=None):
    if base_local is None:
        base_local = Path("/scratch/w40/esh563/THUNER_output")
    year_directory = base_local / f"input_data/raw/d841006/volumes/{year}"
    event_directories = sorted([p for p in year_directory.iterdir() if p.is_dir()])
    return event_directories


def get_event_times(event_directory: Union[str, Path]):
    """
    Get start and end times, and event start date from a GridRad severe event directory.
    """
    if isinstance(event_directory, str):
        event_directory = Path(event_directory)
    event_start = f"{event_directory.name[:4]}-{event_directory.name[4:6]}"
    event_start += f"-{event_directory.name[6:]}"
    files = sorted(event_directory.iterdir())
    times = []
    for file in files:
        # Ignore files
        if "500Z.nc" in str(file):
            continue
        time = str(file.name).split("_")[-1].split(".")[0]
        formatted_time = f"{time[:4]}-{time[4:6]}-{time[6:8]}"
        formatted_time += f"T{time[9:11]}:{time[11:13]}:{time[13:15]}"
        times.append(np.datetime64(formatted_time))
    # Suprisingly it appears we need to sort again to ensure the times are in order
    times = sorted(times)
    start = str(times[0])
    end = str(times[-1])
    return start, end, event_start


def get_gridrad_filepaths(options):
    """
    Get the start and end dates for the cases in the GridRad-Severe dataset
    (doi.org/10.5065/2B46-1A97).
    """

    start = np.datetime64(options.start).astype("datetime64[s]")
    end = np.datetime64(options.end).astype("datetime64[s]")

    filepaths = []

    base_url = utils.get_parent(options)
    base_url += f"/{dataset_id_converter[options.dataset_id]}/volumes"

    times = np.arange(start, end + np.timedelta64(10, "m"), np.timedelta64(10, "m"))
    times = pd.DatetimeIndex(times)
    start, end = pd.Timestamp(start), pd.Timestamp(end)

    # Note gridrad severe directories are organized by the day the event "started"
    if options.dataset_id == "ds841.6":
        event_start = pd.Timestamp(options.event_start)
        base_filepath = f"{base_url}/{event_start.year}/"
        base_filepath += f"{event_start.year}{event_start.month:02}{event_start.day:02}"
        for time in times:
            filepath = (
                f"{base_filepath}/nexrad_3d_{options.version}_"
                f"{time.year}{time.month:02}{time.day:02}T"
                f"{time.hour:02}{time.minute:02}00Z.nc"
            )
            filepaths.append(filepath)
    return sorted(filepaths)


def open_gridrad(path, dataset_options):
    """
    Open a GridRad netcdf file, converting variables with an "Index" dimension back to 3D
    """

    kept_variables = [gridrad_names_dict[f] for f in dataset_options.fields]
    kept_variables += ["Nradobs", "Nradecho", "wReflectivity", "CorrelationCoefficient"]
    ds = xr.open_dataset(path)
    kept_variables = [v for v in kept_variables if v in ds.data_vars]
    dropped_variables = [v for v in ds.data_vars if v not in kept_variables]
    for var in kept_variables:
        if var != "index" and "Index" in ds[var].dims:
            ds = reshape_variable(ds, var)
    ds = ds.drop_vars(dropped_variables + ["index"])
    return ds


def reshape_variable(ds, variable):
    """
    Reshape a variable in a GridRad dataset to a 3D grid. Adapted from code provided by
    Stacey Hitchcock.
    """

    values = ds[variable].values
    attrs = ds[variable].attrs
    alt, lat, lon = ds["Altitude"], ds["Latitude"], ds["Longitude"]
    new_values = np.zeros(len(alt) * len(lat) * len(lon))
    new_values[ds.index.values] = values
    new_values = new_values.astype(ds[variable].dtype)
    new_shape = (len(alt), len(lat), len(lon))
    new_dims = ["Altitude", "Latitude", "Longitude"]
    new_coords = {"Altitude": alt, "Latitude": lat, "Longitude": lon}
    ds[variable] = xr.DataArray(
        new_values.reshape(new_shape), dims=new_dims, coords=new_coords
    )
    ds[variable].attrs = attrs
    return ds


def filter(
    ds,
    weight_thresh=1.2,
    echo_frac_thresh=0.3,
    refl_thresh=0,
    obs_thresh=2,
    variables=None,
):
    """
    Filter a GridRad dataset. Based on code from the GridRad website
    https://gridrad.org/software.html and edits by Stacey Hitchcock.

    Parameters
    ----------
    ds : xarray.Dataset
        The GridRad dataset.
    weight_thresh : float, optional
        The bin weight threshold. Default is 1.5.
    echo_frac_thresh : float, optional
        The echo fraction threshold. Default is 0.6.
    refl_thresh : float, optional
        The reflectivity threshold. Default is 0.
    obs_thresh : int, optional
        The number of observations. Default is 3.

    Returns
    -------
    ds : xarray.Dataset
        The filtered GridRad dataset
    """

    logger.debug("Filtering GridRad data")

    if variables is None:
        variables = [v for v in gridrad_variables if v in ds.variables]

    # Calcualate echo fraction efficiently using where
    args = [ds["Nradobs"] > 0, ds["Nradecho"] / ds["Nradobs"], 0.0]
    kwargs = {"keep_attrs": True}
    echo_fraction = xr.where(*args, **kwargs)
    echo_fraction = echo_fraction.astype(np.float32)

    # Get indices to filter
    weight_cond = xr.where(ds["wReflectivity"] < weight_thresh, True, False, **kwargs)
    refl_cond = xr.where(ds["Reflectivity"] <= refl_thresh, True, False, **kwargs)
    frac_cond = xr.where(echo_fraction < echo_frac_thresh, True, False, **kwargs)
    obs_cond = xr.where(ds["Nradobs"] <= obs_thresh, True, False, **kwargs)
    # Filter cells below weight and reflectivity thresholds
    cond_refl = xr.where(weight_cond & refl_cond, True, False, **kwargs)
    # Filter cells containing at < obs_thresh observations. If at least obs_thresh
    # observations, filter cells with echoes in less than echo_fraction_thresh of the
    # total observations
    cond_frac = xr.where(obs_cond | frac_cond, True, False, **kwargs)
    # Retain values not filtered
    preserved = xr.where(~cond_refl & ~cond_frac, True, False, **kwargs)
    for var in variables:
        ds[var] = ds[var].where(preserved)
    return ds


def remove_speckles(ds, window_size=5, coverage_thresh=0.32, variables=None):
    """
    Remove speckles in GridRad data. Based on code from the GridRad website
    https://gridrad.org/software.html and edits by Stacey Hitchcock. Modified from the
    original to use xr.rolling instead of np.roll to correctly handle edges and corners.
    """

    logger.debug("Removing speckles from the GridRad data")

    if variables is None:
        variables = [v for v in gridrad_variables if v in ds.variables]

    # refl_exists = np.isfinite(ds["Reflectivity"]).astype(float)
    refl_exists = xr.where(~np.isnan(ds["Reflectivity"]), True, False)
    min_size = window_size**3 * coverage_thresh
    speckle_mask = remove_small_objects(refl_exists.values > 0, min_size=min_size)
    for var in variables:
        ds[var] = ds[var].where(speckle_mask)
    return ds


def remove_low_level_clutter(ds, variables=None):
    """
    Remove low level clutter from GridRad data. Based on code from the GridRad website
    https://gridrad.org/software.html and edits by Stacey Hitchcock.
    """

    logger.debug("Removing low level clutter from the GridRad data")

    # Determine max heights of non-nan reflectivity values. If entire column is nan,
    # set max altitude to zero.
    refl_max = ds.Reflectivity.max(dim="Altitude", skipna=True)
    refl_0_alts = ds.Altitude.where(ds.Reflectivity > 0.0, 0.0)
    refl_0_max_alt = refl_0_alts.max(dim="Altitude")
    refl_0_min_alt = refl_0_alts.min(dim="Altitude")
    refl_5_max_alt = ds.Altitude.where(ds.Reflectivity > 5.0, 0.0).max(dim="Altitude")
    refl_15_max_alt = ds.Altitude.where(ds.Reflectivity > 15.0, 0.0).max(dim="Altitude")

    # Check for very weak echos below 4 km
    cond_1 = (refl_max < 20.0) & (refl_0_max_alt <= 4.0) & (refl_0_min_alt <= 3.0)
    # Check for very weak echos below 5 km
    cond_2 = (refl_max < 10.0) & (refl_0_max_alt <= 5.0) & (refl_0_min_alt <= 3.0)
    # Check for weak echos below 5 km. Note the > 0.0 ensures values actually exist
    cond_3 = (refl_5_max_alt <= 5.0) & (refl_5_max_alt > 0.0) & (refl_15_max_alt <= 3.0)
    # Check for weak echos below 2 km
    cond_4 = (refl_15_max_alt < 2.0) & (refl_15_max_alt > 0.0)
    cond = np.logical_not(cond_1 | cond_2 | cond_3 | cond_4)
    for var in variables:
        ds[var] = ds[var].where(cond)
    return ds


def remove_clutter_below_anvils(ds, variables=None):
    """
    Remove clutter below anvils in GridRad data. Based on code from the GridRad website
    https://gridrad.org/software.html and edits by Stacey Hitchcock.
    """

    logger.debug("Removing clutter below anvils from the GridRad data")

    # Check if reflectivity exists at, above and below 4 km
    exists = np.isfinite(ds.Reflectivity)
    exists_above_4 = exists.where(ds.Altitude >= 4.0, drop=True)
    exists_4 = exists_above_4.isel(Altitude=0)
    exists_above_4 = exists_above_4.sum(dim="Altitude") > 0
    exists_below_4 = exists.where(ds.Altitude < 4.0, drop=True).sum(dim="Altitude") > 0

    cond = exists_4 | ~exists_above_4 | ~exists_below_4
    for var in variables:
        ds[var] = ds[var].where(cond)
    return ds


def remove_clutter(ds, variables=None, low_level=True, below_anvil=False):
    """
    Remove clutter from GridRad data. Based on code from the GridRad website
    https://gridrad.org/software.html and edits by Stacey Hitchcock.

    Parameters
    ----------
    ds : xarray.Dataset
        The GridRad dataset.
    variables : list, optional
        The variables to remove clutter from. Default is ["Reflectivity"].

    Returns
    -------
    ds : xarray.Dataset
        The GridRad dataset with clutter removed.
    """

    logger.debug("Removing clutter from the GridRad data")

    if variables is None:
        variables = [v for v in gridrad_variables if v in ds.variables]

    # Remove low reflectivity low level clutter
    cond = (ds.Reflectivity >= 10.0) | (ds.Altitude > 4.0)
    for var in variables:
        ds[var] = ds[var].where(cond)

    # Attempt correlation based clutter removal if relevant variables exist
    correlation_var_list = ["DifferentialReflectivity", "CorrelationCoefficient"]
    if all(corr_var in ds.variables for corr_var in correlation_var_list):

        # Require either high correlation or reflectivity
        cond1 = ds["Reflectivity"] >= 40.0 | ds["r_HV"] >= 0.9
        # Require moderate reflectivity or high correlation or low altitude
        cond2 = ds["Reflectivity"] >= 25.0 | ds["CorrelationCoefficient"] >= 0.95
        cond2 = cond2 | ds["Altitude"] < 10.0
        # Require both conditions above be met
        for var in variables:
            ds[var] = ds[var].where(cond1 & cond2)

    # First pass at speckle removal
    ds = remove_speckles(ds, variables=variables)
    if low_level:
        # Remove low level clutter. Note this can remove some low level cloud/drizzle
        ds = remove_low_level_clutter(ds, variables=variables)
    if below_anvil:
        # Remove clutter below anvils
        ds = remove_clutter_below_anvils(ds, variables=variables)
    # Second pass at speckle removal
    ds = remove_speckles(ds, variables=variables)

    return ds


def get_gridrad(time, input_record, track_options, dataset_options, grid_options):
    filepath = dataset_options.filepaths[input_record["current_file_index"]]
    utils.log_convert(logger, dataset_options.name, filepath)
    args = [time, filepath, track_options, dataset_options, grid_options]
    ds, boundary_coords = convert_gridrad(*args)[:2]
    update_boundary_data(ds, boundary_coords, input_record)
    return ds


def convert_gridrad(time, filepath, track_options, dataset_options, grid_options):
    """Convert gridrad data to the standard format."""

    logger.debug(f"Converting GridRad dataset at time {time}.")

    # Open the dataset and perform preliminary filtering and decluttering
    ds = open_gridrad(filepath, dataset_options)
    ds = filter(ds, obs_thresh=dataset_options.obs_thresh)
    ds = remove_clutter(ds)

    # Ensure the intended time is in the dataset
    if time not in ds.time.values:
        raise ValueError(f"{time} not in {filepath}")

    # Restructure the dataset
    names_dict = {"Latitude": "latitude", "Longitude": "longitude"}
    names_dict.update({"Altitude": "altitude", "Reflectivity": "reflectivity"})
    names_dict.update({"Nradobs": "number_of_observations"})
    names_dict.update({"Nradecho": "number_of_echoes"})

    ds = ds.rename(names_dict)

    for dim in ["latitude", "longitude", "altitude"]:
        ds[dim].attrs["standard_name"] = dim
        ds[dim].attrs["long_name"] = dim
    ds["altitude"] = ds["altitude"] * 1000  # Convert to meters
    kept_fields = dataset_options.fields + ["number_of_observations"]
    kept_fields += ["number_of_echoes"]
    dropped_fields = [f for f in ds.data_vars if f not in kept_fields]
    ds = ds.drop_vars(dropped_fields)

    for field in dataset_options.fields:
        ds[field] = ds[field].expand_dims("time")
        ds[field].attrs["long_name"] = field

    spacing = [ds.latitude.delta, ds.longitude.delta]
    if grid_options.name == "geographic":
        grid_options.latitude = ds.latitude.values
        grid_options.longitude = ds.longitude.values
        grid_options.altitude = ds.altitude.values
        grid_options.geographic_spacing = spacing
        grid_options.shape = [len(ds.latitude), len(ds.longitude)]

    ds["longitude"] = ds["longitude"] % 360

    # Get the domain mask associated with the given object
    # Note the relevant domain mask is a function of how the object is detected, e.g.
    # which levels!
    domain_mask = get_domain_mask(ds, track_options, dataset_options)
    boundary_coords, simple_boundary_coords, boundary_mask = utils.get_mask_boundary(
        domain_mask, grid_options
    )
    ds["domain_mask"] = domain_mask
    ds["boundary_mask"] = boundary_mask

    # Don't mask the gridcell areas
    cell_areas = grid.get_cell_areas(grid_options)
    ds["gridcell_area"] = (["latitude", "longitude"], cell_areas)
    area_attrs = {"units": "km^2", "standard_name": "area", "valid_min": 0}
    ds["gridcell_area"].attrs.update(area_attrs)

    # Apply the domain mask to the current grid
    ds = utils.apply_mask(ds, grid_options)
    ds = ds.drop_vars(["number_of_observations", "number_of_echoes"])
    return ds, boundary_coords, simple_boundary_coords


def get_domain_mask(ds, track_options, dataset_options):
    """
    Get a domain mask for a GridRad dataset.
    """

    domain_masks = []
    dataset_name = dataset_options.name
    for level_options in track_options.levels:
        for object_options in level_options.objects:
            detected = "detection" in object_options.model_fields
            uses_dataset = dataset_name == object_options.dataset
            if detected and uses_dataset:
                mask = utils.mask_from_observations(ds, dataset_options, object_options)
                domain_masks.append(mask)
    # Combine the masks
    if len(domain_masks) == 0:
        message = f"{dataset_name} not used for object detection. Check track_options."
        logger.debug(message)
        raise ValueError(message)
    domain_mask = reduce(lambda x, y: x * y, domain_masks)
    domain_mask = utils.smooth_mask(domain_mask)
    logger.debug(f"Got domain mask for {dataset_name}.")
    return domain_mask


def update_boundary_data(dataset, boundary_coords, input_record):
    previous_domain_mask = copy.deepcopy(input_record["current_domain_mask"])
    previous_boundary_coords = copy.deepcopy(
        input_record["current_boundary_coordinates"]
    )
    previous_boundary_mask = copy.deepcopy(input_record["current_boundary_mask"])

    input_record["previous_domain_masks"].append(previous_domain_mask)
    input_record["previous_boundary_coordinates"].append(previous_boundary_coords)
    input_record["previous_boundary_masks"].append(previous_boundary_mask)

    input_record["current_domain_mask"] = dataset["domain_mask"]
    input_record["current_boundary_coordinates"] = boundary_coords
    input_record["current_boundary_mask"] = dataset["boundary_mask"]


def update_dataset(time, input_record, track_options, dataset_options, grid_options):
    """
    Update a gridrad dataset.

    Parameters
    ----------
    time : datetime64
        The time of the dataset.
    object_tracks : dict
        Dictionary containing the object tracks.
    dataset_options : dict
        Dictionary containing the dataset options.
    grid_options : dict
        Dictionary containing the grid options.

    Returns
    -------
    dataset : object
        The updated dataset.
    """
    utils.log_dataset_update(logger, dataset_options.name, time)
    conv_options = dataset_options.converted_options

    input_record["current_file_index"] += 1
    filepath = dataset_options.filepaths[input_record["current_file_index"]]
    if conv_options.load is False:
        args = [time, input_record, track_options, dataset_options, grid_options]
        dataset = get_gridrad(*args)
    else:
        dataset = xr.open_dataset(filepath)
        domain_mask = dataset["domain_mask"]
        boundary_coords = utils.get_mask_boundary(domain_mask, grid_options)[0]
        update_boundary_data(dataset, boundary_coords, input_record)

    if conv_options.save:
        utils.save_converted_dataset(dataset, dataset_options)
    input_record["dataset"] = dataset


dataset_id_converter = {"ds841.6": "d841006"}


def gridrad_grid_from_dataset(dataset, variable, time):
    """Get a THUNER grid from a GridRad dataset."""
    grid = dataset[variable].sel(time=time)
    logger.debug(f"Got grid from dataset at time {time}.")
    return grid
