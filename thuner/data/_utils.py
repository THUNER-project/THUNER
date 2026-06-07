"""Data processing utilities."""

import os

# Check if system is unix-like, as xESMF is not supported on Windows
if os.name == "posix":
    import xesmf as xe
else:
    message = "Warning: Windows systems cannot run xESMF for regridding."
    message += "If you need regridding, consider using a Linux or MacOS system."
    print(message)


import subprocess
from pathlib import Path
import cv2
import numpy as np
import xarray as xr
from skimage.morphology import remove_small_objects, remove_small_holes
from scipy.ndimage import binary_dilation, binary_erosion
import pyart
import thuner.log as log
import thuner.utils as utils
from thuner.config import get_outputs_directory

logger = log.setup_logger(__name__, level="INFO")
# Set the number of cv2 threads to 0 to avoid crashes.
# See https://github.com/opencv/opencv/issues/5150#issuecomment-675019390
cv2.setNumThreads(0)


def get_demo_data(output_parent=None, remote_directory=None):
    """
    Download the demo data from the AWS s3 thuner-storage bucket.
    """
    if output_parent is None:
        output_parent = get_outputs_directory()
    if remote_directory is None:
        remote_directory = "s3://thuner-storage/THUNER_output"
    if not Path(output_parent).exists():
        Path(output_parent).mkdir(parents=True)
    if not Path(output_parent).is_dir():
        raise ValueError(f"{output_parent} is not a directory.")
    if not Path(output_parent).is_absolute():
        raise ValueError(f"{output_parent} must be an absolute path.")
    # Remove "s3://thuner-storage/" from the remote directory and append result to output parent
    base_url = "s3://thuner-storage/THUNER_output/"
    directory_structure = remote_directory.replace(base_url, "")
    output_directory = output_parent / directory_structure
    # Check the remote directory exists by listing it. `aws s3 ls` on a prefix
    # with no matching keys returns exit code 0 with empty stdout, so verify
    # there is at least one entry before syncing.
    prefix = remote_directory.rstrip("/") + "/"
    check_command = f"aws s3 ls {prefix} --no-sign-request"
    result = subprocess.run(check_command, shell=True, capture_output=True, text=True)
    if result.returncode != 0 or not result.stdout.strip():
        raise FileNotFoundError(
            f"Remote directory {remote_directory} does not exist or is empty."
        )
    command = f"aws s3 sync {remote_directory} {output_directory} --no-sign-request"
    logger.info("Syncing directory %s. Please wait.", output_directory)
    subprocess.run(command, shell=True, check=True)


def log_dataset_update(local_logger, name, time):
    time_str = utils.format_time(time, filename_safe=False)
    local_logger.info(f"Updating {name} dataset for {time_str}.")


def call_ncks(input_filepath, output_filepath, start, end, lat_range, lon_range):
    """Call ncks to subset a large netcdf file."""
    # Read metadata using xr with lazy loading.
    ds = xr.open_dataset(input_filepath, chunks={}, decode_timedelta=True)
    # Check if time variable "time" or "valid_time". If "valid_time" convert to "time".
    if "valid_time" in ds:
        time_var = "valid_time"
    else:
        time_var = "time"
    # Ensure start and end times are within the dataset time range.
    time = ds[time_var].values
    if start < time[0]:
        start = time[0]
    if end > time[-1]:
        end = time[-1]

    lon_range = [(lon + 180) % 360 - 180 for lon in lon_range]
    command = (
        f"ncks -d {time_var},{start},{end} "
        f"-d latitude,{lat_range[0]},{lat_range[1]} "
        f"-d longitude,{lon_range[0]},{lon_range[1]} "
        f"{input_filepath} {output_filepath}"
    )
    result = subprocess.run(command, shell=True, check=True)
    if result.returncode != 0:
        logger.error("ncks failed with return code %d.", result.returncode)
        logger.error("Standard output: %s", result.stdout)
        logger.error("Standard error: %s", result.stderr)
        raise subprocess.CalledProcessError(result.returncode, command)


def apply_mask(ds, grid_options):
    """Apply a domain mask to an xr dataset."""
    domain_mask = ds["domain_mask"]
    if grid_options.name == "cartesian":
        dims = ["y", "x"]
    elif grid_options.name == "geographic":
        dims = ["latitude", "longitude"]
    else:
        raise ValueError("Grid name must be 'cartesian' or 'geographic'.")
    for var in ds.data_vars.keys() - ["gridcell_area", "domain_mask", "boundary_mask"]:
        # Check if the variable has horizontal dimensions
        if not set(dims).issubset(set(ds[var].dims)):
            continue
        # Otherwise apply the mask
        broadcasted_mask = domain_mask.broadcast_like(ds[var])
        # Apply the mask, setting unmasked values to NaN or 0 as appropriate
        dtype = ds[var].dtype
        float_types = [np.floating, np.complexfloating]
        int_types = [np.integer, np.bool_]
        if any(np.issubdtype(dtype, parent_type) for parent_type in float_types):
            ds[var] = ds[var].where(broadcasted_mask)
        elif any(np.issubdtype(dtype, parent_type) for parent_type in int_types):
            ds[var] = ds[var].where(broadcasted_mask, 0)
        else:
            message = f"Cannot apply domain mask to {var}. Unknown data type."
            raise ValueError(message)
    return ds


def mask_from_observations(dataset, dataset_options, object_options=None):
    """Create domain mask based on number of observations in each cell."""

    altitudes = object_options.detection.altitudes
    if altitudes is None:
        logger.warning(
            "No altitudes specified in object options. Using all altitudes in dataset."
        )
        altitudes = [dataset.altitude.values.min(), dataset.altitude.values.max()]
    num_obs = dataset["number_of_observations"].sel(altitude=slice(*altitudes))
    mask = num_obs > dataset_options.obs_thresh
    mask = mask.any(dim="altitude")
    return mask.astype(bool)


def smooth_mask(mask):
    """Smooth a binary mask using morphological operations."""
    # Remove objects smaller than a 150 km radius region.
    # 0.02 lat/lon per pixel and ~100 km per lat/lon gives min area of np.pi*750**2
    # or ~1.8e4 pixels.
    mask_values = remove_small_objects(mask.values, min_size=1.8e4)
    # Fill holes smaller 100 pixels
    mask_values = remove_small_holes(mask_values, area_threshold=2e2)
    # Pad the mask before dilation/erosion to avoid edge effects
    pad_width = 3
    mask_values = np.pad(mask_values, pad_width, mode="edge")
    # Erode and dilate with large element to remove small objects and fill holes
    mask_values = binary_erosion(mask_values, structure=np.ones((20, 20)))
    mask_values = binary_dilation(mask_values, structure=np.ones((20, 20)))
    # Repeat but with a smaller element, and applying dilation first to close lines
    mask_values = binary_dilation(mask_values, structure=np.ones((5, 5)))
    # Erode one more pixel than dilated to ensure objects don't artificially avoid
    # touching the boundary
    mask_values = binary_erosion(mask_values, structure=np.ones((6, 6)))
    mask_values = mask_values[pad_width:-pad_width, pad_width:-pad_width]
    # Another pass at hole filling
    mask_values = remove_small_objects(mask_values, min_size=1.8e4)
    mask_values = remove_small_holes(mask_values, area_threshold=2e2)
    mask.values = mask_values
    return mask


def mask_from_range(dataset, dataset_options, grid_options):
    """Create domain mask for gridcells greater than range from central point."""
    if grid_options.name == "cartesian":
        X, Y = np.meshgrid(grid_options.x, grid_options.y)
        distances = np.sqrt(X**2 + Y**2)
        coords = {"y": dataset.y, "x": dataset.x}
        dims = {"y": dataset.y, "x": dataset.x}
    elif grid_options.name == "geographic":
        lons = grid_options.longitude
        lats = grid_options.latitude
        origin_longitude = float(dataset.attrs["origin_longitude"])
        origin_latitude = float(dataset.attrs["origin_latitude"])
        LON, LAT = np.meshgrid(lons, lats)
        distances = utils.haversine(LAT, LON, origin_latitude, origin_longitude)
        coords = {"latitude": dataset.latitude, "longitude": dataset.longitude}
        dims = {"latitude": dataset.latitude, "longitude": dataset.longitude}
    else:
        raise ValueError("Grid name must be 'cartesian' or 'geographic'.")

    units_dict = {"m": 1, "km": 1e3}
    range = dataset_options.range * units_dict[dataset_options.range_units]
    mask = distances <= range
    mask = xr.DataArray(mask.astype(bool), coords=coords, dims=dims)

    return mask


def get_geographic_regridder(
    dataset,
    grid_options,
    dataset_options,
    latitude=None,
    longitude=None,
    weights_filepath=None,
):
    """Load an xesmf using stored weights if present."""
    weights_filepath = weights_filepath or dataset_options.weights_filepath
    if latitude is None or longitude is None:
        latitude, longitude = grid_options.latitude, grid_options.longitude
    dims_dict = {"latitude": latitude, "longitude": longitude}
    dims = ["latitude", "longitude"]
    ds = xr.Dataset({dim: ([dim], dims_dict[dim]) for dim in dims})
    regrid_options = {"periodic": False, "extrap_method": None}
    if not weights_filepath or not Path(weights_filepath).exists():
        logger.info("Building regridder; this can take a while for large grids.")
        regridder = xe.Regridder(dataset, ds, "bilinear", **regrid_options)
        if dataset_options.reuse_regridder:
            Path(weights_filepath).parent.mkdir(parents=True, exist_ok=True)
            regridder.to_netcdf(weights_filepath)
            # The filepath now exists, so the else case called next time
    else:
        logger.info("Loading regridder weights from file.")
        regrid_options["weights"] = weights_filepath
        regridder = xe.Regridder(dataset, ds, "bilinear", **regrid_options)
    return regridder


def copy_attributes(ds, old_ds):
    """Copy attributes from one xarray dataset to another."""
    for var in ds.data_vars:
        if var in old_ds.data_vars:
            ds[var].attrs = old_ds[var].attrs
    for coord in ds.coords:
        ds[coord].attrs = old_ds[coord].attrs
    ds.attrs.update(old_ds.attrs)
    regrid_log = "Regridded using xesmf on " f"{np.datetime64('now')}"
    if "history" not in ds.attrs:
        ds.attrs["history"] = regrid_log
    else:
        ds.attrs["history"] += f", {regrid_log.lower()}"
    return ds


def read_odim(
    odim_object, weighting_function="Barnes2", grid_shape=None, grid_limits=None
):
    """Process ODIM radar data."""
    # Specify default grid shape and limits.
    grid_shape = grid_shape or (41, 161, 161)
    grid_limits = grid_limits or ((0, 20000), (-200000, 200000), (-200000, 200000))
    fields = ["reflectivity", "reflectivity_horizontal"]

    radar = pyart.aux_io.read_odim_h5(
        odim_object, file_field_names=False, include_fields=fields
    )
    dataset = pyart.map.grid_from_radars(
        radar,
        grid_shape=grid_shape,
        grid_limits=grid_limits,
        weighting_function=weighting_function,
        gridding_algo="map_gates_to_grid",
    )
    dataset = dataset.to_xarray()
    return dataset


def empty_dataset_and_coordinates(grid_options):
    """Build an empty dataset and coordinates for the given grid options."""
    dims = ["time", "latitude", "longitude"]
    coords = {dim: (dim, []) for dim in dims}
    ds = xr.Dataset(coords=coords)
    boundary_coords = [{"latitude": np.array([]), "longitude": np.array([])}]
    return ds, boundary_coords, boundary_coords
