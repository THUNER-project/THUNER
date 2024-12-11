"""Data processing utilities."""

import multiprocessing
import threading
import subprocess
import fcntl
import zipfile
import time
from pathlib import Path
import requests
import cdsapi
import cv2
import numpy as np
import xarray as xr
from skimage.morphology import remove_small_objects, remove_small_holes
from scipy.ndimage import binary_dilation, binary_erosion
import thuner.log as log
import thuner.utils as utils
from thuner.config import get_outputs_directory

logger = log.setup_logger(__name__, level="DEBUG")
# Set the number of cv2 threads to 0 to avoid crashes.
# See https://github.com/opencv/opencv/issues/5150#issuecomment-675019390
cv2.setNumThreads(0)


class DownloadState(utils.SingletonBase):
    """
    Singleton class to manage download state across multiple processes.
    """

    def _initialize(self):
        # Use both process and thread locks to prevent excess simultaneous downloads
        self.process_lock = multiprocessing.Lock()
        self.thread_lock = threading.Lock()
        # Also use a lock file to store time since last download request, to prevent
        # excess requests from all instances of thuner running on a given machine
        with self.process_lock, self.thread_lock:
            lock_directory = get_outputs_directory() / ".locks"
            utils.create_hidden_directory(lock_directory)
            self.lock_filepath = lock_directory / "download_lock"
            if not Path(self.lock_filepath).exists():
                Path(self.lock_filepath).touch()
        # Time of the last download request
        self.last_request_time = 0.0
        # Impose a 1 second wait time between download requests
        self.wait_time = 1

    def wait_for_lockfile(self):
        """Wait for turn to download using filelock."""
        with open(self.lock_filepath, "r+") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                if Path(self.lock_filepath).exists():
                    self._handle_existing_lockfile(lock_file)
                self._update_lockfile_timestamp(lock_file)
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

    def _handle_existing_lockfile(self, lock_file):
        """Handle the case where the lock file already exists."""
        lock_file.seek(0)
        last_request_time = float(lock_file.read() or 0)
        current_time = time.time()
        elapsed_time = current_time - last_request_time
        if elapsed_time < self.wait_time:
            logger.info(f"Recent download Request. Waiting {self.wait_time} seconds.")
            time.sleep(self.wait_time - elapsed_time)

    def _update_lockfile_timestamp(self, lock_file):
        """Update the lock file with the current timestamp."""
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(str(time.time()))


def get_parent(dataset_options):
    """Get the appropriate parent directory."""
    conv_options = dataset_options.converted_options
    local = dataset_options.parent_local
    remote = dataset_options.parent_remote
    if conv_options.load:
        if conv_options.parent_converted is not None:
            parent = conv_options.parent_converted
        elif local is not None:
            conv_options.parent_converted = local.replace("raw", "converted")
            parent = conv_options.parent_converted
        elif conv_options.parent_remote is not None:
            conv_options.parent_converted = remote.replace("raw", "converted")
            parent = conv_options.parent_converted
        else:
            raise ValueError("Could not find/create parent_converted directory.")
    elif local is not None:
        parent = local
    elif remote is not None:
        parent = remote
    else:
        raise ValueError("No parent directory provided.")
    return parent


def url_to_filepath(url, parent_remote, parent_local):
    """Convert remote URL to local file path."""
    if not isinstance(url, str):
        raise TypeError("url must be a string")

    parent_remote = parent_remote.rstrip("/")
    parent_local = parent_local.rstrip("/")
    return url.replace(parent_remote, parent_local)


def handle_response(response, already_downloaded, filepath):
    """Handle the response from a HTTP request."""

    if response.status_code != 200 and response.status_code != 206:
        message = f"Failed to download file to {filepath}. "
        message += f"HTTP status code: {response.status_code}."
        raise ValueError(message)
    partial_filepath = filepath.with_suffix(".part")
    checkpoint_size = 10 * 1024**2  # 10 MB
    mb_downloaded = already_downloaded / 1024**2
    logger.info(f"Downloaded {mb_downloaded:.1f} MB of {filepath.name}.")
    last_checkpoint = already_downloaded
    with open(partial_filepath, "ab") as f:
        block_size = 1024  # 1 KB
        for data in response.iter_content(block_size):
            f.write(data)
            already_downloaded += block_size
            if already_downloaded - last_checkpoint > checkpoint_size:
                mb_downloaded = already_downloaded / 1024**2
                logger.info(f"Downloaded {mb_downloaded:.1f} MB of {filepath.name}.")
                last_checkpoint = already_downloaded
    partial_filepath.rename(filepath)
    logger.info(f"Completed download of {filepath.name}.")


def get_header(url, filepath):
    """Get the header for a HTTP request."""
    partial_filepath = filepath.with_suffix(".part")
    if partial_filepath.exists():
        logger.info("Resuming download of %s", url)
        already_downloaded = partial_filepath.stat().st_size
        resume_header = {"Range": f"bytes={partial_filepath.stat().st_size}-"}
    else:
        logger.info("Initiating download of %s", url)
        already_downloaded = 0
        resume_header = {}
    return already_downloaded, resume_header


def download(url, parent_remote, parent_local, max_retries=10, retry_delay=2):
    """
    Downloads a file from the given URL and saves it to the specified directory,
    preserving the subdirectory structure of the remote filesystem.
    """

    filepath = Path(url_to_filepath(url, parent_remote, parent_local))
    if not filepath.parent.exists():
        filepath.parent.mkdir(parents=True)
    if filepath.exists():
        logger.info("%s already exists.", filepath)
        return str(filepath)

    download_state = DownloadState()
    download_state.wait_for_lockfile()

    for attempt in range(1, max_retries + 1):
        try:
            already_downloaded, resume_header = get_header(url, filepath)
            logger.info("Sending HTTP request to %s.", url)
            response = requests.get(url, headers=resume_header, stream=True, timeout=10)
            handle_response(response, already_downloaded, filepath)
            return str(filepath)
        except Exception as e:
            logger.error(f"Download attempt {attempt} failed: {e}")
            if attempt < max_retries:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                message = "Max retries reached. Download failed."
                raise requests.exceptions.RequestException(message)


def generate_times(dataset_options, attempt_download=False):
    """Get times from data_options."""
    filepaths = dataset_options.filepaths
    parent_local = dataset_options.parent_local
    parent_remote = dataset_options.parent_remote
    for filepath in sorted(filepaths):
        if not Path(filepath).exists() and attempt_download:
            remote_filepath = str(filepath).replace(parent_local, parent_remote)
            download(remote_filepath, parent_remote, parent_local)
        with xr.open_dataset(filepath, chunks={}) as ds:
            for time in ds.time.values:
                yield time


def unzip_file(filepath, directory=None):
    """
    Downloads a .zip file from a URL, extracts the contents of the .zip file into
    directory.

    Parameters
    ----------
    url : str
        The URL of the .zip file.
    directory : str
        The path to the directory where the .zip file contents will be extracted.

    Returns
    -------
    extracted_filepaths : list
        A list of paths to the extracted files.
    dir_size : int
        The total size of the extracted files in bytes.
    """
    if directory is None:
        directory = Path(filepath).parent
    filename = filepath.split("/")[-1]
    out_directory = directory / Path(filename).stem
    out_directory.mkdir(exist_ok=True)

    # Open the .zip file
    with zipfile.ZipFile(Path(directory) / filename, "r") as zip_ref:
        # Extract all the contents of the .zip file in the temporary directory
        zip_ref.extractall(Path(out_directory))
    # Get the list of extracted files
    extracted_filepaths = list(Path(out_directory).rglob("*"))
    extracted_filepaths = [str(file) for file in extracted_filepaths]
    dir_size = get_directory_size(directory)

    return sorted(extracted_filepaths), dir_size


def check_valid_url(url):

    if not isinstance(url, str):
        raise TypeError("url must be a string")

    # Send a HTTP request to the URL
    logger.info("Sending HTTP request to %s.", url)
    try:
        response = requests.head(url, timeout=10)
        # Check if the request is successful
        if response.status_code == 200:
            return True
        else:
            return False
    except requests.RequestException:
        return False


def get_directory_size(directory):
    """
    Get the size of a directory in a human-readable format.

    Parameters
    ----------
    directory : str
        The path to the directory.

    Returns
    -------
    size : str
        The size of the directory in a human-readable format.
    """
    total = 0
    for p in Path(directory).rglob("*"):
        if p.is_file():
            total += p.stat().st_size

    # Convert size to a human-readable format
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if total < 1024.0:
            return f"{total:.1f} {unit}"
        total /= 1024.0


def consolidate_netcdf(filepaths, fields=None, concat_dim="time"):
    """
    Consolidate multiple netCDF files into a single xarray dataset.

    Parameters
    ----------
    filepaths : list of str
        List of filepaths to the netCDF files that need to be consolidated.
    fields : list of str, optional
        List of variable names to include in the consolidated dataset. If not provided,
        all variables in the first file will be included.
    concat_dim : str, optional
        Dimension along which the datasets will be concatenated. Default is "time".

    Returns
    -------
    dataset : xarray.Dataset
        The consolidated xarray dataset containing the selected variables
        from the input files.

    """
    datasets = []
    if fields is None:
        fields = xr.open_dataset(filepaths[0]).data_vars.keys()

    for filepath in filepaths:
        dataset = xr.open_dataset(filepath)
        dataset = dataset[fields]
        datasets.append(dataset)

    logger.info("Concatenating datasets along %s.", concat_dim)
    dataset = xr.concat(datasets, dim=concat_dim)

    return dataset


def get_pyart_grid_shape(grid_options):
    """
    Get the grid shape for pyart grid.

    Parameters
    ----------
    grid_options : dict
        Dictionary containing the grid options.

    Returns
    -------
    tuple
        The grid shape as a tuple of (nz, ny, nx).
    """

    z_min = grid_options.start_z
    z_max = grid_options.end_z
    y_min = grid_options.start_y
    y_max = grid_options.end_y
    x_min = grid_options.start_x
    x_max = grid_options.end_x

    z_count = (z_max - z_min) / grid_options.grid_spacing[0]
    y_count = (y_max - y_min) / grid_options.grid_spacing[1]
    x_count = (x_max - x_min) / grid_options.grid_spacing[2]

    if z_count.is_integer() and y_count.is_integer() and x_count.is_integer():
        z_count = int(z_count)
        y_count = int(y_count)
        x_count = int(x_count)
    else:
        raise ValueError("Grid spacings must divide domain lengths.")

    return (z_count, y_count, x_count)


def get_pyart_grid_limits(grid_options):
    """
    Get the grid limits for pyart grid.

    Parameters
    ----------
    grid_options : dict
        Dictionary containing the grid options.

    Returns
    -------
    tuple
        The grid limits as a tuple of ((z_min, z_max), (y_min, y_max), (x_min, x_max)).
    """
    z_min = grid_options.start_z
    z_max = grid_options.end_z
    y_min = grid_options.start_y
    y_max = grid_options.end_y
    x_min = grid_options.start_x
    x_max = grid_options.end_x

    return ((z_min, z_max), (y_min, y_max), (x_min, x_max))


def cdsapi_retrieval(cds_name, request, local_path):
    """
    Perform a CDS API retrieval.

    Parameters
    ----------
    cds_name : str
        The name argument for the cdsapi retrieval.
    request : dict
        A dictionary containing the cdsapi retrieval options.
    local_path : str
        The local file path where the retrieved data will be saved.

    Returns
    -------
    None
    """
    if Path(local_path).exists():
        logger.info("%s already exists.", local_path)
        return

    if not Path(local_path).parent.exists():
        Path(local_path).parent.mkdir(parents=True)

    cdsc = cdsapi.Client()
    cdsc.retrieve(cds_name, request, local_path)


def log_dataset_update(local_logger, name, time):
    time_str = utils.format_time(time, filename_safe=False)
    local_logger.info(f"Updating {name} dataset for {time_str}.")


def log_convert(local_logger, name, filepath):
    local_logger.info("Converting %s data from %s", name, Path(filepath).name)


def call_ncks(input_filepath, output_filepath, start, end, lat_range, lon_range):

    # Check if time variable "time" or "valid_time". If "valid_time" convert to "time".
    check_command = f"ncks -m {input_filepath} | grep 'valid_time'"
    result = subprocess.run(check_command, shell=True, capture_output=True, text=True)

    if "valid_time" in result.stdout:
        time_var = "valid_time"
    else:
        time_var = "time"

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


def mask_from_input_record(
    track_input_records, dataset_options, object_options, grid_options
):
    """
    Get a domain mask from the input record. This function is used if a single domain
    mask applies to all objects/times in the dataset.
    """

    input_record = track_input_records[dataset_options.name]
    domain_mask = input_record["domain_mask"]
    boundary_coords = input_record["boundary_coordinates"]

    return domain_mask, boundary_coords


def mask_from_observations(dataset, dataset_options, object_options=None):
    """Create domain mask based on number of observations in each cell."""

    altitudes = object_options.detection.altitudes
    if altitudes == [] or altitudes is None:
        altitudes = [dataset.altitude.values.min(), dataset.altitude.values.max()]
    else:
        altitudes = object_options.detection.altitudes
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


def get_mask_boundary(mask, grid_options):
    """Get domain mask boundary using cv2."""

    lons = np.array(grid_options.longitude)
    lats = np.array(grid_options.latitude)
    mask_array = mask.fillna(0).values.astype(np.uint8)
    args = [mask_array, cv2.RETR_LIST]
    # Record the contours with all points, and with only the end points of each line
    # comprising the contour. The former is used to determine boundary overlap,
    # the latter makes plotting the boundary more efficient.
    contours = cv2.findContours(*args, cv2.CHAIN_APPROX_NONE)[0]
    simple_contours = cv2.findContours(*args, cv2.CHAIN_APPROX_SIMPLE)[0]
    boundary_coords = []
    boundary_pixels = []
    simple_boundary_coords = []

    def get_boundary_coords(contour):
        # Append the first point to the end to close the contour
        contour = np.append(contour, [contour[0]], axis=0)
        contour_rows = contour[:, :, 1].flatten()
        contour_cols = contour[:, :, 0].flatten()
        if grid_options.name == "cartesian":
            boundary_lats = lats[contour_rows, contour_rows]
            boundary_lons = lons[contour_rows, contour_cols]
        elif grid_options.name == "geographic":
            boundary_lats = lats[contour_rows]
            boundary_lons = lons[contour_cols]
        boundary_dict = {"latitude": boundary_lats, "longitude": boundary_lons}
        pixel_dict = {"row": contour_rows, "col": contour_cols}
        return boundary_dict, pixel_dict

    for contour in contours:
        boundary_dict, pixel_dict = get_boundary_coords(contour)
        boundary_coords.append(boundary_dict)
        boundary_pixels.append(pixel_dict)
    for contour in simple_contours:
        simple_boundary_coords.append(get_boundary_coords(contour)[0])

    boundary_mask = xr.zeros_like(mask).astype(bool)
    for pixels in boundary_pixels:
        boundary_mask.values[pixels["row"], pixels["col"]] = True

    return boundary_coords, simple_boundary_coords, boundary_mask


def get_encoding(ds):
    """Get encoding for writing masks to file."""
    encoding = {}
    for var in ds.variables:
        encoding[var] = {"zlib": True, "complevel": 5}
    return encoding


def save_converted_dataset(dataset, dataset_options):
    """Save a converted dataset."""
    conv_options = dataset_options.converted_options
    if conv_options.save:
        filepath = dataset_options.filepaths[0]
        parent = get_parent(dataset_options)
        if conv_options["parent_converted"] is None:
            parent_converted = parent.replace("raw", "converted")
        converted_filepath = filepath.replace(parent, parent_converted)
        if not Path(converted_filepath).parent.exists():
            Path(converted_filepath).parent.mkdir(parents=True)
        encoding = get_encoding(dataset)
        dataset.to_netcdf(converted_filepath, mode="w")
    return dataset
