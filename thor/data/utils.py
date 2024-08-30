"""Data processing utilities."""

import subprocess
import zipfile
from pathlib import Path
import requests
from tqdm import tqdm
import cdsapi
import cv2
import numpy as np
import xarray as xr
from skimage.morphology import remove_small_objects, remove_small_holes
from scipy.ndimage import binary_dilation, binary_erosion
from thor.log import setup_logger
from thor.utils import format_time
from thor.utils import haversine


logger = setup_logger(__name__, level="DEBUG")


def generate_times(options, attempt_download=True):
    """Get times from data_options."""
    filepaths = options["filepaths"]
    for filepath in sorted(filepaths):
        if not Path(filepath).exists() and attempt_download:
            remote_filepath = str(filepath).replace(
                options["parent_local"], options["parent_remote"]
            )
            download_file(
                remote_filepath, options["parent_remote"], options["parent_local"]
            )
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


def download_file(url, parent_remote, parent_local):
    """
    Downloads a file from the given URL and saves it to the specified directory.

    Parameters
    ----------
    url : str
        The URL of the file to download.
    directory : str
        The directory where the downloaded file will be saved.

    Returns
    -------
    None

    Raises
    ------
    None
    """

    if not isinstance(url, str):
        raise TypeError("url must be a string")

    parent_remote = parent_remote.rstrip("/")
    parent_local = parent_local.rstrip("/")

    filepath = Path(url.replace(parent_remote, parent_local))
    if not filepath.parent.exists():
        filepath.parent.mkdir(parents=True)

    partial_filepath = filepath.with_suffix(".part")

    if filepath.exists():
        logger.info("%s already exists.", filepath)
        return str(filepath)
    if partial_filepath.exists():
        logger.info("Resuming download of %s...", url)
        already_downloaded = partial_filepath.stat().st_size
        resume_header = {"Range": f"bytes={partial_filepath.stat().st_size}-"}
    else:
        logger.info("Initiating download of %s...", url)
        already_downloaded = 0
        resume_header = {}

    # Send a HTTP request to the URL
    logger.info("Sending HTTP request to %s.", url)
    response = requests.get(url, headers=resume_header, stream=True, timeout=10)
    # Check if the request is successful
    if response.status_code == 200 or response.status_code == 206:
        total_size_in_bytes = (
            int(response.headers.get("content-length", 0)) + already_downloaded
        )
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
        progress_bar.update(already_downloaded)
        # Open a .zip file in the temporary directory
        with open(partial_filepath, "ab") as f:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                f.write(data)
        progress_bar.close()

        partial_filepath.rename(filepath)
    else:
        raise ValueError(
            f"Failed to download {url}. HTTP status code: {response.status_code}."
        )

    return str(filepath)


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


def format_string_list(strings):
    """
    Format a list of strings into a human-readable string.

    Parameters
    ----------
    strings : list of str
        List of strings to be formatted.

    Returns
    -------
    formatted_string : str
        The formatted string.
    """
    if len(strings) > 1:
        formatted_string = ", ".join(strings[:-1]) + " or " + strings[-1]
        return formatted_string
    elif strings:
        return strings[0]
    else:
        raise ValueError("strings must be an iterable of strings'.")


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

    z_min = grid_options["start_z"]
    z_max = grid_options["end_z"]
    y_min = grid_options["start_y"]
    y_max = grid_options["end_y"]
    x_min = grid_options["start_x"]
    x_max = grid_options["end_x"]

    z_count = (z_max - z_min) / grid_options["grid_spacing"][0]
    y_count = (y_max - y_min) / grid_options["grid_spacing"][1]
    x_count = (x_max - x_min) / grid_options["grid_spacing"][2]

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
    z_min = grid_options["start_z"]
    z_max = grid_options["end_z"]
    y_min = grid_options["start_y"]
    y_max = grid_options["end_y"]
    x_min = grid_options["start_x"]
    x_max = grid_options["end_x"]

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
    local_logger.info(
        f"Updating {name} dataset for {format_time(time, filename_safe=False)}."
    )


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

    command = (
        f"ncks -d {time_var},{start},{end} "
        f"-d latitude,{lat_range[0]},{lat_range[1]} "
        f"-d longitude,{lon_range[0]},{lon_range[1]} "
        f"{input_filepath} {output_filepath}"
    )
    subprocess.run(command, shell=True, check=True)


def get_parent(dataset_options):
    conv_options = dataset_options["converted_options"]
    local = dataset_options["parent_local"]
    remote = dataset_options["parent_remote"]
    if conv_options["load"]:
        if conv_options["parent_converted"] is not None:
            parent = conv_options["parent_converted"]
        elif local is not None:
            conv_options["parent_converted"] = local.replace("raw", "converted")
            parent = conv_options["parent_converted"]
        elif conv_options["parent_remote"] is not None:
            conv_options["parent_converted"] = remote.replace("raw", "converted")
            parent = conv_options["parent_converted"]
        else:
            raise ValueError("Could not find/create parent_converted directory.")
    elif local is not None:
        parent = local
    elif remote is not None:
        parent = remote
    else:
        raise ValueError("No parent directory provided.")
    return parent


def mask_from_input_record(
    track_input_records, dataset_options, object_options, grid_options
):
    """
    Get a domain mask from the input record. This function is used if a single domain
    mask applies to all objects/times in the dataset.
    """

    input_record = track_input_records[dataset_options["name"]]
    domain_mask = input_record["domain_mask"]
    boundary_coords = input_record["boundary_coordinates"]

    return domain_mask, boundary_coords


def mask_from_observations(dataset, dataset_options, object_options=None):
    """Create domain mask based on number of observations in each cell."""

    if object_options is None:
        altitudes = [dataset.altitude.values.min(), dataset.altitude.values.max()]
    else:
        altitudes = object_options["detection"]["altitudes"]
    num_obs = dataset["number_of_observations"].sel(altitude=slice(*altitudes))
    num_obs = num_obs.sum(dim="altitude")
    mask = num_obs > dataset_options["obs_thresh"]
    return mask


def smooth_mask(mask):
    mask_values = remove_small_holes(mask.values, area_threshold=50)
    mask_values = remove_small_objects(mask_values, min_size=50)
    # Pad the mask before dilation/erosion to avoid edge effects
    pad_width = 3
    mask_values = np.pad(mask_values, pad_width, mode="edge")
    dilation_element = np.ones((3, 3))
    erosion_element = np.ones((4, 4))
    mask_values = binary_dilation(mask_values, structure=dilation_element)
    mask_values = binary_erosion(mask_values, structure=erosion_element)
    mask_values = mask_values[pad_width:-pad_width, pad_width:-pad_width]
    mask_values = remove_small_holes(mask_values, area_threshold=50)
    mask_values = remove_small_objects(mask_values, min_size=50)
    mask.values = mask_values
    return mask


def mask_from_range(dataset, dataset_options, grid_options):
    """Create domain mask for gridcells greater than range from central point."""
    if grid_options["name"] == "cartesian":
        X, Y = np.meshgrid(grid_options["x"], grid_options["y"])
        distances = np.sqrt(X**2 + Y**2)
        coords = {"y": dataset.y, "x": dataset.x}
        dims = {"y": dataset.y, "x": dataset.x}
    elif grid_options["name"] == "geographic":
        lons = grid_options["longitude"]
        lats = grid_options["latitude"]
        origin_longitude = float(dataset.attrs["origin_longitude"])
        origin_latitude = float(dataset.attrs["origin_latitude"])
        LON, LAT = np.meshgrid(lons, lats)
        distances = haversine(LAT, LON, origin_latitude, origin_longitude)
        coords = {"latitude": dataset.latitude, "longitude": dataset.longitude}
        dims = {"latitude": dataset.latitude, "longitude": dataset.longitude}
    else:
        raise ValueError("Grid name must be 'cartesian' or 'geographic'.")

    units_dict = {"m": 1, "km": 1e3}
    range = dataset_options["range"] * units_dict[dataset_options["range_units"]]
    mask = distances <= range
    mask = xr.DataArray(mask, coords=coords, dims=dims)

    return mask


def get_mask_boundary(mask, grid_options):
    """Get domain mask boundary using cv2."""

    lons = np.array(grid_options["longitude"])
    lats = np.array(grid_options["latitude"])
    contours = cv2.findContours(
        mask.values.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
    )[0]
    boundary_coords = []
    for contour in contours:
        contour = np.append(contour, [contour[0]], axis=0)
        contour_rows = contour[:, :, 1].flatten()
        contour_cols = contour[:, :, 0].flatten()
        if grid_options["name"] == "cartesian":
            boundary_lats = lats[contour_rows, contour_rows]
            boundary_lons = lons[contour_rows, contour_cols]
        elif grid_options["name"] == "geographic":
            boundary_lats = lats[contour_rows]
            boundary_lons = lons[contour_cols]
        boundary_dict = {"latitude": boundary_lats, "longitude": boundary_lons}
        boundary_coords.append(boundary_dict)
    return boundary_coords


def save_converted_dataset(dataset, dataset_options):
    """Save a converted dataset."""
    conv_options = dataset_options["converted_options"]
    if conv_options["save"]:
        filepath = dataset_options["filepaths"][0]
        parent = get_parent(dataset_options)
        if conv_options["parent_converted"] is None:
            parent_converted = parent.replace("raw", "converted")
        converted_filepath = filepath.replace(parent, parent_converted)
        if not Path(converted_filepath).parent.exists():
            Path(converted_filepath).parent.mkdir(parents=True)
        dataset.to_netcdf(converted_filepath, mode="w")
    return dataset
