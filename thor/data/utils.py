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
from thor.log import setup_logger
from thor.utils import format_time
from thor.utils import haversine


logger = setup_logger(__name__, level="DEBUG")


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
        logger.debug("%s already exists.", filepath)
        return str(filepath)
    if partial_filepath.exists():
        logger.debug("Resuming download of %s...", url)
        already_downloaded = partial_filepath.stat().st_size
        resume_header = {"Range": f"bytes={partial_filepath.stat().st_size}-"}
    else:
        logger.debug("Initiating download of %s...", url)
        already_downloaded = 0
        resume_header = {}

    # Send a HTTP request to the URL
    logger.debug("Sending HTTP request to %s.", url)
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

    logger.debug("Concatenating datasets along %s.", concat_dim)
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
        logger.debug("%s already exists.", local_path)
        return

    if not Path(local_path).parent.exists():
        Path(local_path).parent.mkdir(parents=True)

    cdsc = cdsapi.Client()
    cdsc.retrieve(cds_name, request, local_path)


def log_dataset_update(local_logger, name, time):
    local_logger.debug(
        f"Updating {name} dataset for {format_time(time, filename_safe=False)}."
    )


def log_convert(local_logger, name, filepath):
    local_logger.debug("Converting %s data from %s", name, Path(filepath).name)


def call_ncks(input_filepath, output_filepath, start, end, lat_range, lon_range):
    command = (
        f"ncks -d time,{start},{end} "
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


def get_range_mask(dataset, dataset_options):
    """Mask data greater than range from central point."""

    longitudes = dataset.longitude.values
    latitudes = dataset.latitude.values
    origin_longitude = float(dataset.attrs["origin_longitude"])
    origin_latitude = float(dataset.attrs["origin_latitude"])

    LON, LAT = np.meshgrid(longitudes, latitudes)
    distances = haversine(LAT, LON, origin_latitude, origin_longitude)
    units_dict = {"m": 1, "km": 1e3}
    range = dataset_options["range"] * units_dict[dataset_options["range_units"]]
    mask = distances <= range
    contour = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE,
    )[0][0]
    contour = np.append(contour, [contour[0]], axis=0)
    range_latitudes = latitudes[contour[:, :, 1]].flatten()
    range_longitudes = longitudes[contour[:, :, 0]].flatten()
    return mask, range_latitudes, range_longitudes
