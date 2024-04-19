"""Data processing utilities."""

import requests
import zipfile
from pathlib import Path
from thor.log import setup_logger
from tqdm import tqdm
import xarray as xr


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


def download_file(url, directory):
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

    filename = url.split("/")[-1]
    filepath = Path(directory) / filename
    partial_filepath = filepath.with_suffix(".part")

    if filepath.exists():
        logger.debug(f"{filepath} already exists.")
        return str(filepath)
    elif partial_filepath.exists():
        logger.debug(f"Resuming download of {url}...")
        already_downloaded = partial_filepath.stat().st_size
        resume_header = {"Range": f"bytes={partial_filepath.stat().st_size}-"}
    else:
        logger.debug(f"Initiating download of {url}...")
        already_downloaded = 0
        resume_header = {}

    # Send a HTTP request to the URL
    logger.debug(f"Sending HTTP request to {url}.")
    response = requests.get(url, headers=resume_header, stream=True)
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
        raise Exception(
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
        The consolidated xarray dataset containing the selected variables from the input files.

    """
    datasets = []
    if fields is None:
        fields = xr.open_dataset(filepaths[0]).data_vars.keys()

    for filepath in filepaths:
        dataset = xr.open_dataset(filepath)
        dataset = dataset[fields]
        datasets.append(dataset)

    logger.debug(f"Concatenating datasets along {concat_dim}.")
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
