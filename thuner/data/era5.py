"""Process ERA5 data."""

import calendar
import signal
from pathlib import Path
import inspect
import tempfile
import numpy as np
import pandas as pd
import xarray as xr
import cdsapi
from typing import Any
import thuner.log as log
from thuner.utils import get_hour_interval
import thuner.data.utils as utils
from thuner.config import get_outputs_directory


logger = log.setup_logger(__name__)

era5_pressure_levels = ["1000", "975", "950", "925", "900", "875", "850", "825", "800"]
era5_pressure_levels += ["775", "750", "700", "650", "600", "550", "500", "450", "400"]
era5_pressure_levels += ["350", "300", "250", "225", "200", "175", "150", "125", "100"]
era5_pressure_levels += ["70", "50", "30", "20", "10", "7", "5", "3", "2", "1"]


def get_era5_filepaths(options, start=None, end=None, local=True):
    """
    Generate era5 filepaths from dataset options dictionary.

    Parameters
    ----------
    options : dict
        Dictionary containing the input options.

    Returns
    -------
    urls : list
        List of URLs.
    times : list
        Times associated with the URLs.
    """

    # First get the base_path
    base_path = get_base_path(options, local=local)

    if start is None or end is None:
        start = options.start
        # Add an hour to the end time to facilitate temporal interpolation
        end = options.end

    start = pd.Timestamp(start)
    # Add an hour to the end time to facilitate temporal interpolation
    end = pd.Timestamp(end) + pd.Timedelta(hours=1)

    short_data_format = {"pressure-levels": "pl", "single-levels": "sfc"}

    # Get the times corresponding to the filepaths
    times = get_file_datetimes(options, start, end)

    # We will store individual fields in separate files
    filepaths = dict(zip(options.fields, [[] for i in range(len(options.fields))]))

    for field in options.fields:
        for time in times:
            time = pd.Timestamp(time)
            daterange_str = format_daterange(options, time)
            filepath = (
                f"{base_path}/{field}/{time.year}/{field}_era5_oper_"
                f"{short_data_format[options.data_format]}_{daterange_str}.nc"
            )
            filepaths[field].append(filepath)

    for key in filepaths.keys():
        filepaths[key] = sorted(filepaths[key])

    return filepaths


def format_daterange(options, time):
    """
    Format the date range string used in ERA5 file names on NCI Gadi,
    https://dx.doi.org/10.25914/5f48874388857.

    Parameters
    ----------
    options : dict
        Dictionary containing the data options.
    time : np.datetime64, pd.Timestamp or str
        The time to format.

    Returns
    -------
    date_range_str : str
        The formatted date range str.
    """

    time = pd.Timestamp(time)
    last_day = calendar.monthrange(time.year, time.month)[1]
    if options.storage == "daily":
        date_range_str = f"{time.year:04}{time.month:02}{time.day:02}"
    elif options.storage == "monthly":
        date_range_str = (
            f"{time.year:04}{time.month:02}01-{time.year:04}{time.month:02}{last_day}"
        )
    return date_range_str


def get_base_path(options: Any, local: bool = True):
    """Get the base path for the ERA5 data."""
    if local:
        parent = options.parent_local
    else:
        parent = options.parent_remote

    latitude_range = options.latitude_range
    longitude_range = options.longitude_range
    if latitude_range == [-90, 90] and longitude_range == [-180, 180]:
        return f"{parent}/era5/{options.data_format}/{options.mode}"
    area = get_area(options)
    area_str = get_area_string(area)

    if area_str is None:
        group = f"era5_{options.storage}"
    else:
        group = f"era5_{options.storage}_{area_str}"
    return f"{parent}/{group}/era5/{options.data_format}/{options.mode}"


def get_file_datetimes(options, start, end):
    """Get the datetimes corresponding to the filepaths."""
    if options.storage == "daily":
        # Note we typically store data locally in daily files
        range_start = np.datetime64(f"{start.year:04}-{start.month:02}-{start.day:02}")
        range_end = np.datetime64(f"{end.year:04}-{end.month:02}-{end.day:02}")
        time_step = np.timedelta64(1, "D")
    elif options.storage == "monthly":
        # On GADI era5 data is stored in monthly files
        range_start = np.datetime64(f"{start.year:04}-{start.month:02}")
        range_end = np.datetime64(f"{end.year:04}-{end.month:02}")
        time_step = np.timedelta64(1, "M")
    else:
        raise ValueError("options.storage must be either 'daily' or 'monthly'.")
    times = np.arange(range_start, range_end + time_step, time_step)
    return times


def generate_cdsapi_requests(options):
    """
    Retrieve ERA5 data using the CDS API.

    Parameters
    ----------
    options : dict
        A dictionary containing the input options.

    Returns
    -------
    cds_name : str
        The name argument for the cdsapi retrieval.
    requests : dict
        A dictionary containing the cdsapi retrieval options.
    local_paths : dict
        A dictionary containing the local file paths.
    """

    # First get the base_path for where to store the files locally
    base_path = get_base_path(options, local=True)

    short_data_format = {"pressure-levels": "pl", "single-levels": "sfc"}
    short_format = short_data_format[options.data_format]

    requests = dict(zip(options.fields, [[] for i in range(len(options.fields))]))
    local_paths = dict(zip(options.fields, [[] for i in range(len(options.fields))]))

    cds_name = f"reanalysis-era5-{options.data_format}"

    start = pd.Timestamp(options["start"])
    # Add an hour to the end time to facilitate temporal interpolation
    end = pd.Timestamp(options["end"]) + pd.Timedelta(hours=1)

    # Get the times corresponding to the filepaths
    times = get_file_datetimes(options, start, end)
    if options.latitude_range is None:
        latitude_range = [-90, 90]
    else:
        latitude_range = options.latitude_range
    if options.longitude_range is None:
        longitude_range = [-180, 180]
    else:
        longitude_range = options.longitude_range
    area = [latitude_range[1], longitude_range[0]]
    area += [latitude_range[0], longitude_range[1]]

    # Define a function to get the days for the API request for each time
    def get_days(time, options):
        if options.storage == "daily":
            days = [f"{time.day:02}"]
        elif options.storage == "monthly":
            last_day = calendar.monthrange(time.year, time.month)[1]
            days = [f"{i:02}" for i in range(1, last_day + 1)]
        else:
            raise ValueError("options.storage must be either 'daily' or 'monthly'.")
        return days

    for field in options.fields:
        for time in times:
            time = pd.Timestamp(time)
            days = get_days(time, options)

            request = {
                "product_type": [options.mode],
                "data_format": "netcdf",
                "download_format": "unarchived",
                "variable": [field],
                "pressure_level": options["pressure_levels"],
                "year": [f"{time.year:04}"],
                "month": [f"{time.month:02}"],
                "day": days,
                "time": [f"{i:02}" for i in range(0, 24)],
                "area": area,
            }
            daterange_str = format_daterange(options, time)
            local_path = f"{base_path}/{field}/{time.year}/{field}_era5_oper_"
            local_path += f"{short_format}_{daterange_str}.nc"
            requests[field].append(request)
            local_paths[field].append(local_path)

    return cds_name, requests, local_paths


def get_area(options):
    """Get the area for the CDS API request."""
    if options.longitude_range is None:
        max_lon = 180
        min_lon = -180
        logger.warning("No longitude range provided. ERA5 files cover all longitudes.")
    else:
        [min_lon, max_lon] = options.longitude_range
    if options.latitude_range is None:
        max_lat = 90
        min_lat = -90
        logger.warning("No latitude range provided. ERA5 files cover all latitudes.")
    else:
        [min_lat, max_lat] = options.latitude_range
    [max_lat, max_lon] = [int(np.ceil(coord)) for coord in [max_lat, max_lon]]
    [min_lat, min_lon] = [int(np.floor(coord)) for coord in [min_lat, min_lon]]
    if min_lon == -180 and max_lon == 180 and min_lat == -90 and max_lat == 90:
        return None
    else:
        return [max_lat, min_lon, min_lat, max_lon]


def get_area_string(area):
    """Get the area string for the CDS API request."""
    if area is None:
        return None

    # Convert a signed latitude or longitude to a string, e.g. 150E
    def format_lat(lat):
        return "0" if lat == 0 else f"{int(abs(lat))}{'N' if lat > 0 else 'S'}"

    def format_lon(lon):
        return "0" if lon == 0 else f"{int(abs(lon))}{'E' if lon > 0 else 'W'}"

    area_string = f"{format_lat(area[0])}_{format_lon(area[1])}_{format_lat(area[2])}_{format_lon(area[3])}"
    return area_string


def issue_cdsapi_requests(
    cds_name, requests, local_paths, enforce_timeout=False, timeout=5
):
    """Issue cdsapi requests. Note the wait client functionality doesn't appear to work
    yet."""

    def download_data(cds_name, request, local_path):
        c = cdsapi.Client(sleep_max=5, retry_max=1)
        response = c.retrieve(cds_name, request, local_path)
        return response

    def handle_request(cds_name, request, local_path):
        path = Path(local_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if enforce_timeout:

            def signal_handler(signum, frame):
                raise TimeoutError("Request timed out.")

            signal.signal(signal.SIGALRM, signal_handler)
            signal.alarm(timeout)
            try:
                download_data(cds_name, request, local_path)
            except TimeoutError:
                filename = Path(local_path).name
                message = f"Request for {filename} timed out after {timeout} seconds."
                logger.warning(message)
            finally:
                signal.alarm(0)
        else:
            download_data(cds_name, request, local_path)

    for field in requests.keys():
        for i in range(len(local_paths[field])):
            handle_request(cds_name, requests[field][i], local_paths[field][i])


def convert_era5(ds):
    """Convert ERA5 data."""
    if "level" in ds.coords:
        ds = ds.rename({"level": "pressure"})
    if "time_var" in ds.coords:
        ds = ds.rename({"time_var": "time"})
        logger.debug("Renamed time_var to time in era5 dataset.")
    if "r" in ds.data_vars:
        ds = ds.rename({"r": "relative_humidity"})
    if "t" in ds.data_vars:
        ds = ds.rename({"t": "temperature"})
    if "z" in ds.data_vars:
        ds = ds.rename({"z": "geopotential"})
    return ds


def update_dataset(time, input_record, track_options, dataset_options, grid_options):
    """Update ERA5 dataset."""

    utils.log_dataset_update(logger, dataset_options.name, time)

    start, end = get_hour_interval(time)
    filepaths = get_era5_filepaths(dataset_options, start, end, local=True)
    all_files_exist = all(
        Path(filepath).exists() for field in filepaths.values() for filepath in field
    )
    if not all_files_exist and dataset_options.attempt_download:
        logger.warning("One or more filepaths do not exist; attempting download.")
        cds_name, requests, local_paths = generate_cdsapi_requests(dataset_options)
        issue_cdsapi_requests(cds_name, requests, local_paths)

    lat = np.array(grid_options.latitude)
    lon = np.array(grid_options.longitude)
    # Expand the lat and lon ranges to include a buffer to ensure required gridpoints are included
    lat_range = (lat.min() - 0.25, lat.max() + 0.25)
    lon_range = (lon.min() - 0.25, lon.max() + 0.25)

    # Assume user has write privileges in the base_local directory
    logger.info(f"Subsetting {dataset_options.name} data.")
    with tempfile.TemporaryDirectory(dir=str(get_outputs_directory())) as tmp:
        for field in dataset_options.fields:
            for filepath in filepaths[field]:
                output_filename = Path(filepath).name
                logger.debug("Subsetting %s", output_filename)
                args = [filepath, f"{tmp}/{output_filename}.nc", start, end]
                args += [lat_range, lon_range]
                utils.call_ncks(*args)
        logger.debug("Merging files.")
        ds = xr.open_mfdataset(f"{tmp}/*.nc")
        logger.debug("Converting")
        ds = convert_era5(ds)
        input_record["dataset"] = ds.load()
