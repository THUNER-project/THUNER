import itertools
import numpy as np
import pandas as pd
import xarray as xr
import thor.data.option as option
from thor.config import get_outputs_directory
import thor.data.utils as utils
from thor.log import setup_logger
import thor.grid as grid

logger = setup_logger(__name__)


def gridrad_data_options(
    start="2010-01-203T18:00:00",
    end="2010-01-21T03:30:00",
    parent_remote="https://data.rda.ucar.edu",
    save_local=False,
    parent_local=str(get_outputs_directory() / "input_data/raw"),
    converted_options=None,
    filepaths=None,
    use="track",
    dataset_id="ds841.6",
    fields=None,
    version="v4_2",
):
    """
    Generate gridrad radar data options dictionary.
    """

    if fields is None:
        fields = ["reflectivity"]

    options = option.boilerplate_options(
        "gridrad",
        start,
        end,
        parent_remote,
        save_local,
        parent_local,
        converted_options,
        filepaths,
        use=use,
    )

    options.update({"fields": fields, "version": version, "dataset_id": dataset_id})

    return options


def check_data_options(options):
    """
    Check the input options.

    Parameters
    ----------
    options : dict
        Dictionary containing the input options.

    Returns
    -------
    options : dict
        Dictionary containing the input options.
    """
    valid_dataset_ids = ["ds841.6", "ds841.0", "ds841.1"]
    if options["dataset_id"] not in valid_dataset_ids:
        raise ValueError(f"Invalid dataset ID. Must be one of {valid_dataset_ids}.")


def convert_gridrad(time, input_record, dataset_options, grid_options):
    """Convert gridrad data to a standard format."""
    filepath = dataset_options["filepaths"][input_record["current_file_index"]]
    utils.log_convert(logger, dataset_options["name"], filepath)
    gridrad = xr.open_dataset(filepath)

    if time not in gridrad.time.values:
        raise ValueError(f"{time} not in {filepath}")

    gridrad = gridrad.rename(
        {
            "Latitude": "latitude",
            "Longitude": "longitude",
            "Altitude": "altitude",
        }
    )
    altitude = gridrad["altitude"] * 1000  # Convert to meters
    latitude = gridrad["latitude"]
    longitude = gridrad["longitude"]
    time = gridrad["time"]
    array_size = len(longitude) * len(latitude) * len(altitude)
    index = gridrad["index"].values
    da_dict = {}
    names_dict = {"reflectivity": "Reflectivity"}
    coords = [
        ("time", time.values),
        ("altitude", altitude.values),
        ("latitude", latitude.values),
        ("longitude", longitude.values),
    ]
    for field in dataset_options["fields"]:
        values = np.ones(array_size) * np.nan
        values[index] = gridrad[names_dict[field]].values
        shape = (len(time), len(altitude), len(latitude), len(longitude))
        values = values.reshape(shape)
        da = xr.DataArray(values, name=field, coords=coords)
        for coord in ["time", "altitude", "latitude", "longitude"]:
            da[coord].attrs = gridrad[coord].attrs
        da["altitude"].attrs["units"] = "m"
        del da["altitude"].attrs["delta"]
        da.attrs = gridrad[names_dict[field]].attrs
        da.attrs["long_name"] = field
        da_dict[field] = da
    ds = xr.Dataset(da_dict)
    ds.attrs = gridrad.attrs
    cell_areas = grid.get_cell_areas(ds.latitude.values, ds.longitude.values)
    ds["gridcell_area"] = (["latitude", "longitude"], cell_areas)
    ds["gridcell_area"].attrs.update(
        {"units": "km^2", "standard_name": "area", "valid_min": 0}
    )

    grid_options["latitude"] = latitude.values
    grid_options["longitude"] = longitude.values
    grid_options["altitude"] = altitude.values
    grid_options["geographic_spacing"] = [latitude.delta, longitude.delta]
    grid_options["shape"] = [len(latitude), len(longitude)]

    return ds


def update_dataset(time, input_record, dataset_options, grid_options):
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
    utils.log_dataset_update(logger, dataset_options["name"], time)
    conv_options = dataset_options["converted_options"]

    input_record["current_file_index"] += 1
    if conv_options["load"] is False:
        dataset = convert_gridrad(time, input_record, dataset_options, grid_options)
    else:
        dataset = xr.open_dataset(
            dataset_options["filepaths"][input_record["current_file_index"]]
        )
    if conv_options["save"]:
        utils.save_converted_dataset(dataset, dataset_options)
    input_record["dataset"] = dataset


# def get_severe_case_times():
#     """
#     Get the start and end dates for the cases in the GridRad-Severe dataset
#     (doi.org/10.5065/2B46-1A97).
#     """
#     base_url = "https://data.rda.ucar.edu/ds841.6/volumes"
#     years = np.arange(2010, 2011)
#     months = np.arange(1, 2)
#     days = np.arange(19, 22)
#     case_dates = []
#     for year, month, day in itertools.product(years, months, days):
#         url = f"{base_url}/{year:04}/{year:04}{month:02}{day:02}"
#         if utils.check_valid_url(url):
#             case_dates.append(f"{year:04}-{month:02}-{day:02}")
#     return case_dates


def generate_gridrad_filepaths(options):
    """
    Get the start and end dates for the cases in the GridRad-Severe dataset
    (doi.org/10.5065/2B46-1A97).
    """

    start = np.datetime64(options["start"]).astype("datetime64[s]")
    end = np.datetime64(options["end"]).astype("datetime64[s]")

    filepaths = []

    base_url = utils.get_parent(options)
    base_url += f"/{options['dataset_id']}"

    times = np.arange(start, end + np.timedelta64(10, "m"), np.timedelta64(10, "m"))
    times = pd.DatetimeIndex(times)

    for time in times:
        filepath = (
            f"{base_url}/{time.year}/{time.year}{time.month:02}{time.day:02}/"
            f"nexrad_3d_{options['version']}_"
            f"{time.year}{time.month:02}{time.day:02}T"
            f"{time.hour:02}{time.minute:02}00Z.nc"
        )
        filepaths.append(filepath)
    return sorted(filepaths)


def gridrad_grid_from_dataset(dataset, variable, time):
    grid = dataset[variable].sel(time=time)
    # preserved_attributes = dataset.attrs.keys() - ["field_names"]
    # for attr in ["origin_longitude", "origin_latitude", "instrument"]:
    #     grid.attrs[attr] = dataset.attrs[attr]
    return grid
