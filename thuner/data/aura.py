"""Process AURA data."""

import copy
from urllib.parse import urlparse
import xarray as xr
import xesmf as xe
import numpy as np
import pandas as pd
from thuner.log import setup_logger
from thuner.data.odim import convert_odim
import thuner.data.utils as utils
import thuner.grid as grid


logger = setup_logger(__name__)


def get_cpol_filepaths(options):
    """
    Generate CPOL fielpaths.
    """

    start = np.datetime64(options.start).astype("datetime64[s]")
    end = np.datetime64(options.end).astype("datetime64[s]")

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
    # elif options.level == "2":
    #     times = np.arange(
    #         start.astype("datetime64[D]"),
    #         end.astype("datetime64[D]") + np.timedelta64(1, "D"),
    #         np.timedelta64(1, "D"),
    #     )
    #     times = pd.DatetimeIndex(times)

    #     base_url += f"/cpol_level_2/v{options.version}/{options.data_format}"
    #     try:
    #         variable = options.variable
    #         if variable == "equivalent_reflectivity_factor":
    #             variable_short = "reflectivity"
    #     except KeyError:
    #         variable = "equivalent_reflectivity_factor"
    #         variable_short = "reflectivity"

    #     base_url += f"/{variable}"

    #     for time in times:
    #         url = f"{base_url}/twp1440cpol.{variable_short}.c1"
    #         url += f".{time.year}{time.month:02}{time.day:02}.nc"
    #         filepaths.append(filepath)

    return sorted(filepaths)


def get_operational_filepaths(options):
    """
    Generate operational radar URLs from input options dictionary. Note level 1 are
    zipped ODIM files, level 1b are zipped netcdf files.

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


def setup_operational(data_options, grid_options, url, directory):
    """
    Setup operational radar data for a given date.

    Parameters
    ----------
    options : dict
        Dictionary containing the input options.
    url : str
        The URL where the radar data can be found.
    directory : str
        Where to extract the zip file and save the netCDF.

    Returns
    -------
    dataset : object
        The processed radar data.
    """

    if "http" in urlparse(url).scheme:
        filepath = utils.download_file(url, directory)
    else:
        filepath = url
    extracted_filepaths = utils.unzip_file(filepath)[0]
    if data_options.level == "1":
        dataset = convert_odim(
            extracted_filepaths,
            data_options,
            grid_options,
            out_dir=directory,
        )
    elif data_options.level == "1b":
        dataset = utils.consolidate_netcdf(
            extracted_filepaths, fields=data_options.fields, concat_dim="time"
        )

    return dataset


def get_cpol(time, input_record, dataset_options, grid_options):
    """Update the CPOL input_record for tracking."""
    filepath = dataset_options.filepaths[input_record["current_file_index"]]
    ds, boundary_coords, simple_boundary_coords = convert_cpol(
        time, filepath, dataset_options, grid_options
    )

    # Set data outside instrument range to NaN
    keys = ["current_domain_mask", "current_boundary_coordinates"]
    keys += ["current_boundary_mask"]
    if any(input_record[k] is None for k in keys):
        # Get the domain mask and domain boundary. Note this is the region where data
        # exists, not the detected object masks from the detect module.
        input_record["current_domain_mask"] = ds["domain_mask"]
        input_record["current_boundary_coordinates"] = boundary_coords
        input_record["current_boundary_mask"] = ds["boundary_mask"]
    else:
        domain_mask = copy.deepcopy(input_record["current_domain_mask"])
        boundary_mask = copy.deepcopy(input_record["current_boundary_mask"])
        boundary_coords = copy.deepcopy(input_record["current_boundary_coordinates"])
        input_record["previous_domain_masks"].append(domain_mask)
        input_record["previous_boundary_coordinates"].append(boundary_coords)
        input_record["previous_boundary_masks"].append(boundary_mask)
        # Note for AURA data the domain mask is calculated using a fixed range
        # (e.g. 150 km), which is constant for all times. Therefore, the mask is not
        # updated for each new file. Contrast this with, for instance, GridRad, where a
        # new mask is calculated for each time step based on the altitudes of the
        # objects being detected, and the required threshold on number of observations.

    return ds


def convert_cpol(time, filepath, dataset_options, grid_options):
    """Convert CPOL data to a standard format."""
    utils.log_convert(logger, dataset_options.name, filepath)
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
        if grid_options.latitude is None or grid_options.longitude is None:
            # If the lat/lon of the new grid were not specified, construct from spacing
            spacing = grid_options.geographic_spacing
            message = f"Creating new geographic grid with spacing {spacing[0]} m, {spacing[1]} m."
            logger.info(message)
            if spacing is None:
                raise ValueError("Spacing cannot be None if latitude/longitude None.")
            old_lats = cpol["latitude"].values
            old_lons = cpol["longitude"].values
            args = [old_lats, old_lons, spacing[0], spacing[1]]
            latitude, longitude = grid.new_geographic_grid(*args)
            grid_options.latitude = latitude
            grid_options.longitude = longitude
        ds = xr.Dataset({dim: ([dim], getattr(grid_options, dim)) for dim in dims})
        regrid_options = {"periodic": False, "extrap_method": None}
        regridder = xe.Regridder(cpol, ds, "bilinear", **regrid_options)
        ds = regridder(cpol)
        for var in ds.data_vars:
            if var in cpol.data_vars:
                ds[var].attrs = cpol[var].attrs
        for coord in ds.coords:
            ds[coord].attrs = cpol[coord].attrs
        ds.attrs.update(cpol.attrs)
        ds.attrs["history"] += f", regridded using xesmf on " f"{np.datetime64('now')}"

    elif grid_options.name == "cartesian":
        dims = ["y", "x"]
        # Interpolate vertically
        ds = cpol.interp(altitude=grid_options.altitude, method="linear")
        grid_options.latitude = ds["latitude"].values
        grid_options.longitude = ds["longitude"].values
        if grid_options.x is None or grid_options.y is None:
            grid_options.x = ds["x"].values
            grid_options.y = ds["y"].values
        x_spacing = ds["x"].values[1:] - ds["x"].values[:-1]
        y_spacing = ds["y"].values[1:] - ds["y"].values[:-1]
        if np.unique(x_spacing).size > 1 or np.unique(y_spacing).size > 1:
            raise ValueError("x and y must have constant spacing.")
        grid_options.cartesian_spacing = [y_spacing[0], x_spacing[0]]

    # Define grid shape and gridcell areas
    grid_options.shape = [len(ds[dims[0]].values), len(ds[dims[1]].values)]
    cell_areas = grid.get_cell_areas(grid_options)
    ds["gridcell_area"] = (dims, cell_areas)
    ds["gridcell_area"].attrs.update(
        {"units": "km^2", "standard_name": "area", "valid_min": 0}
    )
    if grid_options.altitude is None:
        grid_options.altitude = ds["altitude"].values
    else:
        ds = ds.interp(altitude=grid_options.altitude, method="linear")
    # THUNER convention uses longitude in the range [0, 360]
    ds["longitude"] = ds["longitude"] % 360

    # Get the domain mask and domain boundary. Note this is the region where data
    # exists, not the detected object masks from the detect module.
    domain_mask = utils.mask_from_range(ds, dataset_options, grid_options)
    boundary_coords, simple_boundary_coords, boundary_mask = utils.get_mask_boundary(
        domain_mask, grid_options
    )
    ds["domain_mask"] = domain_mask
    ds["boundary_mask"] = boundary_mask

    ds = utils.apply_mask(ds, grid_options)

    return ds, boundary_coords, simple_boundary_coords


def convert_operational():
    """TBA."""
    ds = None
    return ds


def update_dataset(time, input_record, track_options, dataset_options, grid_options):
    """
    Update an aura dataset.

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
    if conv_options.load is False:
        if dataset_options.name == "cpol":
            dataset = get_cpol(time, input_record, dataset_options, grid_options)
        elif dataset_options.name == "operational":
            dataset = convert_operational(
                time, input_record, dataset_options, grid_options
            )
    else:
        dataset = xr.open_dataset(
            dataset_options.filepaths[input_record["current_file_index"]]
        )
    if conv_options.save:
        utils.save_converted_dataset(dataset, dataset_options)
    input_record["dataset"] = dataset


def cpol_grid_from_dataset(dataset, variable, time):
    grid = dataset[variable].sel(time=time)
    for attr in ["origin_longitude", "origin_latitude", "instrument"]:
        grid.attrs[attr] = dataset.attrs[attr]
    return grid
