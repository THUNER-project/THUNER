"""Process ODIM data."""

import pyart
from thuner.log import setup_logger
import thuner.data.utils as utils
from pathlib import Path
import xarray as xr


logger = setup_logger(__name__)


def convert_odim(
    filepaths, data_options, grid_options, out_dir=None, out_filename=None, save=False
):
    """
    Convert ODIM files to xarray datasets, and save as netCDF if required.

    Parameters
    ----------
    filenames : list
        List of related filenames to be converted, e.g. all the files from a single day.
    data_options : dict
        Dictionary containing the data options.
    grid_options : dict
        Dictionary containing the grid options.
    out_dir : str, optional
        The directory where the converted files will be saved. Default is None.
    out_filename : str
        The name of the output file.
    save : bool, optional
        If True, the converted files will be saved as netCDF files in the specified directory.
        If False, the converted files will only be returned as xarray datasets without saving.
        Default is False.

    Returns
    -------
    dataset: xr.Dataset
        The THUNER compliant xarray dataset containing the converted ODIM files.
    """

    if out_dir is None:
        out_dir = Path(filepaths[0]).parent
    if out_filename is None:
        out_filename = Path(filepaths[0]).parent.name

    grid_shape = utils.get_pyart_grid_shape(grid_options)
    grid_limits = utils.get_pyart_grid_limits(grid_options)

    datasets = []
    for filepath in sorted(filepaths):
        try:
            logger.debug(f"Converting {filepath} to pyart.")
            dataset = pyart.aux_io.read_odim_h5(
                filepath, file_field_names=False, include_fields=data_options.fields
            )
            logger.debug(f"Gridding {filepath}.")
            dataset = pyart.map.grid_from_radars(
                dataset,
                grid_shape=grid_shape,
                grid_limits=grid_limits,
                weighting_function=data_options.weighting_function,
            )
            logger.debug(f"Converting {filepath} to xarray.")
            dataset = dataset.to_xarray()
            datasets.append(dataset)
        except Exception as e:
            logger.warning(f"Failed to convert {filepath}. {e}")

    dataset = xr.concat(datasets, dim="time")

    if save:
        dataset.to_netcdf(f"{out_dir}/{out_filename}")

    return dataset
