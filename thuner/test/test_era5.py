"""Test era5 setup and data download."""

import numpy as np
import thuner.data.era5 as era5
import thuner.grid as grid


def test_cdsapi():
    """
    Test the ECMWF CDS API. Note the wait feature of the cdsapi not yet
    implemented.
    """

    data_options = era5.data_options(start="2002-01-14", end="2002-01-15", fields=["u"])

    lats = np.arange(-14, -10 + 0.025, 0.025).tolist()
    lons = np.arange(129, 133 + 0.025, 0.025).tolist()

    grid_options = grid.create_options(name="geographic", latitude=lats, longitude=lons)
    era5.check_data_options(data_options)
    filepaths = era5.get_era5_filepaths(data_options, grid_options)
    cds_name, requests, local_paths = era5.generate_cdsapi_requests(data_options)
    assert filepaths == local_paths
    era5.issue_cdsapi_requests(cds_name, requests, local_paths, enforce_timeout=True)


if __name__ == "__main__":
    test_cdsapi()
