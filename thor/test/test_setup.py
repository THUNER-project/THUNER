"""Test setup."""

import thor.data.aura as aura
import thor.data.access as access
import thor.grid as grid
import xarray as xr
import tempfile
from thor.log import setup_logger

logger = setup_logger(__name__)


def test_aura_setup():
    """
    Test aura data setup.
    """
    data_options = aura.create_options()
    aura.check_options(data_options)
    grid_options = grid.create_options()
    grid.check_options(grid_options)
    urls = aura.generate_operational_urls(data_options)[0]

    with tempfile.TemporaryDirectory() as tmp_dir:
        dataset = aura.setup_operational(data_options, grid_options, urls[0], tmp_dir)
    assert isinstance(dataset, xr.Dataset)


def test_access_setup():
    """
    Test access data setup.
    """
    data_options = access.create_options()
    access.check_options(data_options)
    grid_options = grid.create_options()
    grid.check_options(grid_options)
    urls = access.generate_access_urls(data_options)[0]
    url = urls[data_options["fields"][0]][0]
    dataset = xr.open_dataset(url)
    assert isinstance(dataset, xr.Dataset)
