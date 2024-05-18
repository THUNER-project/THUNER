"""Methods for creating and modifying default tagging options."""

from pathlib import Path
from thor.log import setup_logger
from thor.option import save_options

logger = setup_logger(__name__)


def boilerplate_options(name, dataset, time_method="linear", space_method="linear"):
    """Boilerplate tagging options.

    Parameters
    ----------
    name : str
        Name of object to track.
    dataset : str
        Name of dataset to draw tags from.
    time_method : str, optional
        Method for looking up time values; default is "linear"
        interpolation.
    space_method : str, optional
        Method for looking up space values; default is "linear"
        interpolation.

    Returns
    -------
    options : dict
        Dictionary of boilerplate tagging options.
    """

    options = {
        "name": name,
        "dataset": dataset,
        "time_method": time_method,
        "space_method": space_method,
    }
    return options


def save_tag_options(
    tag_options, filename=None, options_directory=None, append_time=False
):
    """TBA."""

    if options_directory is None:
        options_directory = Path(__file__).parent / "options/tag_options"
    if filename is None:
        filename = tag_options["name"]
        append_time = True
    save_options(tag_options, filename, options_directory, append_time=append_time)
