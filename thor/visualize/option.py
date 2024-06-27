"""Display options methods."""

from thor.log import setup_logger
from thor.option import save_options
from thor.config import get_outputs_directory

logger = setup_logger(__name__)


def boilerplate_options(
    name,
    save=False,
    parent_local=None,
):
    """
    Generate dataset display dictionary.

    Parameters
    ----------
    object : str
        The name of the object.
    save_local : bool, optional
        Whether to save the raw data locally; default is False.
    parent_local : str, optional
        The local parent directory; default is None.

    Returns
    -------
    options : dict
        Dictionary containing the display options.
    """

    if parent_local is None:
        parent_local = str(get_outputs_directory() / "visualize")

    options = {
        "name": name,
        "save": save,
        "parent_local": parent_local,
    }

    return options


def runtime_options(
    name,
    save=False,
    parent_local=None,
    figures=None,
    style="paper",
):
    """
    Generate dataset display dictionary.

    Parameters
    ----------
    object : str
        The name of the object.
    save : bool, optional
        Whether to save the raw data locally; default is False.
    parent_local : str, optional
        The local parent directory; default is None.

    Returns
    -------
    options : dict
        Dictionary containing the display options.
    """

    if figures is None:
        figures = {obj: {"style": style} for obj in ["mask"]}

    options = {
        **boilerplate_options(name, save, parent_local),
        "figures": figures,
    }

    return options


def save_display_options(
    display_options, filename=None, options_directory=None, append_time=False
):
    """TBA."""

    if options_directory is None:
        options_directory = get_outputs_directory() / "options/visualize_options"
    if filename is None:
        filename = "visualize_options"
        append_time = True
    save_options(display_options, filename, options_directory, append_time=append_time)
