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
    figure_types=["mask", "match"],
    style="paper",
    animate=True,
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

    figures = {
        fig_type: {"style": style, "animate": animate} for fig_type in figure_types
    }

    options = {
        **boilerplate_options(name, save, parent_local),
        "figures": figures,
    }

    return options


def horizontal_attribute_options(
    name,
    save=True,
    parent_local=None,
    attributes=None,
    quality_control=True,
    fields=None,
    extent=None,
    template=None,
    style="paper",
):
    """Default options for horizontal attribute visualization."""

    # Set the default object attributes to display
    if attributes is None:
        attributes = ["velocity", "relative_velocity", "shear", "ambient"]
    # Set the default dataset fields to display
    if fields is None:
        fields = ["reflectivity"]

    options = {
        **boilerplate_options(name, save, parent_local),
        "attributes": attributes,
        "quality_control": quality_control,
        "fields": fields,
        "extent": extent,
        "template": template,
        "style": style,
    }
    return options


def save_display_options(
    display_options, options_directory=None, filename="visualize", append_time=False
):
    """Save the display options."""

    if options_directory is None:
        options_directory = get_outputs_directory() / "options/visualize"
    save_options(display_options, filename, options_directory, append_time=append_time)
