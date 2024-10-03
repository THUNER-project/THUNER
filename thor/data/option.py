"""General data options functions."""

from thor.log import setup_logger
from thor.option import save_options
from thor.utils import check_component_options
from thor.config import get_outputs_directory


logger = setup_logger(__name__)


def boilerplate_options(
    name,
    start,
    end,
    parent_remote=None,
    save_local=False,
    parent_local=None,
    converted_options=None,
    filepaths=None,
    attempt_download=True,
    deque_length=2,
    use="track",
):
    """
    Generate dataset options dictionary.

    Parameters
    ----------
    name : str, optional
        The name of the dataset; default is "operational".
    start : datetime.datetime, optional
        The start time of the dataset; default is None.
    end : datetime.datetime, optional
        The end time of the dataset; default is None.
    parent_remote : str, optional
        The remote parent directory; default is None.
    save_local : bool, optional
        Whether to save the raw data locally; default is False.
    parent_local : str, optional
        The local parent directory; default is None.
    options_converted : dict, optional
        Dictionary containing the converted data options; default is None.
    options_mask : dict, optional
        Dictionary containing the mask options; default is None.
    filepaths : list, optional
        List of filepaths to the files to be converted; default is None.
    attempt_download : bool, optional
        Whether to attempt to download the data; default is True.
    deque_length : int, optional
        The length of the deque; default is 2.
    use : str, optional
        The use of the dataset; default is "track".

    Returns
    -------
    options : dict
        Dictionary containing the data options.
    """

    if converted_options is None:
        converted_options = {"save": False, "load": False, "parent_converted": None}
    else:
        check_component_options(converted_options)

    options = {
        "name": name,
        "start": start,
        "end": end,
        "parent_remote": parent_remote,
        "save_local": save_local,
        "parent_local": parent_local,
        "converted_options": converted_options,
        "filepaths": filepaths,
        "attempt_download": attempt_download,
        "deque_length": deque_length,
        "use": use,
    }

    return options


def save_data_options(
    data_options, options_directory=None, filename="data", append_time=False
):
    """TBA."""

    if options_directory is None:
        options_directory = get_outputs_directory() / "options/data"
    save_options(data_options, filename, options_directory, append_time=append_time)


def check_boilerplate_options(dataset_options):
    if dataset_options["use"] == "track":
        fields = dataset_options["fields"]
        if fields is not None and len(fields) > 1:
            raise ValueError("Only one field can be specified for tracking.")
