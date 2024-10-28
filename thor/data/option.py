"""General data options functions."""

from thor.log import setup_logger
import thor.utils as utils
from thor.config import get_outputs_directory

# from thor.option import BaseOptions
# from pydantic import Field, model_validator


logger = setup_logger(__name__)

# Create convenience dictionary for options descriptions.
_summary = {
    "name": "Name of the dataset.",
    "start": "Tracking start time.",
    "end": "Tracking end time.",
    "parent_remote": "Data parent directory on remote storage.",
    "parent_local": "Data parent directory on local storage.",
    "converted_options": "Options for converted data.",
    "filepaths": "List of filepaths to used for tracking.",
    "attempt_download": "Whether to attempt to download the data.",
    "deque_length": """Number of previous grids from this dataset to keep in memory. 
    Most tracking algorithms require at least two previous grids.""",
    "use": "Whether this dataset will be used for tagging or tracking.",
    "parent_converted": "Parent directory for converted data.",
    "fields": """List of dataset fields, i.e. variables, to use. Fields should be given 
    using their thor, i.e. CF-Conventions, names, e.g. 'reflectivity'.""",
}


# class ConvertedOptions(BaseOptions):
#     """Converted options."""

#     save: bool = Field(False, description="Whether to save the converted data.")
#     load: bool = Field(False, description="Whether to load the converted data.")
#     parent_converted: str | None = Field(None, description=_summary["parent_converted"])


# class BaseDataOptions(BaseOptions):
#     """Base class for data options."""

#     name: str = Field(..., description=_summary["name"])
#     start: str = Field(..., description=_summary["start"])
#     end: str = Field(..., description=_summary["end"])
#     fields: list[str] = Field(..., description=_summary["fields"])
#     parent_remote: str | None = Field(None, description=_summary["parent_remote"])
#     parent_local: str | None = Field(None, description=_summary["parent_local"])
#     converted_options: ConvertedOptions = Field(
#         ConvertedOptions(), description=_summary["converted_options"]
#     )
#     filepaths: list[str] | None = Field(None, description=_summary["filepaths"])
#     attempt_download: bool = Field(False, description=_summary["attempt_download"])
#     deque_length: int = Field(2, description=_summary["deque_length"])
#     use: str = Field("track", description=_summary["use"])

#     @model_validator(mode="after")
#     def _check_parents(cls, values):
#         if values.parent_remote is None and values.parent_local is None:
#             message = "At least one of parent_remote and parent_local must be "
#             message += "specified."
#             raise ValueError(message)
#         if values.converted_options.save or values.converted_options.load:
#             if values.parent_converted is None:
#                 message = "parent_converted must be specified if saving or loading."
#                 raise ValueError(message)
#         if values.attempt_download:
#             if values.parent_remote is None | values.parent_local is None:
#                 message = "parent_remote and parent_local must both be specified if "
#                 message += "attempting to download."
#                 raise ValueError(message)
#         return values

#     @model_validator(mode="after")
#     def _check_fields(cls, values):
#         if values.use == "track" and len(values.fields) != 1:
#             message = "Only one field should be specified if the dataset is used for "
#             message += "tracking. Instead, created grouped objects. See thor.option."
#             raise ValueError(message)
#         return values


def boilerplate_options(
    name,
    start,
    end,
    parent_remote=None,
    save_local=False,
    parent_local=None,
    converted_options=None,
    filepaths=None,
    attempt_download=False,
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
        utils.check_component_options(converted_options)

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
    utils.save_options(
        data_options, filename, options_directory, append_time=append_time
    )


def check_boilerplate_options(dataset_options):
    if dataset_options["use"] == "track":
        fields = dataset_options["fields"]
        if fields is not None and len(fields) > 1:
            raise ValueError("Only one field can be specified for tracking.")
