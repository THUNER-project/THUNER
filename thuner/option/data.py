"""Data options classes, convenience subclasses, and functions."""

import numpy as np
from pathlib import Path
from typing import Literal, Dict
from pydantic import Field, model_validator
import pandas as pd
from thuner.log import setup_logger
from thuner.config import get_outputs_directory
import thuner.option as option

import thuner.data.aura as aura
import thuner.data.gridrad as gridrad


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
    using their thuner, i.e. CF-Conventions, names, e.g. 'reflectivity'.""",
    "start_buffer": """Minutes before interval start time to include. Useful for 
    tagging datasets when one wants to record pre-storm ambient profiles.""",
    "end_buffer": """Minutes after interval end time to include. Useful for 
    tagging datasets when one wants to record post-storm ambient profiles.""",
}


class ConvertedOptions(option.utils.BaseOptions):
    """Converted options."""

    save: bool = Field(False, description="Whether to save the converted data.")
    load: bool = Field(False, description="Whether to load the converted data.")
    parent_converted: str | None = Field(None, description=_summary["parent_converted"])


default_parent_local = str(get_outputs_directory() / "input_data/raw")


class BaseDatasetOptions(option.utils.BaseOptions):
    """Base class for dataset options."""

    name: str = Field(..., description=_summary["name"])
    start: str | np.datetime64 = Field(..., description=_summary["start"])
    end: str | np.datetime64 = Field(..., description=_summary["end"])
    fields: list[str] | None = Field(None, description=_summary["fields"])
    parent_remote: str | None = Field(None, description=_summary["parent_remote"])
    parent_local: str | Path | None = Field(
        default_parent_local, description=_summary["parent_local"]
    )
    converted_options: ConvertedOptions = Field(
        ConvertedOptions(), description=_summary["converted_options"]
    )
    filepaths: list[str] | dict = Field(None, description=_summary["filepaths"])
    attempt_download: bool = Field(False, description=_summary["attempt_download"])
    deque_length: int = Field(2, description=_summary["deque_length"])
    use: Literal["track", "tag"] = Field("track", description=_summary["use"])
    start_buffer: int = Field(-120, description=_summary["start_buffer"])
    end_buffer: int = Field(0, description=_summary["end_buffer"])

    @model_validator(mode="after")
    def _check_parents(cls, values):
        if values.parent_remote is None and values.parent_local is None:
            message = "At least one of parent_remote and parent_local must be "
            message += "specified."
            raise ValueError(message)
        if values.converted_options.save or values.converted_options.load:
            if values.parent_converted is None:
                message = "parent_converted must be specified if saving or loading."
                raise ValueError(message)
        if values.attempt_download:
            if values.parent_remote is None | values.parent_local is None:
                message = "parent_remote and parent_local must both be specified if "
                message += "attempting to download."
                raise ValueError(message)
        return values

    @model_validator(mode="after")
    def _check_fields(cls, values):
        if values.use == "track" and len(values.fields) != 1:
            message = "Only one field should be specified if the dataset is used for "
            message += "tracking. Instead, created grouped objects. See thuner.option."
            raise ValueError(message)
        return values


_summary = {}
_summary["weighting_function"] = "Weighting function used by pyart to reconstruct the "
_summary["weighting_function"] += "grid from ODIM."


_summary["datasets"] = "List of dataset options."


class DataOptions(option.utils.BaseOptions):
    """Class for managing the options for all the datasets of a given run."""

    datasets: list[BaseDatasetOptions] = Field(..., description=_summary["datasets"])
    _dataset_lookup: Dict[str, BaseDatasetOptions] = {}

    @model_validator(mode="after")
    def initialize_dataset_lookup(cls, values):
        values._dataset_lookup = {d.name: d for d in values.datasets}
        return values

    def dataset_by_name(self, dataset_name: str) -> BaseDatasetOptions:
        return self._dataset_lookup.get(dataset_name)
