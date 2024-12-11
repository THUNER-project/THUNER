"""Data options classes, convenience subclasses, and functions."""

import numpy as np
from pathlib import Path
from typing import Literal, Dict
from pydantic import Field, model_validator
import pandas as pd
from thuner.log import setup_logger
from thuner.config import get_outputs_directory
import thuner.option as option
import thuner.data.era5 as era5
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
}


class ConvertedOptions(option.BaseOptions):
    """Converted options."""

    save: bool = Field(False, description="Whether to save the converted data.")
    load: bool = Field(False, description="Whether to load the converted data.")
    parent_converted: str | None = Field(None, description=_summary["parent_converted"])


default_parent_local = str(get_outputs_directory() / "input_data/raw")


class BaseDatasetOptions(option.BaseOptions):
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


_summary = {
    "latitude_range": "Latitude range if accessing a directory of subsetted era5 data.",
    "longitude_range": "Longitude range if accessing a directory of subsetted era5 data.",
    "mode": "Mode of the data, e.g. reannalysis.",
    "data_format": "Data format, e.g. pressure-levels.",
    "pressure_levels": "Pressure levels; required if data_format is pressure-levels.",
    "storage": "Storage format of the data, e.g. monthly.",
}


class ERA5Options(BaseDatasetOptions):
    """Options for ERA5 datasets."""

    # Overwrite the default values from the base class. Note these objects are still
    # pydantic Fields. See https://github.com/pydantic/pydantic/issues/1141
    name: str = "era5_pl"
    parent_remote: str = "/g/data/rt52"
    use: Literal["track", "tag"] = "tag"

    # Define additional fields for era5
    latitude_range: list[float] = Field(
        [-90, 90], description=_summary["latitude_range"]
    )
    longitude_range: list[float] = Field(
        [-180, 180], description=_summary["longitude_range"]
    )
    mode: Literal["reanalysis"] = Field("reanalysis", description=_summary["mode"])
    data_format: Literal["pressure-levels", "single-levels"] = Field(
        "pressure-levels", description=_summary["data_format"]
    )
    pressure_levels: list[str] | list[float] | None = Field(
        None, description=_summary["pressure_levels"]
    )
    storage: str = Field("monthly", description=_summary["storage"])

    @model_validator(mode="after")
    def _check_ranges(cls, values):
        if values.latitude_range[0] < -90 or values.latitude_range[1] > 90:
            raise ValueError("Latitude range must be between -90 and 90.")
        if values.longitude_range[0] < -180 or values.longitude_range[1] > 180:
            raise ValueError("Longitude range must be between -180 and 180.")
        return values

    @model_validator(mode="after")
    def _check_defaults(cls, values):
        if values.data_format == "pressure-levels":
            if values.pressure_levels is None:
                values.pressure_levels = era5.era5_pressure_levels
                logger.debug(f"Assigning default era5 pressure levels.")
            values.pressure_levels = [str(level) for level in values.pressure_levels]
        if values.fields is None:
            message = f"Assigning default era5 {values.data_format} options name "
            message += "and fields."
            logger.debug(message)
            if values.data_format == "pressure-levels":
                values.name = "era5_pl"
                values.fields = ["u", "v", "z", "r", "t"]
            elif values.data_format == "single-levels":
                values.name = "era5_sl"
                values.fields = ["cape", "cin"]
        return values

    @model_validator(mode="after")
    def _check_times(cls, values):
        start_time = np.datetime64("1940-03-01T00:00:00")
        if np.datetime64(values.start) < start_time:
            raise ValueError(f"start must be {str(start_time)} or later.")
        return values

    @model_validator(mode="after")
    def _check_filepaths(cls, values):
        if values.filepaths is None:
            logger.info("Generating era5 filepaths.")
            values.filepaths = era5.get_era5_filepaths(values)
        if values.filepaths is None:
            raise ValueError("filepaths not provided or badly formed.")
        return values


class AURAOptions(BaseDatasetOptions):
    """Base options class for AURA datasets."""

    # Overwrite the default values from the base class. Note these objects are still
    # pydantic Fields. See https://github.com/pydantic/pydantic/issues/1141
    fields: list[str] = ["reflectivity"]

    # Define additional fields for CPOL
    level: Literal["1", "1b", "2"] = Field(..., description="Processing level.")
    data_format: Literal["grid_150km_2500m", "grid_70km_1000m"] = Field(
        ..., description="Data format."
    )
    range: float = Field(142.5, description="Range of the radar in km.")
    range_units: str = Field("km", description="Units of the range.")


class CPOLOptions(AURAOptions):
    """Options for CPOL datasets."""

    # Overwrite the default values from the base class. Note these objects are still
    # pydantic Fields. See https://github.com/pydantic/pydantic/issues/1141
    name: str = "cpol"
    fields: list[str] = ["reflectivity"]
    parent_remote: str = "https://dapds00.nci.org.au/thredds/fileServer/hj10"

    # Define additional fields for CPOL
    level: str = "1b"
    data_format: str = "grid_150km_2500m"
    version: str = Field("v2020", description="Data version.")

    @model_validator(mode="after")
    def _check_times(cls, values):
        if np.datetime64(values.start) < np.datetime64("1998-12-06T00:00:00"):
            raise ValueError("start must be 1998-12-06 or later.")
        if np.datetime64(values.end) > np.datetime64("2017-05-02T00:00:00"):
            raise ValueError("end must be 2017-05-02 or earlier.")
        return values

    @model_validator(mode="after")
    def _check_filepaths(cls, values):
        if values.filepaths is None:
            logger.info("Generating cpol filepaths.")
            values.filepaths = aura.get_cpol_filepaths(values)
        if values.filepaths is None:
            raise ValueError("filepaths not provided or badly formed.")
        return values


_summary = {}
_summary["weighting_function"] = "Weighting function used by pyart to reconstruct the "
_summary["weighting_function"] += "grid from ODIM."


class OperationalOptions(AURAOptions):
    """Options for CPOL datasets."""

    # Overwrite the default values from the base class. Note these objects are still
    # pydantic Fields. See https://github.com/pydantic/pydantic/issues/1141
    name: str = "operational"
    parent_remote: str = "https://dapds00.nci.org.au/thredds/fileServer/rq0"

    # Define additional fields for the operational radar
    level: str = "1"
    data_format: str = "ODIM"
    radar: int = Field(63, description="Radar ID number.")
    weighting_function: str = Field(
        "Barnes2", description=_summary["weighting_function"]
    )


class GridRadSevereOptions(BaseDatasetOptions):
    """Options for GridRad Severe datasets."""

    # Overwrite the default values from the base class. Note these objects are still
    # pydantic Fields. See https://github.com/pydantic/pydantic/issues/1141
    name: str = "gridrad"
    fields: list[str] = ["reflectivity"]
    parent_remote: str = "https://data.rda.ucar.edu"

    # Define additional fields for CPOL
    event_start: str = Field(..., description="Event start date.")
    dataset_id: str = Field("ds841.6", description="UCAR RDA dataset ID.")
    version: str = Field("v4_2", description="GridRad version.")
    obs_thresh: int = Field(2, description="Observation count threshold for filtering.")

    @model_validator(mode="after")
    def _check_times(cls, values):
        start_time = np.datetime64("2010-01-20T18:00:00")
        if np.datetime64(values.start) < start_time:
            raise ValueError(f"start must be {str(start_time)} or later.")
        return values

    @model_validator(mode="after")
    def _check_filepaths(cls, values):
        if values.filepaths is None:
            logger.info("Generating era5 filepaths.")
            values.filepaths = gridrad.get_gridrad_filepaths(values)
        if values.filepaths is None:
            raise ValueError("filepaths not provided or badly formed.")
        return values


AnyDatasetOptions = (
    ERA5Options | CPOLOptions | OperationalOptions | GridRadSevereOptions
)


class DataOptions(option.BaseOptions):
    """Class for managing the options for all the datasets of a given run."""

    datasets: list[AnyDatasetOptions] = Field(
        ..., description="List of dataset options."
    )
    _dataset_lookup: Dict[str, BaseDatasetOptions] = {}

    @model_validator(mode="after")
    def initialize_dataset_lookup(cls, values):
        values._dataset_lookup = {d.name: d for d in values.datasets}
        return values

    def dataset_by_name(self, dataset_name: str) -> BaseDatasetOptions:
        return self._dataset_lookup.get(dataset_name)
