"""
General utilities for the thuner package. We use pydantic extensively. Every persistent
options model inherits BaseOptions, which adds improved serialization to the pydantic
BaseModel. Every transient runtime container inherits BaseModel directly.
"""

import os

# Check if system is unix-like, as xESMF is not supported on Windows
if os.name == "posix":
    import xesmf as xe
else:
    message = "Warning: Windows systems cannot run xESMF for regridding."
    message += "If you need regridding, consider using a Linux or MacOS system."
    print(message)

import inspect
import traceback
import importlib
from datetime import datetime
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import xarray as xr
import cv2
from numba import njit, int32, float32
from numba.typed import List
from scipy.interpolate import interp1d
import os
from typing import Any, Dict, Literal, Generator, Callable, Annotated
from pydantic import (
    Field,
    model_validator,
    PlainSerializer,
    BaseModel,
    ConfigDict,
)
from pydantic._internal._model_construction import ModelMetaclass
import multiprocessing
from thuner.log import setup_logger
from thuner.config import get_outputs_directory, get_zarr_store_name

logger = setup_logger(__name__)

__all__ = ["BaseOptions", "ConvertedOptions", "BaseDatasetOptions"]


DataObject = xr.DataArray | xr.Dataset


def function_to_string(value: Any) -> Any:
    """Serialise a callable as its fully qualified ``module.name`` form."""
    if value is None or isinstance(value, str):
        return value
    if inspect.isroutine(value):
        module = inspect.getmodule(value)
        return f"{module.__name__}.{value.__name__}"
    return value


def type_to_string(value: Any) -> Any:
    """Serialise a Python type as its fully qualified ``module.name`` form."""
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, type):
        return f"{value.__module__}.{value.__name__}"
    return value


def datetime_to_string(value: Any) -> Any:
    """Serialise a ``np.datetime64`` as a formatted UTC string."""
    if isinstance(value, np.datetime64):
        return format_time(value, filename_safe=False, day_only=False)
    return value


def ndarray_to_list(value: Any) -> Any:
    """Serialise an ``np.ndarray`` as a nested list."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


# Annotated field types: declare a field with one of these and serialization
# of the awkward "either a typed value or a string round-trip" is handled
# automatically — no per-class @field_serializer needed.
DatetimeField = Annotated[
    str | np.datetime64,
    PlainSerializer(datetime_to_string, return_type=str, when_used="always"),
]
CallableField = Annotated[
    Callable | str | None,
    PlainSerializer(function_to_string, return_type=str | None, when_used="always"),
]
TypeField = Annotated[
    type | str,
    PlainSerializer(type_to_string, return_type=str, when_used="always"),
]
NDArrayField = Annotated[
    np.ndarray | list | None,
    PlainSerializer(ndarray_to_list, return_type=list | None, when_used="always"),
]


class AutoTypeMeta(ModelMetaclass):
    def __new__(mcls, name, bases, namespace, **kwargs):
        # Skip the abstract root class itself
        if name != "BaseOptions":
            # Inject annotation if the subclass didn't set one explicitly
            annotations = namespace.setdefault("__annotations__", {})
            if "type" not in annotations:
                annotations["type"] = Literal[name]
                namespace["type"] = name  # default value
        # Let Pydantic build the actual model class
        return super().__new__(mcls, name, bases, namespace, **kwargs)


class BaseOptions(BaseModel, metaclass=AutoTypeMeta):
    """
    The base class for all options classes. This class is built on the pydantic
    BaseModel, which is similar to python dataclasses but with type checking.
    """

    type: Literal["BaseOptions"] = Field("BaseOptions")

    # Allow arbitrary types in the options classes, and reject unknown fields so that
    # typos/bad key-value pairs passed to a constructor raise instead of being ignored.
    model_config = ConfigDict(
        arbitrary_types_allowed=True, discriminator="type", extra="forbid"
    )

    def to_json(self, filepath: str, indent: int = 4):
        """Save the options to a JSON file."""
        Path(filepath).parent.mkdir(exist_ok=True, parents=True)
        with open(filepath, "w") as f:
            f.write(self.model_dump_json(indent=indent))

    @classmethod
    def from_json(cls, filepath: str):
        """Load options from a JSON file."""
        with open(filepath, "r") as f:
            return cls.model_validate_json(f.read())

    def revalidate(self):
        """Revalidate the model to ensure all fields are valid."""
        self.model_validate(self)

    def _change_defaults(self, **kwargs):
        """Change the default values of the model fields if not set by user."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                if key not in self.model_fields_set:
                    setattr(self, key, value)
            else:
                raise KeyError(f"{key} is not a valid option.")
        return self

    def model_summary(self) -> str:
        """Return a summary of the model fields and their descriptions."""
        summary_str = "Field Name: Type, Description\n"
        summary_str += "-------------------------------------\n"
        for name, info in self.__class__.model_fields.items():
            field_type = info.annotation if info.annotation else "Any"
            summary_str += f"{name}: {field_type}, {info.description}\n"
        return summary_str


class Retrieval(BaseOptions):
    """Class for retrieval. Generally a function and a dictionary of kwargs."""

    function: CallableField = Field(
        None,
        description="The function used to retrieve the attribute.",
    )
    keyword_arguments: dict = Field(
        {},
        description="Keyword arguments for the retrieval function.",
    )

    @model_validator(mode="after")
    def check_function(self):
        """Ensure that the function is callable, and available to thuner."""
        if isinstance(self.function, str):
            module_name, function_name = self.function.rsplit(".", 1)
            try:
                module = importlib.import_module(module_name)
                self.function = getattr(module, function_name)
            except ImportError:
                message = f"Could not import function {self.function}."
                raise ImportError(message)
            except AttributeError:
                message = f"Function {self.function} not found in {module_name}."
                raise AttributeError(message)
        return self


class ConvertedOptions(BaseOptions):
    """Converted options."""

    save: bool = Field(False, description="Whether to save the converted data.")
    load: bool = Field(False, description="Whether to load the converted data.")
    parent_converted: str | None = Field(
        str(get_outputs_directory() / "input_data/converted"),
        description="Parent directory for converted data.",
    )
    filepaths: Any | None = Field(
        None,
        description=(
            "Filepaths from which to save/load the converted data. If None, will be "
            "inferred from the input filepaths."
        ),
    )


class BaseDatasetOptions(BaseOptions):
    """Base class for dataset options."""

    def model_post_init(self, __context):
        """
        Set the base class post initialization behaviour. Currently this is used to
        build the default converted filepaths if not provided.
        """
        conv_options = self.converted_options
        if (conv_options.load or conv_options.save) and not conv_options.filepaths:
            conv_options.filepaths = self.get_converted_filepaths()

    name: str = Field(None, description="Name of the dataset.")
    start: DatetimeField = Field(..., description="Tracking start time.")
    end: DatetimeField = Field(..., description="Tracking end time.")
    fields: list[str] | None = Field(
        None,
        description=(
            "List of dataset fields, i.e. variables, to use. Fields should be given "
            "using their thuner, i.e. CF-Conventions, names, e.g. 'reflectivity'."
        ),
    )
    parent_remote: str | None = Field(
        None, description="Parent directory of the dataset on remote storage."
    )
    parent_local: str | Path | None = Field(
        str(get_outputs_directory() / "input_data/raw"),
        description="Parent directory of the dataset on local storage.",
    )
    converted_options: ConvertedOptions = Field(
        ConvertedOptions(),
        description="Options for saving and loading converted data.",
    )
    filepaths: Any | None = Field(
        None,
        description=(
            "Collection of filepaths for the dataset. If the dataset has multiple "
            "files for a given time, use a dictionary. If multiple dataset files "
            "are shipped in a zip or archive, use a list of tuples, where the "
            "first element is the zip path, and the second is the filename "
            "inside."
        ),
    )
    attempt_download: bool = Field(
        False, description="Whether to attempt to download the data."
    )
    deque_length: int = Field(
        2,
        description=(
            "Number of current/previous grids from this dataset to keep in memory. "
            "Most tracking algorithms require a 'next' grid, 'current' grid, and at "
            "least two previous grids."
        ),
    )
    use: Literal["track", "tag", "both"] = Field(
        "track",
        description="Whether this dataset will be used for tagging, tracking or both.",
    )
    start_buffer: int = Field(
        -120,
        description=(
            "Minutes before interval start time to include. Useful for tagging when "
            "one wants to record pre-storm ambient profiles."
        ),
    )
    end_buffer: int = Field(
        0,
        description=(
            "Minutes after interval end time to include. Useful for tagging when "
            "one wants to record post-storm ambient profiles."
        ),
    )
    reuse_regridder: bool = Field(
        False,
        description="Whether to save and reuse an xesmf regridder for this dataset.",
    )
    weights_filepath: str | None = Field(
        None,
        description=(
            "Filepath to where the xesmf regridder weights should be saved/loaded. "
            "Should generally be left as None and inferred during tracking."
        ),
    )
    regridder_from: str | None = Field(
        None,
        description=(
            "Name of another dataset whose regridder weights this dataset should reuse."
        ),
    )

    # Create basic functions for getting filepaths etc for already converted datasets.
    # These are overridden in the subclasses.
    def get_filepaths(self):
        """
        Return the subset of the input filepaths that is within the start and end time
        range.
        """
        logger.info(
            (
                "get_filepaths being called from base class BaseDatasetOptions. "
                "In this case get_filepaths just subsets the filepaths list "
                "provided by the user."
            )
        )
        if self.filepaths is None:
            raise ValueError("Filepaths field has not been set.")
        if len(self.filepaths) == 0:
            raise ValueError("Filepaths field is an empty list.")
        time_filepath_lookup = create_time_filepath_lookup(self.filepaths)
        start, end = np.datetime64(self.start), np.datetime64(self.end)
        times = np.array(sorted(list(set(time_filepath_lookup.keys()))))
        new_times = times[(times >= start) & (times <= end)]
        new_filepaths = []
        for time in new_times:
            new_filepaths.append(time_filepath_lookup[time])
        new_filepaths = sorted(list(set(new_filepaths)))
        return new_filepaths

    def get_converted_filepaths(self):
        """
        Get the filepaths for the converted datasets, based on the input filepaths.
        """
        parent_local = self.parent_local
        parent_converted = self.converted_options.parent_converted
        if self.filepaths is None:
            raise ValueError("Filepaths field has not been set.")
        elif all(isinstance(filepath, str) for filepath in self.filepaths):
            # Simplest case, a list of strings
            return [
                filepath.replace(parent_local, parent_converted)
                for filepath in self.filepaths
            ]
        else:
            raise NotImplementedError(
                "get_converted_filepaths not yet implemented for non-string filepaths."
                "Either provide filepaths as a list of strings, or overwrite this "
                "method in a subclass."
            )

    def update_input_record(self, time, input_record, track_options, grid_options):
        """
        Load the next file into the input record.

        Responsible only for loading/converting the file's dataset (which carries the
        domain and boundary masks as data variables) and stashing the per-file boundary
        coordinates. Rotating the per-time-step grid and boundary data into the deques
        is handled by ``thuner.data._update.update_track_input_records``.
        """
        time_str = format_time(time, filename_safe=False)
        logger.info(f"Updating {self.name} input record for {time_str}.")
        conv_options = self.converted_options
        input_record._current_file_index += 1
        filepath = self.filepaths[input_record._current_file_index]
        if conv_options.load is False:
            dataset, boundary_coords, _ = self.convert_dataset(
                time, filepath, track_options, grid_options
            )
            infer_grid_options(dataset, grid_options)
        else:
            dataset = xr.open_dataset(filepath, decode_timedelta=True)
            infer_grid_options(dataset, grid_options)
            if "domain_mask" in dataset:
                domain_mask = dataset["domain_mask"]
                boundary_coords = get_mask_boundary(domain_mask, grid_options)[0]
            else:
                boundary_coords = None
        # Save the dataset if necessary.
        if conv_options.save:
            save_converted_dataset(filepath, dataset, self)
        input_record.dataset = dataset
        # The boundary coordinates are constant over a file (derived from the 2D domain
        # mask) and aren't stored in the dataset, so stash them here for per-time-step
        # rotation in update_track_input_records.
        if boundary_coords is not None:
            input_record.next_boundary_coordinates = boundary_coords

    def grid_from_dataset(self, dataset, variable, time):
        """Get the grid from a generic/pre-converted dataset."""
        grid = dataset[variable].sel(time=time)
        # Copy radar location data to grid if present in dataset
        for attr in ["origin_longitude", "origin_latitude", "instrument"]:
            if attr in dataset.attrs:
                grid.attrs[attr] = dataset.attrs[attr]
        grid.attrs["field_name"] = variable
        return grid

    def convert_dataset(self, time, filepath, track_options, grid_options):
        """
        Convert the dataset. Note if the base class is used directly, the data is
        assumed to be already converted, and hence this function just opens the dataset.
        Function returns the converted dataset, and the boundary coordinates.
        Note the simple boundary coordinates are only used for visualization.
        """
        dataset = xr.open_dataset(filepath, decode_timedelta=True)
        infer_grid_options(dataset, grid_options)
        if time not in dataset.time.values:
            raise ValueError(f"{time} not in dataset time values.")
        if "domain_mask" in dataset:
            logger.info("Domain mask found in dataset. Getting boundary coordinates.")
            all_coords = get_mask_boundary(dataset.domain_mask, grid_options)
            boundary_coords, simple_boundary_coords, boundary_mask = all_coords
            dataset["boundary_mask"] = boundary_mask
        else:
            boundary_coords = None
            simple_boundary_coords = None

        return dataset, boundary_coords, simple_boundary_coords

    @model_validator(mode="after")
    def _check_name(self):
        """
        Check the name field has been created. This should be explicitly provided
        by the user or set in a subclass.
        """
        if self.name is None:
            raise ValueError("The 'name' field has not been set.")
        return self

    @model_validator(mode="after")
    def _check_parents(self):
        """Check the parents fields are correct."""
        if self.parent_remote is None and self.parent_local is None:
            message = "At least one of parent_remote and parent_local must be "
            message += "specified."
            raise ValueError(message)
        if self.converted_options.save or self.converted_options.load:
            if self.converted_options.parent_converted is None:
                message = "parent_converted must be specified if saving or loading."
                raise ValueError(message)
        if self.attempt_download:
            if self.parent_remote is None | self.parent_local is None:
                message = "parent_remote and parent_local must both be specified if "
                message += "attempting to download."
                raise ValueError(message)
        return self

    @model_validator(mode="after")
    def _check_fields(self):
        """Check whether fields compatible with other options."""
        if self.fields is None:
            message = "At least one field must be specified. Ensure fields is set "
            message += "explicitly, or set a default value in the appropriate subclass."
            raise ValueError(message)
        elif self.use == "track" and len(self.fields) != 1:
            message = "Only one field should be specified if the dataset is used for "
            message += "tracking. If you want to define objects built out of multiple "
            message += "components, use grouping. See "
            message += "thuner.option.track.GroupedObjectOptions, thuner.default"
            message += "and the gridrad.ipynb demo."
            raise ValueError(message)
        return self


class BaseHandler(BaseModel):
    """Base class for figure handlers defined in this module."""

    # Allow arbitrary types in the input record classes.
    model_config = ConfigDict(arbitrary_types_allowed=True)


class AttributeHandler(BaseHandler):
    """
    Class for handling the visualization of attributes, e.g. orientation, or groups of
    attributes visualized together, e.g. u, v.
    """

    name: str = Field(
        ...,
        description=(
            "The name of the attribute or attributes being handled, e.g. velocity."
        ),
    )
    axes: list[Any] = Field(
        [],
        description="The axes in which the attributes are to be visualized.",
    )
    label: str = Field(
        ...,
        description="The label to appear in legends etc for this attribute.",
    )
    attributes: list[str] = Field(
        ...,
        description="The names of the attributes to be visualized.",
    )
    filepath: str = Field(
        ...,
        description="Path to the attribute table inside the unified zarr store.",
    )
    method: Retrieval = Field(
        ...,
        description="The method used to visualize the attributes.",
    )
    legend_method: Retrieval | None = Field(
        None,
        description="The method used to create the legend artist for this attribute.",
    )
    quality_filepath: str | None = Field(
        None,
        description="The filepath of the quality control file.",
    )
    quality_variables: list[str] = Field(
        [],
        description="The quality control variables for this attribute.",
    )
    quality_method: Literal["any", "all"] = Field(
        "all",
        description=(
            "The logic used to determine if an object is of sufficient quality."
        ),
    )


def infer_grid_options(dataset: DataObject, grid_options):
    """Infer grid options from the dataset."""
    attrs = ["latitude", "longitude", "shape", "altitude"]
    if all(getattr(grid_options, attr) is not None for attr in attrs):
        # Return early if all grid options are already set.
        return

    logger.info("Grid options not set. Inferring from dataset.")
    if grid_options.name == "geographic":
        grid_options.latitude = dataset.latitude.values.tolist()
        grid_options.longitude = dataset.longitude.values.tolist()
        grid_options.shape = (len(dataset.latitude), len(dataset.longitude))
        lat_spacing = np.round(np.diff(dataset.latitude).flatten(), decimals=8)
        lon_spacing = np.round(np.diff(dataset.longitude).flatten(), decimals=8)
        lat_spacing = np.unique(lat_spacing).tolist()
        lon_spacing = np.unique(lon_spacing).tolist()
        if len(lat_spacing) == 1 and len(lon_spacing) == 1:
            grid_options.geographic_spacing = [lat_spacing[0], lon_spacing[0]]
        else:
            logger.warning("Latitude and longitude spacing not uniform.")
            grid_options.geographic_spacing = None
    elif grid_options.name == "cartesian":
        grid_options.y = dataset.y.values.tolist()
        grid_options.x = dataset.x.values.tolist()
        grid_options.shape = (len(dataset.y), len(dataset.x))
        y_spacing = np.unique(np.diff(grid_options.y).flatten()).tolist()
        x_spacing = np.unique(np.diff(grid_options.x).flatten()).tolist()
        if len(y_spacing) == 1 and len(x_spacing) == 1:
            grid_options.cartesian_spacing = [y_spacing[0], x_spacing[0]]
        else:
            logger.warning("x and y spacing not uniform.")
            grid_options.cartesian_spacing = None
        if "longitude" in dataset and "latitude" in dataset:
            grid_options.latitude = dataset.latitude.values.tolist()
            grid_options.longitude = dataset.longitude.values.tolist()
        else:
            logger.warning("No latitude or longitude coordinates found in dataset.")
    else:
        raise ValueError(f"Grid name {grid_options.name} not recognised.")

    if grid_options.altitude is None:
        if "altitude" in dataset:
            grid_options.altitude = dataset.altitude.values.tolist()
            alt_spacing = np.round(np.diff(dataset.altitude).flatten(), decimals=8)
            alt_spacing = np.unique(alt_spacing).tolist()
            if len(alt_spacing) == 1:
                grid_options.altitude_spacing = alt_spacing[0]
            else:
                logger.warning("Altitude spacing not uniform.")
                grid_options.altitude_spacing = None
        else:
            logger.warning("No altitude coordinates found in dataset.")


def save_converted_dataset(raw_filepath, dataset, dataset_options):
    """Save a converted dataset."""
    conv_options = dataset_options.converted_options
    if conv_options.save:
        parent = get_parent(dataset_options)
        parent_converted = conv_options.parent_converted
        if parent_converted is None:
            raise ValueError("No parent directory provided.")
        parent_converted = parent.replace("raw", "converted")
        conv_options.parent_converted = parent_converted
        converted_filepath = raw_filepath.replace(parent, parent_converted)
        if not Path(converted_filepath).parent.exists():
            Path(converted_filepath).parent.mkdir(parents=True)
        dataset.to_netcdf(converted_filepath, mode="w")
    return dataset


def get_parent(dataset_options: BaseDatasetOptions) -> str:
    """Get the appropriate parent directory."""
    conv_options = dataset_options.converted_options
    local = dataset_options.parent_local
    remote = dataset_options.parent_remote
    if conv_options.load:
        if conv_options.parent_converted is not None:
            parent = conv_options.parent_converted
        elif local is not None:
            conv_options.parent_converted = local.replace("raw", "converted")
            parent = conv_options.parent_converted
        elif conv_options.parent_remote is not None:
            conv_options.parent_converted = remote.replace("raw", "converted")
            parent = conv_options.parent_converted
        else:
            raise ValueError("Could not find/create parent_converted directory.")
    elif local is not None:
        parent = local
    elif remote is not None:
        parent = remote
    else:
        raise ValueError("No parent directory provided.")
    return parent


def store_path(output_directory, *parts):
    """Build a path into a run's zarr store, e.g. ``<output_directory>/output.zarr``.

    ``parts`` are appended as further path components inside the store, so e.g.
    ``store_path(out, "attributes", "mcs", "core")`` gives the path to that group.
    """
    return Path(output_directory).joinpath(get_zarr_store_name(), *parts)


def get_mask_boundary(mask, grid_options):
    """Get domain mask boundary using cv2."""

    lons = np.array(grid_options.longitude)
    lats = np.array(grid_options.latitude)
    mask_array = mask.fillna(0).values.astype(np.uint8)
    # Record the contours with all points, and with only the end points of each line
    # comprising the contour. The former is used to determine boundary overlap,
    # the latter makes plotting the boundary more efficient.
    contours = cv2.findContours(mask_array, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]
    simple_contours = cv2.findContours(
        mask_array, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )[0]
    boundary_coords = []
    boundary_pixels = []
    simple_boundary_coords = []

    def get_boundary_coords(contour):
        # Append the first point to the end to close the contour
        contour = np.append(contour, [contour[0]], axis=0)
        contour_rows = contour[:, :, 1].flatten()
        contour_cols = contour[:, :, 0].flatten()
        if grid_options.name == "cartesian":
            boundary_lats = lats[contour_rows, contour_rows]
            boundary_lons = lons[contour_rows, contour_cols]
        elif grid_options.name == "geographic":
            boundary_lats = lats[contour_rows]
            boundary_lons = lons[contour_cols]
        boundary_dict = {"latitude": boundary_lats, "longitude": boundary_lons}
        pixel_dict = {"row": contour_rows, "col": contour_cols}
        return boundary_dict, pixel_dict

    for contour in contours:
        boundary_dict, pixel_dict = get_boundary_coords(contour)
        boundary_coords.append(boundary_dict)
        boundary_pixels.append(pixel_dict)
    for contour in simple_contours:
        simple_boundary_coords.append(get_boundary_coords(contour)[0])

    boundary_mask = xr.zeros_like(mask).astype(bool)
    for pixels in boundary_pixels:
        boundary_mask.values[pixels["row"], pixels["col"]] = True

    return boundary_coords, simple_boundary_coords, boundary_mask


def generate_times(filepaths: list[str]) -> Generator[np.datetime64, None, None]:
    """Get times from dataset_options."""
    for filepath in sorted(filepaths):
        if not Path(filepath).exists():
            raise ValueError(f"{filepath} does not exist.")
        with xr.open_dataset(filepath, chunks={}, decode_timedelta=True) as ds:
            for time in ds.time.values:
                yield time


def generate_dataset_times(dataset_options: BaseDatasetOptions):
    """Get times from dataset_options."""
    start = np.datetime64(dataset_options.start)
    end = np.datetime64(dataset_options.end)
    filepaths = dataset_options.filepaths
    for filepath in sorted(filepaths):
        if not Path(filepath).exists():
            raise ValueError(f"{filepath} does not exist.")
        with xr.open_dataset(filepath, chunks={}, decode_timedelta=True) as ds:
            for time in ds.time.values:
                if start is not None and time < start:
                    continue
                if end is not None and time > end:
                    return  # files are sorted, so we can stop early
                yield time


def create_time_filepath_lookup(filepaths: list[str]) -> Dict[np.datetime64, str]:
    """Create a time: filepath dictionary from a list of filepaths."""
    if not isinstance(filepaths, list):
        raise TypeError("filepaths must be a list of strings")
    time_filepath_record = {}
    for filepath in sorted(filepaths):
        if not isinstance(filepath, str):
            raise TypeError(f"{filepath} is not a string")
        if not Path(filepath).exists():
            raise ValueError(f"{filepath} does not exist.")
        with xr.open_dataset(filepath, chunks={}, decode_timedelta=True) as ds:
            for time in ds.time.values:
                time_filepath_record[time] = filepath
    return time_filepath_record


def filter_arguments(func, args):
    """Filter arguments for the given attribute retrieval function."""
    sig = inspect.signature(func)
    return {key: value for key, value in args.items() if key in sig.parameters}


def hash_dictionary(dictionary):
    params_str = json.dumps(dictionary, sort_keys=True)
    hash_obj = hashlib.sha256()
    hash_obj.update(params_str.encode("utf-8"))
    return hash_obj.hexdigest()


def almost_equal(numbers, decimal_places=5):
    """Check if all numbers are equal to a certain number of decimal places."""
    rounded_numbers = [round(num, decimal_places) for num in numbers]
    return len(set(rounded_numbers)) == 1


def pad(array, left_pad=1, right_pad=1, kind="linear"):
    """Pad an array by extrapolating."""
    x = np.arange(len(array))
    f = interp1d(x, array, kind=kind, fill_value="extrapolate")
    return f(np.arange(-left_pad, len(array) + right_pad))


def time_in_dataset_range(time, dataset):
    """Check if a time is in a dataset."""

    if dataset is None:
        return False

    condition = time >= dataset.time.values.min() and time <= dataset.time.values.max()
    return condition


def get_hour_interval(time, interval=6, start_buffer=0, end_buffer=0):
    start = (time + np.timedelta64(start_buffer, "m")).astype("M8[h]")
    step = np.max([np.timedelta64(interval, "h"), np.timedelta64(end_buffer, "m")])
    return start, start + step


def format_time(time, filename_safe=True, day_only=False):
    """Format a np.datetime64 object as a string, truncating to seconds."""
    time_seconds = pd.DatetimeIndex([time]).round("s")[0]
    if day_only:
        time_str = time_seconds.strftime("%Y-%m-%d")
    else:
        time_str = time_seconds.strftime("%Y-%m-%dT%H:%M:%S")
    if filename_safe:
        time_str = time_str.replace(":", "").replace("-", "").replace("T", "_")
    return time_str


def now_str(filename_safe=True):
    """Return the current time as a string."""
    return format_time(datetime.now(), filename_safe=filename_safe, day_only=False)


def get_time_interval(next_grid, current_grid):
    """Get the time interval between two grids."""
    if current_grid is not None:
        time_interval = next_grid.time.values - current_grid.time.values
        time_interval = time_interval.astype("timedelta64[s]").astype(int)
        return time_interval
    else:
        return None


_USE_NUMBA = True


def conditional_jit(*jit_args, use_numba=_USE_NUMBA, **jit_kwargs):
    """
    A decorator that applies Numba's JIT compilation to a function if use_numba is True.
    Otherwise, it returns the original function. It also adjusts type aliases based on the
    usage of Numba.
    """

    def decorator(func):
        if use_numba:
            # Define type aliases for use with Numba
            globals()["int32"] = int32
            globals()["float32"] = float32
            globals()["List"] = List
            return njit(*jit_args, **jit_kwargs, cache=True)(func)
        else:
            # Define type aliases for use without Numba
            globals()["int32"] = int
            globals()["float32"] = float
            globals()["List"] = list
            return func

    return decorator


def logging_jit(func):
    """
    A decorator that logs a message before a function is compiled with Numba. This
    decorator should only be applied to the outermost function in a call stack that is
    being compiled with Numba.
    """

    def inner(*args, **kwargs):
        if getattr(func, "signatures", None) == []:
            module = inspect.getmodule(func)
            message = f"Compiling {module.__name__}.{func.__name__} with Numba."
            message += " Please wait."
            logger.info(message)
        result = func(*args, **kwargs)
        return result

    return inner


@conditional_jit(use_numba=_USE_NUMBA)
def meshgrid_numba(x, y):
    """
    Create a meshgrid-like pair of arrays for x and y coordinates.
    This function mimics the behaviour of np.meshgrid but is compatible with Numba.
    """
    m, n = len(y), len(x)
    X = np.empty((m, n), dtype=x.dtype)
    Y = np.empty((m, n), dtype=y.dtype)

    for i in range(m):
        X[i, :] = x
    for j in range(n):
        Y[:, j] = y

    return X, Y


@conditional_jit(use_numba=_USE_NUMBA)
def numba_boolean_assign(array, condition, value=np.nan):
    """
    Assign a value to an array based on a boolean condition.
    """
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if condition[i, j]:
                array[i, j] = value
    return array


@conditional_jit(use_numba=_USE_NUMBA)
def equirectangular(lat1_radians, lon1_radians, lat2_radians, lon2_radians):
    """
    Calculate the equirectangular distance between two points
    on the earth, where lat and lon are expressed in radians.
    """

    # Equirectangular approximation formula
    dlat = lat2_radians - lat1_radians
    dlon = lon2_radians - lon1_radians
    avg_lat = (lat1_radians + lat2_radians) / 2
    r = 6371e3  # Radius of Earth in metres
    x = dlon * np.cos(avg_lat)
    y = dlat
    return np.sqrt(x**2 + y**2) * r


@conditional_jit(use_numba=_USE_NUMBA)
def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance in metres between two points
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371e3  # Radius of earth in metres
    return c * r


def new_angle(angles):
    """
    Get the angle between the two angles that are farthest apart. All angles are
    provided/returned in radians.
    """
    if len(angles) == 0:
        return 0
    sorted_angles = np.sort(angles)
    gaps = np.diff(sorted_angles)
    circular_gap = 2 * np.pi - (sorted_angles[-1] - sorted_angles[0])
    gaps = np.append(gaps, circular_gap)
    max_gap_index = np.argmax(gaps)
    if max_gap_index == len(gaps) - 1:
        # Circular gap case
        angle1 = sorted_angles[-1]
        angle2 = sorted_angles[0] + 2 * np.pi
    else:
        angle1 = sorted_angles[max_gap_index]
        angle2 = sorted_angles[max_gap_index + 1]
    return (angle1 + angle2) / 2 % (2 * np.pi)


def circular_mean(angles, weights=None):
    """
    Calculate a weighted circular mean. Based on the scipy.stats.circmean function.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.circmean.html
    """
    if weights is None:
        weights = np.ones_like(angles)
    angles, weights = np.array(angles), np.array(weights)
    total_weight = np.sum(weights)
    # Convert the angles to complex numbers of unit length
    complex_numbers = np.exp(1j * angles)
    # Get the angle of the weighted sum of the complex numbers
    return np.angle(np.sum(weights * complex_numbers)) % (2 * np.pi)


def check_results(results):
    """Check pool results for exceptions."""
    for result in results:
        try:
            result.get(timeout=5 * 60)
        except Exception as exc:
            print(f"Generated an exception:")
            traceback.print_exc()


def initialize_process():
    """
    Use to set the initializer argument when creating a multiprocessing.Pool object.
    This will ensure that all processes in the pool are non-daemonic, and avoid the
    associated errors.
    """
    multiprocessing.current_process().daemon = False
