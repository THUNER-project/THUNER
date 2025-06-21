"General utilities for the thuner package."

import inspect
import traceback
import importlib
import copy
from datetime import datetime
import yaml
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
import re
import os
import platform
from typing import Any, Dict, Literal, Generator, Callable
from pydantic import Field, model_validator, BaseModel, model_validator, ConfigDict
import multiprocessing
from thuner.log import setup_logger
from thuner.config import get_outputs_directory


logger = setup_logger(__name__)

__all__ = ["BaseOptions", "ConvertedOptions", "BaseDatasetOptions"]


DataObject = xr.DataArray | xr.Dataset


def convert_value(value: Any) -> Any:
    """
    Convenience function to convert options attributes to types serializable as yaml.
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [convert_value(v) for v in value.tolist()]
    if isinstance(value, BaseOptions):
        fields = value.__class__.model_fields.keys()
        return {field: convert_value(getattr(value, field)) for field in fields}
    if isinstance(value, dict):
        return {convert_value(k): convert_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [convert_value(v) for v in value]
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, type):
        # return full name of type, i.e. including module
        return f"{inspect.getmodule(value).__name__}.{value.__name__}"
    if type(value) is np.float32:
        return float(value)
    if inspect.isroutine(value):
        module = inspect.getmodule(value)
        return f"{module.__name__}.{value.__name__}"
    return value


class BaseOptions(BaseModel):
    """
    The base class for all options classes. This class is built on the pydantic
    BaseModel, which is similar to python dataclasses but with type checking.
    """

    type: str = Field(None, description="Type of the options, i.e. the subclass name.")

    # Allow arbitrary types in the options classes.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Ensure that floats in all options classes are np.float32
    @model_validator(mode="after")
    def convert_floats(cls, values):
        """Convert all floats to np.float32."""
        for field in values.__class__.model_fields:
            if type(getattr(values, field)) is float:
                setattr(values, field, np.float32(getattr(values, field)))
        return values

    @model_validator(mode="after")
    def _set_type(cls, values):
        """Set the type of the options class to the subclass name."""
        if values.type is None:
            values.type = cls.__name__
        return values

    def to_dict(self) -> Dict[str, Any]:
        """Convert the options to a dictionary."""
        fields = self.__class__.model_fields.keys()
        return {field: convert_value(getattr(self, field)) for field in fields}

    def to_yaml(self, filepath: str):
        """Save the options to a yaml file."""
        Path(filepath).parent.mkdir(exist_ok=True, parents=True)
        with open(filepath, "w") as f:
            kwargs = {"default_flow_style": False, "allow_unicode": True}
            kwargs = {"sort_keys": False}
            yaml.dump(self.to_dict(), f, **kwargs)

    def revalidate(self):
        """Revalidate the model to ensure all fields are valid."""
        self.model_validate(self)

    def _change_defaults(self, **kwargs):
        """Change the default values of the model fields if not set by user."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                if key not in self.__class__.model_fields_set:
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

    _desc = "The function used to retrieve the attribute."
    function: Callable | str | None = Field(None, description=_desc)
    _desc = "Keyword arguments for the retrieval function."
    keyword_arguments: dict = Field({}, description=_desc)

    @model_validator(mode="after")
    def check_function(cls, values):
        """Ensure that the function is callable, and available to thuner."""
        if isinstance(values.function, str):
            module_name, function_name = values.function.rsplit(".", 1)
            try:
                module = importlib.import_module(module_name)
                values.function = getattr(module, function_name)
            except ImportError:
                message = f"Could not import function {values.function}."
                raise ImportError(message)
            except AttributeError:
                message = f"Function {values.function} not found in {module_name}."
                raise AttributeError(message)
        return values


class ConvertedOptions(BaseOptions):
    """Converted options."""

    save: bool = Field(False, description="Whether to save the converted data.")
    load: bool = Field(False, description="Whether to load the converted data.")
    _desc = "Parent directory for converted data."
    _default_parent_converted = str(get_outputs_directory() / "input_data/converted")
    parent_converted: str | None = Field(_default_parent_converted, description=_desc)


class BaseDatasetOptions(BaseOptions):
    """Base class for dataset options."""

    name: str = Field(None, description="Name of the dataset.")
    start: str | np.datetime64 = Field(..., description="Tracking start time.")
    end: str | np.datetime64 = Field(..., description="Tracking end time.")
    _desc = "List of dataset fields, i.e. variables, to use. Fields should be given "
    _desc += "using their thuner, i.e. CF-Conventions, names, e.g. 'reflectivity'."
    fields: list[str] | None = Field(None, description=_desc)
    _desc = "Parent directory of the dataset on remote storage."
    parent_remote: str | None = Field(None, description=_desc)
    _desc = "Parent directory of the dataset on local storage."
    _default_parent_local = str(get_outputs_directory() / "input_data/raw")
    parent_local: str | Path | None = Field(_default_parent_local, description=_desc)
    _desc = "Options for saving and loading converted data."
    converted_options: ConvertedOptions = Field(ConvertedOptions(), description=_desc)
    _desc = "List of filepaths for the dataset."
    filepaths: list[str] | dict = Field(None, description=_desc)
    _desc = "Whether to attempt to download the data."
    attempt_download: bool = Field(False, description=_desc)
    _desc = "Number of current/previous grids from this dataset to keep in memory. "
    _desc += "Most tracking algorithms require a 'next' grid, 'current' grid, and at "
    _desc += "least two previous grids."
    deque_length: int = Field(2, description=_desc)
    _desc = "Whether this dataset will be used for tagging, tracking or both."
    use: Literal["track", "tag", "both"] = Field("track", description=_desc)
    _desc = "Minutes before interval start time to include. Useful for tagging when "
    _desc += "one wants to record pre-storm ambient profiles."
    start_buffer: int = Field(-120, description=_desc)
    _desc = "Minutes after interval end time to include. Useful for tagging when "
    _desc += "one wants to record post-storm ambient profiles."
    end_buffer: int = Field(0, description=_desc)

    # Create basic functions for getting filepaths etc for already converted datasets.
    # These are overridden in the subclasses.
    def get_filepaths(self):
        """
        Return the subset of the input filepaths that is within the start and end time
        range.
        """
        message = "get_filepaths being called from base class BaseDatasetOptions. "
        message += "In this case get_filepaths just subsets the filepaths list "
        message += "provided by the user."
        logger.info(message)
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

    def update_input_record(self, time, input_record, track_options, grid_options):
        """Update the input record."""
        time_str = format_time(time, filename_safe=False)
        logger.info(f"Updating {self.name} input record for {time_str}.")
        conv_options = self.converted_options
        input_record._current_file_index += 1
        filepath = self.filepaths[input_record._current_file_index]
        if conv_options.load is False:
            args = [time, filepath, track_options, grid_options]
            dataset, boundary_coords = self.convert_dataset(*args)[:2]
            infer_grid_options(dataset, grid_options)
        else:
            dataset = xr.open_dataset(filepath)
            infer_grid_options(dataset, grid_options)
            domain_mask = dataset["domain_mask"]
            boundary_coords = get_mask_boundary(domain_mask, grid_options)[0]
        # Save the dataset if necessary.
        if conv_options.save:
            save_converted_dataset(filepath, dataset, self)
        # Add the dataset to the imput record and update the boundary data.
        input_record.dataset = dataset
        self.update_boundary_data(dataset, input_record, boundary_coords)

    def update_boundary_data(self, dataset, input_record, boundary_coords):
        """Update the boundary data in the input record."""
        current_domain_mask = copy.deepcopy(input_record.next_domain_mask)
        current_boundary_coords = copy.deepcopy(input_record.next_boundary_coordinates)
        current_boundary_mask = copy.deepcopy(input_record.next_boundary_mask)

        input_record.domain_masks.append(current_domain_mask)
        input_record.boundary_coodinates.append(current_boundary_coords)
        input_record.boundary_masks.append(current_boundary_mask)

        input_record.next_domain_mask = dataset["domain_mask"]
        input_record.next_boundary_coordinates = boundary_coords
        input_record.next_boundary_mask = dataset["boundary_mask"]

    def grid_from_dataset(self, dataset, variable, time):
        """Get the grid from a generic/pre-converted dataset."""
        grid = dataset[variable].sel(time=time)
        # Copy radar location data to grid if present in dataset
        for attr in ["origin_longitude", "origin_latitude", "instrument"]:
            if attr in dataset.attrs:
                grid.attrs[attr] = dataset.attrs[attr]
        return grid

    def convert_dataset(self, time, filepath, track_options, grid_options):
        """
        Convert the dataset. Note if the base class is used directly, the data is
        assumed to be already converted, and hence this function just opens the dataset.
        Function returns the converted dataset, and the boundary coordinates.
        Note the simple boundary coordinates are only used for visualization.
        """
        dataset = xr.open_dataset(filepath)
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
    def _check_name(cls, values):
        """
        Check the name field has been created. This should be explicitly provided
        by the user or set in a subclass.
        """
        if values.name is None:
            raise ValueError("The 'name' field has not been set.")
        return values

    @model_validator(mode="after")
    def _check_parents(cls, values):
        """Check the parents fields are correct."""
        if values.parent_remote is None and values.parent_local is None:
            message = "At least one of parent_remote and parent_local must be "
            message += "specified."
            raise ValueError(message)
        if values.converted_options.save or values.converted_options.load:
            if values.converted_options.parent_converted is None:
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
        """Check whether fields compatible with other options."""
        if values.fields is None:
            message = "At least one field must be specified. Ensure fields is set "
            message += "explicitly, or set a default value in the appropriate subclass."
            raise ValueError(message)
        elif values.use == "track" and len(values.fields) != 1:
            message = "Only one field should be specified if the dataset is used for "
            message += "tracking. If you want to define objects built out of multiple "
            message += "components, use grouping. See "
            message += "thuner.option.track.GroupedObjectOptions, thuner.default"
            message += "and the gridrad.ipynb demo."
            raise ValueError(message)
        return values


class BaseHandler(BaseModel):
    """Base class for figure handlers defined in this module."""

    # Allow arbitrary types in the input record classes.
    model_config = ConfigDict(arbitrary_types_allowed=True)


class AttributeHandler(BaseHandler):
    """
    Class for handling the visualization of attributes, e.g. orientation, or groups of
    attributes visualized together, e.g. u, v.
    """

    _desc = "The name of the attribute or attributes being handled, e.g. velocity."
    name: str = Field(..., description=_desc)
    _desc = "The axes in which the attributes are to be visualized."
    axes: list[Any] = Field([], description=_desc)
    _desc = "The label to appear in legends etc for this attribute."
    label: str = Field(..., description=_desc)
    _desc = "The names of the attributes to be visualized."
    attributes: list[str] = Field(..., description=_desc)
    _desc = "The filepath to the attribute file, i.e. an attribute type csv file."
    filepath: str = Field(..., description=_desc)
    _desc = "The method used to visualize the attributes."
    method: Retrieval = Field(..., description=_desc)
    _desc = "The method used to create the legend artist for this attribute."
    legend_method: Retrieval | None = Field(None, description=_desc)
    _desc = "The filepath of the quality control file."
    quality_filepath: str | None = Field(None, description=_desc)
    _desc = "The quality control variables for this attribute."
    quality_variables: list[str] = Field([], description=_desc)
    _desc = "The logic used to determine if an object is of sufficient quality."
    quality_method: Literal["any", "all"] = Field("all", description=_desc)


def infer_grid_options(dataset: DataObject, grid_options):
    """Infer grid options from the dataset."""
    attrs = ["latitude", "longitude", "shape", "altitude"]
    if any(getattr(grid_options, attr) is None for attr in attrs):
        logger.info("Grid options not set. Inferring from dataset.")
        if grid_options.name == "geographic":
            grid_options.latitude = dataset.latitude.values.tolist()
            grid_options.longitude = dataset.longitude.values.tolist()
            grid_options.shape = [len(dataset.latitude), len(dataset.longitude)]
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
            grid_options.shape = [len(dataset.y), len(dataset.x)]
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
            message = "No altitude coordinates found in dataset."
            raise ValueError(message)


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


def get_mask_boundary(mask, grid_options):
    """Get domain mask boundary using cv2."""

    lons = np.array(grid_options.longitude)
    lats = np.array(grid_options.latitude)
    mask_array = mask.fillna(0).values.astype(np.uint8)
    args = [mask_array, cv2.RETR_LIST]
    # Record the contours with all points, and with only the end points of each line
    # comprising the contour. The former is used to determine boundary overlap,
    # the latter makes plotting the boundary more efficient.
    contours = cv2.findContours(*args, cv2.CHAIN_APPROX_NONE)[0]
    simple_contours = cv2.findContours(*args, cv2.CHAIN_APPROX_SIMPLE)[0]
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
        with xr.open_dataset(filepath, chunks={}) as ds:
            for time in ds.time.values:
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
        with xr.open_dataset(filepath, chunks={}) as ds:
            for time in ds.time.values:
                time_filepath_record[time] = filepath
    return time_filepath_record


def camel_to_snake(name):
    """
    Convert camel case string to snake case.

    Parameters:
    name (str): The camel case string to convert.

    Returns:
    str: The converted snake case string.
    """
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def filter_arguments(func, args):
    """Filter arguments for the given attribute retrieval function."""
    sig = inspect.signature(func)
    return {key: value for key, value in args.items() if key in sig.parameters}


class SingletonBase:
    """
    Base class for implementing singletons in python. See for instance the classic
    "Gang of Four" design pattern book for more information on the "singleton" pattern.
    The idea is that only one instance of a "singleton" class can exist at one time,
    making these useful for storing program state.

    Gamma et al. (1995), Design Patterns: Elements of Reusable Object-Oriented Software.

    Note however that if processes are created with, e.g., the multiprocessing module
    different processes will have different instances of the singleton. We can avoid
    this by explicitly passing the singleton instance to the processes.
    """

    # The base class now keeps track of all instances of singleton classes
    _instances = {}

    def __new__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super(SingletonBase, cls).__new__(cls)
            instance._initialize(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

    def _initialize(self, *args, **kwargs):
        """
        Initialize the singleton instance. This method should be overridden by subclasses.
        """
        pass


def format_string_list(strings):
    """
    Format a list of strings into a human-readable string.

    Parameters
    ----------
    strings : list of str
        List of strings to be formatted.

    Returns
    -------
    formatted_string : str
        The formatted string.
    """
    if len(strings) > 1:
        formatted_string = ", ".join(strings[:-1]) + " or " + strings[-1]
        return formatted_string
    elif strings:
        return strings[0]
    else:
        raise ValueError("strings must be an iterable of strings'.")


def create_hidden_directory(path):
    """Create a hidden directory."""
    if not Path(path).name.startswith("."):
        hidden_path = Path(path).parent / f".{Path(path).name}"
    else:
        hidden_path = Path(path)
    if hidden_path.exists() and hidden_path.is_file():
        message = f"{hidden_path} exists, but is a file, not a directory."
        raise FileExistsError(message)
    hidden_path.mkdir(parents=True, exist_ok=True)
    if platform.system() == "Windows":
        os.system(f'attrib +h "{hidden_path}"')
    else:
        os.makedirs(hidden_path, exist_ok=True)
    return hidden_path


def hash_dictionary(dictionary):
    params_str = json.dumps(dictionary, sort_keys=True)
    hash_obj = hashlib.sha256()
    hash_obj.update(params_str.encode("utf-8"))
    return hash_obj.hexdigest()


def drop_time(time):
    """Drop the time component of a datetime64 object."""
    return time.astype("datetime64[D]").astype("datetime64[s]")


def almost_equal(numbers, decimal_places=5):
    """Check if all numbers are equal to a certain number of decimal places."""
    rounded_numbers = [round(num, decimal_places) for num in numbers]
    return len(set(rounded_numbers)) == 1


def pad(array, left_pad=1, right_pad=1, kind="linear"):
    """Pad an array by extrapolating."""
    x = np.arange(len(array))
    f = interp1d(x, array, kind=kind, fill_value="extrapolate")
    return f(np.arange(-left_pad, len(array) + right_pad))


def print_keys(dictionary, indent=0):
    """Print the keys of a nested dictionary."""
    for key, value in dictionary.items():
        print("\t".expandtabs(4) * indent + str(key))
        if isinstance(value, dict):
            print_keys(value, indent + 1)


def check_component_options(component_options):
    """Check options for converted datasets and masks."""

    if not isinstance(component_options, dict):
        raise TypeError("component_options must be a dictionary.")
    if "save" not in component_options:
        raise KeyError("save key not found in component_options.")
    if "load" not in component_options:
        raise KeyError("load key not found in component_options.")
    if not isinstance(component_options["save"], bool):
        raise TypeError("save key must be a boolean.")
    if not isinstance(component_options["load"], bool):
        raise TypeError("load key must be a boolean.")


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


use_numba = True


def conditional_jit(use_numba=True, *jit_args, **jit_kwargs):
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
            return njit(*jit_args, **jit_kwargs)(func)
        else:
            # Define type aliases for use without Numba
            globals()["int32"] = int
            globals()["float32"] = float
            globals()["List"] = list
            return func

    return decorator


@conditional_jit(use_numba=use_numba)
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


@conditional_jit(use_numba=use_numba)
def numba_boolean_assign(array, condition, value=np.nan):
    """
    Assign a value to an array based on a boolean condition.
    """
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if condition[i, j]:
                array[i, j] = value
    return array


@conditional_jit(use_numba=use_numba)
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


@conditional_jit(use_numba=use_numba)
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


def save_options(options, filename=None, options_directory=None, append_time=False):
    """Save the options to a yml file."""
    if filename is None:
        filename = now_str()
        append_time = False
    else:
        filename = Path(filename).stem
    if append_time:
        filename += f"_{now_str()}"
    filename += ".yml"
    if options_directory is None:
        options_directory = get_outputs_directory() / "options"
    if not options_directory.exists():
        options_directory.mkdir(parents=True)
    filepath = options_directory / filename
    logger.debug("Saving options to %s", options_directory / filename)
    with open(filepath, "w") as outfile:
        yaml.dump(
            options,
            outfile,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )


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


def circular_variance(angles, weights=None):
    """
    Calculate a weighted circular variance. Based on the scipy.stats.circvar function.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.circvar.html
    """
    if weights is None:
        weights = np.ones_like(angles)
    angles, weights = np.array(angles), np.array(weights)
    # Convert the angles to complex numbers of unit length
    complex_numbers = np.exp(1j * angles)
    total_weight = np.sum(weights)
    if total_weight == 0:
        return np.nan
    complex_sum = np.sum(weights * complex_numbers / total_weight)
    return 1 - np.abs(complex_sum)


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
