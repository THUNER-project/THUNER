"General utilities for the thuner package."
import inspect
from datetime import datetime
import yaml
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import xarray as xr
from numba import njit, int32, float32
from numba.typed import List
from scipy.interpolate import interp1d
import re
import os
import platform
from typing import Any, Dict, Literal
from pydantic import Field, model_validator, BaseModel, model_validator, PrivateAttr
import multiprocessing
from thuner.log import setup_logger
from thuner.config import get_outputs_directory


logger = setup_logger(__name__)


def convert_value(value: Any) -> Any:
    """
    Convenience function to convert options attributes to types serializable as yaml.
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [convert_value(v) for v in value.tolist()]
    if isinstance(value, BaseOptions):
        fields = value.model_fields.keys()
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
    BaseModel class, which is similar to python dataclasses but with type checking.
    """

    type: str = Field(None, description="Type of the options class.")

    # Allow arbitrary types in the options classes.
    class Config:
        arbitrary_types_allowed = True

    # Ensure that floats in all options classes are np.float32
    @model_validator(mode="after")
    def convert_floats(cls, values):
        for field in values.model_fields:
            if type(getattr(values, field)) is float:
                setattr(values, field, np.float32(getattr(values, field)))
        return values

    @model_validator(mode="after")
    def _set_type(cls, values):
        if values.type is None:
            values.type = cls.__name__
        return values

    def to_dict(self) -> Dict[str, Any]:
        fields = self.model_fields.keys()
        return {field: convert_value(getattr(self, field)) for field in fields}

    def to_yaml(self, filepath: str):
        Path(filepath).parent.mkdir(exist_ok=True, parents=True)
        with open(filepath, "w") as f:
            kwargs = {"default_flow_style": False, "allow_unicode": True}
            kwargs = {"sort_keys": False}
            yaml.dump(self.to_dict(), f, **kwargs)


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
    "deque_length": """Number of current/previous grids from this dataset to keep in memory. 
    Most tracking algorithms require at least two current/previous grids.""",
    "use": "Whether this dataset will be used for tagging or tracking.",
    "parent_converted": "Parent directory for converted data.",
    "fields": """List of dataset fields, i.e. variables, to use. Fields should be given 
    using their thuner, i.e. CF-Conventions, names, e.g. 'reflectivity'.""",
    "start_buffer": """Minutes before interval start time to include. Useful for 
    tagging datasets when one wants to record pre-storm ambient profiles.""",
    "end_buffer": """Minutes after interval end time to include. Useful for 
    tagging datasets when one wants to record post-storm ambient profiles.""",
}


class ConvertedOptions(BaseOptions):
    """Converted options."""

    save: bool = Field(False, description="Whether to save the converted data.")
    load: bool = Field(False, description="Whether to load the converted data.")
    parent_converted: str | None = Field(None, description=_summary["parent_converted"])


default_parent_local = str(get_outputs_directory() / "input_data/raw")


class BaseDatasetOptions(BaseOptions):
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
            print(f"Generated an exception: {exc}")


def initialize_process():
    """
    Use to set the initializer argument when creating a multiprocessing.Pool object.
    This will ensure that all processes in the pool are non-daemonic, and avoid the
    associated errors.
    """
    multiprocessing.current_process().daemon = False
