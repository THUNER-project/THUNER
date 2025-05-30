"""
A package for detecting, tracking, and analyzing meteorological events in gridded
datasets.
"""

import sys
import os
import thuner.config as config


if sys.version_info < (3, 10):
    message = """
    Requires Python 3.10 or later. Check the dependencies, and consider installing
    thuner with a package manager like pip or conda."""
    raise ImportError(message)


if os.name == "nt":
    message = "Warning: Windows systems cannot run xESMF for regridding."
    message += "If you need regridding, consider using a Linux or MacOS system."
    print(message)


# Set version number
__version__ = "0.0.16"

welcome_message = f"""
Welcome to the Thunderstorm Event Reconnaissance (THUNER) package 
v{__version__}! This package is still in testing and development. Please visit 
github.com/THUNER-project/THUNER for examples, and to report issues or contribute.
 
THUNER is a flexible toolkit for performing multi-feature detection, 
tracking, tagging and analysis of events within meteorological datasets. 
The intended application is to convective weather events. For examples 
and instructions, see https://github.com/THUNER-project/THUNER and 
https://thuner.readthedocs.io/en/latest/. If you use THUNER in your research, consider 
citing the following papers;

Short et al. (2023), doi: 10.1175/MWR-D-22-0146.1
Raut et al. (2021), doi: 10.1175/JAMC-D-20-0119.1
Fridlind et al. (2019), doi: 10.5194/amt-12-2979-2019
Whitehall et al. (2015), doi: 10.1007/s12145-014-0181-3
Dixon and Wiener (1993), doi: 10.1175/1520-0426(1993)010<0785:TTITAA>2.0.CO;2
Leese et al. (1971), doi: 10.1175/1520-0450(1971)010<0118:AATFOC>2.0.CO;2
"""

if "THUNER_QUIET" not in os.environ:
    print(welcome_message)
    os.environ["THUNER_QUIET"] = "1"

try:
    config.read_config(config.get_config_path())
except FileNotFoundError:
    config_path = config.create_user_config()

import thuner.track as track
import thuner.parallel as parallel
import thuner.option as option
import thuner.default as default
import thuner.data as data
import thuner.analyze as analyze
import thuner.visualize as visualize


__all__ = [
    "option",
    "default",
    "track",
    "parallel",
    "data",
    "analyze",
    "visualize",
    "config",
    "utils",
]
