import sys
import os

# import thuner.config as config

if sys.version_info < (3, 10):
    message = """
    Requires Python 3.10 or later. Check the dependencies, and consider installing
    thuner with a package manager like pip or conda."""
    raise ImportError(message)

# Set version number
__version__ = "0.0.6"

welcome_message = f"""
Welcome to the Thunderstorm Event Reconnaissance (THUNER) package v{__version__}!
THUNER is a flexible toolkit for performing multi-feature detection, tracking, tagging
and analysis of events within meteorological datasets. The intended application is to
convective weather events. If you use this package in your research, consider citing 
the following papers;

Short et al. (2023), MWR, doi: 10.1175/MWR-D-22-0146.1
Raut et al. (2021), JAMC, doi: 10.1175/JAMC-D-20-0119.1
Fridlind et al. (2019), AMT, doi: 10.5194/amt-12-2979-2019
Dixon and Wiener (1993), JTECH, doi: 10.1175/1520-0426(1993)010<0785:TTITAA>2.0.CO;2
Leese et al. (1971), JAMC, doi: 10.1175/1520-0450(1971)010<0118:AATFOC>2.0.CO;2
"""

if "THUNER_QUIET" not in os.environ:
    print(welcome_message)

# Create config file if one does not exist
# try:
#     print("Reading configuration file.")
#     configuration = config.read_config(config.get_config_path())
# except FileNotFoundError:
#     print("Configuration file not found. Creating a new one.")
#     configuration = config.create_user_config()
