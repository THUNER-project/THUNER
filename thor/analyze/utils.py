"""Utility functions for analyzing thor output."""

from pathlib import Path
import yaml
import glob


def read_options(output_directory):
    """Read run options from yml files."""
    options_directory = Path(output_directory) / "options"
    options_filepaths = glob.glob(str(options_directory / "*.yml"))
    all_options = {}
    for filepath in options_filepaths:
        with open(filepath, "r") as file:
            options = yaml.safe_load(file)
            name = Path(filepath).stem
            all_options[name] = options
    return all_options
