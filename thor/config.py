"""Handle the config file."""

import os
import json
from pathlib import Path


def get_config_path():
    """Get the path to the THOR configuration file."""
    if os.name == "nt":  # Windows
        config_path = Path(os.getenv("LOCALAPPDATA")) / "THOR" / "config.json"
    elif os.name == "posix":
        if "HOME" in os.environ:  # Linux/macOS
            config_path = Path.home() / ".config" / "THOR" / "config.json"
        else:  # Fallback for other POSIX systems
            config_path = Path("/etc") / "THOR" / "config.json"
    else:
        raise Exception("Unsupported operating system.")

    return config_path


def read_config(config_path):
    if config_path.exists():
        with config_path.open() as f:
            config = json.load(f)
            return config
    else:
        raise FileNotFoundError("config.json not found.")


def set_outputs_directory(outputs_directory):
    """Set the THOR outputs directory in the configuration file."""

    # Check if the outputs directory is a valid path
    test = Path(outputs_directory).mkdir(parents=True, exist_ok=True)

    config_path = get_config_path()
    config = read_config(config_path)
    config["outputs_directory"] = str(outputs_directory)
    write_config(config)


def write_config(config):
    config_path = get_config_path()
    with config_path.open("w") as f:
        json.dump(config, f, indent=4)


def get_outputs_directory():
    """Load the THOR outputs directory from the configuration file."""

    config_path = get_config_path()
    config = read_config(config_path)
    return Path(config["outputs_directory"])
