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


def get_outputs_directory():
    """Load the THOR outputs directory from the configuration file."""

    config_path = get_config_path()
    config = read_config(config_path)
    return Path(config["outputs_directory"])
