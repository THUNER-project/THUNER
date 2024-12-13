"""Handle the config file."""

import os
import json
from pathlib import Path


def create_user_config(output_directory=Path.home() / "THUNER_output"):
    # Determine the OS-specific path
    config_path = get_config_path()

    # Ensure the config directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Create a new config.json with initial settings
    write_config({"outputs_directory": str(output_directory)})

    return str(config_path)


def get_config_path():
    """Get the default path to the THUNER configuration file."""
    if os.name == "nt":  # Windows
        config_path = Path(os.getenv("LOCALAPPDATA")) / "THUNER" / "config.json"
    elif os.name == "posix":
        if "HOME" in os.environ:  # Linux/macOS
            config_path = Path.home() / ".config" / "THUNER" / "config.json"
        else:  # Fallback for other POSIX systems
            config_path = Path("/etc") / "THUNER" / "config.json"
    else:
        raise Exception("Unsupported operating system.")

    return config_path


def read_config(config_path):
    if config_path.exists():
        with config_path.open() as f:
            config = json.load(f)
            return config
    else:
        message = f"{config_path} not found. Ensure write_config has been run first."
        raise FileNotFoundError(message)


def set_outputs_directory(outputs_directory):
    """Set the THUNER outputs directory in the configuration file."""

    # Check if the outputs directory is a valid path
    Path(outputs_directory).mkdir(parents=True, exist_ok=True)

    config_path = get_config_path()
    config = read_config(config_path)
    config["outputs_directory"] = str(outputs_directory)
    write_config(config)


def write_config(config):
    config_path = get_config_path()
    with config_path.open("w") as f:
        json.dump(config, f, indent=4)
        print(f"Wrote configuration file to {config_path}")


def get_outputs_directory():
    """Load the THUNER outputs directory from the configuration file."""

    config_path = get_config_path()
    config = read_config(config_path)
    return Path(config["outputs_directory"])
