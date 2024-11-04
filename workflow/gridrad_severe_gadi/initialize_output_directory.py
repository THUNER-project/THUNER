import os
from pathlib import Path
import thor.config as config

if __name__ == "__main__":
    # Set the output directory to scratch if not already set
    username = os.environ["USER"]
    project = os.environ["PROJECT"]
    outputs_directory = f"/scratch/{project}/{username}/THOR_output"
    # Also set this in the thor config file
    config.set_outputs_directory(outputs_directory)
    print(f"Output directory is {config.get_outputs_directory()}")
