import os
import thuner.config as config

if __name__ == "__main__":
    # Set the output directory to scratch if not already set
    username = os.environ["USER"]
    # project = os.environ["PROJECT"]
    # Temporarily reassign to w40
    project = "w40"
    outputs_directory = f"/scratch/{project}/{username}/THUNER_output"
    # Also set this in the thuner config file
    config.set_outputs_directory(outputs_directory)
    print(f"Output directory is {config.get_outputs_directory()}")
