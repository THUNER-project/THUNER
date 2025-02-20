from pathlib import Path
import shutil
import numpy as np
import thuner.data as data
import thuner.default as default
import thuner.track.track as track
import thuner.option as option
import thuner.data.synthetic as synthetic

notebook_name = "synthetic_demo.ipynb"

# Parent directory for saving outputs
base_local = Path.home() / "THUNER_output"
start = "2005-11-13T00:00:00"
end = "2005-11-13T02:00:00"

output_parent = base_local / "runs/synthetic/geographic"
if output_parent.exists():
    shutil.rmtree(output_parent)
options_directory = output_parent / "options"
options_directory.mkdir(parents=True, exist_ok=True)

# Create a grid
lat = np.arange(-14, -6 + 0.025, 0.025).tolist()
lon = np.arange(128, 136 + 0.025, 0.025).tolist()
grid_options = option.grid.GridOptions(name="geographic", latitude=lat, longitude=lon)
grid_options.to_yaml(options_directory / "grid.yml")

# Initialize synthetic objects
starting_objects = []
for i in range(5):
    obj = synthetic.create_object(
        time=start,
        center_latitude=np.mean(lat),
        center_longitude=lon[(i + 1) * len(lon) // 6],
        direction=-np.pi / 4 + i * np.pi / 6,
        speed=30 - 4 * i,
        horizontal_radius=5 + 4 * i,
    )
    starting_objects.append(obj)
# Create data options dictionary
synthetic_options = data.synthetic.SyntheticOptions(starting_objects=starting_objects)
data_options = option.data.DataOptions(datasets=[synthetic_options])
data_options.to_yaml(options_directory / "data.yml")

track_options = default.synthetic_track()
track_options.to_yaml(options_directory / "track.yml")

# Create the display_options dictionary
visualize_options = default.synthetic_runtime(options_directory / "visualize.yml")
visualize_options.to_yaml(options_directory / "visualize.yml")

times = np.arange(
    np.datetime64(start),
    np.datetime64(end) + np.timedelta64(10, "m"),
    np.timedelta64(10, "m"),
)
args = [times, data_options, grid_options, track_options, visualize_options]
track.track(*args, output_directory=output_parent)

central_latitude = -10
central_longitude = 132

y = np.arange(-400e3, 400e3 + 2.5e3, 2.5e3).tolist()
x = np.arange(-400e3, 400e3 + 2.5e3, 2.5e3).tolist()

grid_options = option.grid.GridOptions(
    name="cartesian",
    x=x,
    y=y,
    central_latitude=central_latitude,
    central_longitude=central_longitude,
)
grid_options.to_yaml(options_directory / "grid.yml")

output_parent = base_local / "runs/synthetic/cartesian"
if output_parent.exists():
    shutil.rmtree(output_parent)

times = np.arange(
    np.datetime64(start),
    np.datetime64(end) + np.timedelta64(10, "m"),
    +np.timedelta64(10, "m"),
)

args = [times, data_options, grid_options, track_options, visualize_options]
track.track(*args, output_directory=output_parent)
