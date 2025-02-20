from pathlib import Path
import shutil
import numpy as np
import thuner.data as data
import thuner.grid as grid
import thuner.track.track as track
import thuner.option as option
import thuner.visualize as visualize
import thuner.analyze as analyze
import thuner.option as option
import thuner.default as default

notebook_name = "cpol_demo.ipynb"

# Parent directory for saving outputs
base_local = Path.home() / "THUNER_output"
start = "2005-11-13T18:00:00"
end = "2005-11-13T19:00:00"

output_parent = base_local / "runs/cpol/geographic"
options_directory = output_parent / "options"
visualize_directory = output_parent / "visualize"

if output_parent.exists():
    shutil.rmtree(output_parent)

# Create the dataset options
times_dict = {"start": start, "end": end}
cpol_options = data.aura.CPOLOptions(**times_dict)
era5_dict = {"latitude_range": [-14, -10], "longitude_range": [129, 133]}
era5_pl_options = data.era5.ERA5Options(**times_dict, **era5_dict)
era5_dict.update({"data_format": "single-levels"})
era5_sl_options = data.era5.ERA5Options(**times_dict, **era5_dict)
datasets = [cpol_options, era5_pl_options, era5_sl_options]
data_options = option.data.DataOptions(datasets=datasets)
data_options.to_yaml(options_directory / "data.yml")

# Create the grid_options
grid_options = option.grid.GridOptions()
grid_options.to_yaml(options_directory / "grid.yml")

# Create the track_options
track_options = default.track(dataset="cpol")
track_options.to_yaml(options_directory / "track.yml")

# Create the visualize_options
visualize_options = default.runtime(visualize_directory=visualize_directory)
visualize_options.to_yaml(options_directory / "visualize.yml")

times = data.utils.generate_times(data_options.dataset_by_name("cpol"))
args = [times, data_options, grid_options, track_options, visualize_options]
# parallel.track(*args, output_directory=output_parent)
track.track(*args, output_directory=output_parent)

output_parent = base_local / "runs/cpol/cartesian"
options_directory = output_parent / "options"
options_directory.mkdir(parents=True, exist_ok=True)

if output_parent.exists():
    shutil.rmtree(output_parent)

grid_options = option.grid.GridOptions(name="cartesian", regrid=False)
grid_options.to_yaml(options_directory / "grid.yml")
data_options.to_yaml(options_directory / "data.yml")
track_options.to_yaml(options_directory / "track.yml")

# visualize.option.save_display_options(visualize_options, options_directory)

times = data.utils.generate_times(data_options.dataset_by_name("cpol"))
args = [times, data_options, grid_options, track_options, visualize_options]
track.track(*args, output_directory=output_parent)

analysis_options = analyze.mcs.AnalysisOptions()
analysis_options.to_yaml(options_directory / "analysis.yml")
# utils.save_options(analysis_options, filename="analysis", options_directory=output_directory / "options")
analyze.mcs.process_velocities(output_parent)
analyze.mcs.quality_control(output_parent, analysis_options)
# analyze.mcs.classify_all(output_parent, analysis_options)

figure_name = "mcs_attributes"
kwargs = {"style": "presentation", "attributes": ["velocity", "offset"]}
figure_options = option.visualize.HorizontalAttributeOptions(name=figure_name, **kwargs)

start_time = np.datetime64(start)
end_time = np.datetime64(end)
args = [output_parent, start_time, end_time, figure_options]
args_dict = {"parallel_figure": True, "by_date": False, "num_processes": 4}
visualize.attribute.mcs_series(*args, **args_dict)
