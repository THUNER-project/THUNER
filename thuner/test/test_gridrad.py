from pathlib import Path
import shutil
import numpy as np
import thuner.data as data
import thuner.option as option
import thuner.analyze as analyze
import thuner.parallel as parallel
import thuner.visualize as visualize
import thuner.track.track as track
import thuner.default as default

import thuner.log as log

notebook_name = "gridrad_demo.ipynb"

# Parent directory for saving outputs
base_local = Path.home() / "THUNER_output"
year = 2010
event_directories = data.gridrad.get_event_directories(year, base_local=base_local)
event_directory = event_directories[0]
start, end, event_start = data.gridrad.get_event_times(event_directory)
start = "2010-01-20T21:00:00"
end = "2010-01-21T03:30:00"

period = parallel.get_period(start, end)
intervals = parallel.get_time_intervals(start, end, period=period)

output_parent = base_local / f"runs/gridrad_demo"

if output_parent.exists():
    shutil.rmtree(output_parent)
options_directory = output_parent / "options"

# Create and save the dataset options
times_dict = {"start": start, "end": end}
gridrad_dict = {"event_start": event_start}
gridrad_options = data.gridrad.GridRadSevereOptions(**times_dict, **gridrad_dict)
era5_dict = {"latitude_range": [27, 39], "longitude_range": [-102, -89]}
era5_pl_options = data.era5.ERA5Options(**times_dict, **era5_dict)
era5_dict.update({"data_format": "single-levels"})
era5_sl_options = data.era5.ERA5Options(**times_dict, **era5_dict)
datasets = [gridrad_options, era5_pl_options, era5_sl_options]
data_options = option.data.DataOptions(datasets=datasets)
data_options.to_yaml(options_directory / "data.yml")

# Create and save the grid_options dictionary
kwargs = {"name": "geographic", "regrid": False, "altitude_spacing": None}
kwargs.update({"geographic_spacing": None})
grid_options = option.grid.GridOptions(**kwargs)
grid_options.to_yaml(options_directory / "grid.yml")

# Create the track_options dictionary
track_options = default.track(dataset="gridrad")
# Modify the default options for gridrad. Because grids so large we now use a distinct
# global flow box for each object.
track_options.levels[1].objects[0].tracking.global_flow_margin = 70
track_options.levels[1].objects[0].tracking.unique_global_flow = False
track_options.to_yaml(options_directory / "track.yml")

visualize_options = None

times = data.utils.generate_times(data_options.dataset_by_name("gridrad"))
args = [times, data_options, grid_options, track_options, visualize_options]
parallel.track(*args, output_directory=output_parent, dataset_name="gridrad")
# track.track(*args, output_directory=output_parent)

analysis_options = analyze.mcs.AnalysisOptions()
analyze.mcs.process_velocities(output_parent, profile_dataset="era5_pl")
analyze.mcs.quality_control(output_parent, analysis_options)
# analyze.mcs.classify_all(output_parent, analysis_options)

figure_name = f"mcs_gridrad_{event_start.replace('-', '')}"
kwargs = {"style": "presentation", "attributes": ["velocity", "offset"]}
figure_options = option.visualize.HorizontalAttributeOptions(name=figure_name, **kwargs)

start_time = np.datetime64(start)
end_time = np.datetime64(end)
args = [output_parent, start_time, end_time, figure_options]
args_dict = {"parallel_figure": True, "dt": 7200, "by_date": False, "num_processes": 4}
visualize.attribute.mcs_series(*args, **args_dict)
