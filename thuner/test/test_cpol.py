# # Tracking Methods: CPOL

# This tutorial/demo illustrates how THUNER can be applied to [CPOL](https://www.openradar.io/research-radars/cpol), a C-band dual-polarisation research radar located at Gunn Point near Darwin, in Australia's northern Territory.
#

# ## Setup

import shutil
import glob
import thuner.data as data
import thuner.option as option
import thuner.track.track as track
import thuner.visualize as visualize
import thuner.analyze as analyze
import thuner.default as default
import thuner.attribute as attribute
import thuner.parallel as parallel
import thuner.utils as utils
import thuner.config as config

# Set a flag for whether or not to remove existing output directories
remove_existing_outputs = True

# Specify the local base directory for saving outputs
base_local = config.get_outputs_directory()

output_parent = base_local / "runs/cpol/geographic"
options_directory = output_parent / "options"
visualize_directory = output_parent / "visualize"

# Remove the output parent directory if it already exists
if output_parent.exists() and remove_existing_outputs:
    shutil.rmtree(output_parent)

# Run the cell below to get the demo data for this tutorial, if you haven't already.

# Download the demo data
remote_directory = "s3://thuner-storage/THUNER_output/input_data/raw/cpol"
data.get_demo_data(base_local, remote_directory)
remote_directory = "s3://thuner-storage/THUNER_output/input_data/raw/"
remote_directory += "era5_monthly_10S_129E_14S_133E"
data.get_demo_data(base_local, remote_directory)

# ## Geographic Coordinates

# CPOL level 1b data is provided in cartesian coordinates. We can convert this data to
# geographic coordinates on the fly by specifying default grid options. We will also save
# this converted data to disk for use later.

# Create the dataset options
start = "2005-11-13T14:00:00"
# Note the CPOL times are usually a few seconds off the 10 m interval, so add 30 seconds
# to ensure we capture 19:00:00
end = "2005-11-13T19:00:30"
times_dict = {"start": start, "end": end}
cpol_options = data.aura.CPOLOptions(**times_dict, converted_options={"save": True})
# cpol_options = data.aura.CPOLOptions(**times_dict, converted_options={"load": True})
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
track_options = default.track(dataset_name="cpol")
# Modify the default track options to demonstrate the tracking of both convective
# objects, and mesoscale convective systems, which are built out of convective, middle
# and stratiform echo objects, within the same THUNER run. We will use a larger
# minimum size for the convective objects, as too many very small objects confuses the
# matching algorithm.
core = attribute.core.default_tracked()
attributes = option.attribute.Attributes(name="convective", attribute_types=[core])
track_options.levels[0].object_by_name("convective").attributes = attributes
tint_tracking = option.track.TintOptions(search_margin=5)
track_options.levels[0].object_by_name("convective").tracking = tint_tracking
mask_options = option.track.MaskOptions(save=True)
track_options.levels[0].object_by_name("convective").mask_options = mask_options
track_options.levels[0].object_by_name("convective").detection.min_area = 64
track_options.levels[0].object_by_name("convective").detection.altitudes
track_options.levels[0].object_by_name("convective").revalidate()
track_options.levels[0].revalidate()
# We will also modify the mcs tracking options to save a record of the member object ids
mcs_attributes = track_options.levels[1].object_by_name("mcs").attributes
mcs_group_attr = mcs_attributes.attribute_type_by_name("group")
membership = attribute.group.build_membership_attribute_group()
mcs_group_attr.attributes.append(membership)
mcs_group_attr.revalidate()
track_options.to_yaml(options_directory / "track.yml")

# For this tutorial, we will generate figures during runtime to visualize how THUNER
# is matching both convective and mcs objects.

# Create the visualize_options
kwargs = {"visualize_directory": visualize_directory, "objects": ["convective", "mcs"]}
visualize_options = default.runtime(**kwargs)
visualize_options.to_yaml(options_directory / "visualize.yml")

# We can now perform our tracking run; note the run will be slow as we are generating runtime figures for both convective and MCS objects, and not using parallelization. To make the run go much faster, set `visualize_options = None` and use the the parallel tracking function.

times = utils.generate_times(data_options.dataset_by_name("cpol").filepaths)
args = [times, data_options, grid_options, track_options, visualize_options]
# parallel.track(*args, output_directory=output_parent)
track.track(*args, output_directory=output_parent)

# Once completed, outputs are available in the `output_parent` directory. The visualization
# folder will contain figures like that below, which illustrate the matching process.
# Currently THUNER supports the TINT/MINT matching approach, but the goal is to eventually
# incorporate others. Note that if viewing online, the figures below can be viewed at original scale by right clicking, save image as, and opening locally, or by right clicking, open in new tab, etc.
#
# ![Visualization of the TINT/MINT matching process.](https://raw.githubusercontent.com/THUNER-project/THUNER/refs/heads/main/gallery/cpol_convective_match_20051113.png)
#
# Definitions of terms appearing in the above figure are provided by
# [Raut et al. (2021)](https://doi.org/10.1175/JAMC-D-20-0119.1). Note the displacement
# vector for the central orange object is large due to the object changing shape suddenly.
# Similar jumps occur when objects split and merge, and for this reason, object center displacements are ill suited to define object velocities. Instead, object velocities are calculated by smoothing the corrected local flow vectors, as discussed by [Short et al. (2023)](https://doi.org/10.1175/MWR-D-22-0146.1). Animations of all the runtime matching figures for the convective objects are provided below.
#
# ![Convective object matching.](https://raw.githubusercontent.com/THUNER-project/THUNER/refs/heads/main/gallery/cpol_convective_match_20051113.gif)
#
# We also provide the matching figures for the MCS objects. Note there is only one MCS
# object, which is comprised of multiple disjoint convective objects; the grouping method
# is described by [Short et al. (2023)](https://doi.org/10.1175/MWR-D-22-0146.1).
#
# ![MCS object matching.](https://raw.githubusercontent.com/THUNER-project/THUNER/refs/heads/main/gallery/cpol_mcs_match_20051113.gif)

# Recall that when setting up the options above, we instructed THUNER to keep a record of the IDs of
# each member object (convective, middle and stratiform echoes) comprising each grouped
# mcs object. Note that only the mcs and convective objects are matched between times.

filepath = output_parent / "attributes/mcs/group.csv"
columns = ["convective_ids", "middle_ids", "anvil_ids"]
print(attribute.utils.read_attribute_csv(filepath, columns=columns).to_string())

# We can also perform analysis on, and visualization of, the MCS objects.

analysis_options = analyze.mcs.AnalysisOptions()
analysis_options.to_yaml(options_directory / "analysis.yml")
analyze.mcs.process_velocities(output_parent)
analyze.mcs.quality_control(output_parent, analysis_options)
analyze.mcs.classify_all(output_parent, analysis_options)

style = "presentation"
attribute_handlers = default.grouped_attribute_handlers(output_parent, style)
kwargs = {"name": "mcs_attributes", "object_name": "mcs", "style": style}
kwargs.update({"attribute_handlers": attribute_handlers})
figure_options = option.visualize.GroupedHorizontalAttributeOptions(**kwargs)
args = [output_parent, start, end, figure_options, "cpol"]
args_dict = {"parallel_figure": True, "by_date": False, "num_processes": 4}
visualize.attribute.series(*args, **args_dict)

# ## Pre-Converted Data

# We can also perform THUNER tracking runs on general datasets, we just need to ensure
# they are pre-converted into a format recognized by THUNER, i.e. gridded data files readable by
# ``xarray.open_dataset``, with variables named according to [CF-conventions](https://cfconventions.org/).
# To illustrate, we will use the converted CPOL files that were generated by the code in the
# previous section. We first modify the options used for the geographic coordinates above. Re-run
# the relevant cells above again if necessary. If you get a pydantic error, restart the notebook.

output_parent = base_local / "runs/cpol/pre_converted"
options_directory = output_parent / "options"
options_directory.mkdir(parents=True, exist_ok=True)

if output_parent.exists() & remove_existing_outputs:
    shutil.rmtree(output_parent)

# Get the pre-converted filepaths
base_filepath = base_local / "input_data/converted/cpol/cpol_level_1b/v2020/gridded/"
base_filepath = base_filepath / "grid_150km_2500m/2005/20051113"
filepaths = glob.glob(str(base_filepath / "*.nc"))
filepaths = sorted(filepaths)

# Create the data options.
kwargs = {"name": "cpol", "fields": ["reflectivity"], "filepaths": filepaths}
cpol_options = utils.BaseDatasetOptions(**times_dict, **kwargs)
datasets = [cpol_options, era5_pl_options, era5_sl_options]
data_options = option.data.DataOptions(datasets=datasets)
data_options.to_yaml(options_directory / "data.yml")

# Save other options
grid_options.to_yaml(options_directory / "grid.yml")
track_options.to_yaml(options_directory / "track.yml")

# Switch off the runtime figures
visualize_options = None

times = utils.generate_times(data_options.dataset_by_name("cpol").filepaths)
args = [times, data_options, grid_options, track_options, visualize_options]
kwargs = {"output_directory": output_parent, "dataset_name": "cpol"}
parallel.track(*args, **kwargs, debug_mode=True)

analysis_options = analyze.mcs.AnalysisOptions()
analysis_options.to_yaml(options_directory / "analysis.yml")
analyze.mcs.process_velocities(output_parent)
analyze.mcs.quality_control(output_parent, analysis_options)
analyze.mcs.classify_all(output_parent, analysis_options)

style = "presentation"
attribute_handlers = default.grouped_attribute_handlers(output_parent, style)
kwargs = {"name": "mcs_attributes", "object_name": "mcs", "style": style}
kwargs.update({"attribute_handlers": attribute_handlers})
figure_options = option.visualize.GroupedHorizontalAttributeOptions(**kwargs)
args = [output_parent, start, end, figure_options, "cpol"]
args_dict = {"parallel_figure": True, "by_date": False, "num_processes": 4}
visualize.attribute.series(*args, **args_dict)

# Note we can achieve the same result in this case by modifying `converted_options={"save": True}` to `converted_options={"load": True}` in the [Geographic Coordinates](#geographic-coordinates) section,and rerunning the cells.

# ## Cartesian Coordinates

# Because the CPOL radar domains are small (150 km radii), it is reasonable to perform
# tracking in Cartesian coordinates. This should make the run faster as we are no longer
# performing regridding on the fly. We will also switch off the runtime figure generation.

output_parent = base_local / "runs/cpol/cartesian"
options_directory = output_parent / "options"
options_directory.mkdir(parents=True, exist_ok=True)

if output_parent.exists() & remove_existing_outputs:
    shutil.rmtree(output_parent)

# Recreate the original cpol dataset options
cpol_options = data.aura.CPOLOptions(**times_dict)
datasets = [cpol_options, era5_pl_options, era5_sl_options]
data_options = option.data.DataOptions(datasets=datasets)
data_options.to_yaml(options_directory / "data.yml")

# Create the grid_options
grid_options = option.grid.GridOptions(name="cartesian", regrid=False)
grid_options.to_yaml(options_directory / "grid.yml")

# Save the same track options from earlier
track_options.to_yaml(options_directory / "track.yml")
visualize_options = None

times = utils.generate_times(data_options.dataset_by_name("cpol").filepaths)
args = [times, data_options, grid_options, track_options, visualize_options]
kwargs = {"output_directory": output_parent, "dataset_name": "cpol"}
parallel.track(*args, **kwargs)

analysis_options = analyze.mcs.AnalysisOptions()
analysis_options.to_yaml(options_directory / "analysis.yml")
analyze.mcs.process_velocities(output_parent)
analyze.mcs.quality_control(output_parent, analysis_options)
analyze.mcs.classify_all(output_parent, analysis_options)

style = "presentation"
attribute_handlers = default.grouped_attribute_handlers(output_parent, style)
kwargs = {"name": "mcs_attributes", "object_name": "mcs", "style": style}
kwargs.update({"attribute_handlers": attribute_handlers})
figure_options = option.visualize.GroupedHorizontalAttributeOptions(**kwargs)
args = [output_parent, start, end, figure_options, "cpol"]
args_dict = {"parallel_figure": True, "by_date": False, "num_processes": 4}
visualize.attribute.series(*args, **args_dict)
