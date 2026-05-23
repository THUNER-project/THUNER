import shutil
import glob
import xarray as xr
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


def test_cpol():
    # # Tracking Methods: CPOL
    # This tutorial/demo illustrates how THUNER can be applied to [CPOL](https://www.openradar.io/research-radars/cpol), a C-band dual-polarisation research radar located at Gunn Point near Darwin, in Australia's northern Territory.
    #
    # ## Setup
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
    end = "2005-11-13T16:00:30"
    times_dict = {"start": start, "end": end}
    cpol_options = data.aura.CpolOptions(**times_dict, converted_options={"save": True})
    era5_dict = {"latitude_range": [-14, -10], "longitude_range": [129, 133]}
    era5_pl_options = data.era5.Era5Options(**times_dict, **era5_dict)
    era5_dict.update({"data_format": "single-levels"})
    era5_sl_options = data.era5.Era5Options(**times_dict, **era5_dict)
    datasets = [cpol_options, era5_pl_options, era5_sl_options]
    data_options = option.data.DataOptions(datasets=datasets)
    data_options.to_json(options_directory / "data.json")
    # Create the grid_options
    grid_options = option.grid.GridOptions()
    grid_options.to_json(options_directory / "grid.json")
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
    membership = attribute.group.membership_attribute_group()
    mcs_group_attr.attributes.append(membership)
    mcs_group_attr.revalidate()
    track_options.to_json(options_directory / "track.json")
    # For this tutorial, we will generate figures during runtime to visualize how THUNER
    # is matching both convective and mcs objects. Note the figure generation slows the run down a lot!
    # Create the visualize_options
    kwargs = {
        "visualize_directory": visualize_directory,
        "objects": ["convective", "mcs"],
    }
    visualize_options = default.runtime(**kwargs)
    visualize_options.to_json(options_directory / "visualize.json")
    times = utils.generate_times(data_options.dataset_by_name("cpol").filepaths)
    args = [times, data_options, grid_options, track_options]
    track.track(
        *args, visualize_options=visualize_options, output_directory=output_parent
    )
    # Once the run is completed, outputs are available in the `output_parent` directory. The `output.zarr` store contains the object attributes, masks, and filename records for the run. These can be conveniently explored by loading as an `xr.DataTree`.
    #
    dt = xr.open_datatree(output_parent / "output.zarr", engine="zarr")
    # Print all the group names in the store
    print("\n".join(sorted(dt.groups)))
    # Get the core convective attributes
    ds = dt.attributes.convective.core.to_dataset()
    indices = ds.index_columns
    df = ds.to_dataframe().set_index(indices)
    print(df.head(10).to_string())
    #
    # The visualization folder will contain figures like that below, which illustrate the matching process. Currently THUNER supports the TINT/MINT matching approach, but the goal is to eventually incorporate others.
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
    columns = ["convective_ids", "middle_ids", "anvil_ids"]
    mcs_data = attribute.utils.read_attribute(
        output_parent, "attributes", "mcs", "group", columns=columns
    )
    print(mcs_data.to_string())
    # ## Running in Parallel
    # We can also run THUNER in parallel, which significantly speeds up big runs. Note we cannot create algorithm visualization figures during a parallel run, instead we create figures after the run is complete. The parallelization strategy is simple, we just split the time interval into sub-intervals, run THUNER on each sub-interval, then stitch the results back together at the end.
    # Create the output directories for the parallel run
    output_parent = base_local / "runs/cpol/geographic_parallel"
    options_directory = output_parent / "options"
    visualize_directory = output_parent / "visualize"
    # Remove the output parent directory if it already exists
    if output_parent.exists() and remove_existing_outputs:
        shutil.rmtree(output_parent)
    # Recreate the dataset options with a longer time interval
    start = "2005-11-13T14:00:00"
    end = "2005-11-13T19:00:30"
    times_dict = {"start": start, "end": end}
    cpol_options = data.aura.CpolOptions(**times_dict, converted_options={"save": True})
    era5_dict = {"latitude_range": [-14, -10], "longitude_range": [129, 133]}
    era5_pl_options = data.era5.Era5Options(**times_dict, **era5_dict)
    era5_dict.update({"data_format": "single-levels"})
    era5_sl_options = data.era5.Era5Options(**times_dict, **era5_dict)
    datasets = [cpol_options, era5_pl_options, era5_sl_options]
    data_options = option.data.DataOptions(datasets=datasets)
    data_options.to_json(options_directory / "data.json")
    # All the other options are the same as before
    grid_options.to_json(options_directory / "grid.json")
    track_options.to_json(options_directory / "track.json")
    times = utils.generate_times(data_options.dataset_by_name("cpol").filepaths)
    args = [times, data_options, grid_options, track_options]
    parallel.track(*args, output_directory=output_parent, dataset_name="cpol")
    # After a run, we can also perform analysis and visualization. Here we identify and visualize some Mesoscale Convective System (MCS) objects.
    analysis_options = analyze.mcs.AnalysisOptions()
    analysis_options.to_json(options_directory / "analysis.json")
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
    base_filepath = (
        base_local / "input_data/converted/cpol/cpol_level_1b/v2020/gridded/"
    )
    base_filepath = base_filepath / "grid_150km_2500m/2005/20051113"
    filepaths = glob.glob(str(base_filepath / "*.nc"))
    filepaths = sorted(filepaths)
    # Create the data options.
    kwargs = {"name": "cpol", "fields": ["reflectivity"], "filepaths": filepaths}
    cpol_options = utils.BaseDatasetOptions(**times_dict, **kwargs)
    datasets = [cpol_options, era5_pl_options, era5_sl_options]
    data_options = option.data.DataOptions(datasets=datasets)
    data_options.to_json(options_directory / "data.json")
    # Save other options
    grid_options.to_json(options_directory / "grid.json")
    track_options.to_json(options_directory / "track.json")
    times = utils.generate_times(data_options.dataset_by_name("cpol").filepaths)
    args = [times, data_options, grid_options, track_options]
    kwargs = {"output_directory": output_parent, "dataset_name": "cpol"}
    parallel.track(*args, **kwargs, debug_mode=True)
    analysis_options = analyze.mcs.AnalysisOptions()
    analysis_options.to_json(options_directory / "analysis.json")
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
    cpol_options = data.aura.CpolOptions(**times_dict)
    datasets = [cpol_options, era5_pl_options, era5_sl_options]
    data_options = option.data.DataOptions(datasets=datasets)
    data_options.to_json(options_directory / "data.json")
    # Create the grid_options
    grid_options = option.grid.GridOptions(name="cartesian", regrid=False)
    grid_options.to_json(options_directory / "grid.json")
    # Save the same track options from earlier
    track_options.to_json(options_directory / "track.json")
    visualize_options = None
    times = utils.generate_times(data_options.dataset_by_name("cpol").filepaths)
    args = [times, data_options, grid_options, track_options, visualize_options]
    kwargs = {"output_directory": output_parent, "dataset_name": "cpol"}
    # parallel.track(*args, **kwargs)
    track.track(*args, output_directory=output_parent)
    analysis_options = analyze.mcs.AnalysisOptions()
    analysis_options.to_json(options_directory / "analysis.json")
    analyze.mcs.process_velocities(output_parent)
    analyze.mcs.quality_control(output_parent, analysis_options)
    analyze.mcs.classify_all(output_parent, analysis_options)
    style = "presentation"
    attribute_handlers = default.grouped_attribute_handlers(output_parent, style)
    kwargs = {"name": "mcs_attributes", "object_name": "mcs", "style": style}
    kwargs.update({"attribute_handlers": attribute_handlers})
    figure_options = option.visualize.GroupedHorizontalAttributeOptions(**kwargs)
    args = [output_parent, start, end, figure_options, "cpol"]
    args_dict = {"parallel_figure": False, "by_date": False, "num_processes": 1}
    visualize.attribute.series(*args, **args_dict)


if __name__ == "__main__":
    test_cpol()
