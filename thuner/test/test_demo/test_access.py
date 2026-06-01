import shutil
import xarray as xr
import thuner.data as data
import thuner.option as option
import thuner.analyze as analyze
import thuner.parallel as parallel
import thuner.visualize as visualize
import thuner.default as default
import thuner.config as config
import thuner.utils as utils


def test_access():
    """ACCESS-C demo/test."""
    # Set a flag for whether or not to remove existing output directories
    remove_existing_outputs = True
    # Parent directory for saving outputs
    base_local = config.get_outputs_directory()
    output_parent = base_local / f"runs/access_c/access_c_demo"
    options_directory = output_parent / "options"
    visualize_directory = output_parent / "visualize"
    # Delete the output directory for the run if it already exists
    if output_parent.exists() & remove_existing_outputs:
        shutil.rmtree(output_parent)
    # Create the dataset options
    # For model datasets we generally need to specify which model run we want, in 
    # addition to the start and end times. Typically we want to discard spin up times.
    run_start = "2021-12-01T12:00:00" # The start time of the run we want
    start = "2021-12-02T00:00:00" # The start time of the data we want to analyze. 
    end = "2021-12-02T16:00:00" # The end time of the data we want to analyze.
    times_dict = {"start": start, "end": end, "run_start": run_start}
    access_1km_options = data.access.AccessCOptions(
        **times_dict, name="access_1km", filename="radar_refl_1km.nc"
    )
    access_max_col_options = data.access.AccessCOptions(
        **times_dict, name="access_maxcol", filename="maxcol_refl.nc"
    )
    datasets=[access_1km_options, access_max_col_options]
    data_options = option.data.DataOptions(datasets=datasets)
    data_options.to_json(options_directory / "data.json")
    grid_options = option.grid.GridOptions()
    grid_options.to_json(options_directory / "grid.json")
    track_options = default.track.access_c_track()
    track_options.to_json(options_directory / "track.json")
    times = utils.generate_dataset_times(data_options.dataset_by_name("access_1km"))
    parallel.track(
        times=times,
        data_options=data_options,
        grid_options=grid_options,
        track_options=track_options,
        output_directory=output_parent,
        dataset_name='access_1km',
        num_processes=3,
    )
    analysis_options = analyze.mcs.AnalysisOptions()
    analysis_options.to_json(options_directory / "analysis.json")
    analyze.mcs.process_velocities(output_parent, profile_dataset=None)
    analyze.mcs.quality_control(output_parent, analysis_options)
    style = "presentation"
    attribute_handlers = default.visualize.grouped_attribute_handlers(output_parent, style)
    figure_options = option.visualize.GroupedHorizontalAttributeOptions(
        name='mcs_attributes',
        object_name='mcs',
        style=style,
        attribute_handlers=attribute_handlers,
    )
    visualize.attribute.series(
        output_directory=output_parent,
        start_time=start,
        end_time=end,
        figure_options=figure_options,
        dataset_name="access_1km",
        parallel_figure=True,
        by_date=False,
        num_processes=4
    )
    dt = xr.open_datatree(output_parent / "output.zarr", engine="zarr")
    quality = dt.analysis.quality.ds
    quality_df = quality.to_dataframe().set_index(quality.index_columns)
    raw_sample = quality_df.where(quality_df["duration"]).dropna() 
    mcs_count = len(raw_sample.index.get_level_values("universal_id").unique())
    print(mcs_count)
    velocity = dt.analysis.velocities.ds
    velocity_df = velocity.to_dataframe().set_index(velocity.index_columns)
    raw_sample = velocity_df.where(quality_df["duration"]).dropna()
    average_velocities = raw_sample.reset_index()[["u", "v"]].mean(axis=0)
    print(average_velocities)


if __name__ == '__main__':
    test_access()
