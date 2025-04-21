Tracking Methods: CPOL
======================

This tutorial/demo illustrates how THUNER can be applied to
`CPOL <https://www.openradar.io/research-radars/cpol>`__, a C-band
dual-polarisation research radar located at Gunn Point near Darwin, in
Australia’s northern Territory.

Setup
-----

.. code-block:: python3
    :linenos:

    %load_ext autoreload
    %autoreload 2
    %matplotlib inline
    
    from pathlib import Path
    import shutil
    import thuner.data as data
    import thuner.option as option
    import thuner.track.track as track
    import thuner.visualize as visualize
    import thuner.analyze as analyze
    import thuner.default as default
    import thuner.attribute as attribute
    import thuner.parallel as parallel

.. code-block:: text

    The autoreload extension is already loaded. To reload it, use:
      %reload_ext autoreload

.. code-block:: python3
    :linenos:

    # Parent directory for saving outputs
    base_local = Path.home() / "THUNER_output"
    
    output_parent = base_local / "runs/cpol/geographic"
    options_directory = output_parent / "options"
    visualize_directory = output_parent / "visualize"
    
    # Remove the output parent directory if it already exists
    if output_parent.exists():
        shutil.rmtree(output_parent)

Geographic Coordinates
----------------------

CPOL level 1b data is provided in cartesian coordinates. We can convert
this data to geographic coordinates on the fly by specifying default
grid options. We will also save this converted data to disk for use
later.

.. code-block:: python3
    :linenos:

    # Create the dataset options
    start = "2005-11-13T14:00:00"
    end = "2005-11-13T19:00:00"
    times_dict = {"start": start, "end": end}
    cpol_options = data.aura.CPOLOptions(**times_dict, converted_options={"save": True})
    era5_dict = {"latitude_range": [-14, -10], "longitude_range": [129, 133]}
    era5_pl_options = data.era5.ERA5Options(**times_dict, **era5_dict)
    era5_dict.update({"data_format": "single-levels"})
    era5_sl_options = data.era5.ERA5Options(**times_dict, **era5_dict)
    datasets=[cpol_options, era5_pl_options, era5_sl_options]
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
    track_options.to_yaml(options_directory / "track.yml")
    
    # Create the visualize_options
    kwargs = {"visualize_directory": visualize_directory, "objects": ["convective", "mcs"]}
    visualize_options = default.runtime(**kwargs)
    visualize_options.to_yaml(options_directory / "visualize.yml")

.. code-block:: text

    2025-04-21 21:33:38,061 - thuner.data.aura - INFO - Generating cpol filepaths.
    2025-04-21 21:33:38,062 - thuner.data.era5 - INFO - Generating era5 filepaths.
    2025-04-21 21:33:38,064 - thuner.data.era5 - INFO - Generating era5 filepaths.
    2025-04-21 21:33:38,087 - thuner.option.grid - WARNING - altitude not specified. Using default altitudes.
    2025-04-21 21:33:38,089 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.

We can now perform our tracking run; note the run will be slow as we are
generating runtime figures for both convective and MCS objects, and not
using parallelization. To make the run go much faster, set
``visualize_options = None`` and use the the parallel tracking function.

.. code-block:: python3
    :linenos:

    times = data.generate_times(data_options.dataset_by_name("cpol"))
    args = [times, data_options, grid_options, track_options, visualize_options]
    # parallel.track(*args, output_directory=output_parent)
    track.track(*args, output_directory=output_parent)

.. code-block:: text

    2025-04-21 21:33:39,204 - thuner.track.track - INFO - Beginning thuner tracking. Saving output to /home/ewan/THUNER_output/runs/cpol/geographic.
    2025-04-21 21:33:39,259 - thuner.track.track - INFO - Processing 2005-11-13T14:00:09.
    2025-04-21 21:33:39,261 - thuner.data.aura - INFO - Updating cpol dataset for 2005-11-13T14:00:09.
    2025-04-21 21:33:39,262 - thuner.data.aura - INFO - Converting cpol data from twp10cpolgrid150.b2.20051113.140000.nc
    2025-04-21 21:33:39,318 - thuner.data.aura - INFO - Creating new geographic grid with spacing 0.025 m, 0.025 m.
    2025-04-21 21:33:39,992 - thuner.track.track - INFO - Processing hierarchy level 0.
    2025-04-21 21:33:39,993 - thuner.track.track - INFO - Tracking convective.
    2025-04-21 21:33:40,004 - thuner.match.match - INFO - Matching convective objects.
    2025-04-21 21:33:40,006 - thuner.match.match - INFO - No current mask, or no objects in current mask.
    2025-04-21 21:33:40,011 - thuner.visualize.runtime - INFO - Creating runtime visualization figures.
    2025-04-21 21:33:44,152 - thuner.track.track - INFO - Tracking middle.
    2025-04-21 21:33:44,160 - thuner.track.track - INFO - Tracking anvil.
    2025-04-21 21:33:44,164 - thuner.track.track - INFO - Processing hierarchy level 1.
    2025-04-21 21:33:44,165 - thuner.track.track - INFO - Tracking mcs.
    2025-04-21 21:33:44,186 - thuner.match.match - INFO - Matching mcs objects.
    2025-04-21 21:33:44,188 - thuner.match.match - INFO - No current mask, or no objects in current mask.
    ...

.. code-block:: python3
    :linenos:

    analysis_options = analyze.mcs.AnalysisOptions()
    analysis_options.to_yaml(options_directory / "analysis.yml")
    analyze.mcs.process_velocities(output_parent)
    analyze.mcs.quality_control(output_parent, analysis_options)
    analyze.mcs.classify_all(output_parent, analysis_options)

.. code-block:: text

    2025-04-21 21:38:34,460 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.
    2025-04-21 21:38:34,712 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.

.. code-block:: python3
    :linenos:

    figure_name = "mcs_attributes"
    kwargs = {"style": "presentation", "attributes": ["velocity", "offset"]}
    figure_options = option.visualize.HorizontalAttributeOptions(name=figure_name, **kwargs)
    
    args = [output_parent, start, end, figure_options]
    args_dict = {"parallel_figure": True, "by_date": False, "num_processes": 4}
    visualize.attribute.mcs_series(*args, **args_dict)

.. code-block:: text

    2025-04-21 21:38:52,477 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.
    2025-04-21 21:38:52,715 - thuner.visualize.attribute - INFO - Visualizing MCS at time 2005-11-13T14:00:09.000000000.
    2025-04-21 21:38:52,716 - thuner.data.aura - INFO - Converting cpol data from twp10cpolgrid150.b2.20051113.140000.nc
    2025-04-21 21:38:53,264 - thuner.data.aura - INFO - Creating new geographic grid with spacing 0.025 m, 0.025 m.
    2025-04-21 21:38:53,839 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.
    2025-04-21 21:38:54,206 - thuner.visualize.attribute - INFO - Saving mcs_attributes figure for 2005-11-13T14:00:09.000000000.
    2025-04-21 21:39:00,733 - thuner.visualize.attribute - INFO - Visualizing MCS at time 2005-11-13T14:10:23.000000000.
    2025-04-21 21:39:00,739 - thuner.data.aura - INFO - Converting cpol data from twp10cpolgrid150.b2.20051113.141000.nc
    2025-04-21 21:39:00,896 - thuner.visualize.attribute - INFO - Visualizing MCS at time 2005-11-13T14:20:09.000000000.
    2025-04-21 21:39:00,899 - thuner.data.aura - INFO - Converting cpol data from twp10cpolgrid150.b2.20051113.142000.nc
    2025-04-21 21:39:01,831 - thuner.visualize.attribute - INFO - Visualizing MCS at time 2005-11-13T14:30:09.000000000.
    2025-04-21 21:39:01,833 - thuner.data.aura - INFO - Converting cpol data from twp10cpolgrid150.b2.20051113.143000.nc
    2025-04-21 21:39:03,949 - thuner.visualize.attribute - INFO - Visualizing MCS at time 2005-11-13T14:40:09.000000000.
    2025-04-21 21:39:03,956 - thuner.data.aura - INFO - Converting cpol data from twp10cpolgrid150.b2.20051113.144000.nc
    2025-04-21 21:39:04,087 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.
    ...

Cartesian Coordinates
---------------------

Because the CPOL radar domains are small (150 km radii), it is
reasonable to perform tracking in Cartesian coordinates. This should
make the run faster as we are no longer performing conversions on the
fly. We will also switch off the runtime figure generation.

.. code-block:: python3
    :linenos:

    output_parent = base_local / "runs/cpol/cartesian"
    options_directory = output_parent / "options"
    options_directory.mkdir(parents=True, exist_ok=True)
    
    if output_parent.exists():
        shutil.rmtree(output_parent)
    
    grid_options = option.grid.GridOptions(name="cartesian", regrid=False)
    grid_options.to_yaml(options_directory / "grid.yml")
    data_options.to_yaml(options_directory / "data.yml")
    track_options.to_yaml(options_directory / "track.yml")
    visualize_options = None
    
    times = data.generate_times(data_options.dataset_by_name("cpol"))
    args = [times, data_options, grid_options, track_options, visualize_options]
    kwargs = {"output_directory": output_parent, "dataset_name": "cpol"}
    parallel.track(*args, **kwargs, debug_mode=True)

.. code-block:: text

    2025-04-21 21:43:15,748 - thuner.option.grid - WARNING - altitude not specified. Using default altitudes.
    2025-04-21 21:43:15,749 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.
    2025-04-21 21:43:17,594 - thuner.parallel - INFO - Beginning parallel tracking with 4 processes.
    2025-04-21 21:43:17,756 - thuner.track.track - INFO - Beginning thuner tracking. Saving output to /home/ewan/THUNER_output/runs/cpol/cartesian/interval_0.
    2025-04-21 21:43:17,830 - thuner.track.track - INFO - Processing 2005-11-13T14:00:09.
    2025-04-21 21:43:17,831 - thuner.data.aura - INFO - Updating cpol dataset for 2005-11-13T14:00:09.
    2025-04-21 21:43:17,832 - thuner.data.aura - INFO - Converting cpol data from twp10cpolgrid150.b2.20051113.140000.nc
    2025-04-21 21:43:18,013 - thuner.track.track - INFO - Processing hierarchy level 0.
    2025-04-21 21:43:18,015 - thuner.track.track - INFO - Tracking convective.
    2025-04-21 21:43:18,040 - thuner.match.match - INFO - Matching convective objects.
    2025-04-21 21:43:18,042 - thuner.match.match - INFO - No current mask, or no objects in current mask.
    2025-04-21 21:43:18,046 - thuner.track.track - INFO - Tracking middle.
    2025-04-21 21:43:18,055 - thuner.track.track - INFO - Tracking anvil.
    2025-04-21 21:43:18,065 - thuner.track.track - INFO - Processing hierarchy level 1.
    2025-04-21 21:43:18,066 - thuner.track.track - INFO - Tracking mcs.
    2025-04-21 21:43:18,114 - thuner.match.match - INFO - Matching mcs objects.
    2025-04-21 21:43:18,116 - thuner.match.match - INFO - No current mask, or no objects in current mask.
    ...

.. code-block:: python3
    :linenos:

    analysis_options = analyze.mcs.AnalysisOptions()
    analysis_options.to_yaml(options_directory / "analysis.yml")
    analyze.mcs.process_velocities(output_parent)
    analyze.mcs.quality_control(output_parent, analysis_options)
    analyze.mcs.classify_all(output_parent, analysis_options)

.. code-block:: text

    2025-04-21 21:44:07,801 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.
    2025-04-21 21:44:08,130 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.

.. code-block:: python3
    :linenos:

    figure_name = "mcs_attributes"
    kwargs = {"style": "presentation", "attributes": ["velocity", "offset"]}
    figure_options = option.visualize.HorizontalAttributeOptions(name=figure_name, **kwargs)
    
    args = [output_parent, start, end, figure_options]
    args_dict = {"parallel_figure": True, "by_date": False, "num_processes": 4}
    visualize.attribute.mcs_series(*args, **args_dict)

.. code-block:: text

    2025-04-21 21:44:08,740 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.
    2025-04-21 21:44:09,057 - thuner.visualize.attribute - INFO - Visualizing MCS at time 2005-11-13T14:00:09.000000000.
    2025-04-21 21:44:09,061 - thuner.data.aura - INFO - Converting cpol data from twp10cpolgrid150.b2.20051113.140000.nc
    2025-04-21 21:44:09,218 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.
    2025-04-21 21:44:09,793 - thuner.visualize.attribute - INFO - Saving mcs_attributes figure for 2005-11-13T14:00:09.000000000.
    2025-04-21 21:44:17,959 - thuner.visualize.attribute - INFO - Visualizing MCS at time 2005-11-13T14:10:23.000000000.
    2025-04-21 21:44:17,962 - thuner.data.aura - INFO - Converting cpol data from twp10cpolgrid150.b2.20051113.141000.nc
    2025-04-21 21:44:18,007 - thuner.visualize.attribute - INFO - Visualizing MCS at time 2005-11-13T14:20:09.000000000.
    2025-04-21 21:44:18,011 - thuner.data.aura - INFO - Converting cpol data from twp10cpolgrid150.b2.20051113.142000.nc
    2025-04-21 21:44:18,136 - thuner.visualize.attribute - INFO - Visualizing MCS at time 2005-11-13T14:30:09.000000000.
    2025-04-21 21:44:18,141 - thuner.data.aura - INFO - Converting cpol data from twp10cpolgrid150.b2.20051113.143000.nc
    2025-04-21 21:44:18,803 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.
    2025-04-21 21:44:18,899 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.
    2025-04-21 21:44:18,997 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.
    2025-04-21 21:44:19,815 - thuner.visualize.attribute - INFO - Saving mcs_attributes figure for 2005-11-13T14:10:23.000000000.
    ...