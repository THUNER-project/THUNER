Himawari
========

This tutorial/demo provides a quick and dirty example of how THUNER can
be applied to
`Himawari <https://geonetwork.nci.org.au/geonetwork/srv/eng/catalog.search#/metadata/f8433_0020_1861_5916>`__
observations.

Setup
-----

.. code-block:: python3
    :linenos:

    %load_ext autoreload
    %autoreload 2
    
    %matplotlib inline
    
    import shutil
    import numpy as np
    import thuner.data as data
    import thuner.option as option
    import thuner.track.track as track
    import thuner.visualize as visualize
    import thuner.analyze as analyze
    import thuner.default as default
    import thuner.parallel as parallel
    import thuner.utils as utils
    import thuner.config as config

.. code-block:: text

    The autoreload extension is already loaded. To reload it, use:
      %reload_ext autoreload

.. code-block:: python3
    :linenos:

    # Set a flag for whether or not to remove existing output directories
    remove_existing_outputs = False
    
    # Specify the local base directory for saving outputs
    base_local = config.get_outputs_directory()
    
    output_parent = base_local / "runs/himawari/"
    options_directory = output_parent / "options"
    visualize_directory = output_parent / "visualize"
    
    # Remove the output parent directory if it already exists
    if output_parent.exists() and remove_existing_outputs:
        shutil.rmtree(output_parent)

Run the cell below to get the demo data for this tutorial, if you
haven’t already.

.. code-block:: python3
    :linenos:

    # Download the demo data
    remote_directory = "s3://thuner-storage/THUNER_output/input_data/raw/satellite-products"
    data.get_demo_data(base_local, remote_directory)

.. code-block:: text

    2025-07-09 19:18:47,562 - thuner.data._utils - INFO - Syncing directory /home/ewan/THUNER_output/input_data/raw/satellite-products. Please wait.

Options
-------

.. code-block:: python3
    :linenos:

    # Create the dataset options
    start = "2023-01-01T00:00:00"
    # Note the CPOL times are usually a few seconds off the 10 m interval, so add 30 seconds
    # to ensure we capture 19:00:00
    end = "2023-01-02T00:00:00"
    times_dict = {"start": start, "end": end}
    himawari_options = data.himawari.HimawariOptions(**times_dict)
    data_options = option.data.DataOptions(datasets=[himawari_options])
    data_options.to_yaml(options_directory / "data.yml")
    
    # Setup a grid over New Guinea. 
    # Note the demo data contains the full disk, so vary the lat/lon as you like!
    spacing = [0.025, 0.025]
    latitude = np.arange(-10, 0 + spacing[0], spacing[0])
    longitude = np.arange(130, 150 + spacing[1], spacing[1])
    altitude = None
    grid_options = option.grid.GridOptions(
        name="geographic", latitude=latitude, longitude=longitude, altitude=altitude
    )
    grid_options.to_yaml(options_directory / "grid.yml")
    
    # Create the track_options
    track_options = default.satellite_track(dataset_name="himawari")
    track_options.to_yaml(options_directory / "track.yml")

.. code-block:: text

    2025-07-09 17:20:14,108 - thuner.data.himawari - INFO - Generating Himawari filepaths.
    2025-07-09 17:20:14,119 - thuner.data.himawari - INFO - Generating coordinates filepath.
    2025-07-09 17:20:14,157 - thuner.option.grid - WARNING - altitude not specified. Using default altitudes.
    2025-07-09 17:20:14,158 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.

Track
-----

.. code-block:: python3
    :linenos:

    times = utils.generate_times(data_options.dataset_by_name("himawari").filepaths)
    args = [times, data_options, grid_options, track_options]
    parallel.track(*args, output_directory=output_parent, dataset_name="himawari", num_processes=2)
    # track.track(*args, output_directory=output_parent)

.. code-block:: text

    2025-07-09 17:20:24,849 - thuner.parallel - INFO - Beginning parallel tracking with 2 processes.
    2025-07-09 17:20:31,071 - thuner.track.track - INFO - Beginning thuner tracking. Saving output to /home/ewan/THUNER_output/runs/himawari/interval_0.
    2025-07-09 17:20:31,072 - thuner.track.track - INFO - Beginning thuner tracking. Saving output to /home/ewan/THUNER_output/runs/himawari/interval_1.
    2025-07-09 17:20:31,679 - thuner.track.track - INFO - Processing 2023-01-01T12:00:00.
    2025-07-09 17:20:31,680 - thuner.utils - INFO - Updating himawari input record for 2023-01-01T12:00:00.
    2025-07-09 17:20:31,680 - thuner.data.himawari - INFO - Converting himawari dataset for time 2023-01-01T12:00:00.
    2025-07-09 17:20:31,685 - thuner.track.track - INFO - Processing 2023-01-01T00:00:00.
    2025-07-09 17:20:31,686 - thuner.utils - INFO - Updating himawari input record for 2023-01-01T00:00:00.
    2025-07-09 17:20:31,686 - thuner.data.himawari - INFO - Converting himawari dataset for time 2023-01-01T00:00:00.
    2025-07-09 17:20:36,465 - thuner.data.himawari - INFO - Regridding Himawari data.
    2025-07-09 17:20:36,467 - thuner.data._utils - INFO - Building regridder; this can take a while for large grids.
    2025-07-09 17:20:36,521 - thuner.data.himawari - INFO - Regridding Himawari data.
    2025-07-09 17:20:36,522 - thuner.data._utils - INFO - Building regridder; this can take a while for large grids.
    2025-07-09 17:21:00,395 - thuner.track.track - INFO - Processing hierarchy level 0.
    2025-07-09 17:21:00,395 - thuner.track.track - INFO - Processing hierarchy level 0.
    ...

Analyze/Visualize
-----------------

.. code-block:: python3
    :linenos:

    analysis_options = analyze.mcs.AnalysisOptions()
    analysis_options.to_yaml(options_directory / "analysis.yml")
    core_filepath = output_parent / "attributes/anvil/core.csv"
    analyze.utils.smooth_flow_velocities(core_filepath, output_parent)
    analyze.utils.quality_control("anvil", output_parent, analysis_options)

.. code-block:: python3
    :linenos:

    style = "presentation"
    attribute_handlers = default.detected_attribute_handlers(output_parent, style)
    kwargs = {"name": "himawari_anvil", "object_name": "anvil", "style": style}
    kwargs.update({"attribute_handlers": attribute_handlers})
    figure_options = option.visualize.HorizontalAttributeOptions(**kwargs)
    args = [output_parent, start, end, figure_options, "himawari"]
    args_dict = {"parallel_figure": True, "by_date": False, "num_processes": 4}
    visualize.attribute.series(*args, **args_dict)

.. code-block:: text

    2025-07-09 17:28:04,167 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.
    2025-07-09 17:28:04,289 - thuner.visualize.attribute - INFO - Visualizing attributes at time 2023-01-01T00:00:00.000000000.
    2025-07-09 17:28:06,194 - thuner.data.himawari - INFO - Converting himawari dataset for time 2023-01-01T00:00:00.
    2025-07-09 17:28:09,755 - thuner.data.himawari - INFO - Regridding Himawari data.
    2025-07-09 17:28:09,758 - thuner.data._utils - INFO - Loading regridder from file.
    2025-07-09 17:28:14,670 - thuner.visualize.attribute - INFO - Saving himawari_anvil figure for 2023-01-01T00:00:00.000000000.
    2025-07-09 17:28:26,831 - thuner.visualize.attribute - INFO - Visualizing attributes at time 2023-01-01T00:30:00.000000000.
    2025-07-09 17:28:26,837 - thuner.visualize.attribute - INFO - Visualizing attributes at time 2023-01-01T00:20:00.000000000.
    2025-07-09 17:28:26,837 - thuner.visualize.attribute - INFO - Visualizing attributes at time 2023-01-01T00:10:00.000000000.
    2025-07-09 17:28:28,502 - thuner.data.himawari - INFO - Converting himawari dataset for time 2023-01-01T00:30:00.
    2025-07-09 17:28:28,705 - thuner.visualize.attribute - INFO - Visualizing attributes at time 2023-01-01T00:40:00.000000000.
    2025-07-09 17:28:28,725 - thuner.data.himawari - INFO - Converting himawari dataset for time 2023-01-01T00:10:00.
    2025-07-09 17:28:28,749 - thuner.data.himawari - INFO - Converting himawari dataset for time 2023-01-01T00:20:00.
    2025-07-09 17:28:32,860 - thuner.data.himawari - INFO - Converting himawari dataset for time 2023-01-01T00:40:00.
    2025-07-09 17:28:36,140 - thuner.data.himawari - INFO - Regridding Himawari data.
    ...