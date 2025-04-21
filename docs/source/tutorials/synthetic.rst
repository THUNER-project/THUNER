Testing: Synthetic Data
=======================

For more detailed explanations of THUNER’s usage and features, see the
GridRad Severe demo/tutorial.

.. code-block:: python3
    :linenos:

    """Synthetic data demo/test."""
    
    %load_ext autoreload
    %autoreload 2
    from pathlib import Path
    import shutil
    import numpy as np
    import thuner.data as data
    import thuner.default as default
    import thuner.track.track as track
    import thuner.option as option
    import thuner.data.synthetic as synthetic

.. code-block:: text

    The autoreload extension is already loaded. To reload it, use:
      %reload_ext autoreload

.. code-block:: python3
    :linenos:

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
            center_longitude=lon[(i+1)*len(lon) // 6],
            direction=-np.pi / 4 + i * np.pi / 6,
            speed=30-4*i,
            horizontal_radius=5+4*i,
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

.. code-block:: text

    2025-02-14 21:22:44,245 - thuner.option.grid - WARNING - altitude not specified. Using default altitudes.
    2025-02-14 21:22:44,247 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.

.. code-block:: python3
    :linenos:

    times = np.arange(
        np.datetime64(start),
        np.datetime64(end) + np.timedelta64(10, "m"),
        np.timedelta64(10, "m"),
    )
    args = [times, data_options, grid_options, track_options, visualize_options]
    track.track(*args, output_directory=output_parent)

.. code-block:: text

    2025-02-14 21:22:45,520 - thuner.track.track - INFO - Beginning thuner tracking. Saving output to /home/ewan/THUNER_output/runs/synthetic/geographic.
    2025-02-14 21:22:45,527 - thuner.track.track - INFO - Processing 2005-11-13T00:00:00.
    2025-02-14 21:22:45,529 - thuner.data.synthetic - INFO - Updating synthetic dataset for 2005-11-13T00:00:00.
    2025-02-14 21:22:48,979 - thuner.track.track - INFO - Processing hierarchy level 0.
    2025-02-14 21:22:48,980 - thuner.track.track - INFO - Tracking convective.
    2025-02-14 21:22:49,034 - thuner.match.match - INFO - Matching convective objects.
    2025-02-14 21:22:49,035 - thuner.match.match - INFO - No current mask, or no objects in current mask.
    2025-02-14 21:22:49,040 - thuner.visualize.runtime - INFO - Creating runtime visualization figures.
    2025-02-14 21:22:52,861 - thuner.track.track - INFO - Processing 2005-11-13T00:10:00.
    2025-02-14 21:22:52,863 - thuner.data.synthetic - INFO - Updating synthetic dataset for 2005-11-13T00:10:00.
    2025-02-14 21:22:55,888 - thuner.track.track - INFO - Processing hierarchy level 0.
    2025-02-14 21:22:55,889 - thuner.track.track - INFO - Tracking convective.
    2025-02-14 21:22:55,896 - thuner.write.mask - INFO - Writing convective masks to /home/ewan/THUNER_output/runs/synthetic/geographic/masks/convective.zarr.
    2025-02-14 21:22:55,968 - thuner.match.match - INFO - Matching convective objects.
    2025-02-14 21:22:56,076 - thuner.match.match - INFO - New matchable objects. Initializing match record.
    ...

.. code-block:: python3
    :linenos:

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

.. code-block:: text

    2025-02-14 21:18:32,413 - thuner.option.grid - WARNING - altitude not specified. Using default altitudes.

.. code-block:: python3
    :linenos:

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

.. code-block:: text

    2025-02-14 21:20:50,343 - thuner.track.track - INFO - Beginning thuner tracking. Saving output to /home/ewan/THUNER_output/runs/synthetic/cartesian.
    2025-02-14 21:20:50,346 - thuner.track.track - INFO - Processing 2005-11-13T00:00:00.
    2025-02-14 21:20:50,348 - thuner.data.synthetic - INFO - Updating synthetic dataset for 2005-11-13T00:00:00.
    2025-02-14 21:20:52,888 - thuner.track.track - INFO - Processing hierarchy level 0.
    2025-02-14 21:20:52,890 - thuner.track.track - INFO - Tracking convective.
    2025-02-14 21:20:52,973 - thuner.match.match - INFO - Matching convective objects.
    2025-02-14 21:20:52,978 - thuner.match.match - INFO - No current mask, or no objects in current mask.
    2025-02-14 21:20:52,984 - thuner.visualize.runtime - INFO - Creating runtime visualization figures.
    2025-02-14 21:20:56,862 - thuner.track.track - INFO - Processing 2005-11-13T00:10:00.
    2025-02-14 21:20:56,863 - thuner.data.synthetic - INFO - Updating synthetic dataset for 2005-11-13T00:10:00.
    2025-02-14 21:20:58,888 - thuner.track.track - INFO - Processing hierarchy level 0.
    2025-02-14 21:20:58,889 - thuner.track.track - INFO - Tracking convective.
    2025-02-14 21:20:58,895 - thuner.write.mask - INFO - Writing convective masks to /home/ewan/THUNER_output/runs/synthetic/cartesian/masks/convective.zarr.
    2025-02-14 21:20:58,959 - thuner.match.match - INFO - Matching convective objects.
    2025-02-14 21:20:59,059 - thuner.match.match - INFO - New matchable objects. Initializing match record.
    2025-02-14 21:20:59,069 - thuner.visualize.runtime - INFO - Creating runtime visualization figures.
    2025-02-14 21:21:04,505 - thuner.attribute.attribute - INFO - Recording convective attributes.
    2025-02-14 21:21:04,509 - thuner.track.track - INFO - Processing 2005-11-13T00:20:00.
    ...