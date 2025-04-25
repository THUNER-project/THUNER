Testing: Synthetic Data
=======================

The synthetic module is a work in progress. The idea is to allow
synthetic meteorological datasets to be readily created for testing
purposes. While an entire synthetic dataset could be created first, then
fed into THUNER in the usual way (see previous tutorials/demos) with
this module we instead generate the synthetic data as we go. The
approach avoids the need for storing large datasets.

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

    
    Welcome to the Thunderstorm Event Reconnaissance (THUNER) package 
    v0.0.16! This package is still in testing and development. Please visit 
    github.com/THUNER-project/THUNER for examples, and to report issues or contribute.
     
    THUNER is a flexible toolkit for performing multi-feature detection, 
    tracking, tagging and analysis of events within meteorological datasets. 
    The intended application is to convective weather events. For examples 
    and instructions, see https://github.com/THUNER-project/THUNER and 
    https://thuner.readthedocs.io/en/latest/. If you use THUNER in your research, consider 
    citing the following papers;
    
    Short et al. (2023), doi: 10.1175/MWR-D-22-0146.1
    Raut et al. (2021), doi: 10.1175/JAMC-D-20-0119.1
    Fridlind et al. (2019), doi: 10.5194/amt-12-2979-2019
    ...

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

    2025-04-25 00:32:55,758 - thuner.option.grid - WARNING - altitude not specified. Using default altitudes.
    2025-04-25 00:32:55,759 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.

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

    2025-04-25 00:32:55,915 - thuner.track.track - INFO - Beginning thuner tracking. Saving output to /home/ewan/THUNER_output/runs/synthetic/geographic.
    2025-04-25 00:32:55,925 - thuner.track.track - INFO - Processing 2005-11-13T00:00:00.
    2025-04-25 00:32:55,931 - thuner.data.synthetic - INFO - Updating synthetic dataset for 2005-11-13T00:00:00.
    2025-04-25 00:33:10,261 - thuner.track.track - INFO - Processing hierarchy level 0.
    2025-04-25 00:33:10,262 - thuner.track.track - INFO - Tracking convective.
    2025-04-25 00:33:10,269 - thuner.detect.steiner - INFO - Compiling thuner.detect.steiner.steiner_scheme with Numba. Please wait.
    2025-04-25 00:33:22,544 - thuner.match.match - INFO - Matching convective objects.
    2025-04-25 00:33:22,546 - thuner.match.match - INFO - No current mask, or no objects in current mask.
    2025-04-25 00:33:22,568 - thuner.visualize.runtime - INFO - Creating runtime visualization figures.
    2025-04-25 00:33:26,061 - thuner.track.track - INFO - Processing 2005-11-13T00:10:00.
    2025-04-25 00:33:26,062 - thuner.data.synthetic - INFO - Updating synthetic dataset for 2005-11-13T00:10:00.
    2025-04-25 00:33:40,854 - thuner.track.track - INFO - Processing hierarchy level 0.
    2025-04-25 00:33:40,855 - thuner.track.track - INFO - Tracking convective.
    2025-04-25 00:33:40,864 - thuner.write.mask - INFO - Writing convective masks to /home/ewan/THUNER_output/runs/synthetic/geographic/masks/convective.zarr.
    2025-04-25 00:33:41,070 - thuner.match.match - INFO - Matching convective objects.
    ...

.. figure::
   https://raw.githubusercontent.com/THUNER-project/THUNER/refs/heads/main/gallery/synthetic.gif
   :alt: THUNER applied to synthetic data.

   THUNER applied to synthetic data.

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

    2025-04-25 00:39:35,251 - thuner.option.grid - WARNING - altitude not specified. Using default altitudes.

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

    2025-04-25 00:39:36,701 - thuner.track.track - INFO - Beginning thuner tracking. Saving output to /home/ewan/THUNER_output/runs/synthetic/cartesian.
    2025-04-25 00:39:36,704 - thuner.track.track - INFO - Processing 2005-11-13T00:00:00.
    2025-04-25 00:39:36,707 - thuner.data.synthetic - INFO - Updating synthetic dataset for 2005-11-13T00:00:00.
    2025-04-25 00:39:50,828 - thuner.track.track - INFO - Processing hierarchy level 0.
    2025-04-25 00:39:50,828 - thuner.track.track - INFO - Tracking convective.
    2025-04-25 00:39:50,879 - thuner.match.match - INFO - Matching convective objects.
    2025-04-25 00:39:50,880 - thuner.match.match - INFO - No current mask, or no objects in current mask.
    2025-04-25 00:39:50,885 - thuner.visualize.runtime - INFO - Creating runtime visualization figures.
    2025-04-25 00:39:53,182 - thuner.track.track - INFO - Processing 2005-11-13T00:10:00.
    2025-04-25 00:39:53,185 - thuner.data.synthetic - INFO - Updating synthetic dataset for 2005-11-13T00:10:00.
    2025-04-25 00:40:04,086 - thuner.track.track - INFO - Processing hierarchy level 0.
    2025-04-25 00:40:04,087 - thuner.track.track - INFO - Tracking convective.
    2025-04-25 00:40:04,092 - thuner.write.mask - INFO - Writing convective masks to /home/ewan/THUNER_output/runs/synthetic/cartesian/masks/convective.zarr.
    2025-04-25 00:40:04,171 - thuner.match.match - INFO - Matching convective objects.
    2025-04-25 00:40:04,275 - thuner.match.match - INFO - New matchable objects. Initializing match record.
    2025-04-25 00:40:04,281 - thuner.visualize.runtime - INFO - Creating runtime visualization figures.
    2025-04-25 00:40:08,025 - thuner.attribute.attribute - INFO - Recording convective attributes.
    2025-04-25 00:40:08,029 - thuner.track.track - INFO - Processing 2005-11-13T00:20:00.
    ...