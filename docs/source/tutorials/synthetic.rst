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

    # Set a flag for whether or not to remove existing output directories
    remove_existing_outputs = False
    
    # Parent directory for saving outputs
    base_local = Path.home() / "THUNER_output"
    start = "2005-11-13T00:00:00"
    end = "2005-11-13T02:00:00"
    
    output_parent = base_local / "runs/synthetic/geographic"
    if output_parent.exists() and remove_existing_outputs:
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

    2025-06-21 15:45:06,890 - thuner.option.grid - WARNING - altitude not specified. Using default altitudes.
    2025-06-21 15:45:06,893 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.

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

    2025-06-21 15:45:11,851 - thuner.track.track - INFO - Beginning thuner tracking. Saving output to /home/ewan/THUNER_output/runs/synthetic/geographic.
    2025-06-21 15:45:11,867 - thuner.track.track - INFO - Processing 2005-11-13T00:00:00.
    2025-06-21 15:45:11,869 - thuner.data.synthetic - INFO - Updating synthetic dataset for 2005-11-13T00:00:00.
    2025-06-21 15:45:32,267 - thuner.track.track - INFO - Processing hierarchy level 0.
    2025-06-21 15:45:32,268 - thuner.track.track - INFO - Tracking convective.
    2025-06-21 15:45:32,290 - thuner.detect.steiner - INFO - Compiling thuner.detect.steiner.steiner_scheme with Numba. Please wait.
    2025-06-21 15:45:59,187 - thuner.match.match - INFO - Matching convective objects.
    2025-06-21 15:45:59,201 - thuner.match.match - INFO - No current mask, or no objects in current mask.
    2025-06-21 15:45:59,216 - thuner.visualize.runtime - INFO - Creating runtime visualization figures.
    2025-06-21 15:46:05,368 - thuner.track.track - INFO - Processing 2005-11-13T00:10:00.
    2025-06-21 15:46:05,369 - thuner.data.synthetic - INFO - Updating synthetic dataset for 2005-11-13T00:10:00.
    2025-06-21 15:46:22,396 - thuner.track.track - INFO - Processing hierarchy level 0.
    2025-06-21 15:46:22,397 - thuner.track.track - INFO - Tracking convective.
    2025-06-21 15:46:22,414 - thuner.write.mask - INFO - Writing convective masks to /home/ewan/THUNER_output/runs/synthetic/geographic/masks/convective.zarr.
    2025-06-21 15:46:22,689 - thuner.match.match - INFO - Matching convective objects.
    ...

::

    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    Cell In[3], line 7
          1 times = np.arange(
          2     np.datetime64(start),
          3     np.datetime64(end) + np.timedelta64(10, "m"),
          4     np.timedelta64(10, "m"),
          5 )
          6 args = [times, data_options, grid_options, track_options, visualize_options]
    ----> 7 track.track(*args, output_directory=output_parent)

    File ~/Documents/THUNER/thuner/track/track.py:110, in track(times, data_options, grid_options, track_options, visualize_options, output_directory)
        108         track_level_args += [data_options, grid_options, track_options]
        109         track_level_args += [visualize_options, output_directory]
    --> 110         track_level(*track_level_args)
        112     current_time = next_time
        114 # Write final data to file
        115 # write.mask.write_final(tracks, track_options, output_directory)

    File ~/Documents/THUNER/thuner/track/track.py:155, in track_level(next_time, level_index, tracks, input_records, data_options, grid_options, track_options, visualize_options, output_directory)
        153 for obj in level_tracks.objects.keys():
        154     track_object_args = get_track_object_args(obj, level_options)
    --> 155     track_object(*track_object_args)
        157 return level_tracks

    File ~/Documents/THUNER/thuner/track/track.py:212, in track_object(next_time, level_index, obj, tracks, input_records, dataset_options, grid_options, track_options, visualize_options, output_directory)
        210 if object_tracks.times[-1] is not None:
        211     args = [input_records, tracks, object_options, grid_options]
    --> 212     attribute.record(*args)

    File ~/Documents/THUNER/thuner/attribute/attribute.py:77, in record(input_records, tracks, object_options, grid_options)
         75 for attribute_type in object_options.attributes.attribute_types:
         76     for attribute in attribute_type.attributes:
    ---> 77         attr = retrieve_attribute(kwargs, attribute)
         78         obj_attributes.attribute_types[attribute_type.name].update(attr)
         80 # Append the current attributes to the attributes dictionary

    File ~/Documents/THUNER/thuner/attribute/attribute.py:16, in retrieve_attribute(general_kwargs, attribute, member_object)
         13 def retrieve_attribute(general_kwargs, attribute, member_object=None):
         14     # Get the retrieval function and arguments for the attribute
         15     func_kwargs = general_kwargs.copy()
    ---> 16     keyword_arguments = attribute.retrieval.keyword_arguments
         17     func_kwargs.update(keyword_arguments)
         18     # Retrieval functions expect either "attribute" or "attribute_group"
         19     # keyword arguments. Infer correct argument name from attribute type.

    AttributeError: 'NoneType' object has no attribute 'keyword_arguments'

.. figure:: https://raw.githubusercontent.com/THUNER-project/THUNER/refs/heads/main/gallery/synthetic.gif
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

.. code-block:: python3
    :linenos:

    output_parent = base_local / "runs/synthetic/cartesian"
    if output_parent.exists() & remove_existing_outputs:
        shutil.rmtree(output_parent)
        
    times = np.arange(
        np.datetime64(start),
        np.datetime64(end) + np.timedelta64(10, "m"),
        +np.timedelta64(10, "m"),
    )
    
    args = [times, data_options, grid_options, track_options, visualize_options]
    track.track(*args, output_directory=output_parent)