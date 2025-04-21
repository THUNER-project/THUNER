The TINT/MINT Approach: CPOL
============================

For more detailed explanations of THUNER’s basic usage and features, see
the GridRad Severe demo/tutorial.

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
    
    output_parent = base_local / "runs/cpol/geographic"
    options_directory = output_parent / "options"
    visualize_directory = output_parent / "visualize"
    
    # Remove the output parent directory if it already exists
    if output_parent.exists():
        shutil.rmtree(output_parent)

.. code-block:: python3
    :linenos:

    # Create the dataset options
    start = "2005-11-13T13:00:00"
    end = "2005-11-13T20:00:00"
    times_dict = {"start": start, "end": end}
    cpol_options = data.aura.CPOLOptions(**times_dict)
    
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
    # Modify the default track options to demonstrate the tracking
    # of both convective objects, and mesoscale convective systems, 
    # which are built out of convective, middle and stratiform echo objects, 
    # within the same THUNER run.
    new_convective_options = track_options.levels[0].object_by_name("convective")
    core = attribute.core.default_tracked()
    attributes = option.attribute.Attributes(name="convective", attribute_types=[core])
    track_options.levels[0].object_by_name("convective").attributes = attributes
    mint_tracking = option.track.MintOptions()
    track_options.levels[0].object_by_name("convective").tracking = mint_tracking
    track_options.model_validate(track_options) # Revalidate the track_options model
    track_options.to_yaml(options_directory / "track.yml")
    
    # Create the visualize_options
    visualize_options = default.runtime(visualize_directory=visualize_directory)
    visualize_options.to_yaml(options_directory / "visualize.yml")

.. code-block:: text

    2025-04-21 13:42:31,178 - thuner.data.aura - INFO - Generating cpol filepaths.
    2025-04-21 13:42:31,199 - thuner.data.era5 - INFO - Generating era5 filepaths.
    2025-04-21 13:42:31,209 - thuner.data.era5 - INFO - Generating era5 filepaths.
    2025-04-21 13:42:31,307 - thuner.option.grid - WARNING - altitude not specified. Using default altitudes.
    2025-04-21 13:42:31,313 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.

.. code-block:: python3
    :linenos:

    times = data._utils.generate_times(data_options.dataset_by_name("cpol"))
    args = [times, data_options, grid_options, track_options, visualize_options]
    # parallel.track(*args, output_directory=output_parent)
    track.track(*args, output_directory=output_parent)

.. code-block:: text

    2025-04-21 12:04:38,312 - thuner.track.track - INFO - Beginning thuner tracking. Saving output to /home/ewan/THUNER_output/runs/cpol/geographic.

::

    ---------------------------------------------------------------------------

    ValidationError                           Traceback (most recent call last)

    Cell In[29], line 4
          2 args = [times, data_options, grid_options, track_options, visualize_options]
          3 # parallel.track(*args, output_directory=output_parent)
    ----> 4 track.track(*args, output_directory=output_parent)

    File ~/Documents/THUNER/thuner/track/track.py:68, in track(times, data_options, grid_options, track_options, visualize_options, output_directory)
         45 """
         46 Track objects described in track_options, in the datasets described in
         47 data_options, using the grid described in grid_options.
       (...)
         65     Defaults to None.
         66 """
         67 logger.info("Beginning thuner tracking. Saving output to %s.", output_directory)
    ---> 68 tracks = Tracks(track_options=track_options)
         69 input_records = InputRecords(data_options=data_options)
         71 consolidated_options = consolidate_options(
         72     track_options, data_options, grid_options, visualize_options
         73 )

    File ~/miniconda/envs/THUNER/lib/python3.10/site-packages/pydantic/main.py:253, in BaseModel.__init__(self, **data)
        251 # `__tracebackhide__` tells pytest and some other tools to omit this function from tracebacks
        252 __tracebackhide__ = True
    --> 253 validated_self = self.__pydantic_validator__.validate_python(data, self_instance=self)
        254 if self is not validated_self:
        255     warnings.warn(
        256         'A custom validator is returning a value other than `self`.\n'
        257         "Returning anything other than `self` from a top level model validator isn't supported when validating via `__init__`.\n"
        258         'See the `model_validator` docs (https://docs.pydantic.dev/latest/concepts/validators/#model-validators) for more details.',
        259         stacklevel=2,
        260     )

    ValidationError: 1 validation error for Tracks
    attribute_options
      Input should be a valid dictionary or instance of Attributes [type=model_type, input_value=AttributeType(type='Attri...ities.')], dataset=None), input_type=AttributeType]
        For further information visit https://errors.pydantic.dev/2.11/v/model_type

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
    
    times = data._utils.generate_times(data_options.dataset_by_name("cpol"))
    args = [times, data_options, grid_options, track_options, visualize_options]
    track.track(*args, output_directory=output_parent)

.. code-block:: python3
    :linenos:

    analysis_options = analyze.mcs.AnalysisOptions()
    analysis_options.to_yaml(options_directory / "analysis.yml")
    # utils.save_options(analysis_options, filename="analysis", options_directory=output_directory / "options")
    analyze.mcs.process_velocities(output_parent)
    analyze.mcs.quality_control(output_parent, analysis_options)
    # analyze.mcs.classify_all(output_parent, analysis_options)

.. code-block:: python3
    :linenos:

    figure_name = "mcs_attributes"
    kwargs = {"style": "presentation", "attributes": ["velocity", "offset"]}
    figure_options = option.visualize.HorizontalAttributeOptions(name=figure_name, **kwargs)
    
    args = [output_parent, start, end, figure_options]
    args_dict = {"parallel_figure": True, "by_date": False, "num_processes": 4}
    visualize.attribute.mcs_series(*args, **args_dict)