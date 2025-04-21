Basics: GridRad Severe
======================

This demo/tutorial illustrates the basics of THUNER by tracking and
visualizing mesoscale convective system (MCS) objects in `GridRad
Severe <https://doi.org/10.5065/2B46-1A97>`__ data. See `Short et
al. (2023) <https://doi.org/10.1175/MWR-D-22-0146.1>`__ for
methodological details. By the end of the notebook, you should be able
to generate the animation below.

.. figure::
   https://raw.githubusercontent.com/THUNER-project/THUNER/refs/heads/main/gallery/mcs_gridrad_20100120.gif
   :alt: Animation depicting tracked MCSs.

   Animation depicting tracked MCSs.

Setup
-----

First, import the requisite modules.

.. code-block:: python3
    :linenos:

    """GridRad Severe demo/test."""
    
    %load_ext autoreload
    %autoreload 2
    
    from pathlib import Path
    import shutil
    import yaml
    import numpy as np
    import xarray as xr
    import thuner.data as data
    import thuner.option as option
    import thuner.analyze as analyze
    import thuner.parallel as parallel
    import thuner.visualize as visualize
    import thuner.attribute as attribute
    import thuner.default as default

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

Next, specify the folders where THUNER outputs will be saved.

.. code-block:: python3
    :linenos:

    # Parent directory for saving outputs
    base_local = Path.home() / "THUNER_output"
    output_parent = base_local / f"runs/gridrad/gridrad_demo"
    options_directory = output_parent / "options"
    visualize_directory = output_parent / "visualize"
    
    # Delete the options directory if it already exists
    # if output_parent.exists():
        # shutil.rmtree(output_parent)

Options
-------

We now specify the options for the THUNER run. Options classes in THUNER
are built on the :class:`pydantic.BaseModel`, which provides a simple way to
describe and validate options. Options objects are initialized using
keyword, value pairs. Below we specify the options for a GridRad Severe
dataset.

.. code-block:: python3
    :linenos:

    # Uncomment the line below to download the demo data if not already present
    # data.get_demo_data()
    event_directories = data.gridrad.get_event_directories(year=2010, base_local=base_local)
    event_directory = event_directories[0] # Take the first event from 2010 for the demo
    # Get the start and end times of the event, and the date of the event start
    start, end, event_start = data.gridrad.get_event_times(event_directory)
    times_dict = {"start": start, "end": end}
    gridrad_dict = {"event_start": event_start}
    gridrad_options = data.gridrad.GridRadSevereOptions(**times_dict, **gridrad_dict)

.. code-block:: text

    2025-04-21 17:47:38,591 - thuner.data.gridrad - INFO - Generating GridRad filepaths.

Options instances can be examined using the ``model_dump`` method, which
converts the instance to a dictionary.

.. code-block:: python3
    :linenos:

    gridrad_options.model_dump()

.. code-block:: text

    {'type': 'GridRadSevereOptions',
     'name': 'gridrad',
     'start': '2010-01-20T18:00:00',
     'end': '2010-01-21T03:30:00',
     'fields': ['reflectivity'],
     'parent_remote': 'https://data.rda.ucar.edu',
     'parent_local': '/home/ewan/THUNER_output/input_data/raw',
     'converted_options': {'type': 'ConvertedOptions',
      'save': False,
      'load': False,
      'parent_converted': None},
     'filepaths': ['/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T180000Z.nc',
      '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T181000Z.nc',
      '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T182000Z.nc',
      '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T183000Z.nc',
    ...

The code below illustrates how to examine the metadata associated with
each options type. Note the ``parent_local`` field, which provides the
parent directory on local disk containing the dataset. Analogously,
``parent_remote`` specifies the remote location of the data; which is
useful when one wants to access data from a remote location during the
tracking run. Note also the ``filepaths`` field, which provides a list
of the dataset’s absolute filepaths. The idea is that for standard
datasets, ``filepaths`` can be populated automatically by looking in the
``parent_local`` directory, assuming the same sub-directory structure as
in the dataset’s original location. If the dataset is nonstandard, the
``filepaths`` list can be explicitly provided by the user. For datasets
that do not yet have convenience classes in THUNER, the
:class:`thuner.utils.BaseDatasetOptions` class can be used. Note also the
``use`` field, which tells THUNER whether the dataset will be used to
``track`` or ``tag`` objects. Tracking in THUNER means detecting objects
in a dataset, and matching those objects across time. Tagging means
attaching attributes from potentially different datasets to detected
objects.

.. code-block:: python3
    :linenos:

    for name, info in gridrad_options.__class__.model_fields.items():
        print(f"{name}: {info.description}")

.. code-block:: text

    type: Type of the options, i.e. the subclass name.
    name: Name of the dataset.
    start: Tracking start time.
    end: Tracking end time.
    fields: List of dataset fields, i.e. variables, to use. Fields should be given 
        using their thuner, i.e. CF-Conventions, names, e.g. 'reflectivity'.
    parent_remote: Data parent directory on remote storage.
    parent_local: Data parent directory on local storage.
    converted_options: Options for converted data.
    filepaths: List of filepaths to used for tracking.
    attempt_download: Whether to attempt to download the data.
    deque_length: Number of current/previous grids from this dataset to keep in memory. 
        Most tracking algorithms require at least two current/previous grids.
    use: Whether this dataset will be used for tagging or tracking.
    start_buffer: Minutes before interval start time to include. Useful for 
    ...

We will also create dataset options for ERA5 single-level and
pressure-level data, which we use for tagging the storms detected in the
GridRad Severe dataset with other attributes, e.g. ambient winds and
temperature.

.. code-block:: python3
    :linenos:

    era5_dict = {"latitude_range": [27, 39], "longitude_range": [-102, -89]}
    era5_pl_options = data.era5.ERA5Options(**times_dict, **era5_dict)
    era5_dict.update({"data_format": "single-levels"})
    era5_sl_options = data.era5.ERA5Options(**times_dict, **era5_dict)

.. code-block:: text

    2025-04-21 17:47:46,636 - thuner.data.era5 - INFO - Generating era5 filepaths.
    2025-04-21 17:47:46,639 - thuner.data.era5 - INFO - Generating era5 filepaths.

All the dataset options are grouped into a single
:class:`thuner.option.data.DataOptions` object, which is passed to the THUNER
tracking function. We also save these options as a YAML file.

.. code-block:: python3
    :linenos:

    datasets = [gridrad_options, era5_pl_options, era5_sl_options]
    data_options = option.data.DataOptions(datasets=datasets)
    data_options.to_yaml(options_directory / "data.yml")

Now create and save options describing the grid. If ``regrid`` is
``False`` and grid properties like ``altitude_spacing`` or
``geographic_spacing`` are set to ``None``, THUNER will attempt to infer
these from the tracking dataset.

.. code-block:: python3
    :linenos:

    # Create and save the grid_options dictionary
    kwargs = {"name": "geographic", "regrid": False, "altitude_spacing": None}
    kwargs.update({"geographic_spacing": None})
    grid_options = option.grid.GridOptions(**kwargs)
    grid_options.to_yaml(options_directory / "grid.yml")

.. code-block:: text

    2025-04-21 17:47:49,213 - thuner.option.grid - WARNING - altitude_spacing not specified. Will attempt to infer from input.
    2025-04-21 17:47:49,214 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.

Finally, we create options describing how the tracking should be
performed. In multi-feature tracking, some objects, like mesoscale
convective systems (MCSs), can be defined in terms of others, like
convective and stratiform echoes. THUNER’s approach is to first specify
object options seperately for each object type, e.g. convective echoes,
stratiform echoes, mesoscale convective systems, and so forth. Object
options are specified using ``pydantic`` models which inherit from
:class:`thuner.option.track.BaseObjectOptions`. Related objects are then
grouped together into :class:`thuner.option.track.LevelOptions` models. The
final :class:`thuner.option.track.TrackOptions` model, which is passed to the
tracking function, then contains a list of
:class:`thuner.option.track.LevelOptions` models. The idea is that “lower
level” objects, can comprise the building blocks of “higher level”
objects, with THUNER processing the former before the latter.

In this tutorial, level 0 objects are the convective, middle and
stratiform echo regions, and level 1 objects are mesoscale convective
systems defined by grouping the level 0 objects. Because
:class:`thuner.option.track.TrackOptions` models can be complex to construct,
a function for creating a default :class:`thuner.option.track.TrackOptions`
model matching the approach of `Short et
al. (2023) <https://doi.org/10.1175/MWR-D-22-0146.1>`__ is defined in
the module :mod:`thuner.default`.

.. code-block:: python3
    :linenos:

    # Create the track_options dictionary
    track_options = default.track(dataset_name="gridrad")
    # Show the options for the level 0 objects
    print(f"Level 0 objects list: {track_options.levels[0].object_names}")
    # Show the options for the level 1 objects
    print(f"Level 1 objects list: {track_options.levels[1].object_names}")

.. code-block:: text

    Level 0 objects list: ['convective', 'middle', 'anvil']
    Level 1 objects list: ['mcs']

Note a core component of the options for each object is the
``atributes`` field, which describes how object attributes like
position, velocity and area, are to be retrieved and stored. In THUNER,
the code for collecting object attributes is seperated out from the core
tracking code, allowing different attributes for different objects to be
swapped in and out as needed. Individual attributes are described by the
:class:`thuner.option.attribute.Attribute` model, where each
:class:`thuner.option.attribute.Attribute` will form a column of an output
CSV file.

Sometimes multiple :class:`thuner.option.attribute.Attribute` are grouped
into a :class:`thuner.option.attribute.AttributeGroup` model, in which all
attributes in the group are retrieved at once using the same method. For
instance, attributes based on ellipse fitting, like major and minor
axis, eccentricity and orientation, form a
:class:`thuner.option.attribute.AttributeGroup`. Note however that each
member of the group will still form a seperate column in the output CSV
file.

Finally, collections of attributes and attribute groups are organized
into :class:`thuner.option.attribute.AttributeType` models. Each attribute
type corresponds to related attributes that will be stored in a single
CSV file. This makes the number of columns in each file much smaller,
and THUNER outputs easier to manage and inspect directly. To illustrate,
below we print the MCS object’s “core” attribute type options.

.. code-block:: python3
    :linenos:

    # Show the options for mcs coordinate attributes
    mcs_attributes = track_options.object_by_name("mcs").attributes
    core_mcs_attributes = mcs_attributes.attribute_type_by_name("core")
    core_mcs_attributes.model_dump()

.. code-block:: text

    {'type': 'AttributeType',
     'name': 'core',
     'description': 'Core attributes of tracked object, e.g. position and velocities.',
     'attributes': [{'type': 'Time',
       'name': 'time',
       'retrieval': {'type': 'Retrieval',
        'function': <function thuner.attribute.core.time_from_tracks(attribute: thuner.option.attribute.Attribute, object_tracks)>,
        'keyword_arguments': {}},
       'data_type': numpy.datetime64,
       'precision': None,
       'description': 'Time taken from the tracking process.',
       'units': 'yyyy-mm-dd hh:mm:ss'},
      {'type': 'RecordUniversalID',
       'name': 'universal_id',
       'retrieval': {'type': 'Retrieval',
    ...

The default :class:`thuner.option.track.TrackOptions` use “local” and
“global” cross-correlations to measure object velocities, as described
by `Raut et al. (2021) <https://doi.org/10.1175/JAMC-D-20-0119.1>`__ and
`Short et al. (2023) <https://doi.org/10.1175/MWR-D-22-0146.1>`__. For
GridRad severe, we modify this approach slightly so that “global”
cross-correlations are calculated using boxes encompassing each object,
with a margin of 70 km around the object. Note that pydantic models are
automatically validated when first created. Because we are changing the
model instance, we should revalidate the object options model to check
we haven’t broken anything.

.. code-block:: python3
    :linenos:

    track_options.levels[1].objects[0].tracking.unique_global_flow = False
    track_options.levels[1].objects[0].tracking.global_flow_margin = 70
    track_options.levels[1].objects[0].revalidate()
    track_options.to_yaml(options_directory / "track.yml")

Users can also specify visualization options for generating figures
during a tracking run. Uncomment the line below to generate figures that
visualize the matching algorithm - naturally this makes a tracking run
much slower.

.. code-block:: python3
    :linenos:

    visualize_options = None
    # visualize_options = default.runtime(visualize_directory=visualize_directory)
    # visualize_options.to_yaml(options_directory / "visualize.yml")

Tracking
--------

To perform the tracking run, we need an iterable of the times at which
objects will be detected and tracked. The convenience function
:func:`thuner.data.generate_times` creates a generator from the dataset
options for the tracking dataset. We can then pass this generator, and
the various options, to the tracking function :func:`thuner.parallel.track`.
During the tracking run, outputs will be created in the
``output_parent`` directory, within the subfolders ``interval_0``,
``interval_1`` etc, which represent subintervals of the time period
being tracked. At the end of the run, these outputs are stiched
together.

.. code-block:: python3
    :linenos:

    times = data.generate_times(data_options.dataset_by_name("gridrad"))
    args = [times, data_options, grid_options, track_options, visualize_options]
    num_processes = 4 # If visualize_options is not None, num_processes must be 1
    kwargs = {"output_directory": output_parent, "num_processes": num_processes}
    # In parallel tracking runs, we need to tell the tracking function which dataset to use
    # for tracking, so the subinterval data_options can be generated correctly
    kwargs.update({"dataset_name": "gridrad"})
    parallel.track(*args, **kwargs)

.. code-block:: text

    2025-04-21 16:13:39,189 - thuner.parallel - INFO - Beginning parallel tracking with 4 processes.
    2025-04-21 16:13:45,066 - thuner.parallel - INFO - {'type': 'GridRadSevereOptions', 'name': 'gridrad', 'start': '2010-01-20T18:00:00', 'end': '2010-01-21T03:30:00', 'fields': ['reflectivity'], 'parent_remote': 'https://data.rda.ucar.edu', 'parent_local': '/home/ewan/THUNER_output/input_data/raw', 'converted_options': {'type': 'ConvertedOptions', 'save': False, 'load': False, 'parent_converted': None}, 'filepaths': ['/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T180000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T181000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T182000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T183000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T184000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T185000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T190000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T191000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T192000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T193000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T194000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T195000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T200000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T201000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T202000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T203000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T204000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T205000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T210000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T211000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T212000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T213000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T214000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T215000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T220000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T221000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T222000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T223000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T224000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T225000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T230000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T231000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T232000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T233000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T234000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T235000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T000000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T001000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T002000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T003000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T004000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T005000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T010000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T011000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T012000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T013000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T014000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T015000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T020000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T021000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T022000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T023000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T024000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T025000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T030000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T031000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T032000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T033000Z.nc'], 'attempt_download': False, 'deque_length': 2, 'use': 'track', 'start_buffer': -120, 'end_buffer': 0, 'event_start': '2010-01-20', 'dataset_id': 'ds841.6', 'version': 'v4_2', 'obs_thresh': 2}
    2025-04-21 16:13:45,068 - thuner.parallel - INFO - {'type': 'ERA5Options', 'name': 'era5_pl', 'start': '2010-01-20T18:00:00', 'end': '2010-01-21T03:30:00', 'fields': ['u', 'v', 'z', 'r', 't'], 'parent_remote': '/g/data/rt52', 'parent_local': '/home/ewan/THUNER_output/input_data/raw', 'converted_options': {'type': 'ConvertedOptions', 'save': False, 'load': False, 'parent_converted': None}, 'filepaths': {'u': ['/home/ewan/THUNER_output/input_data/raw/era5_monthly_39N_102W_27N_89W/era5/pressure-levels/reanalysis/u/2010/u_era5_oper_pl_20100101-20100131.nc'], 'v': ['/home/ewan/THUNER_output/input_data/raw/era5_monthly_39N_102W_27N_89W/era5/pressure-levels/reanalysis/v/2010/v_era5_oper_pl_20100101-20100131.nc'], 'z': ['/home/ewan/THUNER_output/input_data/raw/era5_monthly_39N_102W_27N_89W/era5/pressure-levels/reanalysis/z/2010/z_era5_oper_pl_20100101-20100131.nc'], 'r': ['/home/ewan/THUNER_output/input_data/raw/era5_monthly_39N_102W_27N_89W/era5/pressure-levels/reanalysis/r/2010/r_era5_oper_pl_20100101-20100131.nc'], 't': ['/home/ewan/THUNER_output/input_data/raw/era5_monthly_39N_102W_27N_89W/era5/pressure-levels/reanalysis/t/2010/t_era5_oper_pl_20100101-20100131.nc']}, 'attempt_download': False, 'deque_length': 2, 'use': 'tag', 'start_buffer': -120, 'end_buffer': 0, 'latitude_range': [27.0, 39.0], 'longitude_range': [-102.0, -89.0], 'mode': 'reanalysis', 'data_format': 'pressure-levels', 'pressure_levels': ['1000', '975', '950', '925', '900', '875', '850', '825', '800', '775', '750', '700', '650', '600', '550', '500', '450', '400', '350', '300', '250', '225', '200', '175', '150', '125', '100', '70', '50', '30', '20', '10', '7', '5', '3', '2', '1'], 'storage': 'monthly'}
    2025-04-21 16:13:45,068 - thuner.parallel - INFO - {'type': 'ERA5Options', 'name': 'era5_sl', 'start': '2010-01-20T18:00:00', 'end': '2010-01-21T03:30:00', 'fields': ['cape', 'cin'], 'parent_remote': '/g/data/rt52', 'parent_local': '/home/ewan/THUNER_output/input_data/raw', 'converted_options': {'type': 'ConvertedOptions', 'save': False, 'load': False, 'parent_converted': None}, 'filepaths': {'cape': ['/home/ewan/THUNER_output/input_data/raw/era5_monthly_39N_102W_27N_89W/era5/single-levels/reanalysis/cape/2010/cape_era5_oper_sfc_20100101-20100131.nc'], 'cin': ['/home/ewan/THUNER_output/input_data/raw/era5_monthly_39N_102W_27N_89W/era5/single-levels/reanalysis/cin/2010/cin_era5_oper_sfc_20100101-20100131.nc']}, 'attempt_download': False, 'deque_length': 2, 'use': 'tag', 'start_buffer': -120, 'end_buffer': 0, 'latitude_range': [27.0, 39.0], 'longitude_range': [-102.0, -89.0], 'mode': 'reanalysis', 'data_format': 'single-levels', 'pressure_levels': None, 'storage': 'monthly'}
    2025-04-21 16:13:45,120 - thuner.parallel - INFO - {'type': 'GridRadSevereOptions', 'name': 'gridrad', 'start': '2010-01-20T18:00:00', 'end': '2010-01-21T03:30:00', 'fields': ['reflectivity'], 'parent_remote': 'https://data.rda.ucar.edu', 'parent_local': '/home/ewan/THUNER_output/input_data/raw', 'converted_options': {'type': 'ConvertedOptions', 'save': False, 'load': False, 'parent_converted': None}, 'filepaths': ['/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T180000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T181000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T182000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T183000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T184000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T185000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T190000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T191000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T192000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T193000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T194000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T195000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T200000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T201000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T202000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T203000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T204000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T205000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T210000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T211000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T212000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T213000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T214000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T215000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T220000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T221000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T222000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T223000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T224000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T225000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T230000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T231000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T232000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T233000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T234000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T235000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T000000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T001000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T002000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T003000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T004000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T005000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T010000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T011000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T012000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T013000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T014000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T015000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T020000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T021000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T022000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T023000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T024000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T025000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T030000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T031000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T032000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T033000Z.nc'], 'attempt_download': False, 'deque_length': 2, 'use': 'track', 'start_buffer': -120, 'end_buffer': 0, 'event_start': '2010-01-20', 'dataset_id': 'ds841.6', 'version': 'v4_2', 'obs_thresh': 2}
    2025-04-21 16:13:45,122 - thuner.parallel - INFO - {'type': 'ERA5Options', 'name': 'era5_pl', 'start': '2010-01-20T18:00:00', 'end': '2010-01-21T03:30:00', 'fields': ['u', 'v', 'z', 'r', 't'], 'parent_remote': '/g/data/rt52', 'parent_local': '/home/ewan/THUNER_output/input_data/raw', 'converted_options': {'type': 'ConvertedOptions', 'save': False, 'load': False, 'parent_converted': None}, 'filepaths': {'u': ['/home/ewan/THUNER_output/input_data/raw/era5_monthly_39N_102W_27N_89W/era5/pressure-levels/reanalysis/u/2010/u_era5_oper_pl_20100101-20100131.nc'], 'v': ['/home/ewan/THUNER_output/input_data/raw/era5_monthly_39N_102W_27N_89W/era5/pressure-levels/reanalysis/v/2010/v_era5_oper_pl_20100101-20100131.nc'], 'z': ['/home/ewan/THUNER_output/input_data/raw/era5_monthly_39N_102W_27N_89W/era5/pressure-levels/reanalysis/z/2010/z_era5_oper_pl_20100101-20100131.nc'], 'r': ['/home/ewan/THUNER_output/input_data/raw/era5_monthly_39N_102W_27N_89W/era5/pressure-levels/reanalysis/r/2010/r_era5_oper_pl_20100101-20100131.nc'], 't': ['/home/ewan/THUNER_output/input_data/raw/era5_monthly_39N_102W_27N_89W/era5/pressure-levels/reanalysis/t/2010/t_era5_oper_pl_20100101-20100131.nc']}, 'attempt_download': False, 'deque_length': 2, 'use': 'tag', 'start_buffer': -120, 'end_buffer': 0, 'latitude_range': [27.0, 39.0], 'longitude_range': [-102.0, -89.0], 'mode': 'reanalysis', 'data_format': 'pressure-levels', 'pressure_levels': ['1000', '975', '950', '925', '900', '875', '850', '825', '800', '775', '750', '700', '650', '600', '550', '500', '450', '400', '350', '300', '250', '225', '200', '175', '150', '125', '100', '70', '50', '30', '20', '10', '7', '5', '3', '2', '1'], 'storage': 'monthly'}
    2025-04-21 16:13:45,123 - thuner.parallel - INFO - {'type': 'ERA5Options', 'name': 'era5_sl', 'start': '2010-01-20T18:00:00', 'end': '2010-01-21T03:30:00', 'fields': ['cape', 'cin'], 'parent_remote': '/g/data/rt52', 'parent_local': '/home/ewan/THUNER_output/input_data/raw', 'converted_options': {'type': 'ConvertedOptions', 'save': False, 'load': False, 'parent_converted': None}, 'filepaths': {'cape': ['/home/ewan/THUNER_output/input_data/raw/era5_monthly_39N_102W_27N_89W/era5/single-levels/reanalysis/cape/2010/cape_era5_oper_sfc_20100101-20100131.nc'], 'cin': ['/home/ewan/THUNER_output/input_data/raw/era5_monthly_39N_102W_27N_89W/era5/single-levels/reanalysis/cin/2010/cin_era5_oper_sfc_20100101-20100131.nc']}, 'attempt_download': False, 'deque_length': 2, 'use': 'tag', 'start_buffer': -120, 'end_buffer': 0, 'latitude_range': [27.0, 39.0], 'longitude_range': [-102.0, -89.0], 'mode': 'reanalysis', 'data_format': 'single-levels', 'pressure_levels': None, 'storage': 'monthly'}
    2025-04-21 16:13:45,124 - thuner.parallel - INFO - {'type': 'GridRadSevereOptions', 'name': 'gridrad', 'start': '2010-01-20T18:00:00', 'end': '2010-01-21T03:30:00', 'fields': ['reflectivity'], 'parent_remote': 'https://data.rda.ucar.edu', 'parent_local': '/home/ewan/THUNER_output/input_data/raw', 'converted_options': {'type': 'ConvertedOptions', 'save': False, 'load': False, 'parent_converted': None}, 'filepaths': ['/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T180000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T181000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T182000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T183000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T184000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T185000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T190000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T191000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T192000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T193000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T194000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T195000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T200000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T201000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T202000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T203000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T204000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T205000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T210000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T211000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T212000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T213000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T214000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T215000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T220000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T221000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T222000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T223000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T224000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T225000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T230000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T231000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T232000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T233000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T234000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T235000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T000000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T001000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T002000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T003000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T004000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T005000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T010000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T011000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T012000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T013000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T014000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T015000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T020000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T021000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T022000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T023000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T024000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T025000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T030000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T031000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T032000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T033000Z.nc'], 'attempt_download': False, 'deque_length': 2, 'use': 'track', 'start_buffer': -120, 'end_buffer': 0, 'event_start': '2010-01-20', 'dataset_id': 'ds841.6', 'version': 'v4_2', 'obs_thresh': 2}
    2025-04-21 16:13:45,125 - thuner.parallel - INFO - {'type': 'ERA5Options', 'name': 'era5_pl', 'start': '2010-01-20T18:00:00', 'end': '2010-01-21T03:30:00', 'fields': ['u', 'v', 'z', 'r', 't'], 'parent_remote': '/g/data/rt52', 'parent_local': '/home/ewan/THUNER_output/input_data/raw', 'converted_options': {'type': 'ConvertedOptions', 'save': False, 'load': False, 'parent_converted': None}, 'filepaths': {'u': ['/home/ewan/THUNER_output/input_data/raw/era5_monthly_39N_102W_27N_89W/era5/pressure-levels/reanalysis/u/2010/u_era5_oper_pl_20100101-20100131.nc'], 'v': ['/home/ewan/THUNER_output/input_data/raw/era5_monthly_39N_102W_27N_89W/era5/pressure-levels/reanalysis/v/2010/v_era5_oper_pl_20100101-20100131.nc'], 'z': ['/home/ewan/THUNER_output/input_data/raw/era5_monthly_39N_102W_27N_89W/era5/pressure-levels/reanalysis/z/2010/z_era5_oper_pl_20100101-20100131.nc'], 'r': ['/home/ewan/THUNER_output/input_data/raw/era5_monthly_39N_102W_27N_89W/era5/pressure-levels/reanalysis/r/2010/r_era5_oper_pl_20100101-20100131.nc'], 't': ['/home/ewan/THUNER_output/input_data/raw/era5_monthly_39N_102W_27N_89W/era5/pressure-levels/reanalysis/t/2010/t_era5_oper_pl_20100101-20100131.nc']}, 'attempt_download': False, 'deque_length': 2, 'use': 'tag', 'start_buffer': -120, 'end_buffer': 0, 'latitude_range': [27.0, 39.0], 'longitude_range': [-102.0, -89.0], 'mode': 'reanalysis', 'data_format': 'pressure-levels', 'pressure_levels': ['1000', '975', '950', '925', '900', '875', '850', '825', '800', '775', '750', '700', '650', '600', '550', '500', '450', '400', '350', '300', '250', '225', '200', '175', '150', '125', '100', '70', '50', '30', '20', '10', '7', '5', '3', '2', '1'], 'storage': 'monthly'}
    2025-04-21 16:13:45,126 - thuner.parallel - INFO - {'type': 'ERA5Options', 'name': 'era5_sl', 'start': '2010-01-20T18:00:00', 'end': '2010-01-21T03:30:00', 'fields': ['cape', 'cin'], 'parent_remote': '/g/data/rt52', 'parent_local': '/home/ewan/THUNER_output/input_data/raw', 'converted_options': {'type': 'ConvertedOptions', 'save': False, 'load': False, 'parent_converted': None}, 'filepaths': {'cape': ['/home/ewan/THUNER_output/input_data/raw/era5_monthly_39N_102W_27N_89W/era5/single-levels/reanalysis/cape/2010/cape_era5_oper_sfc_20100101-20100131.nc'], 'cin': ['/home/ewan/THUNER_output/input_data/raw/era5_monthly_39N_102W_27N_89W/era5/single-levels/reanalysis/cin/2010/cin_era5_oper_sfc_20100101-20100131.nc']}, 'attempt_download': False, 'deque_length': 2, 'use': 'tag', 'start_buffer': -120, 'end_buffer': 0, 'latitude_range': [27.0, 39.0], 'longitude_range': [-102.0, -89.0], 'mode': 'reanalysis', 'data_format': 'single-levels', 'pressure_levels': None, 'storage': 'monthly'}
    2025-04-21 16:13:45,272 - thuner.parallel - INFO - {'type': 'GridRadSevereOptions', 'name': 'gridrad', 'start': '2010-01-20T18:00:00', 'end': '2010-01-21T03:30:00', 'fields': ['reflectivity'], 'parent_remote': 'https://data.rda.ucar.edu', 'parent_local': '/home/ewan/THUNER_output/input_data/raw', 'converted_options': {'type': 'ConvertedOptions', 'save': False, 'load': False, 'parent_converted': None}, 'filepaths': ['/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T180000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T181000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T182000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T183000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T184000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T185000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T190000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T191000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T192000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T193000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T194000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T195000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T200000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T201000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T202000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T203000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T204000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T205000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T210000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T211000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T212000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T213000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T214000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T215000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T220000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T221000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T222000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T223000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T224000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T225000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T230000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T231000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T232000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T233000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T234000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T235000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T000000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T001000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T002000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T003000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T004000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T005000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T010000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T011000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T012000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T013000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T014000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T015000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T020000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T021000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T022000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T023000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T024000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T025000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T030000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T031000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T032000Z.nc', '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100121T033000Z.nc'], 'attempt_download': False, 'deque_length': 2, 'use': 'track', 'start_buffer': -120, 'end_buffer': 0, 'event_start': '2010-01-20', 'dataset_id': 'ds841.6', 'version': 'v4_2', 'obs_thresh': 2}
    2025-04-21 16:13:45,274 - thuner.parallel - INFO - {'type': 'ERA5Options', 'name': 'era5_pl', 'start': '2010-01-20T18:00:00', 'end': '2010-01-21T03:30:00', 'fields': ['u', 'v', 'z', 'r', 't'], 'parent_remote': '/g/data/rt52', 'parent_local': '/home/ewan/THUNER_output/input_data/raw', 'converted_options': {'type': 'ConvertedOptions', 'save': False, 'load': False, 'parent_converted': None}, 'filepaths': {'u': ['/home/ewan/THUNER_output/input_data/raw/era5_monthly_39N_102W_27N_89W/era5/pressure-levels/reanalysis/u/2010/u_era5_oper_pl_20100101-20100131.nc'], 'v': ['/home/ewan/THUNER_output/input_data/raw/era5_monthly_39N_102W_27N_89W/era5/pressure-levels/reanalysis/v/2010/v_era5_oper_pl_20100101-20100131.nc'], 'z': ['/home/ewan/THUNER_output/input_data/raw/era5_monthly_39N_102W_27N_89W/era5/pressure-levels/reanalysis/z/2010/z_era5_oper_pl_20100101-20100131.nc'], 'r': ['/home/ewan/THUNER_output/input_data/raw/era5_monthly_39N_102W_27N_89W/era5/pressure-levels/reanalysis/r/2010/r_era5_oper_pl_20100101-20100131.nc'], 't': ['/home/ewan/THUNER_output/input_data/raw/era5_monthly_39N_102W_27N_89W/era5/pressure-levels/reanalysis/t/2010/t_era5_oper_pl_20100101-20100131.nc']}, 'attempt_download': False, 'deque_length': 2, 'use': 'tag', 'start_buffer': -120, 'end_buffer': 0, 'latitude_range': [27.0, 39.0], 'longitude_range': [-102.0, -89.0], 'mode': 'reanalysis', 'data_format': 'pressure-levels', 'pressure_levels': ['1000', '975', '950', '925', '900', '875', '850', '825', '800', '775', '750', '700', '650', '600', '550', '500', '450', '400', '350', '300', '250', '225', '200', '175', '150', '125', '100', '70', '50', '30', '20', '10', '7', '5', '3', '2', '1'], 'storage': 'monthly'}
    2025-04-21 16:13:45,275 - thuner.parallel - INFO - {'type': 'ERA5Options', 'name': 'era5_sl', 'start': '2010-01-20T18:00:00', 'end': '2010-01-21T03:30:00', 'fields': ['cape', 'cin'], 'parent_remote': '/g/data/rt52', 'parent_local': '/home/ewan/THUNER_output/input_data/raw', 'converted_options': {'type': 'ConvertedOptions', 'save': False, 'load': False, 'parent_converted': None}, 'filepaths': {'cape': ['/home/ewan/THUNER_output/input_data/raw/era5_monthly_39N_102W_27N_89W/era5/single-levels/reanalysis/cape/2010/cape_era5_oper_sfc_20100101-20100131.nc'], 'cin': ['/home/ewan/THUNER_output/input_data/raw/era5_monthly_39N_102W_27N_89W/era5/single-levels/reanalysis/cin/2010/cin_era5_oper_sfc_20100101-20100131.nc']}, 'attempt_download': False, 'deque_length': 2, 'use': 'tag', 'start_buffer': -120, 'end_buffer': 0, 'latitude_range': [27.0, 39.0], 'longitude_range': [-102.0, -89.0], 'mode': 'reanalysis', 'data_format': 'single-levels', 'pressure_levels': None, 'storage': 'monthly'}
    2025-04-21 16:13:45,390 - thuner.track.track - INFO - Beginning thuner tracking. Saving output to /home/ewan/THUNER_output/runs/gridrad/gridrad_demo/interval_0.
    2025-04-21 16:13:45,448 - thuner.track.track - INFO - Beginning thuner tracking. Saving output to /home/ewan/THUNER_output/runs/gridrad/gridrad_demo/interval_2.
    ...

The outputs of the tracking run are saved in the ``output_parent``
directory. The options for the run are saved in human-readable YAML
files within the ``options`` directory. For reproducibility, Python
objects can be rebuilt from these YAML files by reading the YAML, and
passing this to the appropriate ``pydantic`` model.

.. code-block:: python3
    :linenos:

    with open(options_directory / "data.yml", "r") as f:
        data_options = option.data.DataOptions(**yaml.safe_load(f))
        # Note yaml.safe_load(f) is a dictionary.
        # Prepending with ** unpacks the dictionary into keyword/argument pairs.
    data_options.model_dump()

.. code-block:: text

    {'type': 'DataOptions',
     'datasets': [{'type': 'GridRadSevereOptions',
       'name': 'gridrad',
       'start': '2010-01-20T18:00:00',
       'end': '2010-01-21T03:30:00',
       'fields': ['reflectivity'],
       'parent_remote': 'https://data.rda.ucar.edu',
       'parent_local': '/home/ewan/THUNER_output/input_data/raw',
       'converted_options': {'type': 'ConvertedOptions',
        'save': False,
        'load': False,
        'parent_converted': None},
       'filepaths': ['/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T180000Z.nc',
        '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T181000Z.nc',
        '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T182000Z.nc',
    ...

The convenience function ``thuner.analyze.utils.read_options`` reloads
all options in the above way, storing the different options in a
dictionary.

.. code-block:: python3
    :linenos:

    all_options = analyze.utils.read_options(output_parent)
    all_options["data"].model_dump()

.. code-block:: text

    2025-04-21 16:17:35,297 - thuner.option.grid - WARNING - altitude_spacing not specified. Will attempt to infer from input.
    2025-04-21 16:17:35,299 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.

.. code-block:: text

    {'type': 'DataOptions',
     'datasets': [{'type': 'GridRadSevereOptions',
       'name': 'gridrad',
       'start': '2010-01-20T18:00:00',
       'end': '2010-01-21T03:30:00',
       'fields': ['reflectivity'],
       'parent_remote': 'https://data.rda.ucar.edu',
       'parent_local': '/home/ewan/THUNER_output/input_data/raw',
       'converted_options': {'type': 'ConvertedOptions',
        'save': False,
        'load': False,
        'parent_converted': None},
       'filepaths': ['/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T180000Z.nc',
        '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T181000Z.nc',
        '/home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T182000Z.nc',
    ...

Object attributes, e.g. MCS position, area and velocity, are saved as
CSV files in nested subfolders. Attribute metadata is recorded in YAML
files. One can then load attribute data using ``pandas.read_csv``. One
can also create an appropriately formatted :class:`pandas.DataFrame` using
the convenience function :func:`thuner.attribute.utils.read_attribute_csv`.

.. code-block:: python3
    :linenos:

    core = attribute.utils.read_attribute_csv(output_parent / "attributes/mcs/core.csv")
    print(core.head(20).to_string())

.. code-block:: text

                                     parents  latitude  longitude    area  u_flow  v_flow  u_displacement  v_displacement  echo_top_height
    time                universal_id                                                                                                      
    2010-01-20 18:00:00 1                NaN   30.8229   270.1562   598.6     8.3     7.7            13.3             0.0          13000.0
                        2                NaN   31.6979   270.6979   981.0     9.9     3.9            16.5             0.0          13000.0
    2010-01-20 18:10:00 1                NaN   30.8229   270.2396   589.3    10.0     7.7             3.3            -3.8          12000.0
                        2                NaN   31.6979   270.8021  1053.8     9.9     7.7             NaN             NaN          13000.0
    2010-01-20 18:20:00 1                NaN   30.8021   270.2604   736.9    10.0     7.7            23.3             7.7          13000.0
    2010-01-20 18:30:00 1                NaN   30.8438   270.4062   492.5    16.6     7.7             3.3            11.5          12000.0
    2010-01-20 18:40:00 1                NaN   30.9062   270.4271   460.0    10.0     7.7             NaN             NaN          12000.0
    2010-01-20 18:50:00 3                NaN   29.3854   269.5312   546.4    10.1    11.5             6.7             7.7          14000.0
    2010-01-20 19:00:00 3                NaN   29.4271   269.5729   597.5     6.7     7.7             NaN             NaN          14000.0
                        4                NaN   30.2812   267.0312   486.1    15.0     9.6            13.4            15.4          13000.0
    2010-01-20 19:10:00 4                NaN   30.3646   267.1146   619.8    10.0    11.6            13.3             7.7          13000.0
    2010-01-20 19:20:00 4                NaN   30.4062   267.1979   739.8    13.3     7.7             NaN             NaN          14000.0
    2010-01-20 21:20:00 5                NaN   31.2188   268.4896   779.4     8.3     3.9            23.2             0.0          14000.0
    ...

Records of the filepaths corresponding to each time of the tracking run
are saved in the ``records`` folder. These records are useful for
generating figures after a tracking run.

.. code-block:: python3
    :linenos:

    filepath = output_parent / "records/filepaths/gridrad.csv" 
    records = attribute.utils.read_attribute_csv(filepath)
    print(records.head(20).to_string())

.. code-block:: text

                                                                                                                          gridrad
    time                                                                                                                         
    2010-01-20 18:00:00  /home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T180000Z.nc
    2010-01-20 18:10:00  /home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T181000Z.nc
    2010-01-20 18:20:00  /home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T182000Z.nc
    2010-01-20 18:30:00  /home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T183000Z.nc
    2010-01-20 18:40:00  /home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T184000Z.nc
    2010-01-20 18:50:00  /home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T185000Z.nc
    2010-01-20 19:00:00  /home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T190000Z.nc
    2010-01-20 19:10:00  /home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T191000Z.nc
    2010-01-20 19:20:00  /home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T192000Z.nc
    2010-01-20 19:30:00  /home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T193000Z.nc
    2010-01-20 19:40:00  /home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T194000Z.nc
    2010-01-20 19:50:00  /home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T195000Z.nc
    2010-01-20 20:00:00  /home/ewan/THUNER_output/input_data/raw/d841006/volumes/2010/20100120/nexrad_3d_v4_2_20100120T200000Z.nc
    ...

Object masks are saved as ZARR files, which can be read using
:mod:`xarray`.

.. code-block:: python3
    :linenos:

    xr.open_dataset(output_parent / "masks/mcs.zarr").info()

.. code-block:: text

    xarray.Dataset {
    dimensions:
    	time = 57 ;
    	latitude = 576 ;
    	longitude = 624 ;
    
    variables:
    	uint32 anvil_mask(time, latitude, longitude) ;
    	uint32 convective_mask(time, latitude, longitude) ;
    	float32 latitude(latitude) ;
    	float32 longitude(longitude) ;
    	uint32 middle_mask(time, latitude, longitude) ;
    	datetime64[ns] time(time) ;
    
    // global attributes:
    ...

Analysis and Visualization
--------------------------

We can then perform analysis on the tracking run outputs. Below we
perform the MCS classifications discussed by `Short et
al. (2023) <https://doi.org/10.1175/MWR-D-22-0146.1>`__.

.. code-block:: python3
    :linenos:

    analysis_options = analyze.mcs.AnalysisOptions()
    analyze.mcs.process_velocities(output_parent, profile_dataset="era5_pl")
    analyze.mcs.quality_control(output_parent, analysis_options)
    analyze.mcs.classify_all(output_parent, analysis_options)
    filepath = output_parent / "analysis/classification.csv"
    classifications = attribute.utils.read_attribute_csv(filepath)
    print("\n" + classifications.head(20).to_string())

.. code-block:: text

    2025-04-21 13:40:13,695 - thuner.option.grid - WARNING - altitude_spacing not specified. Will attempt to infer from input.
    2025-04-21 13:40:13,696 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.
    2025-04-21 13:40:14,202 - thuner.option.grid - WARNING - altitude_spacing not specified. Will attempt to infer from input.
    2025-04-21 13:40:14,204 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.

.. code-block:: text

    
                                     stratiform_offset inflow relative_stratiform_offset                 tilt          propagation
    time                universal_id                                                                                              
    2010-01-20 18:00:00 1                      leading  right                       left           down-shear  shear-perpendicular
                        2                      leading  right                       left           down-shear  shear-perpendicular
    2010-01-20 18:10:00 1                      leading  right                       left           down-shear           down-shear
                        2                      leading  right                       left  shear-perpendicular  shear-perpendicular
    2010-01-20 18:20:00 1                      leading  right                       left           down-shear           down-shear
    2010-01-20 18:30:00 1                      leading  right                       left  shear-perpendicular  shear-perpendicular
    2010-01-20 18:40:00 1                      leading  right                       left           down-shear           down-shear
    2010-01-20 18:50:00 3                        right  right                       left           down-shear           down-shear
    2010-01-20 19:00:00 3                     trailing  right                    leading  shear-perpendicular           down-shear
                        4                      leading  right                    leading           down-shear           down-shear
    2010-01-20 19:10:00 4                        right  right                    leading           down-shear           down-shear
    2010-01-20 19:20:00 4                      leading  right                    leading           down-shear           down-shear
    ...

We can also generate figures and animations from the output. Below we
visualize the convective and stratiform regions of each MCS, displaying
each system’s velocity and stratiform-offset, and the boundaries of the
radar mosaic domain, as discussed by `Short et
al. (2023) <https://doi.org/10.1175/MWR-D-22-0146.1>`__. By default,
figures and animations are saved in the ``output_parent`` directory in
the ``visualize`` folder. The code below should generate an animation
``mcs_gridrad_20100120.gif``, matching the animation provided at the
start of the notebook.

.. code-block:: python3
    :linenos:

    figure_name = f"mcs_gridrad_{event_start.replace('-', '')}"
    kwargs = {"name": figure_name, "style": "presentation"}
    kwargs.update({"attributes": ["velocity", "offset"]})
    figure_options = option.visualize.HorizontalAttributeOptions(**kwargs)
    start_time = np.datetime64(start)
    end_time = np.datetime64(end)
    args = [output_parent, start_time, end_time, figure_options]
    args_dict = {"parallel_figure": True, "dt": 7200, "by_date": False, "num_processes": 4}
    visualize.attribute.mcs_series(*args, **args_dict)

.. code-block:: text

    2025-04-21 13:40:53,299 - thuner.option.grid - WARNING - altitude_spacing not specified. Will attempt to infer from input.
    2025-04-21 13:40:53,302 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.
    2025-04-21 13:40:53,778 - thuner.visualize.attribute - INFO - Visualizing MCS at time 2010-01-20T18:00:00.000000000.
    2025-04-21 13:40:57,953 - thuner.option.grid - WARNING - altitude_spacing not specified. Will attempt to infer from input.
    2025-04-21 13:40:57,954 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.
    2025-04-21 13:40:59,213 - thuner.visualize.attribute - INFO - Saving mcs_gridrad_20100120 figure for 2010-01-20T18:00:00.000000000.
    2025-04-21 13:41:09,572 - thuner.visualize.attribute - INFO - Visualizing MCS at time 2010-01-20T18:20:00.000000000.
    2025-04-21 13:41:09,574 - thuner.visualize.attribute - INFO - Visualizing MCS at time 2010-01-20T18:30:00.000000000.
    2025-04-21 13:41:09,576 - thuner.visualize.attribute - INFO - Visualizing MCS at time 2010-01-20T18:10:00.000000000.
    2025-04-21 13:41:10,029 - thuner.visualize.attribute - INFO - Visualizing MCS at time 2010-01-20T18:40:00.000000000.
    2025-04-21 13:41:15,216 - thuner.option.grid - WARNING - altitude_spacing not specified. Will attempt to infer from input.
    2025-04-21 13:41:15,217 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.
    2025-04-21 13:41:15,280 - thuner.option.grid - WARNING - altitude_spacing not specified. Will attempt to infer from input.
    2025-04-21 13:41:15,280 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.
    2025-04-21 13:41:15,308 - thuner.option.grid - WARNING - altitude_spacing not specified. Will attempt to infer from input.
    ...