Basics: GridRad Severe
======================

This demo/tutorial illustrates the basics of THUNER by tracking and
visualizing mesoscale convective system (MCS) objects in `GridRad
Severe <https://doi.org/10.5065/2B46-1A97>`__ data. See `Short et
al. (2023) <https://doi.org/10.1175/MWR-D-22-0146.1>`__ for
methodological details. By the end of the notebook, you should be able
to generate the animation below.

.. figure:: https://raw.githubusercontent.com/THUNER-project/THUNER/refs/heads/main/gallery/mcs_gridrad_20100120.gif
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
    import thuner.config as config
    import thuner.utils as utils

Next, specify the folders where THUNER outputs will be saved. Note that
THUNER stores a fallback output directory in a config file, accessible
via the functions :func:`thuner.config.set_outputs_directory` and
:func:`thuner.config.get_outputs_directory`. By default, this fallback
directory is ``Path.home() / THUNER_output``.

.. code-block:: python3
    :linenos:

    # Set a flag for whether or not to remove existing output directories
    remove_existing_outputs = False
    
    # Parent directory for saving outputs
    base_local = config.get_outputs_directory()
    output_parent = base_local / f"runs/gridrad/gridrad_demo"
    options_directory = output_parent / "options"
    visualize_directory = output_parent / "visualize"
    
    # Delete the output directory for the run if it already exists
    if output_parent.exists() & remove_existing_outputs:
        shutil.rmtree(output_parent)

Next download the demo data for the tutorial, if you haven’t already.

.. code-block:: python3
    :linenos:

    # Download the demo data
    remote_directory = "s3://thuner-storage/THUNER_output/input_data/raw/d81006"
    data.get_demo_data(base_local, remote_directory)
    remote_directory = "s3://thuner-storage/THUNER_output/input_data/raw/"
    remote_directory += "era5_monthly_39N_102W_27N_89W"
    data.get_demo_data(base_local, remote_directory)

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

Options instances can be examined using the ``model_dump`` method, which
converts the instance to a dictionary.

.. code-block:: python3
    :linenos:

    gridrad_options.model_dump()

The ``model_summary()`` method of an options instance returns a string
summary of the fields in the model. Note the ``parent_local`` field,
which provides the parent directory on local disk containing the
dataset. Analogously, ``parent_remote`` specifies the remote location of
the data; which is useful when one wants to access data from a remote
location during the tracking run. Note also the ``filepaths`` field,
which provides a list of the dataset’s absolute filepaths. The idea is
that for standard datasets, ``filepaths`` can be populated automatically
by looking in the ``parent_local`` directory, assuming the same
sub-directory structure as in the dataset’s original location. If the
dataset is nonstandard, the ``filepaths`` list can be explicitly
provided by the user. For datasets that do not yet have convenience
classes in THUNER, the :class:`thuner.utils.BaseDatasetOptions` class can be
used. Note also the ``use`` field, which tells THUNER whether the
dataset will be used to ``track`` or ``tag`` objects. Tracking in THUNER
means detecting objects in a dataset, and matching those objects across
time. Tagging means attaching attributes from potentially different
datasets to detected objects.

.. code-block:: python3
    :linenos:

    print(gridrad_options.model_summary())

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
:func:`thuner.utils.generate_times` creates a generator from the dataset
options for the tracking dataset. We can then pass this generator, and
the various options, to the tracking function :func:`thuner.parallel.track`.
During the tracking run, outputs will be created in the
``output_parent`` directory, within the subfolders ``interval_0``,
``interval_1`` etc, which represent subintervals of the time period
being tracked. At the end of the run, these outputs are stiched
together.

.. code-block:: python3
    :linenos:

    times = utils.generate_times(data_options.dataset_by_name("gridrad").filepaths)
    args = [times, data_options, grid_options, track_options, visualize_options]
    num_processes = 4 # If visualize_options is not None, num_processes must be 1
    kwargs = {"output_directory": output_parent, "num_processes": num_processes}
    # In parallel tracking runs, we need to tell the tracking function which dataset to use
    # for tracking, so the subinterval data_options can be generated correctly
    kwargs.update({"dataset_name": "gridrad"})
    parallel.track(*args, **kwargs)

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

The convenience function ``thuner.analyze.utils.read_options`` reloads
all options in the above way, storing the different options in a
dictionary.

.. code-block:: python3
    :linenos:

    all_options = analyze.utils.read_options(output_parent)
    all_options["data"].model_dump()

Object attributes, e.g. MCS position, area and velocity, are saved as
CSV files in nested subfolders. Attribute metadata is recorded in YAML
files. One can then load attribute data using ``pandas.read_csv``. One
can also create an appropriately formatted :class:`pandas.DataFrame` using
the convenience function :func:`thuner.attribute.utils.read_attribute_csv`.

.. code-block:: python3
    :linenos:

    core = attribute.utils.read_attribute_csv(output_parent / "attributes/mcs/core.csv")
    print(core.head(20).to_string())

Records of the filepaths corresponding to each time of the tracking run
are saved in the ``records`` folder. These records are useful for
generating figures after a tracking run.

.. code-block:: python3
    :linenos:

    filepath = output_parent / "records/filepaths/gridrad.csv" 
    records = attribute.utils.read_attribute_csv(filepath)
    print(records.head(20).to_string())

Object masks are saved as ZARR files, which can be read using
:mod:`xarray`.

.. code-block:: python3
    :linenos:

    xr.open_dataset(output_parent / "masks/mcs.zarr").info()

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

    name = f"mcs_gridrad_{event_start.replace('-', '')}"
    style = "presentation"
    attribute_handlers = default.grouped_attribute_handlers(output_parent, style)
    kwargs = {"name": name, "object_name": "mcs", "style": style}
    kwargs.update({"attribute_handlers": attribute_handlers, "dt": 7200})
    figure_options = option.visualize.GroupedHorizontalAttributeOptions(**kwargs)
    args = [output_parent, start, end, figure_options, "gridrad"]
    args_dict = {"parallel_figure": True, "by_date": False, "num_processes": 4}
    visualize.attribute.series(*args, **args_dict)

Relabelling
-----------

Sometimes we need to define new objects based on the split-merge history
of the objects tracked during a THUNER run.