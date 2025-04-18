GridRad Severe
==============

First, import the requisite modules.

.. code-block:: python3

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

Next, specify the folders where THUNER outputs will be saved.

.. code-block:: python3

    # Parent directory for saving outputs
    base_local = Path.home() / "THUNER_output"
    output_parent = base_local / f"runs/gridrad/gridrad_demo"
    options_directory = output_parent / "options"
    visualize_directory = output_parent / "visualize"
    
    # Delete the options directory if it already exists
    # if output_parent.exists():
        # shutil.rmtree(output_parent)

Now specify the options for the THUNER run, beginning with the GridRad
severe dataset options. Options classes in THUNER are built on the
:class:`pydantic.BaseModel`, and are initialized by passing keyword, value
pairs. Note that GridRad Severe data is organized into “events”. First
load the example events from local disk. If you don’t yet have the demo
data, run thuner.data.get_demo_data().

.. code-block:: python3

    event_directories = data.gridrad.get_event_directories(year=2010, base_local=base_local)
    event_directory = event_directories[0] # Take the first event from 2010 for the demo
    # Get the start and end times of the event, and the date of the event start
    start, end, event_start = data.gridrad.get_event_times(event_directory)
    times_dict = {"start": start, "end": end}
    gridrad_dict = {"event_start": event_start}
    gridrad_options = data.gridrad.GridRadSevereOptions(**times_dict, **gridrad_dict)

All instances of options classes can be examined using the
``model_dump`` method, which converts the class instance to a
dictionary. Note the ``parent_local`` field, which provides the parent
directory on local disk containing the dataset. Analogously,
``parent_remote`` specifies the remote location of the data. Note also
the ``filepaths`` field, which provides a list of the dataset’s absolute
filepaths. If ``filepaths`` is unset when the options instance is
created, it will be populated automatically based on the other input
arguments. Note the ``use`` field, which tells THUNER whether the
dataset will be used to ``track`` or ``tag`` objects.

.. code-block:: python3

    gridrad_options.model_dump()

We will use ERA5 single-level and pressure-level data for tagging the
storms detected in the GridRad Severe dataset with other attributes,
e.g. ambient winds.

.. code-block:: python3

    era5_dict = {"latitude_range": [27, 39], "longitude_range": [-102, -89]}
    era5_pl_options = data.era5.ERA5Options(**times_dict, **era5_dict)
    era5_dict.update({"data_format": "single-levels"})
    era5_sl_options = data.era5.ERA5Options(**times_dict, **era5_dict)

All the dataset options are grouped into a single
:class:`thuner.option.data.DataOptions` object, which is passed to the THUNER
tracking function. We also save these options as a YAML file.

.. code-block:: python3

    datasets = [gridrad_options, era5_pl_options, era5_sl_options]
    data_options = option.data.DataOptions(datasets=datasets)
    data_options.to_yaml(options_directory / "data.yml")

Now create and save options describing the grid. If ``regrid`` is
``False`` and grid properties like ``altitude_spacing`` or
``geographic_spacing`` are set to ``None``, THUNER will attempt to infer
these from the tracking dataset.

.. code-block:: python3

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
with a margin of 70 km around the object.

.. code-block:: python3

    # Modify the default options for gridrad. Because grids so large we now use a distinct
    # global flow box for each object.
    track_options.levels[1].objects[0].tracking.unique_global_flow = False
    track_options.levels[1].objects[0].tracking.global_flow_margin = 70
    track_options.to_yaml(options_directory / "track.yml")

Users can also specify visualization options for generating figures
during a tracking run. Uncomment the line below to generate figures that
visualize the matching algorithm - naturally this makes a tracking run
much slower.

.. code-block:: python3

    visualize_options = None
    # visualize_options = default.runtime(visualize_directory=visualize_directory)
    # visualize_options.to_yaml(options_directory / "visualize.yml")

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

    times = data.generate_times(data_options.dataset_by_name("gridrad"))
    args = [times, data_options, grid_options, track_options, visualize_options]
    num_processes = 4 # If visualize_options is not None, num_processes must be 1
    parallel.track(*args, output_directory=output_parent, num_processes=num_processes)

The outputs of the tracking run are saved in the ``output_parent``
directory. The options for the run are saved in human-readable YAML
files within the ``options`` directory. For reproducibility, Python
objects can be rebuilt from these YAML files by reading the YAML, and
passing this to the appropriate ``pydantic`` model.

.. code-block:: python3

    with open(options_directory / "data.yml", "r") as f:
        data_options = option.data.DataOptions(**yaml.safe_load(f))
        # Note yaml.safe_load(f) is a dictionary.
        # Prepending with ** unpacks the dictionary into keyword/argument pairs.
    data_options

The convenience function ``thuner.analyze.utils.read_options`` reloads
all options in the above way, saving the options in a dictionary.

.. code-block:: python3

    all_options = analyze.utils.read_options(output_parent)
    all_options["data"]

Object attributes, e.g. MCS position, area and velocity, are saved as
CSV files in nested subfolders. Attribute metadata is recorded in YAML
files. One can then load attribute data using ``pandas.read_csv``. One
can also create an appropriately formatted :class:`pandas.DataFrame` using
the convenience function :func:`thuner.attribute.utils.read_attribute_csv`.

.. code-block:: python3

    attribute.utils.read_attribute_csv(output_parent / "attributes/mcs/core.csv")

Records of the filepaths corresponding to each time of the tracking run
are saved in the ``records`` folder. These records are useful for
generating figures after a tracking run.

.. code-block:: python3

    attribute.utils.read_attribute_csv(output_parent / "records/filepaths/gridrad.csv")

Object masks are saved as ZARR files, which can be read using
:mod:`xarray`.

.. code-block:: python3

    xr.open_dataset(output_parent / "masks/mcs.zarr")

We can then perform analysis on the above outputs. Below we perform the
MCS classifications discussed by `Short et
al. (2023) <https://doi.org/10.1175/MWR-D-22-0146.1>`__.

.. code-block:: python3

    analysis_options = analyze.mcs.AnalysisOptions()
    analyze.mcs.process_velocities(output_parent, profile_dataset="era5_pl")
    analyze.mcs.quality_control(output_parent, analysis_options)
    analyze.mcs.classify_all(output_parent, analysis_options)
    attribute.utils.read_attribute_csv(output_parent / "analysis/classification.csv")

We can also generate figures and animations from the output. Below we
visualize the convective and stratiform regions of each MCS, displaying
each system’s velocity and stratiform-offset, and the boundaries of the
radar mosaic domain, as discussed by `Short et
al. (2023) <https://doi.org/10.1175/MWR-D-22-0146.1>`__. By default,
figures and animations are saved in ``output_parent`` directory in the
``visualize`` folder.

.. code-block:: python3

    figure_name = f"mcs_gridrad_{event_start.replace('-', '')}"
    kwargs = {"name": figure_name, "style": "presentation"}
    kwargs.update({"attributes": ["velocity", "offset"]})
    figure_options = option.visualize.HorizontalAttributeOptions(**kwargs)
    start_time = np.datetime64(start)
    end_time = np.datetime64(end)
    args = [output_parent, start_time, end_time, figure_options]
    args_dict = {"parallel_figure": True, "dt": 7200, "by_date": False, "num_processes": 4}
    visualize.attribute.mcs_series(*args, **args_dict)