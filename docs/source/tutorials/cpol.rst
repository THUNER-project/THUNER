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
    
    import shutil
    import glob
    import thuner.data as data
    import thuner.option as option
    import thuner.track.track as track
    import thuner.visualize as visualize
    import thuner.analyze as analyze
    import thuner.default as default
    import thuner.attribute as attribute
    import thuner.parallel as parallel
    import thuner.utils as utils
    import thuner.config as config

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

    # Specify the local base directory for saving outputs
    base_local = config.get_outputs_directory()
    
    output_parent = base_local / "runs/cpol/geographic"
    options_directory = output_parent / "options"
    visualize_directory = output_parent / "visualize"
    
    # Remove the output parent directory if it already exists
    if output_parent.exists():
        shutil.rmtree(output_parent)

Run the cell below to get the demo data for this tutorial, if you
haven’t already.

.. code-block:: python3
    :linenos:

    # Download the demo data
    remote_directory = "s3://thuner-storage/THUNER_output/input_data/raw/cpol"
    data.get_demo_data(base_local, remote_directory)
    remote_directory = "s3://thuner-storage/THUNER_output/input_data/raw/"
    remote_directory += "era5_monthly_10S_129E_14S_133E"
    data.get_demo_data(base_local, remote_directory)

.. code-block:: text

    download: s3://thuner-storage/THUNER_output/input_data/raw/cpol/cpol_level_1b/v2020/gridded/grid_150km_2500m/2005/20051113/twp10cpolgrid150.b2.20051113.004000.nc to ../../../THUNER_output/THUNER_output/input_data/raw/cpol/cpol_level_1b/v2020/gridded/grid_150km_2500m/2005/20051113/twp10cpolgrid150.b2.20051113.004000.nc
    download: s3://thuner-storage/THUNER_output/input_data/raw/cpol/cpol_level_1b/v2020/gridded/grid_150km_2500m/2005/20051113/twp10cpolgrid150.b2.20051113.005000.nc to ../../../THUNER_output/THUNER_output/input_data/raw/cpol/cpol_level_1b/v2020/gridded/grid_150km_2500m/2005/20051113/twp10cpolgrid150.b2.20051113.005000.nc
    download: s3://thuner-storage/THUNER_output/input_data/raw/cpol/cpol_level_1b/v2020/gridded/grid_150km_2500m/2005/20051113/twp10cpolgrid150.b2.20051113.012000.nc to ../../../THUNER_output/THUNER_output/input_data/raw/cpol/cpol_level_1b/v2020/gridded/grid_150km_2500m/2005/20051113/twp10cpolgrid150.b2.20051113.012000.nc
    download: s3://thuner-storage/THUNER_output/input_data/raw/cpol/cpol_level_1b/v2020/gridded/grid_150km_2500m/2005/20051113/twp10cpolgrid150.b2.20051113.010000.nc to ../../../THUNER_output/THUNER_output/input_data/raw/cpol/cpol_level_1b/v2020/gridded/grid_150km_2500m/2005/20051113/twp10cpolgrid150.b2.20051113.010000.nc
    download: s3://thuner-storage/THUNER_output/input_data/raw/cpol/cpol_level_1b/v2020/gridded/grid_150km_2500m/2005/20051113/twp10cpolgrid150.b2.20051113.013000.nc to ../../../THUNER_output/THUNER_output/input_data/raw/cpol/cpol_level_1b/v2020/gridded/grid_150km_2500m/2005/20051113/twp10cpolgrid150.b2.20051113.013000.nc
    download: s3://thuner-storage/THUNER_output/input_data/raw/cpol/cpol_level_1b/v2020/gridded/grid_150km_2500m/2005/20051113/twp10cpolgrid150.b2.20051113.020000.nc to ../../../THUNER_output/THUNER_output/input_data/raw/cpol/cpol_level_1b/v2020/gridded/grid_150km_2500m/2005/20051113/twp10cpolgrid150.b2.20051113.020000.nc
    download: s3://thuner-storage/THUNER_output/input_data/raw/cpol/cpol_level_1b/v2020/gridded/grid_150km_2500m/2005/20051113/twp10cpolgrid150.b2.20051113.011000.nc to ../../../THUNER_output/THUNER_output/input_data/raw/cpol/cpol_level_1b/v2020/gridded/grid_150km_2500m/2005/20051113/twp10cpolgrid150.b2.20051113.011000.nc
    download: s3://thuner-storage/THUNER_output/input_data/raw/cpol/cpol_level_1b/v2020/gridded/grid_150km_2500m/2005/20051113/twp10cpolgrid150.b2.20051113.015000.nc to ../../../THUNER_output/THUNER_output/input_data/raw/cpol/cpol_level_1b/v2020/gridded/grid_150km_2500m/2005/20051113/twp10cpolgrid150.b2.20051113.015000.nc
    download: s3://thuner-storage/THUNER_output/input_data/raw/cpol/cpol_level_1b/v2020/gridded/grid_150km_2500m/2005/20051113/twp10cpolgrid150.b2.20051113.014000.nc to ../../../THUNER_output/THUNER_output/input_data/raw/cpol/cpol_level_1b/v2020/gridded/grid_150km_2500m/2005/20051113/twp10cpolgrid150.b2.20051113.014000.nc
    download: s3://thuner-storage/THUNER_output/input_data/raw/cpol/cpol_level_1b/v2020/gridded/grid_150km_2500m/2005/20051113/twp10cpolgrid150.b2.20051113.023000.nc to ../../../THUNER_output/THUNER_output/input_data/raw/cpol/cpol_level_1b/v2020/gridded/grid_150km_2500m/2005/20051113/twp10cpolgrid150.b2.20051113.023000.nc
    download: s3://thuner-storage/THUNER_output/input_data/raw/cpol/cpol_level_1b/v2020/gridded/grid_150km_2500m/2005/20051113/twp10cpolgrid150.b2.20051113.030000.nc to ../../../THUNER_output/THUNER_output/input_data/raw/cpol/cpol_level_1b/v2020/gridded/grid_150km_2500m/2005/20051113/twp10cpolgrid150.b2.20051113.030000.nc
    download: s3://thuner-storage/THUNER_output/input_data/raw/cpol/cpol_level_1b/v2020/gridded/grid_150km_2500m/2005/20051113/twp10cpolgrid150.b2.20051113.021000.nc to ../../../THUNER_output/THUNER_output/input_data/raw/cpol/cpol_level_1b/v2020/gridded/grid_150km_2500m/2005/20051113/twp10cpolgrid150.b2.20051113.021000.nc
    download: s3://thuner-storage/THUNER_output/input_data/raw/cpol/cpol_level_1b/v2020/gridded/grid_150km_2500m/2005/20051113/twp10cpolgrid150.b2.20051113.031000.nc to ../../../THUNER_output/THUNER_output/input_data/raw/cpol/cpol_level_1b/v2020/gridded/grid_150km_2500m/2005/20051113/twp10cpolgrid150.b2.20051113.031000.nc
    download: s3://thuner-storage/THUNER_output/input_data/raw/cpol/cpol_level_1b/v2020/gridded/grid_150km_2500m/2005/20051113/twp10cpolgrid150.b2.20051113.001000.nc to ../../../THUNER_output/THUNER_output/input_data/raw/cpol/cpol_level_1b/v2020/gridded/grid_150km_2500m/2005/20051113/twp10cpolgrid150.b2.20051113.001000.nc
    download: s3://thuner-storage/THUNER_output/input_data/raw/cpol/cpol_level_1b/v2020/gridded/grid_150km_2500m/2005/20051113/twp10cpolgrid150.b2.20051113.024000.nc to ../../../THUNER_output/THUNER_output/input_data/raw/cpol/cpol_level_1b/v2020/gridded/grid_150km_2500m/2005/20051113/twp10cpolgrid150.b2.20051113.024000.nc
    ...

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
    # Note the CPOL times are usually a few seconds off the 10 m interval, so add 30 seconds
    # to ensure we capture 19:00:00
    end = "2005-11-13T19:00:30" 
    times_dict = {"start": start, "end": end}
    cpol_options = data.aura.CPOLOptions(**times_dict, converted_options={"save": True})
    # cpol_options = data.aura.CPOLOptions(**times_dict, converted_options={"load": True})
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
    # We will also modify the mcs tracking options to save a record of the member object ids
    mcs_attributes = track_options.levels[1].object_by_name("mcs").attributes
    mcs_group_attr = mcs_attributes.attribute_type_by_name("group")
    membership = attribute.group.build_membership_attribute_group()
    mcs_group_attr.attributes.append(membership)
    mcs_group_attr.revalidate()
    track_options.to_yaml(options_directory / "track.yml")

.. code-block:: text

    2025-04-29 19:06:19,008 - thuner.data.aura - INFO - Generating cpol filepaths.
    2025-04-29 19:06:19,009 - thuner.data.era5 - INFO - Generating era5 filepaths.
    2025-04-29 19:06:19,011 - thuner.data.era5 - INFO - Generating era5 filepaths.
    2025-04-29 19:06:19,029 - thuner.option.grid - WARNING - altitude not specified. Using default altitudes.
    2025-04-29 19:06:19,029 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.

For this tutorial, we will generate figures during runtime to visualize
how THUNER is matching both convective and mcs objects.

.. code-block:: python3
    :linenos:

    # Create the visualize_options
    kwargs = {"visualize_directory": visualize_directory, "objects": ["convective", "mcs"]}
    visualize_options = default.runtime(**kwargs)
    visualize_options.to_yaml(options_directory / "visualize.yml")

We can now perform our tracking run; note the run will be slow as we are
generating runtime figures for both convective and MCS objects, and not
using parallelization. To make the run go much faster, set
``visualize_options = None`` and use the the parallel tracking function.

.. code-block:: python3
    :linenos:

    times = utils.generate_times(data_options.dataset_by_name("cpol").filepaths)
    args = [times, data_options, grid_options, track_options, visualize_options]
    # parallel.track(*args, output_directory=output_parent)
    track.track(*args, output_directory=output_parent)

.. code-block:: text

    2025-04-29 19:07:21,779 - thuner.track.track - INFO - Beginning thuner tracking. Saving output to /home/ewan/THUNER_output/runs/cpol/geographic.
    2025-04-29 19:07:21,828 - thuner.track.track - INFO - Processing 2005-11-13T14:00:09.
    2025-04-29 19:07:21,830 - thuner.utils - INFO - Updating cpol input record for 2005-11-13T14:00:09.
    2025-04-29 19:07:21,832 - thuner.data.aura - INFO - Updating cpol dataset for 2005-11-13T14:00:09.
    2025-04-29 19:07:22,379 - thuner.track.track - INFO - Processing hierarchy level 0.
    2025-04-29 19:07:22,380 - thuner.track.track - INFO - Tracking convective.
    2025-04-29 19:07:22,392 - thuner.match.match - INFO - Matching convective objects.
    2025-04-29 19:07:22,393 - thuner.match.match - INFO - No current mask, or no objects in current mask.
    2025-04-29 19:07:22,395 - thuner.visualize.runtime - INFO - Creating runtime visualization figures.
    2025-04-29 19:07:24,310 - thuner.track.track - INFO - Tracking middle.
    2025-04-29 19:07:24,314 - thuner.track.track - INFO - Tracking anvil.
    2025-04-29 19:07:24,317 - thuner.track.track - INFO - Processing hierarchy level 1.
    2025-04-29 19:07:24,317 - thuner.track.track - INFO - Tracking mcs.
    2025-04-29 19:07:24,334 - thuner.match.match - INFO - Matching mcs objects.
    2025-04-29 19:07:24,337 - thuner.match.match - INFO - No current mask, or no objects in current mask.
    ...

Once completed, outputs are available in the ``output_parent``
directory. The visualization folder will contain figures like that
below, which illustrate the matching process. Currently THUNER supports
the TINT/MINT matching approach, but the goal is to eventually
incorporate others. Note that if viewing online, the figures below can
be viewed at original scale by right clicking, save image as, and
opening locally, or by right clicking, open in new tab, etc.

.. figure::
   https://raw.githubusercontent.com/THUNER-project/THUNER/refs/heads/main/gallery/cpol_convective_match_20051113.png
   :alt: Visualization of the TINT/MINT matching process.

   Visualization of the TINT/MINT matching process.

Definitions of terms appearing in the above figure are provided by `Raut
et al. (2021) <https://doi.org/10.1175/JAMC-D-20-0119.1>`__. Note the
displacement vector for the central orange object is large due to the
object changing shape suddenly. Similar jumps occur when objects split
and merge, and for this reason, object center displacements are ill
suited to define object velocities. Instead, object velocities are
calculated by smoothing the corrected local flow vectors, as discussed
by `Short et al. (2023) <https://doi.org/10.1175/MWR-D-22-0146.1>`__.
Animations of all the runtime matching figures for the convective
objects are provided below.

.. figure::
   https://raw.githubusercontent.com/THUNER-project/THUNER/refs/heads/main/gallery/cpol_convective_match_20051113.gif
   :alt: Convective object matching.

   Convective object matching.

We also provide the matching figures for the MCS objects. Note there is
only one MCS object, which is comprised of multiple disjoint convective
objects; the grouping method is described by `Short et
al. (2023) <https://doi.org/10.1175/MWR-D-22-0146.1>`__.

.. figure::
   https://raw.githubusercontent.com/THUNER-project/THUNER/refs/heads/main/gallery/cpol_mcs_match_20051113.gif
   :alt: MCS object matching.

   MCS object matching.

Recall that when setting up the options above, we instructed THUNER to
keep a record of the IDs of each member object (convective, middle and
stratiform echoes) comprising each grouped mcs object. Note that only
the mcs and convective objects are matched between times.

.. code-block:: python3
    :linenos:

    filepath = output_parent / "attributes/mcs/group.csv"
    columns = ["convective_ids", "middle_ids", "anvil_ids"]
    print(attribute.utils.read_attribute_csv(filepath, columns=columns).to_string())

.. code-block:: text

                                     convective_ids     middle_ids anvil_ids
    time                universal_id                                        
    2005-11-13 14:10:23 1                       1 2              1       1 2
    2005-11-13 14:20:09 1                         2              1         1
    2005-11-13 14:30:09 1                         2              1         1
    2005-11-13 14:40:09 1                       2 4              1       1 2
    2005-11-13 14:50:09 1                       2 4              1         1
    2005-11-13 15:00:08 1                     2 4 5              1         1
    2005-11-13 15:10:23 1                   2 4 5 6          1 2 3         1
    2005-11-13 15:20:09 1                     2 4 5            1 2         1
    2005-11-13 15:30:09 1                     2 4 5        1 2 3 4         1
    2005-11-13 15:40:09 1                     2 4 5        1 2 3 4     1 2 3
    2005-11-13 15:50:09 1                     2 4 5      1 2 3 4 6       1 2
    2005-11-13 16:00:08 1                     2 4 5            1 3         1
    2005-11-13 16:10:23 1                   2 4 5 7        1 2 3 4         1
    ...

We can also perform analysis on, and visualization of, the MCS objects.

.. code-block:: python3
    :linenos:

    analysis_options = analyze.mcs.AnalysisOptions()
    analysis_options.to_yaml(options_directory / "analysis.yml")
    analyze.mcs.process_velocities(output_parent)
    analyze.mcs.quality_control(output_parent, analysis_options)
    analyze.mcs.classify_all(output_parent, analysis_options)

.. code-block:: text

    2025-04-29 19:28:06,622 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.
    2025-04-29 19:28:07,519 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.

.. code-block:: python3
    :linenos:

    figure_name = "mcs_attributes"
    kwargs = {"style": "presentation", "attributes": ["id", "velocity", "offset"]}
    figure_options = option.visualize.HorizontalAttributeOptions(name=figure_name, **kwargs)
    
    args = [output_parent, start, end, figure_options]
    args_dict = {"parallel_figure": True, "by_date": False, "num_processes": 4}
    visualize.attribute.mcs_series(*args, **args_dict)

.. code-block:: text

    2025-04-24 23:14:26,564 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.
    2025-04-24 23:14:26,816 - thuner.visualize.attribute - INFO - Visualizing MCS at time 2005-11-13T14:00:09.000000000.
    2025-04-24 23:14:26,831 - thuner.data.aura - INFO - Updating cpol dataset for 2005-11-13T14:00:09.
    2025-04-24 23:14:26,893 - thuner.data.aura - INFO - Creating new geographic grid with spacing 0.025 m, 0.025 m.
    2025-04-24 23:14:27,578 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.
    2025-04-24 23:14:27,959 - thuner.visualize.attribute - INFO - Saving mcs_attributes figure for 2005-11-13T14:00:09.000000000.
    2025-04-24 23:14:36,228 - thuner.visualize.attribute - INFO - Visualizing MCS at time 2005-11-13T14:10:23.000000000.
    2025-04-24 23:14:36,231 - thuner.data.aura - INFO - Updating cpol dataset for 2005-11-13T14:10:23.
    2025-04-24 23:14:36,236 - thuner.visualize.attribute - INFO - Visualizing MCS at time 2005-11-13T14:20:09.000000000.
    2025-04-24 23:14:36,242 - thuner.data.aura - INFO - Updating cpol dataset for 2005-11-13T14:20:09.
    2025-04-24 23:14:36,273 - thuner.visualize.attribute - INFO - Visualizing MCS at time 2005-11-13T14:30:09.000000000.
    2025-04-24 23:14:36,277 - thuner.data.aura - INFO - Updating cpol dataset for 2005-11-13T14:30:09.
    2025-04-24 23:14:37,902 - thuner.visualize.attribute - INFO - Visualizing MCS at time 2005-11-13T14:40:09.000000000.
    2025-04-24 23:14:37,907 - thuner.data.aura - INFO - Updating cpol dataset for 2005-11-13T14:40:09.
    2025-04-24 23:14:39,544 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.
    2025-04-24 23:14:39,545 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.
    ...

Pre-Converted Data
------------------

We can also perform THUNER tracking runs on general datasets, we just
need to ensure they are pre-converted into a format recognized by
THUNER, i.e. gridded data files readable by :func:`xarray.open_dataset`,
with variables named according to
`CF-conventions <https://cfconventions.org/>`__. To illustrate, we will
use the converted CPOL files that were generated by the code in the
previous section. We first modify the options used for the geographic
coordinates above. Re-run the relevant cells above again if necessary.
If you get a pydantic error, restart the notebook.

.. code-block:: python3
    :linenos:

    output_parent = base_local / "runs/cpol/pre_converted"
    options_directory = output_parent / "options"
    options_directory.mkdir(parents=True, exist_ok=True)
    
    if output_parent.exists():
        shutil.rmtree(output_parent)
    
    # Get the pre-converted filepaths
    base_filepath = base_local / "input_data/converted/cpol/cpol_level_1b/v2020/gridded/"
    base_filepath = base_filepath / "grid_150km_2500m/2005/20051113"
    filepaths = glob.glob(str(base_filepath / "*.nc"))
    filepaths = sorted(filepaths)
    
    # Create the data options. 
    kwargs = {"name": "cpol", "fields": ["reflectivity"], "filepaths": filepaths}
    cpol_options = utils.BaseDatasetOptions(**times_dict, **kwargs)
    datasets=[cpol_options, era5_pl_options, era5_sl_options]
    data_options = option.data.DataOptions(datasets=datasets)
    data_options.to_yaml(options_directory / "data.yml")
    
    # Save other options
    grid_options.to_yaml(options_directory / "grid.yml")
    track_options.to_yaml(options_directory / "track.yml")
    
    # Switch off the runtime figures
    visualize_options = None

.. code-block:: python3
    :linenos:

    times = utils.generate_times(data_options.dataset_by_name("cpol").filepaths)
    args = [times, data_options, grid_options, track_options, visualize_options]
    kwargs = {"output_directory": output_parent, "dataset_name": "cpol"}
    parallel.track(*args, **kwargs, debug_mode=True)

.. code-block:: text

    2025-04-24 23:36:43,875 - thuner.parallel - INFO - Beginning parallel tracking with 4 processes.
    2025-04-24 23:36:43,892 - thuner.utils - INFO - get_filepaths being called from base class BaseDatasetOptions. In this case get_filepaths just subsets the filepaths list provided by the user.
    2025-04-24 23:36:44,715 - thuner.track.track - INFO - Beginning thuner tracking. Saving output to /home/ewan/THUNER_output/runs/cpol/pre_converted/interval_0.
    2025-04-24 23:36:44,738 - thuner.track.track - INFO - Processing 2005-11-13T13:10:23.
    2025-04-24 23:36:44,740 - thuner.utils - INFO - Updating cpol input record for 2005-11-13T13:10:23.
    2025-04-24 23:36:44,752 - thuner.utils - INFO - Grid options not set. Inferring from dataset.
    2025-04-24 23:36:44,753 - thuner.utils - INFO - Updating grid_options latitude, longitude and shape using dataset.
    2025-04-24 23:36:44,755 - thuner.utils - INFO - Domain mask found in dataset. Getting boundary coordinates.
    2025-04-24 23:36:44,764 - thuner.track.track - INFO - Processing hierarchy level 0.
    2025-04-24 23:36:44,765 - thuner.track.track - INFO - Tracking convective.
    2025-04-24 23:36:44,773 - thuner.detect.steiner - INFO - Compiling thuner.detect.steiner.steiner_scheme with Numba. Please wait.
    2025-04-24 23:36:56,517 - thuner.match.match - INFO - Matching convective objects.
    2025-04-24 23:36:56,518 - thuner.match.match - INFO - No current mask, or no objects in current mask.
    2025-04-24 23:36:56,522 - thuner.track.track - INFO - Tracking middle.
    2025-04-24 23:36:56,526 - thuner.track.track - INFO - Tracking anvil.
    ...

.. code-block:: python3
    :linenos:

    analysis_options = analyze.mcs.AnalysisOptions()
    analysis_options.to_yaml(options_directory / "analysis.yml")
    analyze.mcs.process_velocities(output_parent)
    analyze.mcs.quality_control(output_parent, analysis_options)
    analyze.mcs.classify_all(output_parent, analysis_options)

.. code-block:: text

    2025-04-24 23:37:48,500 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.
    2025-04-24 23:37:48,823 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.

.. code-block:: python3
    :linenos:

    figure_name = "mcs_attributes"
    kwargs = {"style": "presentation", "attributes": ["id", "velocity", "offset"]}
    figure_options = option.visualize.HorizontalAttributeOptions(name=figure_name, **kwargs)
    
    args = [output_parent, start, end, figure_options]
    args_dict = {"parallel_figure": True, "by_date": False, "num_processes": 4}
    visualize.attribute.mcs_series(*args, **args_dict)

.. code-block:: text

    2025-04-24 23:37:53,724 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.
    2025-04-24 23:37:54,030 - thuner.visualize.attribute - INFO - Visualizing MCS at time 2005-11-13T14:00:09.000000000.
    2025-04-24 23:37:54,064 - thuner.utils - INFO - Grid options not set. Inferring from dataset.
    2025-04-24 23:37:54,065 - thuner.utils - INFO - Updating grid_options latitude, longitude and shape using dataset.
    2025-04-24 23:37:54,067 - thuner.utils - INFO - Domain mask found in dataset. Getting boundary coordinates.
    2025-04-24 23:37:54,103 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.
    2025-04-24 23:37:54,603 - thuner.visualize.attribute - INFO - Saving mcs_attributes figure for 2005-11-13T14:00:09.000000000.
    2025-04-24 23:38:04,312 - thuner.visualize.attribute - INFO - Visualizing MCS at time 2005-11-13T14:10:23.000000000.
    2025-04-24 23:38:04,550 - thuner.visualize.attribute - INFO - Visualizing MCS at time 2005-11-13T14:20:09.000000000.
    2025-04-24 23:38:04,897 - thuner.utils - INFO - Domain mask found in dataset. Getting boundary coordinates.
    2025-04-24 23:38:04,932 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.
    2025-04-24 23:38:05,108 - thuner.utils - INFO - Domain mask found in dataset. Getting boundary coordinates.
    2025-04-24 23:38:05,144 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.
    2025-04-24 23:38:05,452 - thuner.visualize.attribute - INFO - Saving mcs_attributes figure for 2005-11-13T14:10:23.000000000.
    2025-04-24 23:38:05,500 - thuner.visualize.attribute - INFO - Visualizing MCS at time 2005-11-13T14:30:09.000000000.
    ...

Note we can achieve the same result in this case by modifying
``converted_options={"save": True}`` to
``converted_options={"load": True}`` in the `Geographic
Coordinates <#geographic-coordinates>`__ section,and rerunning the
cells.

Cartesian Coordinates
---------------------

Because the CPOL radar domains are small (150 km radii), it is
reasonable to perform tracking in Cartesian coordinates. This should
make the run faster as we are no longer performing regridding on the
fly. We will also switch off the runtime figure generation.

.. code-block:: python3
    :linenos:

    output_parent = base_local / "runs/cpol/cartesian"
    options_directory = output_parent / "options"
    options_directory.mkdir(parents=True, exist_ok=True)
    
    if output_parent.exists():
        shutil.rmtree(output_parent)
    
    # Recreate the original cpol dataset options
    cpol_options = data.aura.CPOLOptions(**times_dict)
    datasets = [cpol_options, era5_pl_options, era5_sl_options]
    data_options = option.data.DataOptions(datasets=datasets)
    data_options.to_yaml(options_directory / "data.yml")
    
    # Create the grid_options
    grid_options = option.grid.GridOptions(name="cartesian", regrid=False)
    grid_options.to_yaml(options_directory / "grid.yml")
    
    # Save the same track options from earlier
    track_options.to_yaml(options_directory / "track.yml")
    visualize_options = None

.. code-block:: text

    2025-04-24 23:39:36,386 - thuner.data.aura - INFO - Generating cpol filepaths.
    2025-04-24 23:39:36,407 - thuner.option.grid - WARNING - altitude not specified. Using default altitudes.
    2025-04-24 23:39:36,409 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.

.. code-block:: python3
    :linenos:

    times = utils.generate_times(data_options.dataset_by_name("cpol").filepaths)
    args = [times, data_options, grid_options, track_options, visualize_options]
    kwargs = {"output_directory": output_parent, "dataset_name": "cpol"}
    parallel.track(*args, **kwargs)

.. code-block:: text

    2025-04-24 23:39:42,172 - thuner.parallel - INFO - Beginning parallel tracking with 4 processes.
    2025-04-24 23:39:48,307 - thuner.track.track - INFO - Beginning thuner tracking. Saving output to /home/ewan/THUNER_output/runs/cpol/cartesian/interval_0.
    2025-04-24 23:39:48,462 - thuner.track.track - INFO - Beginning thuner tracking. Saving output to /home/ewan/THUNER_output/runs/cpol/cartesian/interval_1.
    2025-04-24 23:39:48,952 - thuner.track.track - INFO - Processing 2005-11-13T14:00:09.
    2025-04-24 23:39:48,953 - thuner.utils - INFO - Updating cpol input record for 2005-11-13T14:00:09.
    2025-04-24 23:39:48,953 - thuner.data.aura - INFO - Updating cpol dataset for 2005-11-13T14:00:09.
    2025-04-24 23:39:49,136 - thuner.track.track - INFO - Processing hierarchy level 0.
    2025-04-24 23:39:49,136 - thuner.track.track - INFO - Tracking convective.
    2025-04-24 23:39:49,145 - thuner.detect.steiner - INFO - Compiling thuner.detect.steiner.steiner_scheme with Numba. Please wait.
    2025-04-24 23:39:49,224 - thuner.track.track - INFO - Processing 2005-11-13T15:10:23.
    2025-04-24 23:39:49,227 - thuner.utils - INFO - Updating cpol input record for 2005-11-13T15:10:23.
    2025-04-24 23:39:49,228 - thuner.data.aura - INFO - Updating cpol dataset for 2005-11-13T15:10:23.
    2025-04-24 23:39:49,314 - thuner.track.track - INFO - Beginning thuner tracking. Saving output to /home/ewan/THUNER_output/runs/cpol/cartesian/interval_2.
    2025-04-24 23:39:49,405 - thuner.track.track - INFO - Processing hierarchy level 0.
    2025-04-24 23:39:49,405 - thuner.track.track - INFO - Tracking convective.
    ...

.. code-block:: python3
    :linenos:

    analysis_options = analyze.mcs.AnalysisOptions()
    analysis_options.to_yaml(options_directory / "analysis.yml")
    analyze.mcs.process_velocities(output_parent)
    analyze.mcs.quality_control(output_parent, analysis_options)
    analyze.mcs.classify_all(output_parent, analysis_options)

.. code-block:: text

    2025-04-24 23:41:10,736 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.
    2025-04-24 23:41:10,969 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.

.. code-block:: python3
    :linenos:

    figure_name = "mcs_attributes"
    kwargs = {"style": "presentation", "attributes": ["id", "velocity", "offset"]}
    figure_options = option.visualize.HorizontalAttributeOptions(name=figure_name, **kwargs)
    
    args = [output_parent, start, end, figure_options]
    args_dict = {"parallel_figure": True, "by_date": False, "num_processes": 4}
    visualize.attribute.mcs_series(*args, **args_dict)

.. code-block:: text

    2025-04-24 23:41:11,669 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.
    2025-04-24 23:41:11,916 - thuner.visualize.attribute - INFO - Visualizing MCS at time 2005-11-13T14:00:09.000000000.
    2025-04-24 23:41:11,919 - thuner.data.aura - INFO - Updating cpol dataset for 2005-11-13T14:00:09.
    2025-04-24 23:41:12,081 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.
    2025-04-24 23:41:12,548 - thuner.visualize.attribute - INFO - Saving mcs_attributes figure for 2005-11-13T14:00:09.000000000.
    2025-04-24 23:41:20,079 - thuner.visualize.attribute - INFO - Visualizing MCS at time 2005-11-13T14:10:23.000000000.
    2025-04-24 23:41:20,083 - thuner.data.aura - INFO - Updating cpol dataset for 2005-11-13T14:10:23.
    2025-04-24 23:41:20,199 - thuner.visualize.attribute - INFO - Visualizing MCS at time 2005-11-13T14:20:09.000000000.
    2025-04-24 23:41:20,202 - thuner.data.aura - INFO - Updating cpol dataset for 2005-11-13T14:20:09.
    2025-04-24 23:41:20,801 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.
    2025-04-24 23:41:20,947 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.
    2025-04-24 23:41:21,346 - thuner.visualize.attribute - INFO - Saving mcs_attributes figure for 2005-11-13T14:10:23.000000000.
    2025-04-24 23:41:21,451 - thuner.visualize.attribute - INFO - Visualizing MCS at time 2005-11-13T14:30:09.000000000.
    2025-04-24 23:41:21,454 - thuner.data.aura - INFO - Updating cpol dataset for 2005-11-13T14:30:09.
    2025-04-24 23:41:21,504 - thuner.visualize.attribute - INFO - Saving mcs_attributes figure for 2005-11-13T14:20:09.000000000.
    ...