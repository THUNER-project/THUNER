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

    # Set a flag for whether or not to remove existing output directories
    remove_existing_outputs = False
    
    # Specify the local base directory for saving outputs
    base_local = config.get_outputs_directory()
    
    output_parent = base_local / "runs/cpol/geographic"
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

    2025-06-14 20:08:46,131 - thuner.data.aura - INFO - Generating cpol filepaths.
    2025-06-14 20:08:46,134 - thuner.data.era5 - INFO - Generating era5 filepaths.
    2025-06-14 20:08:46,138 - thuner.data.era5 - INFO - Generating era5 filepaths.
    2025-06-14 20:08:46,169 - thuner.option.grid - WARNING - altitude not specified. Using default altitudes.
    2025-06-14 20:08:46,170 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.

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

.. figure:: https://raw.githubusercontent.com/THUNER-project/THUNER/refs/heads/main/gallery/cpol_convective_match_20051113.png
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

.. figure:: https://raw.githubusercontent.com/THUNER-project/THUNER/refs/heads/main/gallery/cpol_convective_match_20051113.gif
   :alt: Convective object matching.

   Convective object matching.

We also provide the matching figures for the MCS objects. Note there is
only one MCS object, which is comprised of multiple disjoint convective
objects; the grouping method is described by `Short et
al. (2023) <https://doi.org/10.1175/MWR-D-22-0146.1>`__.

.. figure:: https://raw.githubusercontent.com/THUNER-project/THUNER/refs/heads/main/gallery/cpol_mcs_match_20051113.gif
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

    2025-06-13 13:51:36,140 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.
    2025-06-13 13:51:36,448 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.

.. code-block:: python3
    :linenos:

    style = "presentation"
    attribute_handlers = default.grouped_attribute_handlers(output_parent, style)
    kwargs = {"name": "mcs_attributes", "object_name": "mcs", "style": style}
    kwargs.update({"attribute_handlers": attribute_handlers})
    figure_options = option.visualize.GroupedHorizontalAttributeOptions(**kwargs)
    args = [output_parent, start, end, figure_options, "cpol"]
    args_dict = {"parallel_figure": True, "by_date": False, "num_processes": 4}
    visualize.attribute.series(*args, **args_dict)

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
    
    if output_parent.exists() & remove_existing_outputs:
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

    2025-06-14 19:09:29,634 - thuner.parallel - INFO - Beginning parallel tracking with 4 processes.
    2025-06-14 19:09:29,782 - thuner.utils - INFO - get_filepaths being called from base class BaseDatasetOptions. In this case get_filepaths just subsets the filepaths list provided by the user.
    2025-06-14 19:09:32,941 - thuner.track.track - INFO - Beginning thuner tracking. Saving output to /home/ewan/THUNER_output/runs/cpol/pre_converted/interval_0.
    2025-06-14 19:09:33,034 - thuner.track.track - INFO - Processing 2005-11-13T13:10:23.
    2025-06-14 19:09:33,040 - thuner.utils - INFO - Updating cpol input record for 2005-11-13T13:10:23.
    2025-06-14 19:09:33,100 - thuner.utils - INFO - Grid options not set. Inferring from dataset.
    2025-06-14 19:09:33,106 - thuner.utils - INFO - Domain mask found in dataset. Getting boundary coordinates.
    2025-06-14 19:09:33,148 - thuner.track.track - INFO - Processing hierarchy level 0.
    2025-06-14 19:09:33,151 - thuner.track.track - INFO - Tracking convective.
    2025-06-14 19:09:33,181 - thuner.detect.steiner - INFO - Compiling thuner.detect.steiner.steiner_scheme with Numba. Please wait.
    2025-06-14 19:10:17,354 - thuner.match.match - INFO - Matching convective objects.
    2025-06-14 19:10:17,356 - thuner.match.match - INFO - No current mask, or no objects in current mask.
    2025-06-14 19:10:17,371 - thuner.track.track - INFO - Tracking middle.
    2025-06-14 19:10:17,387 - thuner.track.track - INFO - Tracking anvil.
    2025-06-14 19:10:17,408 - thuner.track.track - INFO - Processing hierarchy level 1.
    ...

.. code-block:: python3
    :linenos:

    analysis_options = analyze.mcs.AnalysisOptions()
    analysis_options.to_yaml(options_directory / "analysis.yml")
    analyze.mcs.process_velocities(output_parent)
    analyze.mcs.quality_control(output_parent, analysis_options)
    analyze.mcs.classify_all(output_parent, analysis_options)

.. code-block:: text

    2025-06-14 19:13:48,311 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.
    2025-06-14 19:13:49,628 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.

.. code-block:: python3
    :linenos:

    style = "presentation"
    attribute_handlers = default.grouped_attribute_handlers(output_parent, style)
    kwargs = {"name": "mcs_attributes", "object_name": "mcs", "style": style}
    kwargs.update({"attribute_handlers": attribute_handlers})
    figure_options = option.visualize.GroupedHorizontalAttributeOptions(**kwargs)
    args = [output_parent, start, end, figure_options, "cpol"]
    args_dict = {"parallel_figure": True, "by_date": False, "num_processes": 4}
    visualize.attribute.series(*args, **args_dict)

.. code-block:: text

    2025-06-14 19:13:55,957 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.
    2025-06-14 19:13:57,071 - thuner.visualize.attribute - INFO - Visualizing attributes at time 2005-11-13T14:00:09.000000000.
    2025-06-14 19:13:57,375 - thuner.utils - INFO - Grid options not set. Inferring from dataset.
    2025-06-14 19:13:57,382 - thuner.utils - INFO - Domain mask found in dataset. Getting boundary coordinates.
    2025-06-14 19:13:59,718 - thuner.visualize.attribute - INFO - Saving mcs_attributes figure for 2005-11-13T14:00:09.000000000.
    2025-06-14 19:14:22,438 - thuner.visualize.attribute - INFO - Visualizing attributes at time 2005-11-13T14:10:23.000000000.
    2025-06-14 19:14:23,126 - thuner.visualize.attribute - INFO - Visualizing attributes at time 2005-11-13T14:20:09.000000000.
    2025-06-14 19:14:23,521 - thuner.visualize.attribute - INFO - Visualizing attributes at time 2005-11-13T14:30:09.000000000.
    2025-06-14 19:14:23,586 - thuner.visualize.attribute - INFO - Visualizing attributes at time 2005-11-13T14:40:09.000000000.
    2025-06-14 19:14:25,460 - thuner.utils - INFO - Domain mask found in dataset. Getting boundary coordinates.
    2025-06-14 19:14:26,338 - thuner.utils - INFO - Domain mask found in dataset. Getting boundary coordinates.
    2025-06-14 19:14:26,631 - thuner.utils - INFO - Domain mask found in dataset. Getting boundary coordinates.
    2025-06-14 19:14:26,739 - thuner.utils - INFO - Domain mask found in dataset. Getting boundary coordinates.
    2025-06-14 19:14:30,792 - thuner.visualize.attribute - INFO - Saving mcs_attributes figure for 2005-11-13T14:10:23.000000000.
    2025-06-14 19:14:31,747 - thuner.visualize.attribute - INFO - Saving mcs_attributes figure for 2005-11-13T14:20:09.000000000.
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
    
    if output_parent.exists() & remove_existing_outputs:
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

    2025-06-14 20:08:57,866 - thuner.data.aura - INFO - Generating cpol filepaths.
    2025-06-14 20:08:57,914 - thuner.option.grid - WARNING - altitude not specified. Using default altitudes.
    2025-06-14 20:08:57,915 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.

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

    2025-06-14 19:32:23,367 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.
    2025-06-14 19:32:24,846 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.

.. code-block:: python3
    :linenos:

    style = "presentation"
    attribute_handlers = default.grouped_attribute_handlers(output_parent, style)
    kwargs = {"name": "mcs_attributes", "object_name": "mcs", "style": style}
    kwargs.update({"attribute_handlers": attribute_handlers})
    figure_options = option.visualize.GroupedHorizontalAttributeOptions(**kwargs)
    args = [output_parent, start, end, figure_options, "cpol"]
    args_dict = {"parallel_figure": True, "by_date": False, "num_processes": 4}
    visualize.attribute.series(*args, **args_dict)

.. code-block:: text

    2025-06-14 20:09:09,749 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.
    2025-06-14 20:09:11,010 - thuner.visualize.attribute - INFO - Visualizing attributes at time 2005-11-13T14:00:09.000000000.
    2025-06-14 20:09:11,150 - thuner.data.aura - INFO - Updating cpol dataset for 2005-11-13T14:00:09.
    2025-06-14 20:09:11,334 - thuner.utils - INFO - Grid options not set. Inferring from dataset.
    2025-06-14 20:09:12,541 - thuner.visualize.attribute - INFO - Saving mcs_attributes figure for 2005-11-13T14:00:09.000000000.
    2025-06-14 20:09:27,088 - thuner.visualize.attribute - INFO - Visualizing attributes at time 2005-11-13T14:10:23.000000000.
    2025-06-14 20:09:27,277 - thuner.visualize.attribute - INFO - Visualizing attributes at time 2005-11-13T14:20:09.000000000.
    2025-06-14 20:09:27,299 - thuner.data.aura - INFO - Updating cpol dataset for 2005-11-13T14:10:23.
    2025-06-14 20:09:27,476 - thuner.data.aura - INFO - Updating cpol dataset for 2005-11-13T14:20:09.
    2025-06-14 20:09:28,239 - thuner.visualize.attribute - INFO - Visualizing attributes at time 2005-11-13T14:30:09.000000000.
    2025-06-14 20:09:28,440 - thuner.visualize.attribute - INFO - Visualizing attributes at time 2005-11-13T14:40:09.000000000.
    2025-06-14 20:09:28,485 - thuner.data.aura - INFO - Updating cpol dataset for 2005-11-13T14:30:09.
    2025-06-14 20:09:28,653 - thuner.data.aura - INFO - Updating cpol dataset for 2005-11-13T14:40:09.
    2025-06-14 20:09:30,808 - thuner.visualize.attribute - INFO - Saving mcs_attributes figure for 2005-11-13T14:20:09.000000000.
    2025-06-14 20:09:30,833 - thuner.visualize.attribute - INFO - Saving mcs_attributes figure for 2005-11-13T14:10:23.000000000.
    ...