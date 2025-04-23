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
    
    from pathlib import Path
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
    end = "2005-11-13T19:00:00"
    times_dict = {"start": start, "end": end}
    cpol_options = data.aura.CPOLOptions(**times_dict, converted_options={"save": True})
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
    track_options.to_yaml(options_directory / "track.yml")

.. code-block:: text

    2025-04-22 21:36:25,844 - thuner.data.aura - INFO - Generating cpol filepaths.
    2025-04-22 21:36:25,847 - thuner.data.era5 - INFO - Generating era5 filepaths.
    2025-04-22 21:36:25,901 - thuner.data.era5 - INFO - Generating era5 filepaths.
    2025-04-22 21:36:25,929 - thuner.option.grid - WARNING - altitude not specified. Using default altitudes.
    2025-04-22 21:36:25,929 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.

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

    times = utils.generate_times(data_options.dataset_by_name("cpol"))
    args = [times, data_options, grid_options, track_options, visualize_options]
    # parallel.track(*args, output_directory=output_parent)
    track.track(*args, output_directory=output_parent)

.. code-block:: text

    2025-04-22 21:36:33,112 - thuner.track.track - INFO - Beginning thuner tracking. Saving output to /home/ewan/THUNER_output/runs/cpol/geographic.
    2025-04-22 21:36:33,707 - thuner.track.track - INFO - Processing 2005-11-13T14:00:09.
    2025-04-22 21:36:33,710 - thuner.data.aura - INFO - Updating cpol dataset for 2005-11-13T14:00:09.
    2025-04-22 21:36:33,711 - thuner.data.aura - INFO - Converting cpol data from twp10cpolgrid150.b2.20051113.140000.nc
    2025-04-22 21:36:33,767 - thuner.data.aura - INFO - Creating new geographic grid with spacing 0.025 m, 0.025 m.
    2025-04-22 21:36:37,287 - thuner.track.track - INFO - Processing hierarchy level 0.
    2025-04-22 21:36:37,289 - thuner.track.track - INFO - Tracking convective.
    2025-04-22 21:36:37,295 - thuner.detect.steiner - INFO - Compiling thuner.detect.steiner.steiner_scheme with Numba. Please wait.
    2025-04-22 21:36:53,290 - thuner.match.match - INFO - Matching convective objects.
    2025-04-22 21:36:53,292 - thuner.match.match - INFO - No current mask, or no objects in current mask.
    2025-04-22 21:36:53,296 - thuner.visualize.runtime - INFO - Creating runtime visualization figures.
    2025-04-22 21:36:57,614 - thuner.track.track - INFO - Tracking middle.
    2025-04-22 21:36:57,618 - thuner.track.track - INFO - Tracking anvil.
    2025-04-22 21:36:57,626 - thuner.track.track - INFO - Processing hierarchy level 1.
    2025-04-22 21:36:57,628 - thuner.track.track - INFO - Tracking mcs.
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

We can also perform analysis on, and visualization of, the MCS objects.

.. code-block:: python3
    :linenos:

    analysis_options = analyze.mcs.AnalysisOptions()
    analysis_options.to_yaml(options_directory / "analysis.yml")
    analyze.mcs.process_velocities(output_parent)
    analyze.mcs.quality_control(output_parent, analysis_options)
    analyze.mcs.classify_all(output_parent, analysis_options)

.. code-block:: text

    2025-04-22 21:39:36,283 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.
    2025-04-22 21:39:36,602 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.

.. code-block:: python3
    :linenos:

    figure_name = "mcs_attributes"
    kwargs = {"style": "presentation", "attributes": ["velocity", "offset"]}
    figure_options = option.visualize.HorizontalAttributeOptions(name=figure_name, **kwargs)
    
    args = [output_parent, start, end, figure_options]
    args_dict = {"parallel_figure": True, "by_date": False, "num_processes": 4}
    visualize.attribute.mcs_series(*args, **args_dict)

.. code-block:: text

    2025-04-22 21:39:37,244 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.
    2025-04-22 21:39:37,561 - thuner.visualize.attribute - INFO - Visualizing MCS at time 2005-11-13T14:00:09.000000000.
    2025-04-22 21:39:37,565 - thuner.data.aura - INFO - Converting cpol data from twp10cpolgrid150.b2.20051113.140000.nc
    2025-04-22 21:39:37,613 - thuner.data.aura - INFO - Creating new geographic grid with spacing 0.025 m, 0.025 m.
    2025-04-22 21:39:38,214 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.
    2025-04-22 21:39:38,642 - thuner.visualize.attribute - INFO - Saving mcs_attributes figure for 2005-11-13T14:00:09.000000000.
    2025-04-22 21:39:46,576 - thuner.visualize.attribute - INFO - Visualizing MCS at time 2005-11-13T14:10:23.000000000.
    2025-04-22 21:39:46,579 - thuner.data.aura - INFO - Converting cpol data from twp10cpolgrid150.b2.20051113.141000.nc
    2025-04-22 21:39:46,597 - thuner.visualize.attribute - INFO - Visualizing MCS at time 2005-11-13T14:20:09.000000000.
    2025-04-22 21:39:46,600 - thuner.data.aura - INFO - Converting cpol data from twp10cpolgrid150.b2.20051113.142000.nc
    2025-04-22 21:39:46,797 - thuner.visualize.attribute - INFO - Visualizing MCS at time 2005-11-13T14:30:09.000000000.
    2025-04-22 21:39:46,800 - thuner.data.aura - INFO - Converting cpol data from twp10cpolgrid150.b2.20051113.143000.nc
    2025-04-22 21:39:48,711 - thuner.visualize.attribute - INFO - Visualizing MCS at time 2005-11-13T14:40:09.000000000.
    2025-04-22 21:39:48,714 - thuner.data.aura - INFO - Converting cpol data from twp10cpolgrid150.b2.20051113.144000.nc
    2025-04-22 21:39:50,155 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.
    ...

Pre-Converted Data
------------------

We can also perform THUNER tracking runs on general datasets, we just
need to ensure they are pre-converted into a format recognized by
THUNER, i.e. gridded data files readable by ``xarray.open_dataset``,
with variables named according to
`CF-conventions <https://cfconventions.org/>`__. To illustrate, we will
use the converted CPOL files that were generated by the code in the
previous section.

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
    # Set the dataset name to "thuner" to indicate it is already converted.
    kwargs = {"name": "thuner", "fields": ["reflectivity"], "filepaths": filepaths}
    cpol_options = utils.BaseDatasetOptions(**times_dict, **kwargs)
    datasets=[cpol_options, era5_pl_options, era5_sl_options]
    data_options = option.data.DataOptions(datasets=datasets)
    data_options.to_yaml(options_directory / "data.yml")
    
    # Switch off the runtime figures
    visualize_options = None

.. code-block:: python3
    :linenos:

    # Note the tracking code for thuner (i.e. generic but pre-converted) datasets is yet to 
    # be implemented - this is a priority!  All we need to do is ensure converted/thuner 
    # files have a standard date_time string at the end
    # so that a suitable, generic generate_filepaths function can be created. All we need then are 
    # generic update_dataset functions; no need for a convert_dataset function for thuner datasets,
    # as by definition they are already converted.

Cartesian Coordinates
---------------------

Because the CPOL radar domains are small (150 km radii), it is
reasonable to perform tracking in Cartesian coordinates. This should
make the run faster as we are no longer performing conversions on the
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
    
    # Create the grid_options
    grid_options = option.grid.GridOptions(name="cartesian", regrid=False)
    grid_options.to_yaml(options_directory / "grid.yml")
    data_options.to_yaml(options_directory / "data.yml")
    track_options.to_yaml(options_directory / "track.yml")
    visualize_options = None
    
    times = utils.generate_times(data_options.dataset_by_name("cpol"))
    args = [times, data_options, grid_options, track_options, visualize_options]
    kwargs = {"output_directory": output_parent, "dataset_name": "cpol"}
    parallel.track(*args, **kwargs, debug_mode=True)

.. code-block:: text

    2025-04-22 21:40:46,160 - thuner.data.aura - INFO - Generating cpol filepaths.
    2025-04-22 21:40:46,164 - thuner.option.grid - WARNING - altitude not specified. Using default altitudes.
    2025-04-22 21:40:46,165 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.
    2025-04-22 21:40:48,199 - thuner.parallel - INFO - Beginning parallel tracking with 4 processes.
    2025-04-22 21:40:48,359 - thuner.track.track - INFO - Beginning thuner tracking. Saving output to /home/ewan/THUNER_output/runs/cpol/cartesian/interval_0.
    2025-04-22 21:40:48,430 - thuner.track.track - INFO - Processing 2005-11-13T14:00:09.
    2025-04-22 21:40:48,431 - thuner.data.aura - INFO - Updating cpol dataset for 2005-11-13T14:00:09.
    2025-04-22 21:40:48,432 - thuner.data.aura - INFO - Converting cpol data from twp10cpolgrid150.b2.20051113.140000.nc
    2025-04-22 21:40:48,584 - thuner.track.track - INFO - Processing hierarchy level 0.
    2025-04-22 21:40:48,585 - thuner.track.track - INFO - Tracking convective.
    2025-04-22 21:40:48,606 - thuner.match.match - INFO - Matching convective objects.
    2025-04-22 21:40:48,616 - thuner.match.match - INFO - No current mask, or no objects in current mask.
    2025-04-22 21:40:48,621 - thuner.track.track - INFO - Tracking middle.
    2025-04-22 21:40:48,629 - thuner.track.track - INFO - Tracking anvil.
    2025-04-22 21:40:48,635 - thuner.track.track - INFO - Processing hierarchy level 1.
    ...

.. code-block:: python3
    :linenos:

    analysis_options = analyze.mcs.AnalysisOptions()
    analysis_options.to_yaml(options_directory / "analysis.yml")
    analyze.mcs.process_velocities(output_parent)
    analyze.mcs.quality_control(output_parent, analysis_options)
    analyze.mcs.classify_all(output_parent, analysis_options)

.. code-block:: text

    2025-04-22 21:41:40,931 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.
    2025-04-22 21:41:41,383 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.

.. code-block:: python3
    :linenos:

    figure_name = "mcs_attributes"
    kwargs = {"style": "presentation", "attributes": ["velocity", "offset"]}
    figure_options = option.visualize.HorizontalAttributeOptions(name=figure_name, **kwargs)
    
    args = [output_parent, start, end, figure_options]
    args_dict = {"parallel_figure": True, "by_date": False, "num_processes": 4}
    visualize.attribute.mcs_series(*args, **args_dict)

.. code-block:: text

    2025-04-22 21:41:42,111 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.
    2025-04-22 21:41:42,448 - thuner.visualize.attribute - INFO - Visualizing MCS at time 2005-11-13T14:00:09.000000000.
    2025-04-22 21:41:42,452 - thuner.data.aura - INFO - Converting cpol data from twp10cpolgrid150.b2.20051113.140000.nc
    2025-04-22 21:41:42,661 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.
    2025-04-22 21:41:43,296 - thuner.visualize.attribute - INFO - Saving mcs_attributes figure for 2005-11-13T14:00:09.000000000.
    2025-04-22 21:41:52,744 - thuner.visualize.attribute - INFO - Visualizing MCS at time 2005-11-13T14:10:23.000000000.
    2025-04-22 21:41:52,748 - thuner.data.aura - INFO - Converting cpol data from twp10cpolgrid150.b2.20051113.141000.nc
    2025-04-22 21:41:53,181 - thuner.visualize.attribute - INFO - Visualizing MCS at time 2005-11-13T14:30:09.000000000.
    2025-04-22 21:41:53,191 - thuner.data.aura - INFO - Converting cpol data from twp10cpolgrid150.b2.20051113.143000.nc
    2025-04-22 21:41:53,198 - thuner.visualize.attribute - INFO - Visualizing MCS at time 2005-11-13T14:20:09.000000000.
    2025-04-22 21:41:53,204 - thuner.data.aura - INFO - Converting cpol data from twp10cpolgrid150.b2.20051113.142000.nc
    2025-04-22 21:41:53,683 - thuner.visualize.attribute - INFO - Visualizing MCS at time 2005-11-13T14:40:09.000000000.
    2025-04-22 21:41:53,688 - thuner.data.aura - INFO - Converting cpol data from twp10cpolgrid150.b2.20051113.144000.nc
    2025-04-22 21:41:53,854 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.
    2025-04-22 21:41:54,266 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.
    ...