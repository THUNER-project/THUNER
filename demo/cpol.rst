.. code:: ipython3

    """CPOL demo/test."""
    
    %load_ext autoreload
    %autoreload 2
    %matplotlib inline
    from pathlib import Path
    import shutil
    import numpy as np
    import thuner.data as data
    import thuner.track.track as track
    import thuner.option as option
    import thuner.visualize as visualize
    import thuner.analyze as analyze
    import thuner.option as option
    import thuner.default as default
    
    notebook_name = "cpol_demo.ipynb"


.. parsed-literal::

    
    Welcome to the Thunderstorm Event Reconnaissance (THUNER) package 
    v0.0.16! This is a placeholder version of the package and is not
    yet functional. Please visit github.com/THUNER-project/THUNER for 
    examples, and to report issues or contribute.
    
    THUNER is a flexible toolkit for performing multi-feature detection, 
    tracking, tagging and analysis of events within meteorological datasets. 
    The intended application is to convective weather events. For examples 
    and instructions, see github.com/THUNER-project/THUNER. If you use this 
    package in your research, consider citing the following papers;
    
    Short et al. (2023), doi: 10.1175/MWR-D-22-0146.1
    Raut et al. (2021), doi: 10.1175/JAMC-D-20-0119.1
    Fridlind et al. (2019), doi: 10.5194/amt-12-2979-2019
    Whitehall et al. (2015), doi: 10.1007/s12145-014-0181-3
    Dixon and Wiener (1993), doi: 10.1175/1520-0426(1993)010<0785:TTITAA>2.0.CO;2
    Leese et al. (1971), doi: 10.1175/1520-0450(1971)010<0118:AATFOC>2.0.CO;2
    


test etse

.. code:: ipython3

    # Parent directory for saving outputs
    base_local = Path.home() / "THUNER_output"
    
    output_parent = base_local / "runs/cpol/geographic"
    options_directory = output_parent / "options"
    visualize_directory = output_parent / "visualize"
    
    # Remove the output parent directory if it already exists
    if output_parent.exists():
        shutil.rmtree(output_parent)
    
    # Create the dataset options
    start = "2005-11-13T18:00:00"
    end = "2005-11-13T19:00:00"
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
    track_options = default.track(dataset="cpol")
    track_options.to_yaml(options_directory / "track.yml")
    
    # Create the visualize_options
    visualize_options = default.runtime(visualize_directory=visualize_directory)
    visualize_options.to_yaml(options_directory / "visualize.yml")


.. parsed-literal::

    2025-04-06 16:34:55,135 - thuner.data.aura - INFO - Generating cpol filepaths.
    2025-04-06 16:34:55,138 - thuner.data.era5 - INFO - Generating era5 filepaths.
    2025-04-06 16:34:55,139 - thuner.data.era5 - INFO - Generating era5 filepaths.
    2025-04-06 16:34:55,169 - thuner.option.grid - WARNING - altitude not specified. Using default altitudes.
    2025-04-06 16:34:55,172 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.


.. code:: ipython3

    times = data._utils.generate_times(data_options.dataset_by_name("cpol"))
    args = [times, data_options, grid_options, track_options, visualize_options]
    # parallel.track(*args, output_directory=output_parent)
    track.track(*args, output_directory=output_parent)


.. parsed-literal::

    2025-04-06 16:34:59,996 - thuner.track.track - INFO - Beginning thuner tracking. Saving output to /home/ewan/THUNER_output/runs/cpol/geographic.
    2025-04-06 16:35:00,744 - thuner.track.track - INFO - Processing 2005-11-13T18:00:08.
    2025-04-06 16:35:00,748 - thuner.data.aura - INFO - Updating cpol dataset for 2005-11-13T18:00:08.
    2025-04-06 16:35:00,749 - thuner.data.aura - INFO - Converting cpol data from twp10cpolgrid150.b2.20051113.180000.nc
    2025-04-06 16:35:00,824 - thuner.data.aura - INFO - Creating new geographic grid with spacing 0.025 m, 0.025 m.
    2025-04-06 16:35:04,984 - thuner.track.track - INFO - Processing hierarchy level 0.
    2025-04-06 16:35:04,986 - thuner.track.track - INFO - Tracking convective.
    2025-04-06 16:35:04,992 - thuner.detect.steiner - INFO - Compiling thuner.detect.steiner.steiner_scheme with Numba. Please wait.
    2025-04-06 16:35:19,457 - thuner.track.track - INFO - Tracking middle.
    2025-04-06 16:35:19,462 - thuner.track.track - INFO - Tracking anvil.
    2025-04-06 16:35:19,470 - thuner.track.track - INFO - Processing hierarchy level 1.
    2025-04-06 16:35:19,472 - thuner.track.track - INFO - Tracking mcs.
    2025-04-06 16:35:19,512 - thuner.match.match - INFO - Matching mcs objects.
    2025-04-06 16:35:19,514 - thuner.match.match - INFO - No current mask, or no objects in current mask.
    2025-04-06 16:35:19,529 - thuner.visualize.runtime - INFO - Creating runtime visualization figures.
    2025-04-06 16:35:39,496 - thuner.track.track - INFO - Processing 2005-11-13T18:10:23.
    2025-04-06 16:35:39,501 - thuner.data.aura - INFO - Updating cpol dataset for 2005-11-13T18:10:23.
    2025-04-06 16:35:39,502 - thuner.data.aura - INFO - Converting cpol data from twp10cpolgrid150.b2.20051113.181000.nc
    2025-04-06 16:35:40,649 - thuner.data.era5 - INFO - Updating era5_pl dataset for 2005-11-13T18:00:08.
    2025-04-06 16:35:40,651 - thuner.data.era5 - INFO - Subsetting era5_pl data.
    2025-04-06 16:35:43,701 - thuner.data.era5 - INFO - Updating era5_sl dataset for 2005-11-13T18:00:08.
    2025-04-06 16:35:43,703 - thuner.data.era5 - INFO - Subsetting era5_sl data.
    2025-04-06 16:35:44,146 - thuner.track.track - INFO - Processing hierarchy level 0.
    2025-04-06 16:35:44,147 - thuner.track.track - INFO - Tracking convective.
    2025-04-06 16:35:44,187 - thuner.track.track - INFO - Tracking middle.
    2025-04-06 16:35:44,194 - thuner.track.track - INFO - Tracking anvil.
    2025-04-06 16:35:44,204 - thuner.track.track - INFO - Processing hierarchy level 1.
    2025-04-06 16:35:44,205 - thuner.track.track - INFO - Tracking mcs.
    2025-04-06 16:35:44,215 - thuner.write.mask - INFO - Writing mcs masks to /home/ewan/THUNER_output/runs/cpol/geographic/masks/mcs.zarr.
    2025-04-06 16:35:44,601 - thuner.match.match - INFO - Matching mcs objects.
    2025-04-06 16:35:44,654 - thuner.match.match - INFO - New matchable objects. Initializing match record.
    2025-04-06 16:35:44,671 - thuner.visualize.runtime - INFO - Creating runtime visualization figures.
    2025-04-06 16:35:48,335 - thuner.attribute.attribute - INFO - Recording mcs attributes.
    2025-04-06 16:35:48,663 - thuner.track.track - INFO - Processing 2005-11-13T18:20:09.
    2025-04-06 16:35:48,665 - thuner.data.aura - INFO - Updating cpol dataset for 2005-11-13T18:20:09.
    2025-04-06 16:35:48,668 - thuner.data.aura - INFO - Converting cpol data from twp10cpolgrid150.b2.20051113.182000.nc
    2025-04-06 16:35:49,516 - thuner.track.track - INFO - Processing hierarchy level 0.
    2025-04-06 16:35:49,517 - thuner.track.track - INFO - Tracking convective.
    2025-04-06 16:35:49,576 - thuner.track.track - INFO - Tracking middle.
    2025-04-06 16:35:49,585 - thuner.track.track - INFO - Tracking anvil.
    2025-04-06 16:35:49,597 - thuner.track.track - INFO - Processing hierarchy level 1.
    2025-04-06 16:35:49,598 - thuner.track.track - INFO - Tracking mcs.
    2025-04-06 16:35:49,605 - thuner.write.mask - INFO - Writing mcs masks to /home/ewan/THUNER_output/runs/cpol/geographic/masks/mcs.zarr.
    2025-04-06 16:35:49,669 - thuner.match.match - INFO - Matching mcs objects.
    2025-04-06 16:35:49,712 - thuner.match.match - INFO - Updating match record.
    2025-04-06 16:35:49,721 - thuner.visualize.runtime - INFO - Creating runtime visualization figures.
    2025-04-06 16:35:53,438 - thuner.attribute.attribute - INFO - Recording mcs attributes.
    2025-04-06 16:35:53,603 - thuner.track.track - INFO - Processing 2005-11-13T18:30:09.
    2025-04-06 16:35:53,604 - thuner.data.aura - INFO - Updating cpol dataset for 2005-11-13T18:30:09.
    2025-04-06 16:35:53,605 - thuner.data.aura - INFO - Converting cpol data from twp10cpolgrid150.b2.20051113.183000.nc
    2025-04-06 16:35:54,162 - thuner.track.track - INFO - Processing hierarchy level 0.
    2025-04-06 16:35:54,163 - thuner.track.track - INFO - Tracking convective.
    2025-04-06 16:35:54,193 - thuner.track.track - INFO - Tracking middle.
    2025-04-06 16:35:54,197 - thuner.track.track - INFO - Tracking anvil.
    2025-04-06 16:35:54,201 - thuner.track.track - INFO - Processing hierarchy level 1.
    2025-04-06 16:35:54,202 - thuner.track.track - INFO - Tracking mcs.
    2025-04-06 16:35:54,205 - thuner.write.mask - INFO - Writing mcs masks to /home/ewan/THUNER_output/runs/cpol/geographic/masks/mcs.zarr.
    2025-04-06 16:35:54,226 - thuner.match.match - INFO - Matching mcs objects.
    2025-04-06 16:35:54,240 - thuner.match.match - INFO - Updating match record.
    2025-04-06 16:35:54,243 - thuner.visualize.runtime - INFO - Creating runtime visualization figures.
    2025-04-06 16:35:57,371 - thuner.attribute.attribute - INFO - Recording mcs attributes.
    2025-04-06 16:35:57,500 - thuner.track.track - INFO - Processing 2005-11-13T18:40:09.
    2025-04-06 16:35:57,501 - thuner.data.aura - INFO - Updating cpol dataset for 2005-11-13T18:40:09.
    2025-04-06 16:35:57,503 - thuner.data.aura - INFO - Converting cpol data from twp10cpolgrid150.b2.20051113.184000.nc
    2025-04-06 16:35:58,214 - thuner.track.track - INFO - Processing hierarchy level 0.
    2025-04-06 16:35:58,215 - thuner.track.track - INFO - Tracking convective.
    2025-04-06 16:35:58,263 - thuner.track.track - INFO - Tracking middle.
    2025-04-06 16:35:58,271 - thuner.track.track - INFO - Tracking anvil.
    2025-04-06 16:35:58,281 - thuner.track.track - INFO - Processing hierarchy level 1.
    2025-04-06 16:35:58,283 - thuner.track.track - INFO - Tracking mcs.
    2025-04-06 16:35:58,291 - thuner.write.mask - INFO - Writing mcs masks to /home/ewan/THUNER_output/runs/cpol/geographic/masks/mcs.zarr.
    2025-04-06 16:35:58,339 - thuner.match.match - INFO - Matching mcs objects.
    2025-04-06 16:35:58,374 - thuner.match.match - INFO - Updating match record.
    2025-04-06 16:35:58,381 - thuner.visualize.runtime - INFO - Creating runtime visualization figures.
    2025-04-06 16:36:01,866 - thuner.attribute.attribute - INFO - Recording mcs attributes.
    2025-04-06 16:36:01,995 - thuner.track.track - INFO - Processing 2005-11-13T18:50:08.
    2025-04-06 16:36:01,996 - thuner.data.aura - INFO - Updating cpol dataset for 2005-11-13T18:50:08.
    2025-04-06 16:36:01,997 - thuner.data.aura - INFO - Converting cpol data from twp10cpolgrid150.b2.20051113.185000.nc
    2025-04-06 16:36:02,638 - thuner.track.track - INFO - Processing hierarchy level 0.
    2025-04-06 16:36:02,641 - thuner.track.track - INFO - Tracking convective.
    2025-04-06 16:36:02,698 - thuner.track.track - INFO - Tracking middle.
    2025-04-06 16:36:02,707 - thuner.track.track - INFO - Tracking anvil.
    2025-04-06 16:36:02,713 - thuner.track.track - INFO - Processing hierarchy level 1.
    2025-04-06 16:36:02,714 - thuner.track.track - INFO - Tracking mcs.
    2025-04-06 16:36:02,720 - thuner.write.mask - INFO - Writing mcs masks to /home/ewan/THUNER_output/runs/cpol/geographic/masks/mcs.zarr.
    2025-04-06 16:36:02,751 - thuner.match.match - INFO - Matching mcs objects.
    2025-04-06 16:36:02,769 - thuner.match.match - INFO - Updating match record.
    2025-04-06 16:36:02,773 - thuner.visualize.runtime - INFO - Creating runtime visualization figures.
    2025-04-06 16:36:05,938 - thuner.attribute.attribute - INFO - Recording mcs attributes.
    2025-04-06 16:36:06,080 - thuner.track.track - INFO - Processing 2005-11-13T19:00:08.
    2025-04-06 16:36:06,081 - thuner.data.aura - INFO - Updating cpol dataset for 2005-11-13T19:00:08.
    2025-04-06 16:36:06,082 - thuner.data.aura - INFO - Converting cpol data from twp10cpolgrid150.b2.20051113.190000.nc
    2025-04-06 16:36:06,512 - thuner.write.filepath - INFO - Writing cpol filepaths from 2005-11-13T18:00:00 to 2005-11-13T19:00:00, inclusive and non-inclusive, respectively.
    2025-04-06 16:36:06,528 - thuner.track.track - INFO - Processing hierarchy level 0.
    2025-04-06 16:36:06,529 - thuner.track.track - INFO - Tracking convective.
    2025-04-06 16:36:06,568 - thuner.track.track - INFO - Tracking middle.
    2025-04-06 16:36:06,574 - thuner.track.track - INFO - Tracking anvil.
    2025-04-06 16:36:06,578 - thuner.track.track - INFO - Processing hierarchy level 1.
    2025-04-06 16:36:06,578 - thuner.track.track - INFO - Tracking mcs.
    2025-04-06 16:36:06,583 - thuner.write.mask - INFO - Writing mcs masks to /home/ewan/THUNER_output/runs/cpol/geographic/masks/mcs.zarr.
    2025-04-06 16:36:06,598 - thuner.write.attribute - INFO - Writing mcs attributes from 2005-11-13T18:00:00 to 2005-11-13T19:00:00, inclusive and non-inclusive, respectively.
    2025-04-06 16:36:06,685 - thuner.match.match - INFO - Matching mcs objects.
    2025-04-06 16:36:06,702 - thuner.match.match - INFO - Updating match record.
    2025-04-06 16:36:06,707 - thuner.visualize.runtime - INFO - Creating runtime visualization figures.
    2025-04-06 16:36:09,148 - thuner.attribute.attribute - INFO - Recording mcs attributes.
    2025-04-06 16:36:09,239 - thuner.write.attribute - INFO - Writing mcs attributes from 2005-11-13T19:00:08 to 2005-11-13T20:00:08, inclusive and non-inclusive, respectively.
    2025-04-06 16:36:09,339 - thuner.write.filepath - INFO - Writing cpol filepaths from 2005-11-13T19:00:00 to 2005-11-13T20:00:00, inclusive and non-inclusive, respectively.
    2025-04-06 16:36:09,343 - thuner.write.attribute - INFO - Aggregating attribute files.
    2025-04-06 16:36:09,656 - thuner.write.filepath - INFO - Aggregating filepath records.
    2025-04-06 16:36:09,663 - thuner.visualize.visualize - INFO - Animating match figures for mcs objects.
    2025-04-06 16:36:09,664 - thuner.visualize.visualize - INFO - Saving animation to /home/ewan/THUNER_output/runs/cpol/geographic/visualize/match/mcs_20051113.gif.


.. code:: ipython3

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


.. parsed-literal::

    2025-03-06 23:56:45,291 - thuner.option.grid - WARNING - altitude not specified. Using default altitudes.
    2025-03-06 23:56:45,292 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.
    2025-03-06 23:56:45,430 - thuner.track.track - INFO - Beginning thuner tracking. Saving output to /home/ewan/THUNER_output/runs/cpol/cartesian.
    2025-03-06 23:56:45,513 - thuner.track.track - INFO - Processing 2005-11-13T18:00:08.
    2025-03-06 23:56:45,515 - thuner.data.aura - INFO - Updating cpol dataset for 2005-11-13T18:00:08.
    2025-03-06 23:56:45,515 - thuner.data.aura - INFO - Converting cpol data from twp10cpolgrid150.b2.20051113.180000.nc
    2025-03-06 23:56:45,683 - thuner.track.track - INFO - Processing hierarchy level 0.
    2025-03-06 23:56:45,685 - thuner.track.track - INFO - Tracking convective.
    2025-03-06 23:56:45,749 - thuner.track.track - INFO - Tracking middle.
    2025-03-06 23:56:45,763 - thuner.track.track - INFO - Tracking anvil.
    2025-03-06 23:56:45,777 - thuner.track.track - INFO - Processing hierarchy level 1.
    2025-03-06 23:56:45,778 - thuner.track.track - INFO - Tracking mcs.
    2025-03-06 23:56:45,825 - thuner.match.match - INFO - Matching mcs objects.
    2025-03-06 23:56:45,827 - thuner.match.match - INFO - No current mask, or no objects in current mask.
    2025-03-06 23:56:45,843 - thuner.visualize.runtime - INFO - Creating runtime visualization figures.
    2025-03-06 23:56:50,655 - thuner.track.track - INFO - Processing 2005-11-13T18:10:23.
    2025-03-06 23:56:50,658 - thuner.data.aura - INFO - Updating cpol dataset for 2005-11-13T18:10:23.
    2025-03-06 23:56:50,659 - thuner.data.aura - INFO - Converting cpol data from twp10cpolgrid150.b2.20051113.181000.nc
    2025-03-06 23:56:50,793 - thuner.data.era5 - INFO - Updating era5_pl dataset for 2005-11-13T18:00:08.
    2025-03-06 23:56:50,794 - thuner.data.era5 - INFO - Subsetting era5_pl data.
    2025-03-06 23:56:53,428 - thuner.data.era5 - INFO - Updating era5_sl dataset for 2005-11-13T18:00:08.
    2025-03-06 23:56:53,430 - thuner.data.era5 - INFO - Subsetting era5_sl data.
    2025-03-06 23:56:53,710 - thuner.track.track - INFO - Processing hierarchy level 0.
    2025-03-06 23:56:53,711 - thuner.track.track - INFO - Tracking convective.
    2025-03-06 23:56:53,754 - thuner.track.track - INFO - Tracking middle.
    2025-03-06 23:56:53,760 - thuner.track.track - INFO - Tracking anvil.
    2025-03-06 23:56:53,772 - thuner.track.track - INFO - Processing hierarchy level 1.
    2025-03-06 23:56:53,773 - thuner.track.track - INFO - Tracking mcs.
    2025-03-06 23:56:53,778 - thuner.write.mask - INFO - Writing mcs masks to /home/ewan/THUNER_output/runs/cpol/cartesian/masks/mcs.zarr.
    2025-03-06 23:56:53,830 - thuner.match.match - INFO - Matching mcs objects.
    2025-03-06 23:56:53,872 - thuner.match.match - INFO - New matchable objects. Initializing match record.
    2025-03-06 23:56:53,877 - thuner.visualize.runtime - INFO - Creating runtime visualization figures.
    2025-03-06 23:56:56,227 - thuner.attribute.attribute - INFO - Recording mcs attributes.
    2025-03-06 23:56:56,352 - thuner.track.track - INFO - Processing 2005-11-13T18:20:09.
    2025-03-06 23:56:56,353 - thuner.data.aura - INFO - Updating cpol dataset for 2005-11-13T18:20:09.
    2025-03-06 23:56:56,354 - thuner.data.aura - INFO - Converting cpol data from twp10cpolgrid150.b2.20051113.182000.nc
    2025-03-06 23:56:56,433 - thuner.track.track - INFO - Processing hierarchy level 0.
    2025-03-06 23:56:56,434 - thuner.track.track - INFO - Tracking convective.
    2025-03-06 23:56:56,462 - thuner.track.track - INFO - Tracking middle.
    2025-03-06 23:56:56,466 - thuner.track.track - INFO - Tracking anvil.
    2025-03-06 23:56:56,471 - thuner.track.track - INFO - Processing hierarchy level 1.
    2025-03-06 23:56:56,471 - thuner.track.track - INFO - Tracking mcs.
    2025-03-06 23:56:56,475 - thuner.write.mask - INFO - Writing mcs masks to /home/ewan/THUNER_output/runs/cpol/cartesian/masks/mcs.zarr.
    2025-03-06 23:56:56,497 - thuner.match.match - INFO - Matching mcs objects.
    2025-03-06 23:56:56,509 - thuner.match.match - INFO - Updating match record.
    2025-03-06 23:56:56,513 - thuner.visualize.runtime - INFO - Creating runtime visualization figures.
    2025-03-06 23:56:59,144 - thuner.attribute.attribute - INFO - Recording mcs attributes.
    2025-03-06 23:56:59,265 - thuner.track.track - INFO - Processing 2005-11-13T18:30:09.
    2025-03-06 23:56:59,266 - thuner.data.aura - INFO - Updating cpol dataset for 2005-11-13T18:30:09.
    2025-03-06 23:56:59,267 - thuner.data.aura - INFO - Converting cpol data from twp10cpolgrid150.b2.20051113.183000.nc
    2025-03-06 23:56:59,389 - thuner.track.track - INFO - Processing hierarchy level 0.
    2025-03-06 23:56:59,390 - thuner.track.track - INFO - Tracking convective.
    2025-03-06 23:56:59,423 - thuner.track.track - INFO - Tracking middle.
    2025-03-06 23:56:59,428 - thuner.track.track - INFO - Tracking anvil.
    2025-03-06 23:56:59,433 - thuner.track.track - INFO - Processing hierarchy level 1.
    2025-03-06 23:56:59,434 - thuner.track.track - INFO - Tracking mcs.
    2025-03-06 23:56:59,437 - thuner.write.mask - INFO - Writing mcs masks to /home/ewan/THUNER_output/runs/cpol/cartesian/masks/mcs.zarr.
    2025-03-06 23:56:59,460 - thuner.match.match - INFO - Matching mcs objects.
    2025-03-06 23:56:59,477 - thuner.match.match - INFO - Updating match record.
    2025-03-06 23:56:59,484 - thuner.visualize.runtime - INFO - Creating runtime visualization figures.
    2025-03-06 23:57:02,374 - thuner.attribute.attribute - INFO - Recording mcs attributes.
    2025-03-06 23:57:02,488 - thuner.track.track - INFO - Processing 2005-11-13T18:40:09.
    2025-03-06 23:57:02,489 - thuner.data.aura - INFO - Updating cpol dataset for 2005-11-13T18:40:09.
    2025-03-06 23:57:02,490 - thuner.data.aura - INFO - Converting cpol data from twp10cpolgrid150.b2.20051113.184000.nc
    2025-03-06 23:57:02,565 - thuner.track.track - INFO - Processing hierarchy level 0.
    2025-03-06 23:57:02,566 - thuner.track.track - INFO - Tracking convective.
    2025-03-06 23:57:02,594 - thuner.track.track - INFO - Tracking middle.
    2025-03-06 23:57:02,598 - thuner.track.track - INFO - Tracking anvil.
    2025-03-06 23:57:02,603 - thuner.track.track - INFO - Processing hierarchy level 1.
    2025-03-06 23:57:02,604 - thuner.track.track - INFO - Tracking mcs.
    2025-03-06 23:57:02,607 - thuner.write.mask - INFO - Writing mcs masks to /home/ewan/THUNER_output/runs/cpol/cartesian/masks/mcs.zarr.
    2025-03-06 23:57:02,627 - thuner.match.match - INFO - Matching mcs objects.
    2025-03-06 23:57:02,639 - thuner.match.match - INFO - Updating match record.
    2025-03-06 23:57:02,642 - thuner.visualize.runtime - INFO - Creating runtime visualization figures.
    2025-03-06 23:57:04,929 - thuner.attribute.attribute - INFO - Recording mcs attributes.
    2025-03-06 23:57:05,041 - thuner.track.track - INFO - Processing 2005-11-13T18:50:08.
    2025-03-06 23:57:05,042 - thuner.data.aura - INFO - Updating cpol dataset for 2005-11-13T18:50:08.
    2025-03-06 23:57:05,043 - thuner.data.aura - INFO - Converting cpol data from twp10cpolgrid150.b2.20051113.185000.nc
    2025-03-06 23:57:05,121 - thuner.track.track - INFO - Processing hierarchy level 0.
    2025-03-06 23:57:05,122 - thuner.track.track - INFO - Tracking convective.
    2025-03-06 23:57:05,153 - thuner.track.track - INFO - Tracking middle.
    2025-03-06 23:57:05,157 - thuner.track.track - INFO - Tracking anvil.
    2025-03-06 23:57:05,164 - thuner.track.track - INFO - Processing hierarchy level 1.
    2025-03-06 23:57:05,164 - thuner.track.track - INFO - Tracking mcs.
    2025-03-06 23:57:05,168 - thuner.write.mask - INFO - Writing mcs masks to /home/ewan/THUNER_output/runs/cpol/cartesian/masks/mcs.zarr.
    2025-03-06 23:57:05,189 - thuner.match.match - INFO - Matching mcs objects.
    2025-03-06 23:57:05,201 - thuner.match.match - INFO - Updating match record.
    2025-03-06 23:57:05,205 - thuner.visualize.runtime - INFO - Creating runtime visualization figures.
    2025-03-06 23:57:07,474 - thuner.attribute.attribute - INFO - Recording mcs attributes.
    2025-03-06 23:57:07,612 - thuner.track.track - INFO - Processing 2005-11-13T19:00:08.
    2025-03-06 23:57:07,614 - thuner.data.aura - INFO - Updating cpol dataset for 2005-11-13T19:00:08.
    2025-03-06 23:57:07,615 - thuner.data.aura - INFO - Converting cpol data from twp10cpolgrid150.b2.20051113.190000.nc
    2025-03-06 23:57:07,700 - thuner.write.filepath - INFO - Writing cpol filepaths from 2005-11-13T18:00:00 to 2005-11-13T19:00:00, inclusive and non-inclusive, respectively.
    2025-03-06 23:57:07,703 - thuner.track.track - INFO - Processing hierarchy level 0.
    2025-03-06 23:57:07,704 - thuner.track.track - INFO - Tracking convective.
    2025-03-06 23:57:07,740 - thuner.track.track - INFO - Tracking middle.
    2025-03-06 23:57:07,751 - thuner.track.track - INFO - Tracking anvil.
    2025-03-06 23:57:07,760 - thuner.track.track - INFO - Processing hierarchy level 1.
    2025-03-06 23:57:07,763 - thuner.track.track - INFO - Tracking mcs.
    2025-03-06 23:57:07,770 - thuner.write.mask - INFO - Writing mcs masks to /home/ewan/THUNER_output/runs/cpol/cartesian/masks/mcs.zarr.
    2025-03-06 23:57:07,789 - thuner.write.attribute - INFO - Writing mcs attributes from 2005-11-13T18:00:00 to 2005-11-13T19:00:00, inclusive and non-inclusive, respectively.
    2025-03-06 23:57:07,890 - thuner.match.match - INFO - Matching mcs objects.
    2025-03-06 23:57:07,907 - thuner.match.match - INFO - Updating match record.
    2025-03-06 23:57:07,916 - thuner.visualize.runtime - INFO - Creating runtime visualization figures.
    2025-03-06 23:57:10,947 - thuner.attribute.attribute - INFO - Recording mcs attributes.
    2025-03-06 23:57:11,112 - thuner.write.attribute - INFO - Writing mcs attributes from 2005-11-13T19:00:08 to 2005-11-13T20:00:08, inclusive and non-inclusive, respectively.
    2025-03-06 23:57:11,174 - thuner.write.filepath - INFO - Writing cpol filepaths from 2005-11-13T19:00:00 to 2005-11-13T20:00:00, inclusive and non-inclusive, respectively.
    2025-03-06 23:57:11,179 - thuner.write.attribute - INFO - Aggregating attribute files.
    2025-03-06 23:57:11,470 - thuner.write.filepath - INFO - Aggregating filepath records.
    2025-03-06 23:57:11,478 - thuner.visualize.visualize - INFO - Animating match figures for mcs objects.
    2025-03-06 23:57:11,480 - thuner.visualize.visualize - INFO - Saving animation to /home/ewan/THUNER_output/runs/cpol/cartesian/visualize/match/mcs_20051113.gif.


.. code:: ipython3

    analysis_options = analyze.mcs.AnalysisOptions()
    analysis_options.to_yaml(options_directory / "analysis.yml")
    # utils.save_options(analysis_options, filename="analysis", options_directory=output_directory / "options")
    analyze.mcs.process_velocities(output_parent)
    analyze.mcs.quality_control(output_parent, analysis_options)
    # analyze.mcs.classify_all(output_parent, analysis_options)


.. parsed-literal::

    2025-03-06 23:57:12,187 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.
    2025-03-06 23:57:12,545 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.


.. code:: ipython3

    figure_name = "mcs_attributes"
    kwargs = {"style": "presentation", "attributes": ["velocity", "offset"]}
    figure_options = option.visualize.HorizontalAttributeOptions(name=figure_name, **kwargs)
    
    start_time = np.datetime64(start)
    end_time = np.datetime64(end)
    args = [output_parent, start_time, end_time, figure_options]
    args_dict = {"parallel_figure": True, "by_date": False, "num_processes": 4}
    visualize.attribute.mcs_series(*args, **args_dict)


.. parsed-literal::

    2025-03-06 23:57:13,037 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.
    2025-03-06 23:57:13,256 - thuner.visualize.attribute - INFO - Visualizing MCS at time 2005-11-13T18:00:08.000000000.
    2025-03-06 23:57:13,265 - thuner.data.aura - INFO - Converting cpol data from twp10cpolgrid150.b2.20051113.180000.nc
    2025-03-06 23:57:13,435 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.
    2025-03-06 23:57:14,122 - thuner.visualize.attribute - INFO - Saving mcs_attributes figure for 2005-11-13T18:00:08.000000000.
    2025-03-06 23:57:21,449 - thuner.visualize.attribute - INFO - Visualizing MCS at time 2005-11-13T18:10:23.000000000.
    2025-03-06 23:57:21,451 - thuner.data.aura - INFO - Converting cpol data from twp10cpolgrid150.b2.20051113.181000.nc
    2025-03-06 23:57:21,520 - thuner.visualize.attribute - INFO - Visualizing MCS at time 2005-11-13T18:20:09.000000000.
    2025-03-06 23:57:21,525 - thuner.data.aura - INFO - Converting cpol data from twp10cpolgrid150.b2.20051113.182000.nc
    2025-03-06 23:57:21,899 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.
    2025-03-06 23:57:21,979 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.
    2025-03-06 23:57:22,413 - thuner.visualize.attribute - INFO - Saving mcs_attributes figure for 2005-11-13T18:10:23.000000000.
    2025-03-06 23:57:22,481 - thuner.visualize.attribute - INFO - Saving mcs_attributes figure for 2005-11-13T18:20:09.000000000.
    2025-03-06 23:57:23,346 - thuner.visualize.attribute - INFO - Visualizing MCS at time 2005-11-13T18:30:09.000000000.
    2025-03-06 23:57:23,348 - thuner.data.aura - INFO - Converting cpol data from twp10cpolgrid150.b2.20051113.183000.nc
    2025-03-06 23:57:23,897 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.
    2025-03-06 23:57:24,427 - thuner.visualize.attribute - INFO - Saving mcs_attributes figure for 2005-11-13T18:30:09.000000000.
    2025-03-06 23:57:25,358 - thuner.visualize.attribute - INFO - Visualizing MCS at time 2005-11-13T18:40:09.000000000.
    2025-03-06 23:57:25,360 - thuner.data.aura - INFO - Converting cpol data from twp10cpolgrid150.b2.20051113.184000.nc
    2025-03-06 23:57:25,790 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.
    2025-03-06 23:57:26,282 - thuner.visualize.attribute - INFO - Saving mcs_attributes figure for 2005-11-13T18:40:09.000000000.
    2025-03-06 23:57:27,018 - thuner.visualize.attribute - INFO - Visualizing MCS at time 2005-11-13T18:50:08.000000000.
    2025-03-06 23:57:27,021 - thuner.data.aura - INFO - Converting cpol data from twp10cpolgrid150.b2.20051113.185000.nc
    2025-03-06 23:57:27,141 - thuner.option.grid - WARNING - shape not specified. Will attempt to infer from input.
    2025-03-06 23:57:27,453 - thuner.visualize.attribute - INFO - Saving mcs_attributes figure for 2005-11-13T18:50:08.000000000.
    2025-03-06 23:57:29,694 - thuner.visualize.visualize - INFO - Animating mcs_attributes figures for mcs objects.
    2025-03-06 23:57:29,696 - thuner.visualize.visualize - INFO - Saving animation to /home/ewan/THUNER_output/runs/cpol/cartesian/visualize/mcs_attributes.gif.


