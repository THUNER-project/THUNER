"""Track storm objects in a dataset."""

from thor.log import setup_logger
import thor.option as option
import thor.data.data as data


logger = setup_logger(__name__)


def initialise_tracks(track_options, data_options):
    """
    Initialise the tracks dictionary.

    Parameters
    ----------
    track_options : dict
        Dictionary containing the track options.

    Returns
    -------
    dict
        Dictionary that will contain the tracks of each object.

    """

    # initialise tracks as a list of dictionaries
    # one dictionary for each hierarchy level
    tracks = []
    for level_options in track_options:
        level_tracks = {obj: {} for obj in level_options.keys()}
        for obj in level_options.keys():
            level_tracks[obj]["tracks"] = []
            dataset = level_options[obj]["dataset"]
            if dataset is not None:
                filepaths = data.generate_filepaths(data_options[dataset])
            else:
                filepaths = None
            level_tracks[obj]["filepaths"] = filepaths
        tracks.append(level_tracks)
    return tracks


def simultaneous_track(times, data_options, grid_options, track_options):
    """
    Track objects across the hierachy simultaneously.

    Parameters
    ----------
    filenames : list of str
        List of filepaths to the netCDF files that need to be consolidated.
    data_options : dict
        Dictionary containing the data options.
    grid_options : dict
        Dictionary containing the grid options.
    track_options : dict
        Dictionary containing the track options.

    Returns
    -------
    pandas.DataFrame
        The pandas dataframe containing the object tracks.
    xarray.Dataset
        The xarray dataset containing the object masks.

    """

    option.check_options(track_options)
    # tracks will be a list containing dictionaries
    # each dictionary contains the tracks at a given hierarchy level
    tracks = initialise_tracks(track_options)

    for time in times:
        logger.info(f"Processing {time}.")
        # loop over levels
        for index, level_options in enumerate(track_options):
            logger.debug(f"Processing hierarchy level {index}.")
            tracks[index] = track_level(
                time, tracks[index], data_options, grid_options, level_options
            )
    return tracks


def track_level(time, level_tracks, data_options, grid_options, level_options):

    # loop over objects in level
    for obj in level_tracks.keys():
        logger.debug(f"Tracking {obj}.")
        level_tracks[obj] = track_object(
            time, level_tracks[obj], data_options, grid_options, level_options[obj]
        )

    return level_tracks


def track_object(time, tracks, data_options, grid_options, object_options):

    # track the object
    tracks["tracks"].append(f"Tracks at {time}.")
    tracks["current_file"] = None
    tracks["ds"] = None
    tracks["previous_ds"] = None

    # store the parent ds?
    # store the current timeslice "grid"

    # load the relevent file? push deque

    return tracks
