from collections import deque
import numpy as np
import pandas as pd
import xarray as xr
from thuner.log import setup_logger
import thuner.object.object as thuner_object
import thuner.match.tint as tint
from thuner.match.utils import get_masks

logger = setup_logger(__name__)


def initialise_match_records(object_tracks, object_options):
    """Initialise the match records dictionary for the object tracks."""
    object_tracks.next_matched_mask = None
    deque_length = object_options.deque_length
    object_tracks.matched_masks = deque([None] * deque_length, maxlen=deque_length)
    object_tracks.match_record = thuner_object.empty_match_record()
    object_tracks.previous_match_records = deque(
        [None] * deque_length, maxlen=deque_length
    )


def match(object_tracks, object_options, grid_options):
    """Match objects between current and next masks."""
    if object_options.tracking is None:
        return
    next_mask, current_mask = get_masks(object_tracks, object_options)
    logger.info(f"Matching {object_options.name} objects.")
    current_ids = np.unique(next_mask)
    current_ids = current_ids[current_ids != 0]

    def reset_match_record():
        object_tracks.match_record = thuner_object.empty_match_record()
        args = (object_tracks, object_options, grid_options)
        get_matched_mask(*args, current_ids=current_ids)

    if current_mask is None or np.max(current_mask) == 0:
        logger.info("No current mask, or no objects in current mask.")
        reset_match_record()
        return
    if object_tracks.next_time_interval > object_options.allowed_gap * 60:
        logger.info("Time gap too large. Resetting match record.")
        reset_match_record()
        return

    logger.debug("Getting matches.")
    match_data = tint.get_matches(object_tracks, object_options, grid_options)
    # Get the ids from the previous mask, i.e. the current mask of
    # the last matching iteration, to see whether objects detected in the current
    # mask of the current matching iteration are new.
    ids = np.array(object_tracks.match_record["ids"])
    ids[ids > 0]

    if len(ids) == 0:
        logger.info("New matchable objects. Initializing match record.")
        thuner_object.initialize_match_record(match_data, object_tracks, object_options)
    else:
        logger.info("Updating match record.")
        thuner_object.update_match_record(match_data, object_tracks, object_options)

    get_matched_mask(object_tracks, object_options, grid_options)


def get_matched_mask(object_tracks, object_options, grid_options, current_ids=None):
    """Get the matched mask for the current time."""
    next_mask = get_masks(object_tracks, object_options)[0]

    match_record = object_tracks.match_record
    if current_ids is None:
        current_ids = np.unique(next_mask.values)
        current_ids = current_ids[current_ids != 0]
    universal_id_dict = dict(
        zip(match_record["next_ids"], match_record["universal_ids"])
    )

    # Not all the objects in the next mask are in the current objects list of the
    # match record. These are new objects in the next mask, unmatched with those in
    # the current mask. These new object ids will be created in the match record in
    # the next iteration of the tracking loop. However, to update the next
    # matched mask, we need to premptively assign new universal ids to these new objects.
    unmatched_ids = [id for id in current_ids if id not in match_record["next_ids"]]
    new_universal_ids = np.arange(
        object_tracks.object_count + 1,
        object_tracks.object_count + len(unmatched_ids) + 1,
    )
    new_universal_id_dict = dict(zip(unmatched_ids, new_universal_ids))
    universal_id_dict.update(new_universal_id_dict)
    universal_id_dict[0] = 0

    def replace_values(data_array, value_dict):
        series = pd.Series(data_array.ravel())
        replaced = series.map(value_dict).values.reshape(data_array.shape)
        return replaced

    if grid_options.name == "cartesian":
        core_dims = [["y", "x"]]
    elif grid_options.name == "geographic":
        core_dims = [["latitude", "longitude"]]
    else:
        raise ValueError(f"Grid name must be 'cartesian' or 'geographic'.")

    next_matched_mask = xr.apply_ufunc(
        replace_values,
        object_tracks.next_mask,
        kwargs={"value_dict": universal_id_dict},
        input_core_dims=core_dims,
        output_core_dims=core_dims,
        vectorize=True,
    )
    # Update the matched mask deque with the next mask from the previous iteration
    matched_mask = object_tracks.next_matched_mask
    object_tracks.matched_masks.append(matched_mask)
    # Update the next matched mask with the next matched mask of the current iteration
    object_tracks.next_matched_mask = next_matched_mask
