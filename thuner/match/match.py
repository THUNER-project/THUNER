"""Module for matching objects between current and next masks."""

from typing import Callable, NamedTuple
import numpy as np
from thuner.log import setup_logger
import thuner.match.tint as tint
from thuner.match.utils import get_masks, TintMatchRecord

logger = setup_logger(__name__)


__all__ = ["match", "tint"]


class MatchMethod(NamedTuple):
    """A registered matching method: its match record class and matching function."""

    record_class: type
    get_matches: Callable


# Registry mapping a tracking method's name to its matching implementation. Extend this
# when adding new tracking/matching methods. TINT and MINT share the same matching
# function; the differences between them are handled within thuner.match.tint.
match_methods = {
    "tint": MatchMethod(TintMatchRecord, tint.get_matches),
    "mint": MatchMethod(TintMatchRecord, tint.get_matches),
}


def get_match_method(object_options):
    """Return the registered MatchMethod for an object's tracking options."""
    tracking = object_options.tracking
    name = getattr(tracking, "name", None)
    if name not in match_methods:
        message = f"Matching not implemented for tracking method '{name}'. "
        message += f"Supported methods: {sorted(match_methods)}."
        raise ValueError(message)
    return match_methods[name]


def match(object_tracks, object_options, grid_options):
    """Match objects between current and next masks."""
    if object_options.tracking is None:
        return
    match_method = get_match_method(object_options)
    next_mask, current_mask = get_masks(object_tracks, object_options)
    logger.info(f"Matching {object_options.name} objects.")
    next_ids = np.unique(next_mask.values)
    next_ids = next_ids[next_ids != 0]

    def reset_match_record():
        """Start a fresh, empty match record, with no matches at the current time."""
        record = match_method.record_class()
        object_tracks.match_record = record
        args = [object_tracks, object_options, grid_options]
        record.get_matched_mask(*args, current_ids=next_ids)

    if current_mask is None or np.max(current_mask) == 0:
        logger.info("No current mask, or no objects in current mask.")
        reset_match_record()
        return
    if object_tracks.next_time_interval > object_options.allowed_gap * 60:
        logger.info("Time gap too large. Resetting match record.")
        reset_match_record()
        return

    logger.debug("Getting matches.")
    # The matching function reads the previous iteration's record from object_tracks, so
    # capture it before overwriting object_tracks.match_record below.
    previous_record = object_tracks.match_record
    record = match_method.get_matches(object_tracks, object_options, grid_options)
    record.assign_ids(object_tracks, object_options, previous_record)
    object_tracks.match_record = record
    record.get_matched_mask(object_tracks, object_options, grid_options)
