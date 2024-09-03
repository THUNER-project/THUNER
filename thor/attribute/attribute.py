"""Methods for getting object attributes."""

from thor.attribute import core
from thor.log import setup_logger

logger = setup_logger(__name__)

record_dispatcher = {
    "core": core.record,
}


def record(time, object_tracks, object_options, grid_options):
    """Get object attributes."""
    logger.info("Recording object attributes.")
    if object_options["attribute"] is None:
        return

    # Get the object attributes of each type, e.g. core, tag, profile
    for attributes_type in object_options["attribute"].keys():
        record_attributes = record_dispatcher[attributes_type]
        record_attributes(time, object_tracks, object_options, grid_options)
