# thuner/option/option.py
"""Overall options class."""

from typing import List
from pydantic import Field, model_validator
from thuner.utils import BaseOptions
from thuner.option.grid import GridOptions
from thuner.option.track import TrackOptions
from thuner.option.data import DataOptions


def _get_object_datasets(object_options, referenced_datasets):
    """
    Helper function to collect datasets from object options.
    """
    if object_options.dataset:
        referenced_datasets.add(object_options.dataset)
    if object_options.attributes:
        for attr_type in object_options.attributes.attribute_types:
            if attr_type.dataset:
                referenced_datasets.add(attr_type.dataset)


class Options(BaseOptions):
    """
    Main options class containing grid, track, and data options with validation
    to ensure mutual consistency between different option types.
    """

    grid: GridOptions = Field(..., description="Grid options.")
    track: TrackOptions = Field(..., description="Object options.")
    data: DataOptions = Field(..., description="Data options.")

    @model_validator(mode="after")
    def _check_datasets(cls, values):
        """
        Validate that all datasets referenced in tracking options exist in data options.
        """
        # Collect all dataset names referenced in tracking options
        referenced_datasets = set()
        for level in values.track.levels:
            for obj in level.objects:
                _get_object_datasets(obj, referenced_datasets)

        # Check for missing datasets
        missing_datasets = referenced_datasets - set(values.data.dataset_names)
        if missing_datasets:
            missing_list = sorted(list(missing_datasets))
            message = f"Tracking options reference datasets {missing_list} that don't "
            message += "exist in data options."
            raise ValueError(message)

        return values
