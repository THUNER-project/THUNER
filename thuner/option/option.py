# thuner/option/option.py
"""Overall options class."""

from typing import List
from pydantic import Field, model_validator
from thuner.utils import BaseOptions
from thuner.option.grid import GridOptions
from thuner.option.track import TrackOptions
from thuner.option.data import DataOptions
import numpy as np


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
    def _check_datasets(self):
        """
        Validate that all datasets referenced in tracking options exist in data options.
        """
        # Collect all dataset names referenced in tracking options
        referenced_datasets = set()
        for level in self.track.levels:
            for obj in level.objects:
                _get_object_datasets(obj, referenced_datasets)

        # Check for missing datasets
        missing_datasets = referenced_datasets - set(self.data.dataset_names)
        if missing_datasets:
            missing_list = sorted(list(missing_datasets))
            message = f"Tracking options reference datasets {missing_list} that don't "
            message += "exist in data options."
            raise ValueError(message)

        return self

    @model_validator(mode="after")
    def _check_himawari_grid(self):
        """
        Check latitude/longitude have been provided for Himawari data.
        """
        if "himawari" not in self.data.dataset_names:
            return self
        if self.grid.name != "geographic":
            message = "Cartesian coordiantes not yet implemented for Himawari data."
            raise ValueError(message)
        if self.grid.latitude is None or self.grid.longitude is None:
            message = "Please set latitude/longitude grid options explicitly, as "
            message += "regridding the entire Himawari disk uses a lot of memory!"
            raise ValueError(message)
        lon = np.array(self.grid.longitude)
        lat = np.array(self.grid.latitude)
        if lon.min() < -180 or lon.max() > 180:
            message = "Himawari longitude must be between -180 and 180 degrees."
            raise ValueError(message)
        if (lat.min() < -81.13867) or (lat.max() > 81.13867):
            message = (
                "Himawari latitude must be between -81.13867 and 81.13867 degrees."
            )
            raise ValueError(message)
        return self
