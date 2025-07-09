"""Data options classes, convenience subclasses, and functions."""

from typing import Dict, Union, List
from pydantic import Field, model_validator
from thuner.log import setup_logger
from thuner.utils import BaseOptions, BaseDatasetOptions
import thuner.data.gridrad as gridrad
import thuner.data.aura as aura
import thuner.data.era5 as era5
import thuner.data.himawari as himawari

logger = setup_logger(__name__)

__all__ = ["DataOptions"]

_summary = {"datasets": "List of dataset options."}


AnyDatasetOptions = Union[
    gridrad.GridRadSevereOptions,
    aura.CPOLOptions,
    aura.OperationalOptions,
    era5.ERA5Options,
    himawari.HimawariOptions,
    BaseDatasetOptions,
]


class DataOptions(BaseOptions):
    """Class for managing the options for all the datasets of a given run."""

    datasets: list[AnyDatasetOptions] = Field(..., description=_summary["datasets"])
    _dataset_lookup: Dict[str, AnyDatasetOptions] = {}
    _desc = "List of dataset names to be used in the run. This is set automatically."
    dataset_names: List[str] = Field([], description=_desc)

    @model_validator(mode="after")
    def initialize_dataset_lookup(cls, values):
        """Initialize the dataset lookup dictionary."""
        values._dataset_lookup = {d.name: d for d in values.datasets}
        values.dataset_names = list(values._dataset_lookup.keys())
        return values

    def dataset_by_name(self, dataset_name: str) -> AnyDatasetOptions:
        """Return the dataset options for a given dataset name."""
        return self._dataset_lookup.get(dataset_name)
