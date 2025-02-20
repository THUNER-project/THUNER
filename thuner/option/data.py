"""Data options classes, convenience subclasses, and functions."""

from typing import Dict, Union
from pydantic import Field, model_validator
from thuner.log import setup_logger
from thuner.utils import BaseOptions, BaseDatasetOptions
from thuner.data.gridrad import GridRadSevereOptions
from thuner.data.aura import CPOLOptions, OperationalOptions
from thuner.data.era5 import ERA5Options
from thuner.data.synthetic import SyntheticOptions

logger = setup_logger(__name__)


_summary = {"datasets": "List of dataset options."}


AnyDatasetOptions = Union[
    BaseDatasetOptions,
    GridRadSevereOptions,
    CPOLOptions,
    OperationalOptions,
    ERA5Options,
    SyntheticOptions,
]


__all__ = ["DataOptions"]


class DataOptions(BaseOptions):
    """Class for managing the options for all the datasets of a given run."""

    datasets: list[AnyDatasetOptions] = Field(..., description=_summary["datasets"])
    _dataset_lookup: Dict[str, AnyDatasetOptions] = {}

    @model_validator(mode="after")
    def initialize_dataset_lookup(cls, values):
        values._dataset_lookup = {d.name: d for d in values.datasets}
        return values

    def dataset_by_name(self, dataset_name: str) -> AnyDatasetOptions:
        return self._dataset_lookup.get(dataset_name)
