"""Data options classes, convenience subclasses, and functions."""

from typing import Dict, Union, List
from pydantic import Field, model_validator
from thuner.log import setup_logger
from thuner.utils import BaseOptions, BaseDatasetOptions
import thuner.data.gridrad as gridrad
import thuner.data.aura as aura
import thuner.data.era5 as era5
import thuner.data.himawari as himawari
import thuner.data.access as access
import thuner.data.synthetic as synthetic

logger = setup_logger(__name__)

__all__ = ["DataOptions"]


AnyDatasetOptions = Union[
    gridrad.GridRadSevereOptions,
    aura.CpolOptions,
    era5.Era5Options,
    himawari.HimawariOptions,
    access.AccessCOptions,
    synthetic.SyntheticOptions,
    BaseDatasetOptions,
]


class DataOptions(BaseOptions):
    """Class for managing the options for all the datasets of a given run."""

    datasets: list[AnyDatasetOptions] = Field(
        ..., description="List of dataset options."
    )
    _dataset_lookup: Dict[str, AnyDatasetOptions] = {}
    dataset_names: List[str] = Field(
        [],
        description=(
            "List of dataset names to be used in the run. This is set automatically."
        ),
    )

    @model_validator(mode="after")
    def check_unique_names(self):
        """Check that all dataset names are unique."""
        names = [dataset.name for dataset in self.datasets]
        seen, duplicates = set(), set()
        for name in names:
            if name in seen:
                duplicates.add(name)
            seen.add(name)
        if duplicates:
            raise ValueError(f"Duplicate dataset names: {sorted(duplicates)}.")
        return self

    @model_validator(mode="after")
    def check_regridder_from(self):
        """Check the regridder_from references are valid and acyclic."""
        options_by_name = {dataset.name: dataset for dataset in self.datasets}
        for dataset in self.datasets:
            source = dataset.regridder_from
            if source is None:
                continue
            if not dataset.reuse_regridder:
                message = f"Dataset {dataset.name!r} sets regridder_from but has "
                message += "reuse_regridder=False."
                raise ValueError(message)
            if source == dataset.name:
                raise ValueError(f"Dataset {dataset.name!r} has regridder_from set to itself.")
            if source not in options_by_name:
                message = f"Dataset {dataset.name!r} has regridder_from={source!r}, which is "
                message += "not a dataset in this DataOptions."
                raise ValueError(message)
            if not options_by_name[source].reuse_regridder:
                message = f"Dataset {dataset.name!r} reuses the regridder of {source!r}, but "
                message += f"{source!r} has reuse_regridder=False."
                raise ValueError(message)
        # Walk each regridder_from chain to detect cycles.
        for dataset in self.datasets:
            seen, current = [], dataset
            while current.regridder_from is not None:
                if current.name in seen:
                    message = f"Cyclic regridder_from references: {seen + [current.name]}."
                    raise ValueError(message)
                seen.append(current.name)
                current = options_by_name[current.regridder_from]
        return self

    @model_validator(mode="after")
    def initialize_dataset_lookup(self):
        """Initialize the dataset lookup dictionary."""
        self._dataset_lookup = {dataset.name: dataset for dataset in self.datasets}
        self.dataset_names = list(self._dataset_lookup.keys())
        return self

    def dataset_by_name(self, dataset_name: str) -> AnyDatasetOptions:
        """Return the dataset options for a given dataset name."""
        return self._dataset_lookup.get(dataset_name)
