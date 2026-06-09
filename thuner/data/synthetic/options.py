"""
Synthetic dataset options.

:class:`SyntheticOptions` is the :class:`~thuner.utils.BaseDatasetOptions` adapter for
synthetic data. It lets the tracker treat a synthetic dataset exactly like a real one:
there are no files on disk, and each grid is generated on the fly as tracking proceeds.
The scene itself is owned by a :class:`~thuner.data.synthetic.generator.SyntheticGenerator`
(its evolving run-time state lives in the generator's private attributes).
"""

from pathlib import Path
from pydantic import Field
from thuner.log import setup_logger
import thuner.data._utils as _utils
from thuner.utils import (
    BaseDatasetOptions,
    InputUnit,
    save_converted_dataset,
    format_time,
)
from thuner.data.synthetic.generator import AnyGenerator

logger = setup_logger(__name__)


class SyntheticOptions(BaseDatasetOptions):
    """Options for an on-the-fly synthetic dataset."""

    name: str = Field("synthetic")
    start: str = Field("2005-11-13T00:00:00")
    end: str = Field("2005-11-14T00:00:00")
    fields: list[str] = Field(["reflectivity"])
    use: str = Field("track")
    generator: AnyGenerator = Field(
        ...,
        description="Scene generator producing the data. See thuner.data.synthetic.",
    )
    target_objects: list[str | tuple[str, str]] | None = Field(
        None,
        description=(
            "Tracked objects to match synthetic ground-truth objects against (by "
            "containment of each truth object's centre in the detected masks). Each "
            "entry is either an object name — matched against that object's own masks "
            "— or a (grouped_object, member) tuple — matched against that member's mask "
            "within the grouped object. None disables matching. Targets' masks must be "
            "saved (mask_options.save=True)."
        ),
    )

    def get_filepaths(self):
        """Synthetic data has no files on disk; return an empty list."""
        return []

    def converted_filepath(self, unit):
        """A saved synthetic grid *is* its own source, so return the path directly.

        Unlike a real dataset (whose converted path is derived from a raw source), a
        synthetic unit's single source is already the saved ``.nc``; returning it lets
        ``save_converted_dataset`` and ``load_or_convert`` resolve the same path.
        """
        return unit.sources[0]

    def _converted_filepath_for_time(self, time):
        """Deterministic path for the saved synthetic grid at ``time``."""
        parent = Path(self.converted_options.parent_converted) / self.name
        return str(parent / f"{format_time(time)}.nc")

    def update_input_record(self, time, input_record, track_options, grid_options):
        """Generate the synthetic dataset for ``time`` into the input record.

        With ``converted_options.save`` set, the generated grid is also written to disk
        and a filepath record buffered, so synthetic data loads and visualizes through
        the same path as a real converted dataset (it has no source files of its own to
        record otherwise).
        """
        _utils.log_dataset_update(logger, self.name, time)
        # The generator resets itself when the grid changes (e.g. the same data_options
        # reused for a geographic then a cartesian run), so a step is always valid here.
        dataset = self.generator.step(time, grid_options)
        input_record.dataset = dataset
        if self.converted_options.save:
            unit = InputUnit(sources=[self._converted_filepath_for_time(time)])
            save_converted_dataset(unit, dataset, self)
            input_record._time_list.append(time)
            input_record._filepath_list.append(unit.record_str())
