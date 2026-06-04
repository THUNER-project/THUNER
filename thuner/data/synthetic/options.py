"""
Synthetic dataset options.

:class:`SyntheticOptions` is the :class:`~thuner.utils.BaseDatasetOptions` adapter for
synthetic data. It lets the tracker treat a synthetic dataset exactly like a real one:
there are no files on disk, and each grid is generated on the fly as tracking proceeds.
The scene itself is owned by a :class:`~thuner.data.synthetic.generator.SyntheticGenerator`
(its evolving run-time state lives in the generator's private attributes).
"""

from pydantic import Field
from thuner.log import setup_logger
import thuner.data._utils as _utils
from thuner.utils import BaseDatasetOptions
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
        ..., description="Scene generator producing the data. See thuner.data.synthetic."
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

    def update_input_record(self, time, input_record, track_options, grid_options):
        """Generate the synthetic dataset for ``time`` into the input record."""
        _utils.log_dataset_update(logger, self.name, time)
        # The generator resets itself when the grid changes (e.g. the same data_options
        # reused for a geographic then a cartesian run), so a step is always valid here.
        input_record.dataset = self.generator.step(time, grid_options)
