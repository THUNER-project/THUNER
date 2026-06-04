"""
Synthetic dataset options.

:class:`SyntheticOptions` is the :class:`~thuner.utils.BaseDatasetOptions` adapter for
synthetic data. It lets the tracker treat a synthetic dataset exactly like a real one:
there are no files on disk, and each grid is generated on the fly as tracking proceeds.
The evolving generation state is owned by a private :class:`SyntheticGenerator`.
"""

from pydantic import Field, PrivateAttr
from thuner.log import setup_logger
import thuner.data._utils as _utils
from thuner.utils import BaseDatasetOptions
from thuner.data.synthetic.objects import EllipsoidObject
from thuner.data.synthetic.generator import SyntheticGenerator

logger = setup_logger(__name__)


class SyntheticOptions(BaseDatasetOptions):
    """Options for an on-the-fly synthetic dataset."""

    name: str = Field("synthetic")
    start: str = Field("2005-11-13T00:00:00")
    end: str = Field("2005-11-14T00:00:00")
    fields: list[str] = Field(["reflectivity"])
    use: str = Field("track")
    # As more object types are added this becomes a discriminated union, e.g.
    # list[Union[EllipsoidObject, AnvilObject]] keyed on the auto-injected "type".
    objects: list[EllipsoidObject] = Field(
        ..., description="Synthetic objects to generate. See thuner.data.synthetic."
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
    _generator: SyntheticGenerator | None = PrivateAttr(default=None)

    def get_filepaths(self):
        """Synthetic data has no files on disk; return an empty list."""
        return []

    def update_input_record(self, time, input_record, track_options, grid_options):
        """Generate the synthetic dataset for ``time`` into the input record."""
        _utils.log_dataset_update(logger, self.name, time)
        # The options object can outlive a single run (e.g. the same data_options
        # reused for a geographic then a cartesian run), so rebuild the generator
        # whenever the grid changes. This resets object positions to their start,
        # which is the desired behaviour at the beginning of a new run.
        if self._generator is None or self._generator.grid_options is not grid_options:
            self._generator = SyntheticGenerator(self.objects, grid_options)
        input_record.dataset = self._generator.step(time)
