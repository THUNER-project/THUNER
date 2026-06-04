"""On-the-fly synthetic datasets for testing THUNER tracking configurations.

Objects with known kinematics and geometry are generated as gridded data on the fly,
so configurations can be exercised against features with known velocities, sizes and
orientations. See :mod:`thuner.data.synthetic.objects` for object types,
:mod:`thuner.data.synthetic.generator` for the generator, and
:mod:`thuner.data.synthetic.options` for the dataset options adapter.
"""

import thuner.data.synthetic.objects as objects
import thuner.data.synthetic.generator as generator
import thuner.data.synthetic.options as options
import thuner.data.synthetic.truth as truth
from thuner.data.synthetic.objects import SyntheticObject, EllipsoidObject
from thuner.data.synthetic.generator import (
    SyntheticGenerator,
    FixedGenerator,
    RandomEllipseGenerator,
)
from thuner.data.synthetic.options import SyntheticOptions
from thuner.data.synthetic.truth import synthetic_ground_truth

__all__ = [
    "objects",
    "generator",
    "options",
    "truth",
    "SyntheticObject",
    "EllipsoidObject",
    "SyntheticGenerator",
    "FixedGenerator",
    "RandomEllipseGenerator",
    "SyntheticOptions",
    "synthetic_ground_truth",
]
