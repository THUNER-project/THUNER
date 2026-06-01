"""Default options configurations.

Curated, runnable default configurations built on top of the :mod:`thuner.option`
vocabulary. Tracking-option presets live in :mod:`thuner.default.track`;
visualization presets and attribute handlers live in :mod:`thuner.default.visualize`.
"""

import thuner.default.track as track
import thuner.default.visualize as visualize
import thuner.default.utils as utils

__all__ = ["track", "visualize", "utils"]
