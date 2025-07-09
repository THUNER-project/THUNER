"""Options classes and functions for managing standard datasets."""

import thuner.data.access as access
import thuner.data.aura as aura
import thuner.data.era5 as era5
import thuner.data.gridrad as gridrad
import thuner.data.himawari as himawari
import thuner.data.odim as odim
import thuner.data._utils as _utils
import thuner.data.wrf as wrf
import thuner.data.synthetic as synthetic
from thuner.data._utils import get_demo_data

__all__ = ["aura", "era5", "gridrad", "synthetic", "himawari", "get_demo_data"]
