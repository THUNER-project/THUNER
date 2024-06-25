"""General display methods."""

import numpy as np
import pyart.graph.cm_colorblind as pcm

grid_formats = {
    "reflectivity": {
        "cmap": pcm.HomeyerRainbow,
        "levels": np.arange(-10, 60 + 5, 5),
    },
}

map_colors = {
    "paper": {
        "land_color": tuple(np.array([249.0, 246.0, 216.0]) / (256)),
        "sea_color": tuple(np.array([240.0, 240.0, 256.0]) / (256)),
        "coast_color": "black",
        "leg_color": "w",
    },
    "presentation": {
        "land_color": tuple(np.array([249.0, 246.0, 216.0]) / (256 * 3.5)),
        "sea_color": tuple(np.array([245.0, 245.0, 256.0]) / (256 * 3.5)),
        "coast_color": "white",
        "leg_color": tuple(np.array([249.0, 246.0, 216.0]) / (256 * 3.5)),
    },
}

base_styles = {"paper": "default", "presentation": "dark_background"}

styles = {
    style: [base_styles[style], f"../thor/visualize/styles/{style}.mplstyle"]
    for style in base_styles.keys()
}
