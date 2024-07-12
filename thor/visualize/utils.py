"""Visualization convenience functions."""

import string
import subprocess
import matplotlib.pyplot as plt


def make_subplot_labels(axes, x_shift=-0.15, y_shift=0, fontsize=12):
    labels = list(string.ascii_lowercase)
    labels = [label + ")" for label in labels]
    for i in range(len(axes)):
        axes[i].text(
            x_shift,
            1.0 + y_shift,
            labels[i],
            transform=axes[i].transAxes,
            fontsize=plt.rcParams["axes.titlesize"],
        )


def call_convert(input_filepaths, output_filepath):
    command = f"magick -delay 20 -loop 0 {input_filepaths} {output_filepath}"
    subprocess.run(command, shell=True, check=True)
