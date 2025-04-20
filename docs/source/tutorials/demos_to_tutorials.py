import nbformat
from pathlib import Path
from nbconvert import RSTExporter
import re


def convert_notebook_to_tutorial(notebook_path, rst_path):
    # Load the notebook
    with open(Path(notebook_path), "r", encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=4)

    # Create an .rst exporter
    rst_exporter = RSTExporter()
    rst_exporter.exclude_output = True

    # Convert the notebook to an .rst file
    rst = rst_exporter.from_notebook_node(notebook)[0]

    lines = rst.split("\n")
    cleaned_lines = []
    for line in lines:
        # Remove IPython magic commands
        if re.match(r"#!/usr/bin/env python", line):
            continue
        if re.match(r"# coding: utf-8", line):
            continue
        if re.match(r"get_ipython\(\)\.run_line_magic", line):
            continue
        # Remove cell markers
        if re.match(r"# In\[.*\]:", line):
            continue
        cleaned_lines.append(line)
    rst = "\n".join(cleaned_lines)
    # Remove leading and trailing whitespace
    rst = rst.strip()
    # Remove duplicate empty lines
    rst = re.sub(r"\n{3,}", "\n\n", rst)
    rst = rst.replace(".. code:: ipython3", ".. code-block:: python3")
    classes = [
        "pydantic.BaseModel",
        "thuner.option.data.DataOptions",
        "thuner.option.track.BaseObjectOptions",
        "thuner.option.track.LevelOptions",
        "thuner.option.track.TrackOptions",
        "pandas.DataFrame",
        "thuner.option.attribute.Attribute",
        "thuner.option.attribute.AttributeGroup",
        "thuner.option.attribute.AttributeType",
        "thuner.utils.BaseDatasetOptions",
    ]
    for class_name in classes:
        rst = rst.replace(f"``{class_name}``", f":class:`{class_name}`")
    modules = [
        "thuner.default",
        "xarray",
    ]
    for module_name in modules:
        rst = rst.replace(f"``{module_name}``", f":mod:`{module_name}`")
    functions = [
        "thuner.data.generate_times",
        "thuner.parallel.track",
        "thuner.analyze.read_options",
        "thuner.attribute.utils.read_attribute_csv",
    ]
    for function_name in functions:
        rst = rst.replace(f"``{function_name}``", f":func:`{function_name}`")

    # Save the script to a file
    with open(rst_path, "w", encoding="utf-8") as f:
        f.write(rst)


demo_dir = Path(__file__).parent.parent.parent.parent / "demo"
tutorial_dir = Path(__file__).parent

# Iterate over items in the demo directory
for item in demo_dir.iterdir():
    if item.is_file() and item.suffix == ".ipynb":
        # Convert the notebook to a test script
        convert_notebook_to_tutorial(item, tutorial_dir / f"{item.stem}.rst")
