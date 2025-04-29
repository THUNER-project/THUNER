import nbformat
from pathlib import Path
from nbconvert import RSTExporter
import re


def truncate_output(output, max_lines=15):
    """Truncate output to a maximum number of lines, adding '...' if truncated."""
    lines = output.splitlines()
    if len(lines) > max_lines:
        return "\n".join(lines[:max_lines]) + "\n..."
    return output


def convert_notebook_to_tutorial(notebook_path, rst_path):
    """
    Convert a Jupyter notebook to a reStructuredText (.rst) file for Sphinx
    documentation.
    """
    # Load the notebook
    with open(Path(notebook_path), "r", encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=4)

    # Process cell outputs to truncate them
    for cell in notebook.cells:
        if "outputs" in cell:
            for output in cell.outputs:
                if "text" in output:
                    output["text"] = truncate_output(output["text"])
                elif "data" in output:
                    if "text/plain" in output["data"]:
                        plain_text = output["data"]["text/plain"]
                        output["data"]["text/plain"] = truncate_output(plain_text)

    # Create an .rst exporter
    rst_exporter = RSTExporter()

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
    replacement_string = ".. code-block:: python3\n    :linenos:"
    rst = rst.replace(".. code:: ipython3", replacement_string)
    rst = rst.replace(".. parsed-literal::", ".. code-block:: text")
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
        "thuner.utils.generate_times",
        "thuner.parallel.track",
        "thuner.analyze.read_options",
        "thuner.attribute.utils.read_attribute_csv",
        "thuner.config.set_outputs_directory",
        "thuner.config.get_outputs_directory",
        "xarray.open_dataset",
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
