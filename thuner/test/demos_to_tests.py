import nbformat
from pathlib import Path
from nbconvert import PythonExporter
import re


def convert_notebook_to_script(notebook_path, script_path):
    # Load the notebook
    with open(Path(notebook_path), "r", encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=4)

    # Create a Python exporter
    python_exporter = PythonExporter()

    # Convert the notebook to a Python script
    script = python_exporter.from_notebook_node(notebook)[0]

    # Remove the first line of the script
    lines = script.split("\n")
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
    script = "\n".join(cleaned_lines)
    # Remove leading and trailing whitespace
    script = script.strip()
    # Remove duplicate empty lines
    script = re.sub(r"\n{3,}", "\n\n", script)

    # Save the script to a file
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(script)


demo_dir = Path(__file__).parent.parent.parent / "demo"
test_dir = Path(__file__).parent

# Iterate over items in the demo directory
for item in demo_dir.iterdir():
    if item.is_file() and item.suffix == ".ipynb":
        # Convert the notebook to a test script
        convert_notebook_to_script(item, test_dir / f"test_{item.stem}.py")
