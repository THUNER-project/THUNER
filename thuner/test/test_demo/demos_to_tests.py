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
        # Find the remove_existing_outputs line and set it to True
        if re.match(r"remove_existing_outputs\s*=\s*False", line):
            line = "remove_existing_outputs = True"

        cleaned_lines.append(line)

    script = "\n".join(cleaned_lines)
    # Remove leading and trailing whitespace
    script = script.strip()
    # Remove duplicate empty lines
    script = re.sub(r"\n{3,}", "\n\n", script)
    lines = script.split("\n")
    lines = [l + "\n" for l in lines if l.strip() != ""]

    import_lines = []
    code_lines = []
    for line in lines:
        if line.strip().startswith("import ") or line.strip().startswith("from "):
            import_lines.append(line)
        else:
            code_lines.append(line)

    with open(script_path, "w", encoding="utf-8") as f:
        for line in import_lines:
            f.write(line)
        f.write(f"\n\ndef test_{notebook_path.stem}():\n")
        for line in code_lines:
            if line.strip():
                f.write("    " + line)
            else:
                f.write(line)
        f.write("\n\nif __name__ == '__main__':\n")
        f.write(f"    test_{notebook_path.stem}()\n")


if __name__ == "__main__":
    demo_dir = Path(__file__).parent.parent.parent.parent / "demo"
    test_demo_dir = Path(__file__).parent

    # Iterate over items in the demo directory
    for item in demo_dir.iterdir():
        if item.is_file() and item.suffix == ".ipynb":
            # Convert the notebook to a test script
            convert_notebook_to_script(item, test_demo_dir / f"test_{item.stem}.py")
