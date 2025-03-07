import re


def contains_name_arrow_name(text):
    """Check if the string contains the pattern <name> --> <name>."""
    pattern = r"([a-zA-Z0-9_.]+) --> \1\n"
    return bool(re.search(pattern, text))


def replace_module_with_subpackage(text):
    """Replace strings of the form thuner.subpackage.module with thuner.subpackage."""
    pattern = r"(\bthuner\.[a-zA-Z0-9_]+)\.[a-zA-Z0-9_]+\b"
    replacement = r"\1"
    return re.sub(pattern, replacement, text)


filepath = "./packages.puml"
with open(filepath, "r") as file:
    lines = file.readlines()

# Exclude private modules
excluded_names = ["thuner.test", "thuner.log", "thuner.config", "thuner.utils"]

new_lines = []
for line in lines:
    new_line = replace_module_with_subpackage(line)
    cond = new_line != "}\n" and not contains_name_arrow_name(new_line)
    cond = (new_line not in new_lines) and cond
    cond = cond & all(name not in new_line for name in excluded_names)
    if cond:
        new_lines.append(new_line)
        if new_line[-2:] == "{\n":
            new_lines.append("}\n")

style_text = """scale 0.5
<style>
package {
    FontSize 26
}
</style>
"""

new_lines.insert(2, style_text)
# Write new_lines to file
filepath = "./edited_packages.puml"
with open(filepath, "w") as file:
    file.writelines(new_lines)
