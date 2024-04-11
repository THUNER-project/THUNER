import yaml
from pathlib import Path
from thor.option import default_track


def test_gen_options():
    """Test that default options are generated correctly."""

    # Map object names to functions
    option_functions = {
        "cell": default_track.cell,
        "anvil": default_track.anvil,
        "mcs": default_track.mcs,
    }

    for obj in ["cell", "anvil", "mcs"]:
        regenerated_options = option_functions[obj](save=False)
        filepath = Path(__file__).parent.parent / f"option/default/{obj}.yaml"
        with open(filepath, "r") as infile:
            saved_options = yaml.safe_load(infile)
        assert regenerated_options == saved_options
