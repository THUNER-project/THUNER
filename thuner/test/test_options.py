import unittest
import thuner.data as data
import thuner.option as option
import thuner.default as default
import yaml
from pathlib import Path
from pydantic import ValidationError
import numpy as np
import tempfile


def build_example_data_options():
    start = "2005-11-13T14:00:00"
    end = "2005-11-13T19:00:30"
    times_dict = {"start": start, "end": end}
    cpol_options = data.aura.CPOLOptions(**times_dict, converted_options={"save": True})
    era5_dict = {"latitude_range": [-14, -10], "longitude_range": [129, 133]}
    era5_pl_options = data.era5.ERA5Options(**times_dict, **era5_dict)
    era5_dict.update({"data_format": "single-levels"})
    era5_sl_options = data.era5.ERA5Options(**times_dict, **era5_dict)
    datasets = [cpol_options, era5_pl_options, era5_sl_options]
    data_options = option.data.DataOptions(datasets=datasets)
    return data_options


class TestOptionsClasses(unittest.TestCase):
    """
    Test the options classes, ensuring validation works as expected, and options can be
    saved and loaded as YAML files.
    """

    def test_grid_options_save_load(self):
        """Test grid options can be saved and loaded correctly."""
        with tempfile.TemporaryDirectory() as _test_output:
            options_directory = Path(_test_output) / "grid_options_test"
            options_directory.mkdir(parents=True, exist_ok=True)
            # Check good grid options can be created and saved
            grid_options = option.grid.GridOptions()
            grid_options.to_yaml(options_directory / "grid.yml")
            # Check grid options can be loaded from the yml file
            with open(options_directory / "grid.yml", "r") as f:
                loaded_grid_options = option.grid.GridOptions(**yaml.safe_load(f))
            self.assertEqual(grid_options, loaded_grid_options)

    def test_grid_options_validation(self):
        """Test the grid options class validation."""
        with self.assertRaises(ValidationError):
            # Check bad grid type raises ValidationError
            option.grid.GridOptions(name="polar")
            # Check inconsistent spacing raises ValidationError
            kwargs = {"x": np.arange(0, 100, 1), "cartesian_spacing": [100, 100]}
            option.grid.GridOptions(**kwargs)
            kwargs = {"longitude": np.arange(0, 100, 1), "geographic_spacing": [2, 2]}
            option.grid.GridOptions(**kwargs)

    def test_data_options_save_load(self):
        """Test data options can be saved and loaded correctly."""
        with tempfile.TemporaryDirectory() as _test_output:
            options_directory = Path(_test_output) / "data_options_test"
            options_directory.mkdir(parents=True, exist_ok=True)
            # Create a good dataset options instance
            data_options = build_example_data_options()
            # Check data options can be saved
            data_options.to_yaml(options_directory / "data.yml")
            with open(options_directory / "data.yml", "r") as f:
                loaded_data_options = option.data.DataOptions(**yaml.safe_load(f))
            self.assertEqual(data_options, loaded_data_options)

    def test_cpol_data_options_validation(self):
        """Test the data options class validation."""
        # Check bad data options raise ValidationError
        with self.assertRaises(ValidationError):
            start = "2005-11-13T14:00:00"
            end = "2005-11-13T19:00:30"
            times_dict = {"start": start, "end": end}
            # Check bad level raises ValidationError
            data.aura.CPOLOptions(**times_dict, level="7")
            # Check bad time range raises ValidationError
            times_dict = {"start": "1998-12-05T00:00", "end": "2005-11-13T19:00"}
            data.aura.CPOLOptions(**times_dict)

    def test_himawari_data_options_validation(self):
        """Test the Himawari data options class validation."""
        # Check bad data options raise ValidationError
        with self.assertRaises(ValidationError):
            start = "2019-01-01T00:00:00"
            end = "2019-01-02T00:00:00"
            times_dict = {"start": start, "end": end}
            data.himawari.HimawariOptions(**times_dict, time_frame="abc")
            data.himawari.HimawariOptions(**times_dict, band="B21")
            data.himawari.HimawariOptions(**times_dict, resolution="1500")
            data.himawari.HimawariOptions(**times_dict, version="alpha")
            times_dict = {"start": "2011-12-05T00:00", "end": "2011-12-06T00:00"}
            data.himawari.HimawariOptions(**times_dict)

    def test_track_options_save_load(self):
        """Test track options can be saved and loaded correctly."""
        with tempfile.TemporaryDirectory() as _test_output:
            options_directory = Path(_test_output) / "track_options_test"
            options_directory.mkdir(parents=True, exist_ok=True)
            # Create some good track options
            track_options = default.track(dataset_name="cpol")
            track_options.revalidate()
            track_options.to_yaml(options_directory / "track.yml")
            with open(options_directory / "track.yml", "r") as f:
                loaded_track_options = option.track.TrackOptions(**yaml.safe_load(f))
            # Check the loaded options match the original
            self.assertEqual(track_options, loaded_track_options)

    def test_track_options_validation(self):
        """Test the track options class validation."""
        track_options = default.track(dataset_name="cpol")
        with self.assertRaises(ValidationError):
            # Masks need to be saved for the default configuration to work!
            track_options.levels[1].objects[0].mask_options.save = False
            track_options.levels[1].objects[0].revalidate()

    def test_options_save_load(self):
        """Test the overall options class can be saved and loaded correctly."""
        with tempfile.TemporaryDirectory() as _test_output:
            options_directory = Path(_test_output) / "options_test"
            options_directory.mkdir(parents=True, exist_ok=True)
            grid_options = option.grid.GridOptions()
            data_options = build_example_data_options()
            track_options = default.track(dataset_name="cpol")
            kwargs = {"grid": grid_options, "data": data_options}
            kwargs.update({"track": track_options})
            # Check the options class actually contains all the requisite options!
            options = option.option.Options(**kwargs)
            field_names = set(options.__class__.model_fields)
            field_names.remove("type")
            self.assertEqual(set(kwargs.keys()), field_names)
            options.to_yaml(options_directory / "options.yml")
            with open(options_directory / "options.yml", "r") as f:
                loaded_options = option.option.Options(**yaml.safe_load(f))
            # Check the saved and reloaded options match the original
            self.assertEqual(options, loaded_options)

    def test_options_validation(self):
        """Test the overall options class validation."""
        grid_options = option.grid.GridOptions()
        data_options = build_example_data_options()
        track_options = default.track(dataset_name="cpol")
        # Set the main dataset name to a bad value
        data_options.dataset_by_name("cpol").name = "bad_dataset_name"
        kwargs = {"grid": grid_options, "data": data_options, "track": track_options}
        with self.assertRaises(ValidationError):
            option.option.Options(**kwargs)
        data_options.dataset_by_name("bad_dataset_name").name = "cpol"
        # Set the tag dataset to an invalid value
        mcs_options = track_options.levels[1].objects[0]
        mcs_options.attributes.attribute_types[2].dataset = "bad_dataset_name"
        kwargs = {"grid": grid_options, "data": data_options, "track": track_options}
        with self.assertRaises(ValidationError):
            option.option.Options(**kwargs)
        mcs_options.attributes.attribute_types[2].dataset = "era5_pl"
        mcs_options.dataset = "bad_dataset_name"
        kwargs = {"grid": grid_options, "data": data_options, "track": track_options}
        with self.assertRaises(ValidationError):
            option.option.Options(**kwargs)


if __name__ == "__main__":
    unittest.main()
