"Preprocess data before detection."

__all__ = ["vertical_max", "cross_section"]


def vertical_max(grid, object_options):
    """Return the maximum over the specified altitude range."""

    # DetectionOptions validation guarantees a (start, end) range here.
    start_alt, end_alt = object_options.detection.altitudes
    flat_grid = grid.sel(altitude=slice(start_alt, end_alt)).max(
        dim="altitude", keep_attrs=True
    )
    flat_grid.attrs["flatten_method"] = "vertical_max"
    flat_grid.attrs["start_altitude"] = start_alt
    flat_grid.attrs["end_altitude"] = end_alt
    flat_grid.attrs["altitude_units"] = "m"
    return flat_grid


def cross_section(grid, object_options):
    """Return the cross section at the specified altitude."""
    # DetectionOptions validation guarantees a single altitude here.
    altitude = object_options.detection.altitudes
    flat_grid = grid.sel(altitude=altitude)
    flat_grid = flat_grid.reset_coords("altitude", drop=True)
    flat_grid.attrs["flatten_method"] = "cross_section"
    flat_grid.attrs["altitude"] = altitude
    flat_grid.attrs["altitude_units"] = "m"
    return flat_grid
