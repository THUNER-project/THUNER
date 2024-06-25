"Preprocess data before detection."


def vertical_max(grid, object_options):
    """Return the maximum over the specified altitude range."""

    [start_alt, end_alt] = object_options["detection"]["altitudes"]
    return grid.sel(altitude=slice(start_alt, end_alt)).max(
        dim="altitude", keep_attrs=True
    )


def cross_section(grid, object_options):
    """Return the cross section at the specified altitude."""
    return grid.sel(altitude=object_options["altitude"])
