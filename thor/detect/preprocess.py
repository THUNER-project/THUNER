"Preprocess data before detection."


def vertical_max(grid, start_alt, end_alt):
    """Return the maximum over the specified altitude range."""
    return grid.sel(altitude=slice(start_alt, end_alt)).max(dim="altitude")


def cross_section(grid, alt):
    """Return the cross section at the specified altitude."""
    return grid.sel(altitude=alt)
