"Preprocess data before detection."


def vertical_max(grid, object_options):
    """Return the maximum over the specified altitude range."""

    altitudes = object_options["detection"]["altitudes"]
    if len(altitudes) == 2:
        [start_alt, end_alt] = object_options["detection"]["altitudes"]
        return grid.sel(altitude=slice(start_alt, end_alt)).max(
            dim="altitude", keep_attrs=True
        )
    else:
        raise ValueError("altitudes must have 2 elements.")


def cross_section(grid, object_options):
    """Return the cross section at the specified altitude."""
    altitude = object_options["detection"]["altitudes"]
    if len(altitude) == 1:
        altitude = object_options["detection"]["altitudes"][0]
        return grid.sel(altitude=altitude)
    else:
        raise ValueError("altitudes must have 1 element.")
