"""Convenience functions for processing grids."""

import numpy as np
from pyproj import Geod, Proj, Transformer
from thuner.utils import almost_equal, pad
from thuner.log import setup_logger


logger = setup_logger(__name__)


def subset_curvilinear(dataset, lat_min, lat_max, lon_min, lon_max):
    """
    Subset a curvilinear dataset based on latitude and longitude bounds.
    """
    # Ensure the dataset has latitude and longitude coordinates
    if "latitude" not in dataset.coords or "longitude" not in dataset.coords:
        raise ValueError("Dataset must contain 'latitude' and 'longitude' coordinates.")
    if "y" not in dataset.coords or "x" not in dataset.coords:
        raise ValueError("Dataset must have 'y' and 'x' dimensions.")

    latitude = dataset["latitude"].values
    longitude = dataset["longitude"].values

    lat_cond = (latitude >= lat_min) & (latitude <= lat_max)
    lon_cond = (longitude >= lon_min) & (longitude <= lon_max)
    mask = lat_cond & lon_cond

    rows, cols = np.where(mask)
    if len(rows) == 0 or len(cols) == 0:
        raise ValueError("No points found in the specified lat/lon range!")
    row_min, row_max = rows.min(), rows.max()
    col_min, col_max = cols.min(), cols.max()

    return dataset.isel(y=slice(row_min, row_max + 1), x=slice(col_min, col_max + 1))


def cartesian_to_geographic_lcc(grid_options, x, y):
    """Get latitude, longitude coordinates from cartesian coordinates."""

    if grid_options.name != "cartesian":
        raise ValueError("Grid name must be 'cartesian'.")
    if grid_options.latitude is not None and grid_options.longitude is not None:
        logger.warning("Latitude and longitude already specified.")
    if grid_options.central_latitude is None or grid_options.central_longitude is None:
        message = "Central latitude and longitude must be specified."
        raise ValueError(message)

    # Define the LCC projection
    lcc_proj = Proj(
        proj="lcc",
        lat_1=grid_options.central_latitude,
        lat_2=grid_options.central_latitude,
        lat_0=grid_options.central_latitude,
        lon_0=grid_options.central_longitude,
    )

    # Create a transformer to convert from LCC to geographic coordinates
    transformer = Transformer.from_proj(lcc_proj, Proj(proj="latlong", datum="WGS84"))

    # Transform the Cartesian coordinates to geographic coordinates
    longitude, latitude = transformer.transform(x, y)
    return longitude, latitude


def geographic_to_cartesian_lcc(grid_options, latitude, longitude):
    """Get x, y coordinates from geographic coordinates."""

    if grid_options.name != "geographic":
        raise ValueError("Grid name must be 'geographic'.")
    if grid_options.x is not None and grid_options.y is not None:
        logger.warning("x and y already specified.")
    if grid_options.central_latitude is None or grid_options.central_longitude is None:
        grid_options.central_latitude = np.mean(grid_options.latitude)
        grid_options.central_longitude = np.mean(grid_options.longitude)

    # Define the LCC projection
    lcc_proj = Proj(
        proj="lcc",
        lat_1=grid_options.central_latitude,
        lat_2=grid_options.central_latitude,
        lat_0=grid_options.central_latitude,
        lon_0=grid_options.central_longitude,
    )

    # Create a transformer to convert from geographic to LCC coordinates
    transformer = Transformer.from_proj(Proj(proj="latlong", datum="WGS84"), lcc_proj)

    # Transform the geographic coordinates to Cartesian coordinates
    x, y = transformer.transform(longitude, latitude)
    return x, y


def check_spacing(array, dx):
    """Check if array equally spaced."""
    if not almost_equal(np.diff(array)):
        raise ValueError("Grid not equally spaced.")
    elif not almost_equal(list(np.diff(array)) + [dx]):
        raise ValueError("Grid spacing does not match prescribed gridlengths.")


def new_geographic_grid(lats, lons, dlat, dlon):
    """
    Get the geographic grid.
    """

    min_lat = int(np.floor(np.nanmin(lats) / dlat)) * dlat
    max_lat = int(np.ceil(np.nanmax(lats) / dlat)) * dlat
    min_lon = int(np.floor(np.nanmin(lons) / dlon)) * dlon
    max_lon = int(np.ceil(np.nanmax(lons) / dlon)) * dlon
    new_lats = np.arange(min_lat, max_lat + dlat, dlat).round(8)
    new_lons = np.arange(min_lon, max_lon + dlon, dlon).round(8)

    return list(new_lats), list(new_lons)


def get_cell_areas(grid_options):
    """
    Get the cell areas.

    Parameters
    ----------
    grid_options : dict
        Dictionary containing the grid grid_options.

    Returns
    -------
    numpy.ndarray
        The cell areas in km^2.
    """

    if grid_options.name == "cartesian":
        area = np.prod(grid_options.cartesian_spacing) / 1e6  # Convert to km^2
        cell_areas = np.ones((len(grid_options.y), len(grid_options.x))) * area
        cell_areas = cell_areas.astype(np.float32)
        return cell_areas
    elif grid_options.name == "geographic":
        return get_geographic_cell_areas(grid_options.latitude, grid_options.longitude)


def get_coordinate_names(grid_options):
    """Get the names of the horizontal coordinates."""
    if grid_options.name == "cartesian":
        return ["y", "x"]
    elif grid_options.name == "geographic":
        return ["latitude", "longitude"]


def get_coordinates(grid_options, row, col):
    """Get coordinates."""
    row, col = int(row), int(col)
    if grid_options.name == "cartesian":
        latitude = grid_options.longitude[row, col]
        longitude = grid_options.latitude[row, col]
    elif grid_options.name == "geographic":
        latitude = grid_options.latitude[row]
        longitude = grid_options.longitude[col]
    return latitude, longitude


def get_distance(row_1, col_1, row_2, col_2, grid_options):
    """Get the distance in meters between two grid cells."""
    row_coords, col_coords = get_horizontal_coordinates(grid_options)
    row_coord_1, col_coord_1 = [row_coords[row_1], col_coords[col_1]]
    row_coord_2, col_coord_2 = [row_coords[row_2], col_coords[col_2]]

    if grid_options.name == "cartesian":
        return np.sqrt(
            (row_coord_2 - row_coord_1) ** 2 + (col_coord_2 - col_coord_1) ** 2
        )
    elif grid_options.name == "geographic":
        return geodesic_distance(col_coord_1, row_coord_1, col_coord_2, row_coord_2)


def get_geographic_cell_areas(lats, lons):
    """Get cell areas in km^2."""

    lats, lons = np.array(lats), np.array(lons)
    d_lon = lons[1:] - lons[:-1]
    d_lat = lats[1:] - lats[:-1]

    if almost_equal(d_lon, 5) and almost_equal(d_lat, 5):

        dx = geodesic_distance(lons[2], lats, lons[0], lats) / 2
        dy = geodesic_distance(lons[0], lats[2:], lons[0], lats[:-2]) / 2
        dy = pad(dy)

        areas = dx * dy
        areas = np.tile(areas, (len(lons), 1)).T
    else:
        logger.warning("Irregular lat/lon grid. May be slow to calculate areas.")
        LONS, LATS = np.meshgrid(lons, lats)
        dx = geodesic_distance(
            LONS[1:-1, 2:], LATS[1:-1, 1:-1], LONS[1:-1, :-2], LATS[1:-1, 1:-1]
        )
        dx = dx / 2
        dy = geodesic_distance(
            LONS[1:-1, 1:-1], LATS[2:, 1:-1], LONS[1:-1, 1:-1], LATS[:-2, 1:-1]
        )
        dy = dy / 2
        areas = dx * dy
        areas = np.apply_along_axis(pad, axis=0, arr=areas)
        areas = np.apply_along_axis(pad, axis=1, arr=areas)
    areas = (areas / 1e6).astype(np.float32)  # Convert to km^2
    return areas


def get_horizontal_coordinates(grid_options):
    """
    Get the coordinates for the grid.

    Parameters
    ----------
    grid_options : dict
        Dictionary containing the grid grid_options.

    Returns
    -------
    tuple
        The coordinates as a tuple of (lats, lons, alts).
    """

    if grid_options.name == "cartesian":
        [col_coords, row_coords] = [getattr(grid_options, var) for var in ["x", "y"]]
    elif grid_options.name == "geographic":
        [col_coords, row_coords] = [
            getattr(grid_options, var) for var in ["longitude", "latitude"]
        ]

    return np.array(row_coords), np.array(col_coords)


def get_horizontal_spacing(grid_options):
    """
    Get the grid spacing.
    """

    if grid_options.name == "cartesian":
        return grid_options.cartesian_spacing
    elif grid_options.name == "geographic":
        return grid_options.geographic_spacing


def pixel_to_cartesian_vector(row, col, vector, grid_options):
    """
    Convert a vector from gridcell, i.e. "pixel", coordinates to cartesian coordinates.
    Note that while mathematically a vector does not have a "start" position, if the
    grid is in geographic coordinates, the "location" of the vector in pixel coordinates
    determines what distances it corresponds to in cartesian coordinates.

    Parameters
    ----------
    row : int
        The start row of the vector.
    col : int
        The start column of the vector
    vector : tuple
        The vector to be converted, represented as a tuple of (delta_row, delta_col)
        in pixel coordinates.
    grid_options : dict
        The grid grid_options dictionary.

    Returns
    -------
    tuple
        The converted vector in cartesian coordinates, represented as a tuple
        (delta_y, delta_x), both in units of metres.
    """

    if grid_options.name == "cartesian":
        return vector * grid_options.cartesian_spacing
    elif grid_options.name == "geographic":
        grid_options.geographic_spacing
        lats = grid_options.latitude
        lons = grid_options.longitude
        start_lat = lats[row]
        start_lon = lons[col]
        end_lat = start_lat + vector[0] * grid_options.geographic_spacing[0]
        end_lon = start_lon + vector[1] * grid_options.geographic_spacing[1]
        return geographic_to_cartesian_displacement(
            start_lat, start_lon, end_lat, end_lon
        )


geod = Geod(ellps="WGS84")
geodesic_inverse = np.vectorize(
    lambda lon1, lat1, lon2, lat2: geod.inv(lon1, lat1, lon2, lat2)
)
geodesic_forward = np.vectorize(
    lambda lon, lat, direction, distance: geod.fwd(lon, lat, direction, distance)
)
geodesic_distance = lambda lon1, lat1, lon2, lat2: geodesic_inverse(
    lon1, lat1, lon2, lat2
)[2]


def geographic_to_cartesian_displacement(start_lat, start_lon, end_lat, end_lon):
    """
    Calculate the y and x displacements in metres in cartesian coordinates between two
    points on the Earth's surface given in geographic coordinates.
    """
    direction, backward_direction, distance = geodesic_inverse(
        start_lon, start_lat, end_lon, end_lat
    )
    y_displacement = distance * np.cos(np.radians(direction))
    x_displacement = distance * np.sin(np.radians(direction))
    return y_displacement, x_displacement


def get_pixels_geographic(rows, cols, grid_options):
    """Get the geographic coordinates of the gridcells, i.e. "pixels" at rows, cols."""
    if np.array(rows).shape != np.array(cols).shape:
        raise ValueError("row and col must have the same shape.")
    scalar_input = np.isscalar(rows) and np.isscalar(cols)
    rows = np.array([rows]).flatten()
    cols = np.array([cols]).flatten()
    latitudes = np.array(grid_options.latitude)
    longitudes = np.array(grid_options.longitude)
    if grid_options.name == "cartesian":
        # lats, lons are 2D arrays
        lats = [latitudes[row, col] for row, col in zip(rows, cols)]
        lons = [longitudes[row, col] for row, col in zip(rows, cols)]
    elif grid_options.name == "geographic":
        # lats, lons are 1D arrays
        lats, lons = [latitudes[row] for row in rows], [longitudes[col] for col in cols]
    if scalar_input:
        lats = lats[0]
        lons = lons[0]
    return lats, lons
