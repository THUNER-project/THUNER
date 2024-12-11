"""
functions ellipse attributes.
"""

import numpy as np
import xarray as xr
import cv2
from skimage.morphology.convex_hull import convex_hull_image
from thuner.log import setup_logger
from thuner.attribute import core
import thuner.grid as grid
import thuner.attribute.utils as utils

logger = setup_logger(__name__)
# Set the number of cv2 threads to 0 to avoid crashes.
# See https://github.com/opencv/opencv/issues/5150#issuecomment-675019390
cv2.setNumThreads(0)


def coordinate(name, method=None, description=None):
    """
    Options for coordinate attributes.
    """
    data_type = float
    precision = 4
    if name == "latitude":
        units = "degrees_north"
    elif name == "longitude":
        units = "degrees_east"
    else:
        raise ValueError(f"Coordinate must be 'latitude' or 'longitude'.")
    if method is None:
        method = {"function": "from_mask"}
    if description is None:
        description = f"{name} coordinate of the center of the ellipse fit. "
    args = [name, method, data_type, precision, description, units]
    return utils.get_attribute_dict(*args)


def axis(name="major", method=None, description=None):
    """
    Options for major or minor axis length attributes.
    """
    data_type = float
    precision = 1
    units = "km"
    if method is None:
        method = {"function": "from_mask"}
    if description is None:
        description = f"{name} axis from ellipse fitted to object mask."
    args = [name, method, data_type, precision, description, units]
    return utils.get_attribute_dict(*args)


def orientation(method=None, description=None):
    """
    Options for orientation attribute.
    """
    name = "orientation"
    data_type = float
    precision = 4
    units = "radians"
    if method is None:
        method = {"function": "from_mask"}
    if description is None:
        description = f"The orientation of the ellipse fit to the object mask."
    args = [name, method, data_type, precision, description, units]
    return utils.get_attribute_dict(*args)


def eccentricity(method=None, description=None):
    """
    Options for orientation attribute.
    """
    name = "eccentricity"
    data_type = float
    precision = 4
    units = None
    if method is None:
        method = {"function": "from_mask"}
    if description is None:
        description = f"The eccentricity of the ellipse fit to the object mask."
    args = [name, method, data_type, precision, description, units]
    return utils.get_attribute_dict(*args)


# Convenience functions for creating default core attribute options dictionaries
def default(names=None, matched=True):
    """Create a dictionary of default quality control attribute options."""

    if names is None:
        names = ["time", "latitude", "longitude", "major", "minor", "orientation"]
        names += ["eccentricity"]
    if matched:
        id_type = "universal_id"
    else:
        id_type = "id"
    core_method = {"function": "attribute_from_core"}
    attributes = {}
    # Reuse core attributes, just replace the default functions method
    attributes["time"] = core.time(method=core_method)
    attributes[id_type] = core.identity(id_type, method=core_method)
    if "latitude" in names:
        attributes["latitude"] = coordinate("latitude")
    if "longitude" in names:
        attributes["longitude"] = coordinate("longitude")
    if "major" in names:
        attributes["major"] = axis("major")
    if "minor" in names:
        attributes["minor"] = axis("minor")
    if "orientation" in names:
        attributes["orientation"] = orientation()
    if "eccentricity" in names:
        attributes["eccentricity"] = eccentricity()

    return attributes


def cartesian_pixel_to_distance(spacing, axis, orientation):
    x_distance = axis * np.cos(orientation) * spacing[1]
    y_distance = axis * np.sin(orientation) * spacing[0]
    return np.sqrt(x_distance**2 + y_distance**2) / 1e3


def geographic_pixel_to_distance(latitude, longitude, spacing, axis, orientation):
    lon_distance = axis * np.cos(orientation) * spacing[1]
    lat_distance = axis * np.sin(orientation) * spacing[0]
    new_latitude = latitude + lat_distance
    new_longitude = longitude + lon_distance
    distance = grid.geodesic_distance(longitude, latitude, new_longitude, new_latitude)
    return distance / 1e3


def cv2_ellipse(mask, id, grid_options):
    lats, lons = grid_options.latitude, grid_options.longitude
    hull = convex_hull_image(mask == id).astype(np.uint8)
    contours = cv2.findContours(hull, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

    # Check if small object, and pad if necessary
    if len(contours[0]) > 6:
        ellipse_properties = cv2.fitEllipseDirect(contours[0])
    else:
        print("Object too small to fit ellipse. Retrying with padded contour.")
        new_contour = []
        for r in contours[0]:
            [new_contour.append(r) for i in range(3)]
        new_contour = np.array(new_contour)
        ellipse_properties = cv2.fitEllipseDirect(new_contour)
    [(column, row), (axis_1, axis_2), orientation] = ellipse_properties
    orientation = np.deg2rad(orientation)

    if grid_options.name == "cartesian":
        lats = xr.DataArray(lats, dims=("row", "column"))
        lons = xr.DataArray(lons, dims=("row", "column"))
        latitude = lats.interp(row=row, column=column, method="linear")
        longitude = lons.interp(row=row, column=column, method="linear")
        spacing = grid_options.cartesian_spacing
        axis_1 = cartesian_pixel_to_distance(spacing, axis_1, orientation)
        axis_2 = cartesian_pixel_to_distance(spacing, axis_2, orientation)
    elif grid_options.name == "geographic":
        lats = xr.DataArray(lats, dims=("row"))
        lons = xr.DataArray(lons, dims=("column"))
        latitude = lats.interp(row=row, method="linear")
        longitude = lons.interp(column=column, method="linear")
        spacing = grid_options.geographic_spacing
        args = [latitude, longitude, spacing, axis_1, orientation]
        axis_1 = geographic_pixel_to_distance(*args)
        args[3] = axis_2
        axis_2 = geographic_pixel_to_distance(*args)
    else:
        raise ValueError("Grid must be 'cartesian' or 'geographic'.")

    if axis_1 >= axis_2:
        major = axis_1
        minor = axis_2
    else:
        major = axis_2
        minor = axis_1
        orientation = orientation - np.pi / 2
    orientation = orientation % np.pi
    eccentricity = np.sqrt(1 - (minor / major) ** 2)
    return latitude, longitude, major, minor, orientation, eccentricity


def from_mask(
    attributes,
    attribute_options,
    object_tracks,
    grid_options,
    member_object=None,
):
    """
    Get ellipse properties from object mask.
    """
    mask = utils.get_previous_mask(attribute_options, object_tracks)
    # If examining just a member of a grouped object, get masks for that object
    if member_object is not None and isinstance(mask, xr.Dataset):
        mask = mask[f"{member_object}_mask"]

    if "universal_id" in attribute_options:
        id_type = "universal_id"
    elif "id" in attribute_options:
        id_type = "id"

    ids = attributes[id_type]

    all_names = ["latitude", "longitude", "major", "minor", "orientation"]
    all_names += ["eccentricity"]
    all_attributes = {name: [] for name in all_names}

    for id in ids:
        ellipse_properties = cv2_ellipse(mask, id, grid_options)
        for i, name in enumerate(all_names):
            all_attributes[name].append(ellipse_properties[i])

    ellipse_attributes = {key: all_attributes[key] for key in all_attributes.keys()}
    return ellipse_attributes


# Dispatch dictionary for getting core attributes
get_attributes_dispatcher = {
    "from_mask": from_mask,
    "attribute_from_core": utils.attribute_from_core,
}


def record_ellipse(
    attributes,
    attribute_options,
    object_tracks,
    grid_options,
    method,
    member_object,
):
    """Record ellipse properties."""
    method = utils.tuple_to_dict(method)
    get_ellipse = get_attributes_dispatcher.get(method["function"])
    if get_ellipse is None:
        message = f"Function {method['function']} for obtaining ellipse properties "
        message += "not recognised."
        raise ValueError(message)
    from_mask_args = [attributes, attribute_options, object_tracks, grid_options]
    from_mask_args += [member_object]
    args_dispatcher = {"from_mask": from_mask_args}
    args = args_dispatcher[method["function"]]
    ellipse = get_ellipse(*args)
    attributes.update(ellipse)


def record(
    attributes,
    object_tracks,
    attribute_options,
    grid_options,
    member_object=None,
):
    """Get ellipse object attributes."""
    # Get core attributes
    previous_time = object_tracks["previous_times"][-1]
    if previous_time is None:
        return
    core_attributes = ["time", "id", "universal_id"]
    keys = attributes.keys()
    core_attributes = [attr for attr in core_attributes if attr in keys]
    remaining_attributes = [attr for attr in keys if attr not in core_attributes]
    # Get the appropriate core attributes
    for name in core_attributes:
        attr_function = attribute_options[name]["method"]["function"]
        get_attr = get_attributes_dispatcher.get(attr_function)
        if get_attr is not None:
            attr = get_attr(name, object_tracks, member_object)
            attributes[name] += list(attr)
        else:
            message = f"Function {attr_function} for obtaining attribute {name} not recognised."
            raise ValueError(message)

    if attributes["time"] is None or len(attributes["time"]) == 0:
        return

    # Get profiles efficiently by processing attributes with same method together
    ellipse_attributes = {key: attribute_options[key] for key in remaining_attributes}
    grouped_by_method = utils.group_by_method(ellipse_attributes)
    for method in grouped_by_method.keys():
        args = [attributes, attribute_options, object_tracks, grid_options]
        args += [method, member_object]
        record_ellipse(*args)
