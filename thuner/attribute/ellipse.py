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
from thuner.option.attribute import Attribute, AttributeGroup, Retrieval, AttributeType

logger = setup_logger(__name__)
# Set the number of cv2 threads to 0 to avoid crashes.
# See https://github.com/opencv/opencv/issues/5150#issuecomment-675019390
cv2.setNumThreads(0)


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
        latitude = lats.interp(row=row, column=column, method="linear").values
        longitude = lons.interp(row=row, column=column, method="linear").values
        spacing = grid_options.cartesian_spacing
        axis_1 = cartesian_pixel_to_distance(spacing, axis_1, orientation)
        axis_2 = cartesian_pixel_to_distance(spacing, axis_2, orientation)
    elif grid_options.name == "geographic":
        lats = xr.DataArray(lats, dims=("row"))
        lons = xr.DataArray(lons, dims=("column"))
        latitude = lats.interp(row=row, method="linear").values
        longitude = lons.interp(column=column, method="linear").values
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
    attribute_group,
    object_tracks,
    grid_options,
    member_object=None,
    matched=True,
):
    """
    Get ellipse properties from object mask.
    """
    mask = utils.get_current_mask(object_tracks, matched=matched)
    # If examining just a member of a grouped object, get masks for that object
    if member_object is not None and isinstance(mask, xr.Dataset):
        mask = mask[f"{member_object}_mask"]

    if matched:
        ids = object_tracks.match_record["universal_ids"]
    else:
        ids = object_tracks.match_record["ids"]

    all_names = ["latitude", "longitude", "major", "minor", "orientation"]
    all_names += ["eccentricity"]
    all_attributes = {name: [] for name in all_names}

    for id in ids:
        ellipse_properties = cv2_ellipse(mask, id, grid_options)
        for i, name in enumerate(all_names):
            all_attributes[name].append(ellipse_properties[i])

    # Subset to just those attributes requested
    names = [attr.name for attr in attribute_group.attributes]
    ellipse_attributes = {key: all_attributes[key] for key in all_names if key in names}
    return ellipse_attributes


class Latitude(Attribute):
    name: str = "latitude"
    data_type: type = float
    precision: int = 4
    retrieval: Retrieval | None = None
    units: str = "degrees_north"
    description: str = "Latitude of the center of the ellipse fit."


class Longitude(Attribute):
    name: str = "longitude"
    data_type: type = float
    precision: int = 4
    retrieval: Retrieval | None = None
    units: str = "degrees_east"
    description: str = "Longitude of the center of the ellipse fit."


class Major(Attribute):
    name: str = "major"
    data_type: type = float
    precision: int = 1
    units: str = "km"
    retrieval: Retrieval | None = None
    description: str = "Major axis from ellipse fitted to object mask."


class Minor(Attribute):
    name: str = "minor"
    data_type: type = float
    precision: int = 1
    units: str = "km"
    retrieval: Retrieval | None = None
    description: str = "Minor axis from ellipse fitted to object mask."


class Orientation(Attribute):
    name: str = "orientation"
    data_type: type = float
    precision: int = 4
    units: str = "radians"
    retrieval: Retrieval | None = None
    description: str = "Orientation of the ellipse fit to the object mask."


class Eccentricity(Attribute):
    name: str = "eccentricity"
    data_type: type = float
    precision: int = 4
    units: str | None = None
    retrieval: Retrieval | None = None
    description: str = "Eccentricity of the ellipse fit to the object mask."


class EllipseFit(AttributeGroup):
    name: str = "ellipse_fit"
    retrieval: Retrieval = Retrieval(function=from_mask)
    description: str = "Properties of ellipse fit to object mask."
    attributes: list[Attribute] = [
        Latitude(),
        Longitude(),
        Major(),
        Minor(),
        Orientation(),
        Eccentricity(),
    ]


# Convenience function for creating default ellipse attribute type
def default(matched=True):
    """Create the default ellipse attribute type."""

    attributes_list = core.retrieve_core([core.Time()], matched)
    attributes_list += [EllipseFit()]
    description = "Ellipse fit attributes of the object, e.g. major axis length."
    kwargs = {"name": "ellipse", "attributes": attributes_list}
    kwargs.update({"description": description})

    return AttributeType(**kwargs)
