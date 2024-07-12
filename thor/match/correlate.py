"""Methods for calcualting cross correlation, a.k.a. optical flow."""

import numpy as np
from scipy import ndimage
import thor.object.box as box
from thor.utils import get_cartesian_displacement, geodesic_forward
from thor.match.utils import get_grids


def get_flow(bounding_box, object_tracks, object_options, grid_options, flow_margin):
    """Get the optical flow within bounding_box."""

    current_grid, previous_grid = get_grids(object_tracks, object_options)
    current_grid = current_grid.copy()
    previous_grid = previous_grid.copy()
    grid_spacing = grid_options["geographic_spacing"]
    latitudes = current_grid.latitude.values
    longitudes = current_grid.longitude.values
    flow_margin_row, flow_margin_col = box.get_gridcell_margins(
        bounding_box, latitudes, longitudes, flow_margin, grid_spacing
    )
    flow_box = bounding_box.copy()
    flow_box = box.expand_box(flow_box, flow_margin_row, flow_margin_col)
    flow_box = box.clip_box(flow_box, current_grid.shape)

    box_previous = previous_grid[
        flow_box["row_min"] : flow_box["row_max"] + 1,
        flow_box["col_min"] : flow_box["col_max"] + 1,
    ]

    box_current = current_grid[
        flow_box["row_min"] : flow_box["row_max"] + 1,
        flow_box["col_min"] : flow_box["col_max"] + 1,
    ]
    box_previous = box_previous.fillna(0)
    box_current = box_current.fillna(0)

    return calculate_flow(box_previous, box_current), flow_box


def calculate_flow(grid1, grid2, global_flow=False):
    """Calculate optical flow vector using cross covariance."""
    cross_covariance = get_cross_covariance(grid1, grid2)
    sigma = (1 / 8) * min(cross_covariance.shape)
    smoothed_covariance = ndimage.filters.gaussian_filter(cross_covariance, sigma)
    dims = np.array(grid1.shape)
    flow = np.argwhere(smoothed_covariance == np.max(smoothed_covariance))[0]

    row_centre = np.ceil(grid1.shape[0] / 2).astype("int")
    column_centre = np.ceil(grid1.shape[1] / 2).astype("int")

    # Calculate flow relative to center - see fft_flow.
    flow = flow - (dims - np.array([row_centre, column_centre]))
    return flow


def convert_flow_cartesian(flow, previous_lat, previous_lon, geographic_spacing):
    current_lat = previous_lat + flow[0] * geographic_spacing[0]
    current_lon = previous_lon + flow[1] * geographic_spacing[1]

    flow_meters = get_cartesian_displacement(
        previous_lat, previous_lon, current_lat, current_lon
    )
    return flow_meters


def get_cross_covariance(grid1, grid2):
    """Compute cross covariance matrix."""
    fourier_previous_conj = np.conj(np.fft.fft2(grid1))
    fourier_current = np.fft.fft2(grid2)
    normalize = abs(fourier_current * fourier_previous_conj)
    normalize[normalize == 0] = 1  # prevent divide by zero error
    cross_power_spectrum = (fourier_current * fourier_previous_conj) / normalize
    cross_covariance = np.fft.ifft2(cross_power_spectrum)
    cross_covariance = np.real(cross_covariance)
    return shift(cross_covariance)


def shift(cross_covariance):
    """Rearranges the cross covariance matrix so that the zero frequency
    is in the middle of the matrix."""
    if type(cross_covariance) is np.ndarray:
        row_centre = np.ceil(cross_covariance.shape[0] / 2).astype("int")
        column_centre = np.ceil(cross_covariance.shape[1] / 2).astype("int")
        box_1 = cross_covariance[:row_centre, :column_centre]
        box_2 = cross_covariance[:row_centre, column_centre:]
        box_3 = cross_covariance[row_centre:, column_centre:]
        box_4 = cross_covariance[row_centre:, :column_centre]
        centered_top = np.concatenate((box_4, box_1), axis=0)
        centered_bottom = np.concatenate((box_3, box_2), axis=0)
        centered = np.concatenate((centered_bottom, centered_top), axis=1)
        return centered
    else:
        print("input to shift() should be a matrix")
        return
