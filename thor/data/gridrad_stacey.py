# This program will open downloaded Gridrad files, run the filtering and decluttering
# based on gridrad.py the Cameron Homeyer wrote, but updated to use Xarray
# The ipython notebook for testing this code is ReadandCleanGridrad.ipynb

# This code also calculates ETH and two measures of dividing convective/stratiform regions
# Classic classification done via Steiner 1995, modern classification done via PyREClass by Raut et al. https://github.com/vlouf/PyREClass

# You should do labeling and convective characteristics in another piece of code so you don't need to re run
# processing step for sensitivity tests and other small changes

import xarray as xr
import matplotlib as mpl
import numpy as np
import glob
import sys
import os

sys.path.insert(0, "/home/563/sh1269/ProgramLibrary/general")
from dist_GC import _distGC
from PyREClass import pyreclass
import steiner


def read_fileXR(infile):
    data = xr.open_dataset(infile)

    nx = len(data.Longitude)
    ny = len(data.Latitude)
    nz = len(data.Altitude)

    ref = np.zeros(nx * ny * nz)
    wref = np.zeros(nx * ny * nz)

    ref[:] = np.nan
    wref[:] = np.nan

    # Add values to arrays
    ref[data.index.values] = data.Reflectivity.values
    wref[data.index.values] = data.wReflectivity.values

    # replace reflectivity with the 3d array
    data["Reflectivity"] = (
        ["Altitude", "Latitude", "Longitude"],
        ref.reshape(nz, ny, nx),
    )
    data["wReflectivity"] = (
        ["Altitude", "Latitude", "Longitude"],
        wref.reshape(nz, ny, nx),
    )

    return data


def radfilterXR(data0):
    # Extract year from GridRad analysis time string
    year = data0.time.dt.year

    wmin = (
        0.1  # Set absolute minimum weight threshold for an observation (dimensionless)
    )
    wthresh = 1.33 - 1.0 * (
        year < 2009
    )  # Set default bin weight threshold for filtering by year (dimensionless)
    freq_thresh = 0.6  # Set echo frequency threshold (dimensionless)
    Z_H_thresh = 18.5  # Reflectivity threshold (dBZ)
    nobs_thresh = 2  # Number of observations threshold

    # Extract dimension sizes
    nx = len(data0.Longitude)
    ny = len(data0.Latitude)
    nz = len(data0.Altitude)

    echo_frequency = np.zeros(
        (nz, ny, nx)
    )  # Create array to compute frequency of radar obs in grid volume with echo

    ipos = np.where(data0.Nradobs > 0)  # Find bins with obs
    npos = len(ipos[0])  # Count number of bins with obs

    if npos > 0:
        echo_frequency[ipos] = (
            data0.Nradecho.values[ipos] / data0.Nradobs.values[ipos]
        )  # Compute echo frequency (number of scans with echo out of total

    inan = np.where(np.isnan(data0.Reflectivity))  # Find bins with NaNs
    nnan = len(inan[0])  # Count number of bins with NaNs
    data0["Reflectivity"] = data0.Reflectivity.fillna(0)  # fill nans with zero

    # Find observations with low weight
    ifilter = np.where(
        (data0.wReflectivity.values < wmin)
        | (
            (data0.wReflectivity.values < wthresh.values)
            & (data0.Reflectivity.values <= Z_H_thresh)
        )
        | ((echo_frequency < freq_thresh) & (data0.Nradobs > nobs_thresh))
    )

    nfilter = len(ifilter[0])  # Count number of bins that need to be removed

    # Remove low confidence observations
    if nfilter > 0:
        data0["Reflectivity"].values[ifilter] = np.nan

    # Replace NaNs that were previously removed
    if nnan > 0:
        data0["Reflectivity"].values[inan] = np.nan

    # Return filtered data0
    return data0


def rm_speckles(data0):
    # Code to remove speckles
    # Set fractional areal coverage threshold for speckle identification
    areal_coverage_thresh = 0.32

    # Extract dimension sizes
    nx = len(data0.Longitude)
    ny = len(data0.Latitude)
    nz = len(data0.Altitude)

    # First pass at removing speckles
    fin = np.isfinite(data0.Reflectivity)

    # Compute fraction of neighboring points with echo
    cover = np.zeros((nz, ny, nx))
    for i in range(-2, 3):
        for j in range(-2, 3):
            cover += np.roll(np.roll(fin, i, axis=2), j, axis=1)
    cover = cover / 25.0

    # Find bins with low nearby areal echo coverage (i.e., speckles) and remove (set to NaN).
    data0["Reflectivity"] = data0.Reflectivity.where(cover > areal_coverage_thresh)

    return data0


def remove_clutterXR(data0, **kwargs):
    # Set defaults for optional parameters
    if "skip_weak_ll_echo" not in kwargs:
        skip_weak_ll_echo = 0

    # Extract dimension sizes
    nx = len(data0.Longitude)
    ny = len(data0.Latitude)
    nz = len(data0.Altitude)

    # Copy altitude array to 3 dimensions
    zzz = np.meshgrid(data0.Longitude, data0.Latitude, data0.Altitude)[2]
    zzz = np.moveaxis(zzz, 2, 0)

    data0 = rm_speckles(data0)

    # Attempts to mitigate ground clutter and biological scatterers
    if skip_weak_ll_echo == 0:

        # Find weak low-level echo and remove (set to NaN)
        clutter1 = data0.Reflectivity.where(data0.Reflectivity < 10.0).where(zzz <= 4.0)
        data0["Reflectivity"] = data0.Reflectivity.where(np.isnan(clutter1) == True)

        refl_max = data0.Reflectivity.max(dim="Altitude", skipna=True)
        echo0_max = np.nanmax((data0.Reflectivity > 0.0) * zzz, axis=0)
        echo0_min = np.nanmin((data0.Reflectivity > 0.0) * zzz, axis=0)
        echo5_max = np.nanmax((data0.Reflectivity > 5.0) * zzz, axis=0)
        echo15_max = np.nanmax((data0.Reflectivity > 15.0) * zzz, axis=0)

        # Find weak and/or shallow echo
        ibad = np.where(
            ((refl_max < 20.0) & (echo0_max <= 4.0) & (echo0_min <= 3.0))
            | ((refl_max < 10.0) & (echo0_max <= 5.0) & (echo0_min <= 3.0))
            | ((echo5_max <= 5.0) & (echo5_max > 0.0) & (echo15_max <= 3.0))
            | ((echo15_max < 2.0) & (echo15_max > 0.0))
        )
        nbad = len(ibad[0])
        if nbad > 0:
            kbad = (np.zeros((nbad))).astype(int)
            for k in range(0, nz):
                data0["Reflectivity"].values[(k + kbad), ibad[0], ibad[1]] = np.nan

    # Find clutter below convective anvils
    k4km = ((np.where(data0.Altitude >= 4.0))[0])[0]
    fin = np.isfinite(data0.Reflectivity)
    ibad = np.where(
        (fin[k4km, :, :] == False)
        & (np.sum(fin[k4km:, :, :], axis=0) > 0)
        & (np.sum(fin[0:k4km, :, :], axis=0) > 0)
    )  # edited here bc python counting (1:5 = 1,2,3,4)
    nbad = len(ibad[0])
    if nbad > 0:
        kbad = (np.zeros((nbad))).astype(int)
        for k in range(0, k4km + 1):
            data0["Reflectivity"].values[(k + kbad), ibad[0], ibad[1]] = np.nan

    data0 = rm_speckles(data0)

    return data0


def distNS_EW(lon, lat):
    # compute dx/dy at each box, can be used for things like in area above, or average resolution
    from pyproj import Geod

    g = Geod(ellps="WGS84")
    lon2D, lat2D = np.meshgrid(lon, lat)
    _, _, distEW = g.inv(lon2D[:, :-1], lat2D[:, 1:], lon2D[:, 1:], lat2D[:, 1:])
    _, _, distNS = g.inv(lon2D[1:, :], lat2D[1:, :], lon2D[1:, :], lat2D[:-1, :])

    return distEW, distNS


def calc_ETH(ds, refthresh=10):
    # takes xarray dataset with Alt, Lat, Lon dimensions, but this is easily edited
    # refthresh is the reflectivity threshold used to determine ETH, 10 is consistent
    # with Cooney et al 2018 for same datset
    # outputs ETH in km
    zzzgrid = np.array(
        [[list(ds.Altitude.values)] * len(ds.Latitude)] * len(ds.Longitude)
    )
    zzzgrid = np.swapaxes(zzzgrid, 0, 2)

    dbz10 = ds.Reflectivity.where(ds.Reflectivity >= refthresh)
    # create a 3d grid of where each point is the altitude
    zzzgrid = xr.DataArray(
        data=zzzgrid,
        dims=["Altitude", "Latitude", "Longitude"],
    )
    # 3D grid of altitudes where dbz is greater than 10
    zzzgrid = zzzgrid.where(np.isnan(dbz10) == False)

    # Cooney et al 2018 has additional check that near topopause, 20 dbz
    # threshold to reduce identifying anvil as oversohot. for now leave this out bc no ERAI data involved
    # This eliminates many scans in which the vertical profile of ZH is discontinuous or the convective
    # anvil echo top altitude lies above the tropopause over broad regions. We actually want anvil identified

    # they also have a condition: require that the two altitude levels immediately below a potential echo
    # top also contain valid ZH measurements.

    # use a rolling count with a window of 3 to evaluate where it exists for 3 concecutive points (brilliant soln found online)
    # starts count at low levels, so should not artificially lower heights
    real = zzzgrid.rolling(Altitude=3).count()

    # find zzzgrid where at least 3 levels exist
    zzzgrid = zzzgrid.where(np.isnan(real) == False)

    # ETH=ETH.where(ETH>2) #anything shallower than 2 km is probably not real
    # highest altitude with greater than 10 dbz (where above is true)
    ETH = zzzgrid.max(dim="Altitude")
    return ETH


def _mapsetup(AX):
    # AX.set_extent([lonmin, lonmax, latmin, latmax])
    # Create a feature for States/Admin 1 regions at 1:50m from Natural Earth
    states_provinces = cfeature.NaturalEarthFeature(
        category="cultural",
        name="admin_1_states_provinces_lines",
        scale="50m",
        facecolor="none",
    )

    # create feature for melbourne urban area
    # melb_urban=cfeature.NaturalEarthFeature(
    # category='cultural',
    # name='urban_areas',
    # scale='10m',
    # facecolor='none')

    # AX.coastlines(resolution='10m')
    AX.add_feature(states_provinces, edgecolor="gray")
    # AX.add_feature(melb_urban,edgecolor='#FF6347')
    return AX


# maybe do one day's worth of data at a time, but run each day in parallel?
year = sys.argv[1]
mo = sys.argv[2]
day = sys.argv[3]

fdir = (
    "/g/data/w40/sh1269/turbulence/gridrad/downloaded/V3.1/" + year + "/"
)  # enter path to data
savedir = "/g/data/w40/sh1269/turbulence/gridrad/corrected/" + year + "/"

first = True
for fpath in glob.glob(fdir + "nexrad_3d_v3_1_" + year + mo + day + "*Z.nc"):
    print(fpath)

    # plot using SMH method w/t xarray
    data = read_fileXR(fpath)
    timestr = data.time.dt.strftime("%Y%m%d_%H%M%S").values[0]

    # did you already do this one?
    exists = os.path.exists(savedir + timestr + ".nc")
    if exists == True:
        print("file already exists")
        continue

    # run filter and clutter code
    data = radfilterXR(data)
    data = remove_clutterXR(data)

    # now that we've used what we needed, lets only save most essential.
    data = data.drop_vars(
        ["files_merged", "Nradobs", "Nradecho", "index", "wReflectivity", "datehour"]
    )

    # CLASSIFICATION: Identify convective/stratiform regions
    # do both methods at 3 km,
    ref3km = data.Reflectivity[2].where(data.Reflectivity[2] > 0).data.copy()

    # both methods assume regularly spaced x,y grid, so there may be some minor issues esp near edges
    distEW, distNS = distNS_EW(data.Longitude.data, data.Latitude.data)

    # Identify convective/stratiform regions using Steiner 1995 method.
    # Note: code is edited version of Joshua Soderholm code for AURA level 2 Steiner
    # Intensity and peakedness
    # once again going to use averages, but here we can be slightly more specific since allows for dx and dy
    avgdx = distEW.mean()
    avgdy = distNS.mean()
    steiner_class = steiner.steiner_classification(
        ref3km,
        data.Longitude.data,
        data.Latitude.data,
        avgdx,
        avgdy,
        intense=40,
        peak_relation=0,
        area_relation=1,
        bkg_rad=11000,
        use_intense=True,
        latlon=True,
    )
    data["steiner_class"] = xr.DataArray(
        data=steiner_class, dims=["Latitude", "Longitude"]
    )

    # for PyREClass: use average resolution overall for now, and adjust later if needed.
    avgres = 0.5 * (distEW.mean() + distNS.mean())

    # PyREClass method: Raut et al. https://github.com/vlouf/PyREClass,
    # Get the classification. #note resolution in km
    # based on the code, I think conv_scale_km and res_km are used to compute scale break,
    # a unitless interger ratio in pixels. So should be able to just use degrees too
    # must do this order, bc for some reason pyreclass screws with ref3km directly.
    wt_class = pyreclass.getWTClass(ref3km, res_km=avgres / 1000.0, conv_scale_km=20)

    data["wt_class"] = xr.DataArray(data=wt_class, dims=["Latitude", "Longitude"])

    # Now compute ETH
    data["ETH"] = calc_ETH(data)

    # try saving as a netcdf. Use compression to save space
    comp = dict(zlib=True, complevel=5)
    encoding = {var: comp for var in data.data_vars}
    data.to_netcdf(savedir + timestr + ".nc", encoding=encoding)

    if first == True:
        first = False
        # test a plot on the first file
        import matplotlib.pyplot as plt
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        from cartopy import config

        lon_0 = data.Longitude.mean().values
        lat_0 = data.Latitude.mean().values

        plt.figure(figsize=(8, 6))
        ax = plt.axes(
            projection=ccrs.AzimuthalEquidistant(
                central_longitude=lon_0, central_latitude=lat_0
            )
        )
        # ax.set_extent(bbox)
        radplt = ax.pcolormesh(
            data.Longitude,
            data.Latitude,
            np.nanmax(data.Reflectivity, axis=0),
            vmin=0,
            vmax=65,
            cmap="gist_ncar",
            transform=ccrs.PlateCarree(),
        )

        _mapsetup(ax)
        cbar = plt.colorbar(radplt, shrink=0.8)
        cbar.set_label("dBZ")
        plt.tight_layout()
        plt.savefig(
            "/scratch/w40/sh1269/turbulence/testradarimages/"
            + timestr
            + "_corrected.png",
            facecolor="w",
        )
        plt.close()

    data.close()
