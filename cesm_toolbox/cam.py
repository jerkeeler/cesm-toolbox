import xarray as xr

from cartopy import crs as ccrs
from matplotlib import pyplot as plt

from .consts import KELVIN_OFFSET, DAY_IN_SECONDS, G, EARTH_RADIUS
from .utils import cyclitize, fix_dates
from .paleoclimate import plot_land


def delta_18O(cam_data: xr.Dataset) -> xr.DataArray:
    """
    Compute the d18O of precipitation from CAM output.
    """
    p16O = (
        cam_data.PRECRC_H216Or
        + cam_data.PRECSC_H216Os
        + cam_data.PRECRL_H216OR
        + cam_data.PRECSL_H216OS
    )
    p18O = (
        cam_data.PRECRC_H218Or
        + cam_data.PRECSC_H218Os
        + cam_data.PRECRL_H218OR
        + cam_data.PRECSL_H218OS
    )
    d18Op = (p18O / p16O - 1) * 1000
    d18Op.attrs["units"] = "per mil"
    d18Op.attrs["name"] = "$\\delta^{18}O$ of precipitation$"
    return d18Op


def total_precip(dataset: xr.Dataset) -> xr.DataArray:
    """
    Calculate total precipitation from convection and large scale movement.
    """
    precip = dataset.PRECL + dataset.PRECC
    precip.attrs["units"] = "m/s"
    precip.attrs["name"] = "total precipitation"
    return precip


def total_precip_mm(dataset: xr.Dataset) -> xr.DataArray:
    """
    Calculate total precipitation from convection and large scale movement.
    """
    precip = (dataset.PRECL + dataset.PRECC) * DAY_IN_SECONDS * 1000
    precip.attrs["units"] = "mm/day"
    precip.attrs["name"] = "total precipitation"
    return precip


def precip_weighted_d18o(dataset: xr.Dataset) -> xr.DataArray:
    """
    Calculate d18O weighted by total precipitation for a given dataset where
    d18O has already be calculated.
    """
    d18Op = dataset.d18o * (dataset.PRECT / dataset.PRECT.sum(dim="time"))
    d18Op.attrs["units"] = "per mil"
    d18Op.attrs["name"] = "$\\delta^{18}O_p$ weighted by precipitation"
    return d18Op


def delta_d(dataset: xr.Dataset) -> xr.DataArray:
    """
    Calculate deuterium excess (dD) of precipitation for a given dataset and cam output.
    """
    p16o = (
        dataset.PRECRC_H216Or
        + dataset.PRECSC_H216Os
        + dataset.PRECRL_H216OR
        + dataset.PRECSL_H216OS
    )
    Dp = (
        dataset.PRECRC_HDOr
        + dataset.PRECSC_HDOs
        + dataset.PRECRL_HDOR
        + dataset.PRECSL_HDOS
    )
    dDp = (Dp / p16o - 1.0) * 1000.0
    dDp.attrs["units"] = "per mil"
    dDp.attrs["name"] = "$\\delta D$ of precipitation"
    return dDp


def elevation(dataset: xr.Dataset) -> xr.DataArray:
    """
    Calculate elevation (in meters) from surface geopotential height.
    """
    phis = dataset.PHIS.max(dim="time")
    elevation = (phis * EARTH_RADIUS) / (G * EARTH_RADIUS - phis)
    elevation.attrs["units"] = "m"
    elevation.attrs["name"] = "elevation"
    return elevation


def net_precip(dataset: xr.Dataset) -> xr.DataArray:
    """
    Calculates the net precipitation by determining the evaportion from the
    QFLX data attribute.
    """
    precip = dataset.PRECTmm - dataset.QFLX * DAY_IN_SECONDS
    precip.attrs["units"] = "mm/day"
    precip.attrs["name"] = "net precipitation (precipitation - evaporation)"
    return precip


def read_cam_data(
    path: str,
    with_fixed_dates: bool = True,
    with_extra_data: bool = True,
) -> xr.Dataset:
    """
    Reads in a cam dataset and performs some common operations. Namely,
    precribing dates, calculating total precipitation, calculating d18O
    """
    data = xr.open_dataset(path)
    if with_fixed_dates:
        data = fix_dates(data)
    if with_extra_data:
        data = data.assign(
            d18o=delta_18O(data),
            PRECT=total_precip(data),
            PRECTmm=total_precip_mm(data),
            TSC=(data.TS - KELVIN_OFFSET).assign_attrs(units="C", name="temperature"),
            ELE=elevation(data),
        )
        data = data.assign(
            d18o_weighted=precip_weighted_d18o(data), NET_PRECT=net_precip(data)
        )
    return data


def plot_cam_mask(cam_data, land, projection=ccrs.PlateCarree):
    mean_TS = cam_data.TS.mean(dim="time")

    def wrapped(mask, contour=False):
        data = cyclitize(mean_TS.where(mask))
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection=projection())
        if contour:
            ax.contourf(data.lon, data.lat, data, transform=ccrs.PlateCarree())
        else:
            ax.pcolormesh(data.lon, data.lat, data, transform=ccrs.PlateCarree())
        plot_land(ax, land)
        ax.gridlines(draw_labels=True, alpha=0.5)
        plt.show()

    return wrapped
