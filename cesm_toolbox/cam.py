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
    return d18Op


def total_precip(dataset: xr.Dataset) -> xr.Dataset:
    """
    Calculate total precipitation from convection and large scale movement.
    """
    return dataset.PRECL + dataset.PRECC


def precip_weighted_d18o(dataset: xr.Dataset) -> xr.Dataset:
    """
    Calculate d18O weighted by total precipitation for a given dataset where
    d18O has already be calculated.
    """
    return dataset.d18o * (dataset.PRECT / dataset.PRECT.sum(dim="time"))


def elevation(dataset: xr.Dataset) -> xr.Dataset:
    """
    Calculate elevation (in meters) from surface geopotential height.
    """
    phis = dataset.PHIS.max(dim="time")
    elevation = (phis * EARTH_RADIUS) / (G * EARTH_RADIUS - phis)
    return elevation


def net_precip(dataset: xr.Dataset) -> xr.Dataset:
    """
    Calculates the net precipitation by determining the evaportion from the
    QFLX data attribute.
    """
    return dataset.PRECTmm - dataset.QFLX * DAY_IN_SECONDS


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
        data = data.assign(d18o=delta_18O(data).assign_attrs(units="per thousand"))
        data = data.assign(PRECT=total_precip(data).assign_attrs(units="m/s"))
        data = data.assign(
            PRECTmm=(data["PRECT"] * DAY_IN_SECONDS * 1000).assign_attrs(units="mm/day")
        )
        data = data.assign(
            d18o_weighted=precip_weighted_d18o(data).assign_attrs(units="per thousand")
        )
        data = data.assign(TSC=(data.TS - KELVIN_OFFSET).assign_attrs(units="C"))
        data = data.assign(
            ELE=elevation(data).assign_attrs(units="m", long_name="Elevation")
        )
        data = data.assign(
            NET_PRECT=net_precip(data).assign_attrs(
                units="mm/day",
                long_name="Net precipitation (precipitation - evaporation)",
            )
        )
    return data


def plot_cam_mask(cam_data, land, projection=ccrs.PlateCarree):
    def wrapped(mask):
        mean_TS = cyclitize(cam_data.TS.where(mask).mean(dim="time"))
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection=projection())
        ax.pcolormesh(mean_TS.lon, mean_TS.lat, mean_TS, transform=ccrs.PlateCarree())
        plot_land(ax, land)
        ax.gridlines(draw_labels=True, alpha=0.5)
        plt.show()

    return wrapped
