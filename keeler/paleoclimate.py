import colorsys
from datetime import datetime
from typing import List, Union

import cartopy.crs as ccrs
import cartopy.util as cutil
import matplotlib.colors as mc
import numpy as np
import xarray as xr
from dateutil.relativedelta import relativedelta

from keeler.consts import KELVIN_OFFSET, YEARS_IN_SECONDS


def plot_land(ax, land_frac, threshold=0.5):
    land = land_frac >= threshold
    cyclic_land, cyclic_land_lon = cutil.add_cyclic_point(land, coord=land.lon)
    ax.contour(
        cyclic_land_lon,
        land.lat,
        cyclic_land,
        transform=ccrs.PlateCarree(),
        colors="black",
        levels=[True],
        linewidths=0.5,
    )


def fix_dates(climatology_data, date_coord="time"):
    fixed_dates = [datetime(2000, 1, 1) + relativedelta(months=i) for i in range(12)]
    update_dict = {
        date_coord: fixed_dates,
    }
    return climatology_data.assign_coords(**update_dict)


def seasonal_plot(figure, season_func, projection=None):
    projection = projection if projection is not None else ccrs.Robinson()
    axes = [
        figure.add_subplot(subplot, projection=projection)
        for subplot in (221, 222, 223, 224)
    ]
    seasons = ["DJF", "MAM", "JJA", "SON"]
    func_returns = [season_func(season, ax) for season, ax in zip(seasons, axes)]
    return axes, func_returns


def delta_18O(cam_data):
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


def clustering(cluster_algo, data, reshape=True):
    stacked = data.stack(z=("lat", "lon"))
    reshaped = stacked.values
    if reshape:
        reshaped = np.reshape(reshaped, (-1, 1))

    clusters = cluster_algo.fit(reshaped)
    labelled_array = xr.DataArray(
        clusters.labels_, dims=stacked.dims, coords=stacked.coords, attrs=stacked.attrs
    )
    return labelled_array.unstack()


def read_cam_data(
    path: str,
    with_fixed_dates: bool = True,
    with_extra_data: bool = True,
) -> xr.Dataset:
    data = xr.open_dataset(path)
    if with_fixed_dates:
        data = fix_dates(data)
    if with_extra_data:
        data = data.assign(d18o=delta_18O(data).assign_attrs(units="per thousand"))
        data = data.assign(PRECT=total_precip(data).assign_attrs(units="m/s"))
        data = data.assign(
            PRECTmm=(data["PRECT"] * YEARS_IN_SECONDS * 1000).assign_attrs(
                units="mm/year"
            )
        )
        data = data.assign(
            d18o_weighted=precip_weighted_d18o(data).assign_attrs(units="per thousand")
        )
        data = data.assign(TSC=(data.TS - KELVIN_OFFSET).assign_attrs(units="C"))
    return data


def total_precip(dataset: xr.Dataset) -> xr.Dataset:
    return dataset.PRECL + dataset.PRECC


def precip_weighted_d18o(dataset: xr.Dataset) -> xr.Dataset:
    return dataset.d18o * (dataset.PRECT / dataset.PRECT.sum(dim="time"))


def get_value_from_datasets(
    datasets: List[Union[xr.Dataset, xr.DataArray]],
    function_name: str = "min",
    placeholder_coord: str = "experiment",
) -> xr.Dataset:
    datasets = [
        dataset.expand_dims({placeholder_coord: 1}).assign_coords(
            {placeholder_coord: [i]}
        )
        for (i, dataset) in enumerate(datasets)
    ]
    merged_data = xr.merge(datasets)
    return getattr(merged_data, function_name)(dim=placeholder_coord)


def adjust_lightness(color, amount=1.5):
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])
