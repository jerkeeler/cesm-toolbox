from typing import List

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cartopy import crs as ccrs, util as cutil
from matplotlib.figure import Figure
from matplotlib import ticker


from cesm_toolbox.paleoclimate import plot_land, seasonal_plot
from cesm_toolbox.utils import get_max_colormap_value


def plot_geospatial_scalar(
    data: List[List[xr.DataArray]],
    labels: List[List[str]],
    land: xr.DataArray,
    colormap: str = "RdBu_r",
    levels: int = 20,
) -> Figure:
    num_rows, num_cols = len(data), len(data[0])
    fig = plt.figure(figsize=(num_cols * 8, num_rows * 4))
    max_value = max(get_max_colormap_value(col) for row in data for col in row)
    for idx_row, row in enumerate(data):
        for idx_col, col in enumerate(row):
            subplot_num = idx_row * num_cols + idx_col + 1
            ax = fig.add_subplot(
                f"{num_rows}{num_cols}{subplot_num}", projection=ccrs.Robinson()
            )
            cyclic_data, cyclic_lon = cutil.add_cyclic_point(col, coord=col.lon)
            contour = ax.contourf(
                cyclic_lon,
                col.lat,
                cyclic_data,
                transform=ccrs.PlateCarree(),
                cmap=colormap,
                levels=levels,
                vmin=-max_value,
                vmax=max_value,
            )
            plot_land(ax, land)
            cbar = fig.colorbar(contour, ax=ax, pad=0.1)
            cbar.set_label(col.attrs["units"])
            ax.gridlines(draw_labels=True)
            title = labels[idx_row][idx_col]
            ax.set_title(f"{title} {col.name}", size=15)
    return fig


def plot_geospatial_seasonal_scalar(
    data: xr.DataArray,
    label: str,
    land: xr.DataArray,
    colormap: str = "RdBu_r",
    levels: int = 20,
):
    fig = plt.figure(figsize=(12, 12))

    def season_func(season, ax):
        season_data = data.sel(season=season)
        cyclic_data, cyclic_lon = cutil.add_cyclic_point(
            season_data, coord=season_data.lon
        )
        max_value = get_max_colormap_value(data)
        contour = ax.contourf(
            cyclic_lon,
            season_data.lat,
            cyclic_data,
            transform=ccrs.PlateCarree(),
            cmap=colormap,
            levels=levels,
            vmin=-max_value,
            vmax=max_value,
        )
        plot_land(ax, land)
        cbar = fig.colorbar(contour, ax=ax, pad=0.1)
        cbar.set_label(season_data.attrs["units"])

        ax.gridlines(draw_labels=True)
        ax.set_title(season, size=15)

    seasonal_plot(fig, season_func)
    fig.suptitle(f"{label} Seasonal {data.name}")

    return fig


def plot_geospatial_scalar_differences(
    data: List[List[xr.DataArray]],
    labels: List[List[str]],
    land: xr.DataArray,
    colormap: str = "RdBu_r",
    levels: int = 20,
    col_diffs: bool = False,
) -> Figure:
    num_rows, num_cols = len(data), len(data[0])
    fig = plt.figure(figsize=(num_cols * 8, num_rows * 4))
    max_value = max(get_max_colormap_value(col) for row in data for col in row)
    for idx_row, row in enumerate(data):
        for idx_col, col in enumerate(row):
            subplot_num = idx_row * num_cols + idx_col + 1
            ax = fig.add_subplot(
                f"{num_rows}{num_cols}{subplot_num}", projection=ccrs.Robinson()
            )
            cyclic_data, cyclic_lon = cutil.add_cyclic_point(col, coord=col.lon)
            contour = ax.contourf(
                cyclic_lon,
                col.lat,
                cyclic_data,
                transform=ccrs.PlateCarree(),
                cmap=colormap,
                levels=levels,
                vmin=-max_value,
                vmax=max_value,
            )
            plot_land(ax, land)
            cbar = fig.colorbar(contour, ax=ax, pad=0.1)
            cbar.set_label(col.attrs["units"])
            ax.gridlines(draw_labels=True)
            title = labels[idx_row][idx_col]
            control_title = labels[idx_row][0] if col_diffs else labels[0][idx_col]
            ax.set_title(f"{title} - {control_title} {col.name}", size=15)

    fig.suptitle(f"Difference Plots for {col.name}")
    return fig


@ticker.FuncFormatter
def lat_formatter(x, pos):
    if x > 0:
        return f"{x:.0f}N"
    elif x < 0:
        return f"{abs(x):.0f}S"
    else:
        return f"{x:.0f}"
