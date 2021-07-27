from typing import List

import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from cesm_toolbox.paleoclimate import plot_land
from cesm_toolbox.utils import cyclitize
from cesm_toolbox.pop import regrid


def map_difference_plot(
    datasets: List[xr.DataArray],
    titles: List[str],
    data_label: str,
    land: xr.DataArray,
    data_func=None,
    target_projection=ccrs.PlateCarree,
    input_projection=ccrs.PlateCarree,
    should_cyclitize=True,
    figsize=(30, 5),
    should_regrid=False,
    regrid_size=(1, 1),
    cmap="viridis",
    norm=None,
) -> Figure:
    # Data maniupulation here
    if data_func is not None:
        datasets = [data_func(d) for d in datasets]
    base_data = datasets[0]
    diffs = [d - base_data for d in datasets[1:]]
    if should_cyclitize:
        base_data = cyclitize(base_data)
        diffs = [cyclitize(d) for d in diffs]
    if should_regrid:
        base_data = regrid(base_data, regrid_size)
        diffs = [regrid(d, regrid_size) for d in diffs]

    # Plot stuff here
    target_proj, input_proj = target_projection(), input_projection()
    num_plots = len(datasets)
    subplots = [int(f"1{num_plots}{i + 2}") for i in range(len(datasets) - 1)]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(int(f"1{num_plots}1"), projection=target_proj)
    contour = ax.contourf(
        base_data.lon,
        base_data.lat,
        base_data,
        transform=input_proj,
        cmap=cmap,
        norm=norm,
    )
    ax.gridlines(draw_labels=True, linestyle="--", alpha=0.5)
    plot_land(ax, land)
    ax.set_title(titles[0], size=15)
    cbar = fig.colorbar(contour, ax=ax, pad=0.15)
    cbar.set_label(data_label)

    vmax = max(np.nanmax(d.values) for d in diffs)
    vmin = min(np.nanmin(d.values) for d in diffs)
    vmax = max(abs(vmin), abs(vmax))
    vmin = -vmax
    axes = []
    for data, title, subplot in zip(diffs, titles[1:], subplots):
        ax = fig.add_subplot(subplot, projection=target_proj)
        ax.contourf(
            data.lon,
            data.lat,
            data,
            transform=input_proj,
            cmap="RdBu_r",
            norm=colors.CenteredNorm(),
            vmin=vmin,
            vmax=vmax,
        )
        ax.gridlines(draw_labels=True, linestyle="--", alpha=0.5)
        plot_land(ax, land)
        ax.set_title(f"{title} - {titles[0]}", size=15)
        axes.append(ax)

    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="RdBu_r"), ax=axes)
    cbar.set_label(data_label)
    return fig
