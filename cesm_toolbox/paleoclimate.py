from typing import List, Union
import warnings

import cartopy.crs as ccrs
import cartopy.util as cutil
import numpy as np
import xarray as xr
from matplotlib.figure import Axes


def plot_land(ax: Axes, land_frac: xr.DataArray, threshold=0.5):
    """
    Plot the outline of grid cells that contain a land fraction greater than the
    provided threshold. Useful for plotting the paleo outline of continents.

    land_frac: A DataArray from CESM that contains the fraction of land at each
    given grid cell
    threshold: The fraction of land that needs to be in a grid cell to be considered
    in the outline
    """
    land = land_frac >= threshold
    cyclic_land, cyclic_land_lon = cutil.add_cyclic_point(land, coord=land.lon)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ax.contour(
            cyclic_land_lon,
            land.lat,
            cyclic_land,
            transform=ccrs.PlateCarree(),
            colors="black",
            levels=[1],
            linewidths=0.5,
        )


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
