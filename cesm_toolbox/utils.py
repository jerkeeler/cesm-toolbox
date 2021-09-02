import colorsys
import os
import time
from contextlib import ContextDecorator
from datetime import datetime
from typing import List, Tuple, Union, Optional

import matplotlib.colors as mc
import numpy as np
import xarray as xr
from cartopy import util as cutil
from cartopy.geodesic import Geodesic
from dateutil.relativedelta import relativedelta
from matplotlib.figure import Figure
from shapely.geometry import LineString

Color = Tuple[float, float, float]
IMAGE_DIR = os.getenv("SAVED_FIG_DIR", default=".")
GEODESIC = Geodesic()


def set_image_dir(image_dir):
    global IMAGE_DIR
    IMAGE_DIR = image_dir


def cyclitize(dataset: xr.DataArray) -> xr.DataArray:
    """
    Take a dataarray and add cyclic longitude points for easy plotting. Return
    a new dataarray to make life easier for everyone.
    """
    cyclic_data, cyclic_lon = cutil.add_cyclic_point(dataset, coord=dataset.lon)
    new_dataset = xr.DataArray(
        cyclic_data,
        coords={
            **dataset.coords,
            "lon": cyclic_lon,
        },
        dims=dataset.dims,
        attrs=dataset.attrs,
        name=dataset.name,
    )
    return new_dataset


def fix_dates(
    climatology_data: xr.Dataset, date_coord: str = "time", fake_year: int = 1994
) -> xr.Dataset:
    """
    Prescribe dates for a given dataset since the timestamps are meaningless in
    the output file and only the order of the months matter (e.g. the first
    month is always January)

    fake_year - the year to set the dates to, this should be a NON-leap year
    as per https://www.cesm.ucar.edu/models/cesm1.0/cesm/cesm_doc_1_0_4/x3088.html
    which states that leap years are not being used
    """
    fixed_dates = [
        datetime(fake_year, 1, 1) + relativedelta(months=i) for i in range(12)
    ]
    update_dict = {
        date_coord: fixed_dates,
    }
    return climatology_data.assign_coords(**update_dict)


def to_lower_snake_case(*args: List[str]) -> str:
    """
    Convenience function for converting one or more strings to lower snake case. Useful
    for creating filenames.
    """
    return "_".join([input_str.lower().replace(" ", "_") for input_str in args])


def get_max_colormap_value(data: xr.DataArray) -> float:
    """
    Get the maximum absolute value from a dataset. Useful for setting the vmax and vmin
    values when creating a colormap for plotting in matplotlib.
    """
    return max(np.abs(np.max(data)), np.abs(np.min(data)))


def adjust_lightness(color: Union[Color, str], amount: float = 1.5) -> Color:
    """
    Change the lightness of the provided color. The second optional color determines
    how much to lighten the color. The higher the value, the lighter it will be.

    Adapted from: https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib/49601444
    """
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def savefig(
    fig: Figure,
    filename: str,
    dpi: int = 100,
    version: Optional[int] = None,
    facecolor: Optional[str] = "white",
) -> None:
    """
    Save a figure to a specified directory and version it to ensure that it doesn't overwrite any
    previously saved figures.
    """
    if facecolor is not None:
        fig.patch.set_facecolor(facecolor)
    figure_path = os.path.join(
        IMAGE_DIR, f"{filename}-v{str(version or 1).zfill(3)}.png"
    )
    if version is None:
        version = 1
        while os.path.exists(figure_path):
            version += 1
            if version > 100:
                raise Exception(
                    f"To many files named {filename} in directory, tried up to version {version}"
                )
            figure_path = os.path.join(
                IMAGE_DIR, f"{filename}-v{str(version or 1).zfill(3)}.png"
            )
    fig.savefig(figure_path, dpi=dpi)


class timer(ContextDecorator):
    """
    Use as an annotation to wrap a function and create a timer for that function
    to measure wall clock timing. Can also be used as a contextmanager:

    with timer(name="potato"):
        time.sleep(2) # do stuff

    @timer(name="funcname")
    def test():
        time.sleep(3) # do stuff
    """

    def __init__(self, name="timer", format="s"):
        self.name = name
        self.format = format

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *exc):
        elapsed = time.time() - self.start

        if format == "ms":
            elapsed *= 1000

        print(f"{self.name} took {elapsed:.2f}{self.format}")
        return False


def combine_datasets(
    datasets: List[xr.Dataset], labels=None, new_coord="experiment"
) -> xr.Dataset:
    """
    Merge similarly dimensioned dataset together along a new artifical coordinate
    (default "experiment").
    """
    if labels is None:
        labels = list(range(len(datasets)))
    datasets = [
        dataset.expand_dims({new_coord: 1}).assign_coords({new_coord: [label]})
        for (label, dataset) in zip(labels, datasets)
    ]
    merged_data = xr.merge(datasets)
    return merged_data


def central_lon(extent: List[Union[int, float]]) -> float:
    """
    Return the central longitude given an array defining extents such as [x0, x1, y0, y1]. This is useful for
    ensuring that your projection is always centered over your extent area to reduce distortions.
    """
    return sum(extent[:2]) / 2


def get_region_mask_from_point(
    grid: xr.Dataset,
    lat: float,
    lon: float,
    lon_offset: float = 5,
    lat_offset: float = 2.5,
):
    """
    Return lat/lon mask around a given lat/lon pair. Assuming the lat/lon pair is in the center of a grid point, this
    mask should provide a 3x3 grid cell mask over that point (with default lon/lat offsets). Useful for conducting
    point analysis.
    """
    return (
        (grid.lat < lat + lat_offset)
        & (grid.lat > lat - lat_offset)
        & (grid.lon < lon + lon_offset)
        & (grid.lon > lon - lon_offset)
    )


def find_closest_idx(values: np.ndarray, to_find: float) -> int:
    """Returns the index of the value that is closest to the one you're trying to find."""
    return (np.abs(values - to_find)).argmin()


def find_closest_cell(
    grid: xr.Dataset, lat: float, lon: float
) -> Tuple[Tuple[int, int], Tuple[float, float]]:
    """
    Given a grid that has lat/lon, find the closest cell center that match the lat/lon pair provided.
    """
    lat_idx = find_closest_idx(grid.lat.values, lat)
    lon_idx = find_closest_idx(grid.lon.values, lon)
    grid_lat, grid_lon = grid.lat.values[lat_idx], grid.lon.values[lon_idx]
    return (lat_idx, lon_idx), (grid_lat, grid_lon)


def find_all_closest_cells(grid: xr.Dataset, lat_lon_pairs: np.ndarray):
    """
    For each lat/lon pair find the grid point that is closest to the pair, compile a list and return it.
    """
    indicies = [
        (find_closest_idx(grid.lat.values, lat), find_closest_idx(grid.lon.values, lon))
        for (lat, lon) in lat_lon_pairs
    ]
    return np.array(
        [
            (grid.lat.values[lat_idx], grid.lon.values[lon_idx])
            for (lat_idx, lon_idx) in indicies
        ]
    )


def great_circle_dist(
    point1: Tuple[float, float], point2: Tuple[float, float]
) -> float:
    """
    Finds geodesic great cricle distance between two pairs of lat/lon points. The "true" distance between the points
    on a sphere. Or at least as close an approximation as possible.

    Returns values in meters.
    """
    line = LineString([point1, point2])
    return GEODESIC.geometry_length(line)
