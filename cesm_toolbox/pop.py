from enum import Enum
from typing import Tuple

import xarray as xr
import xesmf as xe

from .utils import fix_dates


class GridType(Enum):
    U = "U"
    T = "T"


def regrid(
    dataset: xr.DataArray,
    output_dims: Tuple[int, int],
    grid_type: GridType = GridType.T,
    interpolation_method: str = "bilinear",
    periodic: bool = True,
    reuse_weights: bool = True,
) -> xr.DataArray:
    """
    This function takes in a dataarray that is in the curvilinear grid that POP
    uses and regrids it to a rectilinear grid using xesmf. The dimensions of
    the output grid are given as a tuple in degrees latitude and longitude
    (respectively). POP data should be regridded before plotting and before
    performing a zonal average.
    """
    lon_str, lat_str = f"{grid_type.value}LONG", f"{grid_type.value}LAT"
    output_grid = xe.util.grid_global(*output_dims)
    input_data = dataset.rename({lon_str: "lon", lat_str: "lat"})
    regridder = xe.Regridder(
        input_data,
        output_grid,
        interpolation_method,
        periodic=periodic,
        reuse_weights=reuse_weights,
    )
    output_data = regridder(input_data)
    return output_data


def delta_18o(pop_data: xr.Dataset) -> xr.DataArray:
    d18o = (pop_data.R18O - 1) * 1000
    d18o = d18o.assign_attrs(units="per thousand")
    return d18o


def read_pop_data(
    path: str, with_fixed_dates: bool = True, with_extra_data: bool = False
) -> xr.Dataset:
    data = xr.open_dataset(path)
    if with_fixed_dates:
        data = fix_dates(data)
    if with_extra_data:
        data = data.assign(d18o=delta_18o(data))
    return data
