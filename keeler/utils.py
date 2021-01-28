from typing import List

import numpy as np
import xarray as xr


def to_lower_snake_case(*args: List[str]) -> str:
    return "_".join([input_str.lower().replace(" ", "_") for input_str in args])


def get_max_colormap_value(data: xr.DataArray) -> float:
    return max(np.abs(np.max(data)), np.abs(np.min(data)))
