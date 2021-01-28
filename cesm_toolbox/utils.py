import colorsys
from typing import List, Tuple, Union

import matplotlib.colors as mc
import numpy as np
import xarray as xr

Color = Tuple[float, float, float]


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
