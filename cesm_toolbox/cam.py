from datetime import datetime

import xarray as xr
from dateutil.relativedelta import relativedelta

from .consts import KELVIN_OFFSET, YEAR_IN_SECONDS


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
            PRECTmm=(data["PRECT"] * YEAR_IN_SECONDS * 1000 / 365).assign_attrs(
                units="mm/day"
            )
        )
        data = data.assign(
            d18o_weighted=precip_weighted_d18o(data).assign_attrs(units="per thousand")
        )
        data = data.assign(TSC=(data.TS - KELVIN_OFFSET).assign_attrs(units="C"))
    return data