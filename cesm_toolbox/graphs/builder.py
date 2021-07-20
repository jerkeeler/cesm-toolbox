import os
from typing import List, Tuple, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.figure import Figure

from cesm_toolbox.graphs.plotting import (
    plot_geospatial_scalar,
    plot_geospatial_scalar_differences,
    plot_geospatial_seasonal_scalar,
)
from cesm_toolbox.cam import read_cam_data
from cesm_toolbox.paleoclimate import plot_land, seasonal_plot
from cesm_toolbox.utils import to_lower_snake_case


Self = TypeVar("Self")


class CAMGraphBuilder:
    """
    This class follows the builder pattern for defining features, limits, and attributes from which to makes graphs
    given an xarray dataset. You can use this class to quickly and easily generate a wide variety of plots for a
    variety of CESM output.

    Usage:
    ------
    CAMGraphBuilder()
        .with_dimensions([["path/data1.nc", "path/data2.nc"]
                          ["path/data3.nc", "path/data4.nc"]])
        .with_dimension_labels([["Control 3x CO2", "Control 6x CO2"]
                                ["OrbMaxN 3x CO2", "OrbMaxN 6x CO2"]])
        .with_attributes(["TS", "d18o"])
        .with_output_path("output/")
        .with_zonal_graphs(True)
        .with_dpi(300)
        .generate_graphs()
    """

    def __init__(self):
        xr.set_options(keep_attrs=True)
        self.attributes: List[str] = []
        self.dimensions: List[List[str]] = []
        self.dimension_labels: List[List[str]] = []
        self.zonal_graphs: bool = False
        self.output_path = None
        self.graphs_generated = False
        self.dpi = 100

    def with_attribute(self: Self, attribute: str) -> Self:
        self.attributes.append(attribute)
        return self

    def with_attributes(self: Self, attributes: List[str]) -> Self:
        self.attributes += attributes
        return self

    def with_dimensions(self: Self, dimensions: List[List[str]]) -> Self:
        """
        Add dimensions to the graphs. These are the filepaths corresponding to each dimension in your matrix.
        """
        self.dimensions = dimensions
        return self

    def with_dimension_labels(self: Self, labels: List[List[str]]) -> Self:
        self.dimension_labels = labels
        return self

    def with_zonal_graphs(self: Self, zonal: bool) -> Self:
        self.zonal_graphs = zonal
        return self

    def with_output_path(self: Self, output_path: str) -> Self:
        self.output_path = output_path
        return self

    def with_dpi(self: Self, dpi: int) -> Self:
        self.dpi = dpi
        return self

    def generate_graphs(self) -> None:
        if self.graphs_generated:
            raise Exception(
                "This builder has already been used! Cannot regenerate graphs."
            )
        if not self.output_path:
            raise Exception("No output path specified!")

        # Read in all the data
        # Select desired attributes
        # Calculate means for all grid cells (sum if d18o_weighted)

        data = self._read_data()
        self.land = data[0][0].LANDFRAC.isel(time=0)
        data = [[d[self.attributes] for d in row] for row in data]
        mean_data = [[d.mean(dim="time", keep_attrs=True) for d in row] for row in data]
        seasonal_data = [
            [d.groupby("time.season").mean(dim="time", keep_attrs=True) for d in row]
            for row in data
        ]

        self._plot_mean(mean_data)
        self._plot_mean_seasonal(seasonal_data)
        self._plot_row_diff(mean_data)
        self._plot_row_diff_seasonal(seasonal_data)
        self._plot_col_diff(mean_data)
        self._plot_col_diff_seasonal(seasonal_data)

        # raw values plotted for each dimension
        # i want difference plots between second column (second - first)
        # i want difference plots between first row and all other rows (row x - row 1)
        # i want seasonal plots too...
        # seasonal plots of all data?
        # seasonal difference plots between first and second column (second - first for each season)
        # seasonal difference plots between first row and all other rows (row x - row 1 for each season)

    def _read_data(self) -> List[List[xr.Dataset]]:
        data = []
        for row in self.dimensions:
            data.append([read_cam_data(col) for col in row])
        return data

    def _plot_mean(self, data: List[List[xr.Dataset]]):
        all_figures: List[Tuple[Figure, str, str]] = []
        for attribute in self.attributes:
            selected_data = [[d[attribute] for d in row] for row in data]
            all_figures.append(
                (
                    plot_geospatial_scalar(
                        selected_data,
                        self.dimension_labels,
                        self.land,
                    ),
                    attribute,
                )
            )

        for figure, attribute in all_figures:
            filename = f"{attribute}_mean.png"
            self._savefig(figure, filename)

    def _plot_mean_seasonal(self, data: List[List[xr.Dataset]]):
        all_figures: List[Tuple[Figure, str, str]] = []
        flattened_titles = [label for row in self.dimension_labels for label in row]
        for attribute in self.attributes:
            selected_data = [d[attribute] for row in data for d in row]
            all_figures += [
                (plot_geospatial_seasonal_scalar(d, label, self.land), attribute, label)
                for (d, label) in zip(selected_data, flattened_titles)
            ]

        for figure, attribute, label in all_figures:
            filename = f"{attribute}_seasonal_{to_lower_snake_case(label)}.png"
            self._savefig(figure, filename)

    def _plot_row_diff(self, data: List[List[xr.Dataset]]):
        all_figures: List[Tuple[Figure, str, str]] = []
        for attribute in self.attributes:
            selected_data = [[d[attribute] for d in row] for row in data]
            row1 = selected_data[0]
            selected_data = [
                [(col - row1[col_idx]) for (col_idx, col) in enumerate(row)]
                for row in selected_data
            ]
            all_figures.append(
                (
                    plot_geospatial_scalar_differences(
                        selected_data,
                        self.dimension_labels,
                        self.land,
                    ),
                    attribute,
                )
            )

        for figure, attribute in all_figures:
            filename = f"{attribute}_row_differences.png"
            self._savefig(figure, filename)

    def _plot_row_diff_seasonal(self, data: List[List[xr.Dataset]]):
        pass

    def _plot_col_diff(self, data: List[List[xr.Dataset]]):
        all_figures: List[Tuple[Figure, str, str]] = []
        for attribute in self.attributes:
            selected_data = [[d[attribute] for d in row] for row in data]
            col1 = np.array(selected_data)[:, 0]
            selected_data = [
                [col - col1[row_idx] for col in row]
                for (row_idx, row) in enumerate(selected_data)
            ]
            all_figures.append(
                (
                    plot_geospatial_scalar_differences(
                        selected_data,
                        self.dimension_labels,
                        self.land,
                        col_diffs=True,
                    ),
                    attribute,
                )
            )

        for figure, attribute in all_figures:
            filename = f"{attribute}_col_differences.png"
            self._savefig(figure, filename)

    def _plot_col_diff_seasonal(self, data: List[List[xr.Dataset]]):
        pass

    def _savefig(self, figure: Figure, filename: str) -> None:
        filepath = os.path.join(self.output_path, filename)
        figure.savefig(filepath, dpi=self.dpi)
        plt.close(fig=figure)
