# cesm_toolbox

This is a collection of utilitiy classes and functions for easily manipulating,
plotting, and examining CESM output. This is primarily my (Jeremy's) personal
repository for code snippets that help me explore paleoclimate simulations.

This code is provided *as is*. You are more than welcome to use it for your own
purposes, request features, submit bugs/pull requests, but there is no guarantee
that I will respond or incorporate changes. However, I will do my best to respond
and help out if requested.

## Development

All code is formatted using [Black](https://github.com/psf/black). Please ensure
code follows those style guidelines before making changes or submitting pull requests.

Please make code as Pythonic and idiomatic as possible.

There are, currently, no tests, but if this toolbox becomes more involved I would
like to add them.

## Capabilities and functions

Easily read in CAM data, plot paleo outlines, calculate d18O, total precip, create seasonal plots, 
easily split and combine datasets, and more!

### CAM module

Useful functions for working with CAM output.

`delta_18O` - calculates $\delta^{18}O$ of precipitaiton

`total_precip` - calculates total precipitation in mm/day

`precip_weighted_d18o` - calcualtes $\delta^{18}O$ of precipitation weighted by total preciptation

`delta_d` - calculates deuterium excess of precipitation

`elevation` - calculates the elevation in meters at all grid points

`net_precip` - calculates net precipitation (precipitation - evaporation)

