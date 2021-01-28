from distutils.core import setup

setup(
    name="cesm_toolbox",
    version="0.1",
    description="Utility for working with CESM during my PhD.",
    author="Jeremy Keeler",
    author_email="j@keeler.dev",
    packages=["cesm_toolbox"],
    install_requires=["cartopy", "numpy", "xarray", "python-dateutil", "matplotlib"],
)
