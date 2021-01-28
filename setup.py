from distutils.core import setup

setup(
    name="keeler",
    version="0.1",
    description="Utility for working with CESM during my PhD.",
    author="Jeremy Keeler",
    author_email="j@keeler.dev",
    packages=["keeler"],
    install_requires=["cartopy", "numpy", "xarray", "python-dateutil", "matplotlib"],
)
