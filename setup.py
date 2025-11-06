# --------------------------- ONLY FOR DEVELOPMENT MODE ---------------------------
# This setup.py file is used to configure the setup for the pymesh2d project.
# It specifies metadata about the project and its dependencies.
# This file is ONLY useful for development mode, where you can install
# the project in an editable state using the command `pip install -e .`.
# This allows you to make changes to the code and have them immediately reflected
# without needing to reinstall the package.
# To install the package, change name of pyproject.toml to pyproject.toml.bak

from setuptools import find_packages, setup

setup(
    name="pymesh2d",
    author="GeoOcean Group",
    author_email="faugere@unican.es",
    # long_description=open("README.md").read(),
    # long_description_content_type="text/markdown",
    # url="https://github.com/GeoOcean/pymesh2d",
    packages=find_packages(),  # Automatically find packages in the directory
    include_package_data=True,
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "os",
        "typing",
        "sys",
        "math",
        "time",
        "netCDF4",
        "rasterio",
    ],
    classifiers=["Programming Language :: Python :: 3.3.7"],
    python_requires=">=3.7",  # Specify the Python version required
)
