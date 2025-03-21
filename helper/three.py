"""This file contains helper functions for the three.py file."""

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.spatial import cKDTree


def display_image_from_nc(nc_path, var_name, time_index=0, cmap="viridis"):
    """

    Display an image from a NetCDF (.nc) file.

    Display an image from a NetCDF (.nc) file for the specified
    variable at a specific time index.

    Args:
        nc_path (str): Path to the NetCDF file.
        var_name (str): The variable name in the NetCDF file to display.
        time_index (int): The index of the time dimension to plot (default =0).
        cmap (str): Colormap for the plot.
    """
    # Open the NetCDF file
    ds = xr.open_dataset(nc_path)

    # Get the variable data (assumes it has dimensions [time, lat, lon])
    var_data = ds[var_name]

    # If it's a 3D array, slice it by time
    if var_data.ndim == 3:
        var_data = var_data.isel(time=time_index)

    # Plot the variable data as an image
    plt.figure(figsize=(10, 6))
    var_data.plot.imshow(cmap=cmap)
    plt.title(f"{var_name} from {nc_path} at time index {time_index}")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()


def get_value_from_nc(nc_path, var_name, lon, lat):
    """

    Get value from nc file.

    Use longitude and latitude, return the value for that location
    from a NetCDF (.nc) file.

    Args:
        nc_path (str): Path to the NetCDF file.
        var_name (str): The variable name in the NetCDF file.
        lon (float): Longitude of the point.
        lat (float): Latitude of the point.

    Returns:
        float: The value at the specified location.
    """
    # Open the NetCDF file
    ds = xr.open_dataset(nc_path)

    # Get the variable data (assumes it has dimensions [lat, lon])
    var_data = ds[var_name]

    # Extract latitudes and longitudes
    lons = ds["longitude"].values
    lats = ds["latitude"].values

    # Use KDTree to find the nearest grid point (efficient search)
    tree = cKDTree(np.vstack((lons, lats)).T)
    dist, idx = tree.query([lon, lat])
    print(dist, idx)  # debug dist useless
    # Get the value for the closest point
    value = var_data.values[idx]

    return value
