"""TIF file helper functions."""

import matplotlib.pyplot as plt
import rasterio
from pyproj import Transformer
from rasterio.plot import show


def render_tif_image(tif_path):
    """
    Render a .tif file as an image.

    This function reads a raster TIF file and
    displays it using a specified colormap.

    Args:
        tif_path (str): Path to the .tif file.
    """
    with rasterio.open(tif_path) as src:
        plt.figure(figsize=(10, 6))
        show(src, cmap="viridis")  # Adjust colormap as needed
        plt.title("Copernicus Forest Type 2018 (UK).")
        plt.show()


def get_tif_value(tif_path, lon, lat):
    """
    Retrieve the raster value from a .tif file at a given geographic location.

    This function converts longitude and latitude coordinates into the raster's
    coordinate system and extracts the corresponding pixel value.

    Args:
        tif_path (str): Path to the .tif file.
        lon (float): Longitude of the point.
        lat (float): Latitude of the point.

    Returns:
        float: The raster value at the specified location.
    """
    with rasterio.open(tif_path) as src:
        # Convert geographic coordinates (lon, lat) to raster image coordinates
        transformer = Transformer.from_crs(
            "EPSG:4326", src.crs.to_string(), always_xy=True
        )
        x, y = transformer.transform(lon, lat)

        # Get the row and column in the raster
        row, col = src.index(x, y)

        # Read the value at the specified location
        value = src.read(1)[row, col]  # Assuming a single-band raster

        return value
