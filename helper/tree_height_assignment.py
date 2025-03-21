"""Tree height assignment to nodes in a power grid graph."""

# import networkx as nx
import numpy as np
import rasterio
from scipy.spatial import cKDTree


def load_raster_data(raster_path):
    """Load raster data (e.g., tree height) from Earth observation datasets."""
    with rasterio.open(raster_path) as src:
        data = src.read(1)  # Read first band
        transform = src.transform
        no_data = src.nodata
    return data, transform, no_data


def get_grid_coordinates(transform, shape):
    """Generate latitude and longitude for each grid cell."""
    rows, cols = shape
    x_coords = np.arange(cols) * transform.a + transform.c
    y_coords = np.arange(rows) * transform.e + transform.f
    return np.meshgrid(x_coords, y_coords)


def assign_tree_heights_to_nodes(graph, raster_data, transform, no_data):
    """Assign approximate tree height values to nodes in a grid graph."""
    # Generate grid coordinates
    x_grid, y_grid = get_grid_coordinates(transform, raster_data.shape)

    # Flatten data for KDTree lookup
    valid_mask = raster_data != no_data
    tree_points = np.column_stack((x_grid[valid_mask], y_grid[valid_mask]))
    tree_heights = raster_data[valid_mask]
    tree_kdtree = cKDTree(tree_points)

    # Assign nearest tree height to each node
    for node, data in graph.nodes(data=True):
        node_x, node_y = data["coordinates"]
        _, idx = tree_kdtree.query([node_x, node_y])
        graph.nodes[node]["tree_height"] = tree_heights[idx]

    return graph
