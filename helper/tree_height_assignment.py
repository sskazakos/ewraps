"""Tree height assignment to nodes in a power grid graph."""

# pylint: disable=R0912, R0915, R0914
# pylint: disable=W0511, W1203, W0715, W0718, W0012, W0707, W0012.

import logging

import numpy as np
import rasterio
from scipy.spatial import cKDTree

try:
    import networkx as nx
except ImportError:
    raise ImportError(
        "NetworkX is required for this module. "
        "Install it with 'pip install networkx'."
    )

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_raster_data(raster_path):
    """
    Load raster data (e.g., tree height) from Earth observation datasets.

    Args:
        raster_path (str): Path to the raster file

    Returns:
        tuple: (data, transform, no_data, crs) or
        (None, None, None, None) if error
    """
    try:
        with rasterio.open(raster_path) as src:
            data = src.read(1)  # Read first band
            print(f"Raster length: {len(data)}")
            transform = src.transform
            no_data = src.nodata
            crs = src.crs

            if data.size == 0:
                logger.error("Raster file %s contains no data.", raster_path)
                return None, None, None, None

            logger.info("Successfully loaded raster data from %s", raster_path)
            return data, transform, no_data, crs
    except rasterio.errors.RasterioIOError:
        logger.error("Could not open raster file: %s", raster_path)
        return None, None, None, None
    except Exception as e:
        logger.error("Error loading raster data: %s", str(e))
        return None, None, None, None


# TODO: This is currently not working as called in
# 'assign_tree_heights_to_nodes'.
# This is due to a mismatch in the raster data shape attributes
# and the Affine object for passage into rows, cols attributes.


def get_grid_coordinates(transform, raster_data):
    """
    Generate latitude and longitude for each grid cell.

    Args:
        transform: The affine transform from the raster
        shape: (rows, cols) of the raster

    Returns:
        tuple: x_coords and y_coords as 2D arrays
    """
    try:
        logger.info("Raster data shape: %s", raster_data)
        # rows, cols = shape
        # print(raster_data)
        rows, cols = raster_data.shape
        # rows, cols = raster_data.height, raster_data.width
        # For very large rasters, consider memory usage
        logger.info("Rows: %s, Cols: %s", rows, cols)
        if rows * cols > 10_000_000:  # Arbitrary threshold for "large" raster
            logger.warning("Large raster detected. Memory impact.")

        x_coords = np.arange(cols) * transform.a + transform.c
        y_coords = np.arange(rows) * transform.e + transform.f
        logger.info("X coords: %s, Y coords: %s", x_coords, y_coords)
        return np.meshgrid(x_coords, y_coords)
    except Exception as e:
        logger.error("Error generating grid coordinates: %s", str(e))
        return None, None


# TODO: Complete. Changing due to changes in get_grid_coordinates.
def assign_tree_heights_to_nodes(
    graph, raster_path=None, raster_data=None, transform=None, no_data=None
):
    """
    Assign approximate tree height values to nodes in a grid graph.

    Args:
        graph: NetworkX graph with nodes having 'coordinates' attribute
        raster_path: Path to raster file (used if raster_data is None)
        raster_data: Pre-loaded raster data (optional)
        transform: Affine transform (required if raster_data is provided)
        no_data: No data value (required if raster_data is provided)

    Returns:
        nx.Graph: Graph with tree_height attributes added to nodes
    """
    # Validate the graph
    if not isinstance(graph, nx.Graph) and not isinstance(graph, nx.DiGraph):
        raise TypeError("Input must be a NetworkX graph")

    if len(graph) == 0:
        logger.warning("Empty graph provided. No nodes to process.")
        return graph

    # Load raster data if not provided
    if raster_data is None:
        if raster_path is None:
            raise ValueError("Raster_data or raster_path required.")
        raster_data, transform, no_data, _ = load_raster_data(raster_path)
        if raster_data is None:
            raise ValueError("Failed to load raster data from %s", raster_path)

    # Verify all nodes have coordinates
    missing_coords = [
        node
        for node, data in graph.nodes(data=True)
        if "coordinates" not in data or data["coordinates"] is None
    ]
    if missing_coords:
        logger.error(
            "%s nodes missing coordinates attribute",
            len(missing_coords),
        )
        if len(missing_coords) <= 5:
            logger.error("Nodes missing coordinates: %s", missing_coords)
        else:
            logger.error(
                "First 5 nodes missing coords: %s..",
                missing_coords[:5],
            )
        raise ValueError("Some nodes are missing 'coordinates' attribute")

    try:
        # Generate grid coordinates
        # x_grid, y_grid = get_grid_coordinates(transform, raster_data.shape)
        x_grid, y_grid = get_grid_coordinates(transform, raster_data)

        # Flatten data for KDTree lookup
        valid_mask = raster_data != no_data
        valid_count = np.sum(valid_mask)

        if valid_count == 0:
            logger.error(
                "No valid data points found in raster (all values are no_data)"
            )
            raise ValueError("Raster contains no valid data points")

        logger.info(f"Found {valid_count} valid data points in raster")
        tree_points = np.column_stack((x_grid[valid_mask], y_grid[valid_mask]))
        tree_heights = raster_data[valid_mask]

        logger.info("Building KDTree for spatial search")
        tree_kdtree = cKDTree(tree_points)

        # Track nodes that might be outside raster extent
        outside_extent_count = 0
        max_distance_threshold = (
            np.sqrt(transform.a**2 + transform.e**2) * 10
        )  # Fixed: was a2

        # Assign nearest tree height to each node
        nodes_processed = 0
        nodes_with_errors = 0

        for node, data in graph.nodes(data=True):
            try:
                node_x, node_y = data["coordinates"]
                distance, idx = tree_kdtree.query([node_x, node_y])

                # Check if node is far from any raster point
                # (possibly outside extent)
                if distance > max_distance_threshold:
                    outside_extent_count += 1
                    logger.warning(
                        "Node %s may be outside raster (distance: %.2f)",
                        node,
                        distance,
                    )

                graph.nodes[node]["tree_height"] = float(tree_heights[idx])
                graph.nodes[node]["tree_height_distance"] = float(
                    distance
                )  # Store distance for quality assessment
                nodes_processed += 1

            except (ValueError, IndexError, TypeError) as e:
                logger.error("Error processing node %s: %s", node, str(e))
                graph.nodes[node]["tree_height"] = None
                graph.nodes[node]["tree_height_distance"] = None
                nodes_with_errors += 1

        if outside_extent_count > 0:
            logger.warning(
                "%s nodes may be outside raster ",
                outside_extent_count,
            )

        logger.info(
            "Tree height assignment complete: %s successful, %s failed",
            nodes_processed,
            nodes_with_errors,
        )
        return graph

    except Exception as e:
        logger.error("Error in assign_tree_heights_to_nodes: %s", str(e))
        raise
