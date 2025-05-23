# Extreme Weather Resilience Analysis of Power Systems (EWRAPS)
A comprehensive toolkit for analyzing and modeling the resilience of power distribution systems subjected to extreme weather events, with special focus on tree-related failures.

## Overview
This repository contains a collection of helper modules for analyzing power distribution systems and their resilience to environmental factors, particularly extreme winds and tree failures. The toolkit implements the integrated framework described in:
> Hou, G. and Muraleetharan, K.K., 2023. Modeling the resilience of power distribution systems subjected to extreme winds considering tree failures: An integrated framework. International Journal of Disaster Risk Science, 14(2), pp.194-208.

## Features
- Geospatial data processing and visualization
- Integration with multiple geospatial data formats (CSV, TIF, NetCDF)
- Geocoding capabilities for location data
- Tree height assignment to power grid nodes
- Fragility curve modeling for power distribution components
- Resilience index estimation for power systems
- Line coordinate interpolation for network analysis

## Module Structure
The repository is organized into helper modules, each with specific functionality:

### Geocoding and CSV Processing (one.py)
Read and process CSV files containing location data
- Geocode locations to obtain latitude and longitude coordinates
- Handle geocoding errors and retries

### TIF File Operations (two.py)
- Render TIF images for visualization
- Extract values from TIF files at specific geographic coordinates

### NetCDF File Operations (three.py)
- Display images from NetCDF files
- Extract values from NetCDF files at specific geographic coordinates
- Spatial interpolation using KDTree

### Coordinate Interpolation (four.py)
- Generate intermediate coordinates between points using linear interpolation

### Fragility Curve Analysis (fragility_curves.py)
- Model probability of failure for system components
Compute tree failure probabilities based on height and wind speed
Compute power distribution system component failure probabilities
- Estimate resilience indices based on system performance
### Tree Height Assignment (tree_height_assignment.py)
Load raster data from Earth observation datasets
Assign tree height values to nodes in a power grid graph
Spatial analysis using KDTree for efficient nearest neighbor searches

## Installation

```sh
#Clone the repository
git clone https://github.com/username/power-distribution-resilience.git
cd power-distribution-resilience
```
```sh
# Create and activate a virtual environment using uv (optional but recommended)
python pip install uv
uv pip install
```
```sh
# Alternative: Create and activate a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
# Install dependencies
pip install -r requirements.txt
```

## Contributing
- Contributions are welcome! Please feel free to submit a Pull Request.
- Fork the repository
- Create your feature branch (git checkout -b feature/amazing-feature)
- Commit your changes (git commit -m 'Add some amazing feature')
- Push to the branch (git push origin feature/amazing-feature)
- Open a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
For questions or feedback, please open an issue in the repository.
