"""Module for handling primary helper functions in the GCPB project."""

import time
from typing import Optional, Tuple

import pandas as pd
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from geopy.geocoders import Nominatim


def read_csv(file_path: str) -> Optional[pd.DataFrame]:
    """Read data from a CSV file.

    Args:
        file_path (str): Path to the CSV file

    Returns:
        Optional[pd.DataFrame]: DataFrame containing the CSV data,
        or None if error
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except (IOError, pd.errors.EmptyDataError) as e:
        print(f"Error reading CSV file: {str(e)}")
        return None


def geocode_location(
    location: str, geolocator: Nominatim, retries: int = 3, delay: int = 1
) -> Tuple[str, str]:
    """Geocode a location string using Nominatim.

    Args:
        location (str): Location string to geocode
        geolocator: Initialized Nominatim geolocator
        retries (int): Number of retries if geocoding fails
        delay (int): Delay between retries in seconds

    Returns:
        tuple: (latitude, longitude) or ('Not found', 'Not found')
           if geocoding fails.
    """
    if not location or pd.isna(location):
        return "Not found", "Not found"

    for attempt in range(retries):
        try:
            geocode_result = geolocator.geocode(location)
            if geocode_result is None:
                return "Not found", "Not found"
            return geocode_result.latitude, geocode_result.longitude
        except (GeocoderTimedOut, GeocoderUnavailable) as e:
            print(
                f"Geocoding error for '{location}': {str(e)}. "
                f"Attempt {attempt+1}/{retries}"
            )
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                return "Not found", "Not found"

    # Ensure we always return something even if the loop completes
    return "Not found", "Not found"


def process_locations_csv(input_file: str, output_file: str) -> bool:
    """Process a CSV file to add latitude and longitude columns.

    Args:
        input_file (str): Path to the input CSV file
        output_file (str): Path to the output CSV file

    Returns:
        bool: True if processing was successful, False otherwise
    """
    # Read CSV file
    df = read_csv(input_file)
    if df is None:
        return False

    # Check if 'Location' column exists
    if "Location" not in df.columns:
        print("Error: CSV file does not contain a 'Location' column")
        return False

    # Initialize Nominatim geolocator
    # Adding a meaningful user agent is important for Nominatim usage policy
    geolocator = Nominatim(user_agent="location_geocoder_script")

    # Create new columns for latitude and longitude
    df["Latitude"] = "Not found"
    df["Longitude"] = "Not found"

    # Process each location
    for index, row in df.iterrows():
        print(f"Geocoding {index+1}/{len(df)}: {row['Location']}")
        lat, lon = geocode_location(row["Location"], geolocator)
        df.at[index, "Latitude"] = lat
        df.at[index, "Longitude"] = lon

        # Add a small delay to comply with Nominatim usage policy
        time.sleep(1)

    # Write to new CSV file
    try:
        df.to_csv(output_file, index=False)
        print(f"Successfully wrote geocoded data to {output_file}")
        return True
    except IOError as e:
        print(f"Error writing to output CSV file: {str(e)}")
        return False


def main() -> None:
    """Execute the geocoding process with example input/output files."""
    input_file = "input.csv"  # Replace with your input file path
    output_file = "output_geocoded.csv"  # Replace with desired output path

    process_locations_csv(input_file, output_file)


if __name__ == "__main__":
    main()
