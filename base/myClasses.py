"""
A set of helper classes created to aid data analysis and visualisation of events from the "Adverse Weather Scenarios for
Future Electricity Systems" dataset. The development was part of the CIReN project at the University of Sussex, under the
supervision of Dr. Spyros Skarvelis-Kazakos.

Main author: Petros Zantis
May - July 2023
"""

# pylint: skip-file
# flake8: noqa

from datetime import date, timedelta

import cartopy
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

plt.rcParams["figure.figsize"] = (15, 7)
plt.rcParams["font.size"] = 15


class Event:
    """An instance of this class represents a specific event from the Adverse Weather Scenarios Dataset

    event_type (str) : the type of event (e.g. summer/winter wind drought)
    location (str) : the location of the event (europe/uk)
    extremity (str) : the extremity of the event (return period)
    duration_severity (str) : duration/severity used to classify the event
    global_warming_level (str) : the corresponding global warming level
    event_no (str) : event number
    """

    def __init__(
        self,
        event_type,
        location,
        extremity,
        duration_severity,
        global_warming_level,
        event_no,
    ):

        self.event_type = event_type
        self.location = location
        self.extremity = extremity
        self.duration_severity = duration_severity
        self.global_warming_level = global_warming_level
        self.event_no = event_no
        self.weather_vars = []

    def get_event_details(self):

        details = {
            "Event type": self.event_type,
            "Location": self.location,
            "Extremity": self.extremity,
            "Duration/Severity": self.duration_severity,
            "Global Warming Level": self.global_warming_level,
            "Event no.": self.event_no,
        }

        return details

    def add_weather_variable(self, weather_var):
        """Add a weather variable (e.g. wind, surface temp, ssr) into the list for this event

        weather_var (WeatherVariable) : an instance of the WeatherVariable class - see below
        """
        if isinstance(weather_var, WeatherVariable):
            self.weather_vars.append(weather_var)
        else:
            print("The supplied argument is not of WeatherVariable type.")


class WeatherVariable:
    """An instance of this class represents a weather variable (e.g. wind, surface temp, ssr)

    var (str) : the variable as represented in the CEDA data
    name (str) : the variable's name in a more readable format
    units (str) : the variable's units
    cmap (matplotlib.colors.Colormap) : chosen colormap for this variable
    """

    def __init__(self, var, name, units, cmap, clr):

        self.var = var
        self.name = name
        self.units = units
        self.cmap = cmap
        self.clr = clr

    def get_attributes(self):

        details = {
            "Variable": self.var,
            "Name": self.name,
            "Units": self.units,
            "Color map": self.cmap,
            "Plot color": self.clr,
        }

        return details

    def set_data(self, gridded_lons, gridded_lats, lons, lats, var_data, times):
        """Sets the data extracted from the .nc file, to be used for visualisation and analysis"""

        self.gridded_lons = gridded_lons
        self.gridded_lats = gridded_lats
        self.lons = lons
        self.lats = lats
        self.var_data = var_data
        self.times = times

    def convert_to_C(self):
        """Converts Temp data from Kelvin to Celsius"""

        self.var_data = self.var_data - 273

    def view_snapshot(self, title="Example Snapshot", frame=0):
        """A function to view a snapshot of the data on a 2D map, of the desired frame

        title (str) : the title of the snapshot - usually the event details
        frame (int) : the desired frame to view - usually the day of the year
        """

        # setup plot

        plt.rcParams["font.size"] = 15
        fig = plt.figure(figsize=(13, 13), facecolor="w")
        ax1 = fig.add_subplot(111, projection=ccrs.PlateCarree())
        cmap = self.cmap
        bounds = np.linspace(-10, 20, 31)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        cs = plt.pcolor(
            self.gridded_lons,
            self.gridded_lats,
            self.var_data[frame, :, :],
            cmap=cmap,
            transform=ccrs.PlateCarree(),
        )  # deleted norm=norm for now
        cb = plt.colorbar(cs, ax=ax1, shrink=0.7)
        cb.ax.set_ylabel(self.name + " " + self.units)
        ax1.set_xlabel("Longitude")
        ax1.set_ylabel("Latitude")
        ax1.set_title(title)
        ax1.coastlines(resolution="50m")
        ax1.add_feature(cartopy.feature.BORDERS)
        plt.show()

    def view_animation(self, title="Example Animation", frames=10):
        """A function which animates the data over the 2D map up to the desired frame, to dynamically observe the event

        title (str) : the title for the animation - usually the event details
        frames (int) : how many frames to animate - usually bounded by 365, but recommended low values to avoid crashing
        """
        plt.rcParams["font.size"] = 15
        fig = plt.figure(figsize=(13, 13))
        ax1 = fig.add_subplot(111, projection=ccrs.PlateCarree())
        cmap = self.cmap
        bounds = np.linspace(self.var_data.min(), self.var_data.max(), 101)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        cs = ax1.pcolor(
            self.gridded_lons,
            self.gridded_lats,
            self.var_data[0, :, :],
            cmap=cmap,
            transform=ccrs.PlateCarree(),
            norm=norm,
        )
        cb = plt.colorbar(cs, ax=ax1, shrink=0.8)
        cb.ax.set_ylabel(self.name + " " + self.units)

        def animation_func(i):
            ax1.cla()  # Clear the previous frame
            cs = ax1.pcolor(
                self.gridded_lons,
                self.gridded_lats,
                self.var_data[i, :, :],
                cmap=cmap,
                transform=ccrs.PlateCarree(),
                norm=norm,
            )
            ax1.set_title(title)
            ax1.coastlines(resolution="50m")
            ax1.add_feature(cartopy.feature.BORDERS)

        anim = animation.FuncAnimation(
            fig,
            animation_func,
            frames=frames,
            interval=100,
            repeat_delay=2000,
            blit=True,
        )

        return anim

    def find_location(self, lat, lon, printLoc=True):
        """A function which locates the exact grid point within the dataset, given a desired location (lat, lon)
        If the desired values are out of range, it defaults to the closest ones within range

        lat, lon (floats) : desired Latitude and Longitude
        printLoc (Bool) : Boolean to indicate whether to print the actual closest location within the dataset
        """

        lons_tol = np.abs(self.lons[1] - self.lons[0]) / 2
        lats_tol = np.abs(self.lats[1] - self.lats[0]) / 2

        new_lat_idx = np.where(np.isclose(self.lats, lat, atol=lats_tol))[0]
        new_lon_idx = np.where(np.isclose(self.lons, lon, atol=lons_tol))[0]

        if len(new_lat_idx) == 0:
            print("Desired latitude ({}) is out of range".format(lat))

            if lat < min(self.lats):
                new_lat_idx = 0
                new_lat = self.lats[new_lat_idx]
            elif lat > max(self.lats):
                new_lat_idx = -1
                new_lat = self.lats[new_lat_idx]
        else:
            new_lat = self.lats[new_lat_idx][0]

        if len(new_lon_idx) == 0:
            print("Desired longitude ({}) is out of range".format(lon))

            if lon < min(self.lons):
                new_lon_idx = 0
                new_lon = self.lons[new_lon_idx]
            elif lon > max(self.lons):
                new_lon_idx = -1
                new_lon = self.lons[new_lon_idx]
        else:
            new_lon = self.lons[new_lon_idx][0]

        if printLoc:
            print("The closest location is (Lat: {}, Lon: {})".format(new_lat, new_lon))

        return new_lat_idx, new_lon_idx, new_lat, new_lon

    def set_thresholds(self, min_t, max_t):
        """Sets lower and upper thresholds for the specific weather variable

        min_t, max_t (floats) : lower and upper thresholds respectively
        """

        self.min_t = min_t
        self.max_t = max_t

    def get_thresholds(self):

        return self.min_t, self.max_t

    def local_detector(self, grid_var_data, dates):
        """A function which scans the variable data and detects when the values are above or below the set thresholds

        grid_var_data (list of floats) : variable data of a specific grid point of interest
        dates (list of dates) : the corresponding dates of the above data
        """

        dates = np.array(dates)

        if self.var == "t2m":
            units = "oC"  # hardcoded this for now, fix later
        elif self.var == "wind_speed":
            units = "m/s"

        print(
            "\n{} above max threshold of {:.2f} {}:\n".format(
                self.name, self.max_t, units
            )
        )
        for high_w in list(
            zip(
                grid_var_data[grid_var_data > self.max_t],
                dates[np.where(grid_var_data > self.max_t)[0]],
            )
        ):
            print("{}: {:.3f} {} on {}".format(self.name, high_w[0], units, high_w[1]))

        print(
            "Total high day counts: {}".format(
                len(grid_var_data[grid_var_data > self.max_t])
            )
        )

        print(
            "\n{} below min threshold of {:.2f} {}:\n".format(
                self.name, self.min_t, units
            )
        )
        for low_w in list(
            zip(
                grid_var_data[grid_var_data < self.min_t],
                dates[np.where(grid_var_data < self.min_t)[0]],
            )
        ):
            print("{}: {:.3f} {} on {}".format(self.name, low_w[0], units, low_w[1]))

        print(
            "Total low day counts: {}".format(
                len(grid_var_data[grid_var_data < self.min_t])
            )
        )

    def get_var_time_data(self, latitude, longitude, printLoc=True):
        """A function which returns the converted datetimes and the weather variable data, at the selected location (lat, lon)

        latitude, longitude (floats) : desired Latitude and Longitude
        """

        # Turn time into a date (time is hours since 1970-01-01 00:00:00)
        start = date(1970, 1, 1)
        date_conv = []
        for i in range(0, len(self.times)):
            delta_temp = timedelta(hours=int(self.times[i]))
            date_conv.append(start + delta_temp)

        lat_idx, lon_idx, lat, lon = self.find_location(latitude, longitude, printLoc)

        var_data = self.var_data[:, lat_idx, lon_idx]

        return date_conv, var_data, lat, lon

    def time_series(
        self, latitude, longitude, title="Example Time Series", thresholds=False
    ):
        """A function which plots the time series of the quantity of interest, at the selected location (lat, lon)

        latitude, longitude (floats) : desired Latitude and Longitude
        thresholds (Bool) : a Boolean to set or ignore the thresholds
        title (str) : the title of the time series - usually the event details & lat, lon
        """

        date_conv, var_data, lat, lon = self.get_var_time_data(
            latitude, longitude, printLoc=True
        )

        # Time series plot in one grid cell
        plt.rcParams["font.size"] = 15
        fig = plt.figure(figsize=(15, 7), facecolor="w", dpi=130)
        ax1 = fig.add_subplot(111)

        ax1.plot(date_conv, var_data, color=self.clr)

        ax1.set_title("{}\n Lat: {:.3f}, Lon: {:.3f}".format(title, lat, lon))
        ax1.set_xlabel("Date")
        ax1.set_ylabel(self.name + " " + self.units)

        if thresholds:

            ax1.axhline(
                self.max_t,
                linestyle="--",
                color="r",
                label="Max threshold: {:.2f} {}".format(self.max_t, self.units),
            )
            ax1.axhline(
                self.min_t,
                linestyle="--",
                color="g",
                label="Min threshold: {:.2f} {}".format(self.min_t, self.units),
            )
            ax1.legend()

            self.local_detector(var_data, date_conv)

        ax1.grid(axis="y", linestyle="--", alpha=0.7)
        plt.show()

    def histogram(
        self, latitude, longitude, bins=10, title="Example Histogram", thresholds=False
    ):
        """A function which plots a histogram of the quantity of interest, at the selected location (lat, lon)

        latitude, longitude (floats) : desired Latitude and Longitude
        thresholds (Bool) : a Boolean to set or ignore the thresholds
        title (str) : the title of the histogram - usually the event details & lat, lon
        """

        # Turn time into a date (time is hours since 1970-01-01 00:00:00)
        start = date(1970, 1, 1)
        date_conv = []
        for i in range(0, len(self.times)):
            delta_temp = timedelta(hours=int(self.times[i]))
            date_conv.append(start + delta_temp)

        lat_idx, lon_idx, lat, lon = self.find_location(latitude, longitude)

        plt.rcParams["font.size"] = 15
        fig = plt.figure(figsize=(15, 7), facecolor="w", dpi=130)
        ax1 = fig.add_subplot(111)

        ax1.hist(
            self.var_data[:, lat_idx, lon_idx], bins=bins, color=self.clr, rwidth=0.95
        )

        ax1.set_title("{}\n Lat: {:.3f}, Lon: {:.3f}".format(title, lat, lon))
        ax1.set_ylabel("Day counts")
        ax1.set_xlabel(self.name + " " + self.units)

        if thresholds:

            ax1.axvline(
                self.max_t,
                linestyle="--",
                color="r",
                label="Max threshold: {:.2f} {}".format(self.max_t, self.units),
            )
            ax1.axvline(
                self.min_t,
                linestyle="--",
                color="g",
                label="Min threshold: {:.2f} {}".format(self.min_t, self.units),
            )
            ax1.legend()

            self.local_detector(self.var_data[:, lat_idx, lon_idx], date_conv)

        ax1.grid(axis="y", linestyle="--", alpha=0.7)
        plt.show()


class WindTurbine:
    """An instance of this class represents a wind turbine, with its specs given as arguments

    cutin (float): cut-in speed in m/s (power is ~ 0 below this value)
    cutout (float): cut-out speed in m/s (power is 0 above this value)
    ros (float): rated output speed in m/s
    rop (float): rated output power in W (at the rated output speed)
    """

    def __init__(self, cutin=4, cutout=25, ros=13, rop=1e6):

        assert (
            0 < cutin < ros < cutout
        ), "Make sure the values are correct! (0 < Cut-in < Rated speed < Cut-out)"

        self.cutin = cutin
        self.cutout = cutout
        self.ros = ros
        self.rop = rop

    def wind_power(self, v):
        """A function to calculate the expected power output in Watts of the specific wind turbine
        This function uses a rescaled sigmoid function to model the power curve of the wind turbine,
        except when the wind speed is higher than the cutout speed, where the output power will be 0

        v (float) : input wind speed in m/s
        """

        x0 = (
            self.ros + self.cutin
        ) / 2  # centre of the sigmoid - assumption holds for abs(ros-cutin)>5
        k = np.sqrt((np.pi**2) / (x0 - self.cutin))  # curve steepness

        wp = self.rop / (
            1 + np.exp(-k * (v - x0))
        )  # basically a rescaled sigmoid function

        # condition for turbine protection
        return np.where(v < self.cutout, wp, 0)

    def wind_power_old(self, v, rho=1.3, A=452, cp=16 / 27):
        """A function to calculate wind turbine power in Watts, based on the basic physics equation
        *** replaced this with the sigmoid function above - more intuitive modelling ***

        v (float) : input wind speed in m/s
        rho (float) : density of the air in kg/m^3
        A (float) : cross-sectional area of the wind turbine in m^2
        cp (float) : power coefficient (maximum is 16/27 according to Betz's law)
        """
        # A = np.pi*(r**2)
        wp = cp * 0.5 * rho * A * (v**3)

        return wp

    def power_curve(self, wind_speeds=np.linspace(0, 30, 500)):
        """A function which plots the power curve of the specific wind turbine over the input wind speeds

        wind_speeds (array of floats) : desired range of wind speeds in m/s to plot
        """
        fig = plt.figure(figsize=(15, 7), facecolor="w", dpi=130)
        plt.rcParams["font.size"] = 15
        ax1 = fig.add_subplot(111)

        ax1.plot(wind_speeds, self.wind_power(wind_speeds) / 1e6, color="teal")

        ax1.set_title("Wind Turbine Power Curve")
        ax1.set_xlabel("Wind speed in m/s")
        ax1.set_ylabel("Power in MW")

        ax1.axvline(
            self.cutin,
            linestyle="--",
            color="g",
            label="Cut-in speed:  {:.2f} m/s".format(self.cutin),
        )
        ax1.axvline(
            self.ros,
            linestyle="--",
            color="b",
            label="Rated speed:  {:.2f} m/s".format(self.ros),
        )
        ax1.axvline(
            self.cutout,
            linestyle="--",
            color="r",
            label="Cut-out speed: {:.2f} m/s".format(self.cutout),
        )
        ax1.legend()
        ax1.grid(axis="y", linestyle="--", alpha=0.7)

        plt.show()

    def power_time_series(
        self, date_conv, wind_speeds, lat, lon, title="Example Power Time Series"
    ):
        """A function to plot the power generated over time, using the wind data from CEDA

        wind_speeds (array of floats) : desired range of wind speeds in m/s to plot
        """
        plt.rcParams["font.size"] = 15
        fig = plt.figure(figsize=(15, 7), facecolor="w", dpi=130)
        ax1 = fig.add_subplot(111)

        ax1.scatter(
            date_conv, self.wind_power(wind_speeds) / 1e6, color="teal", marker="x"
        )
        ax1.plot(date_conv, self.wind_power(wind_speeds) / 1e6, color="teal", alpha=0.7)

        ax1.set_title("{}\n Lat: {:.3f}, Lon: {:.3f}".format(title, lat, lon))
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Power generated in MW")

        ax1.grid(axis="y", linestyle="--", alpha=0.7)

        plt.show()

    def power_histogram(
        self, wind_speeds, lat, lon, bins=15, title="Example Power Histogram"
    ):
        """A function to plot a histogram of the powers generated, using the wind data from CEDA

        wind_speeds (array of floats) : desired range of wind speeds in m/s to plot
        """
        plt.rcParams["font.size"] = 15

        fig = plt.figure(figsize=(15, 7), facecolor="w", dpi=130)
        ax1 = fig.add_subplot(111)

        ax1.hist(
            self.wind_power(wind_speeds) / 1e6, bins=bins, color="teal", rwidth=0.95
        )

        ax1.set_title("{}\n Lat: {:.3f}, Lon: {:.3f}".format(title, lat, lon))
        ax1.set_xlabel("Power generated in MW")
        ax1.set_ylabel("Day counts")

        ax1.grid(axis="y", linestyle="--", alpha=0.7)
        plt.show()
