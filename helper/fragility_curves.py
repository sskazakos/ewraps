"""
This script contains the code for the fragility curves in the paper.

Paper:Hou, G. and Muraleetharan, K.K., 2023.
Modeling the resilience of power distribution systems subjected to
extreme winds considering tree failures: An integrated framework.
International Journal of Disaster Risk Science, 14(2), pp.194-208.
"""

import numpy as np
from scipy.optimize import curve_fit

# pylint: disable=W0632


def logistic_function(x, a, b, c):
    """
    Logistic function for fragility curves.

    Args:
        x: Input variable
        a: Intercept
        b: Slope
        c: Shape parameter
    """
    return 1 / (1 + np.exp(-(a + b * x[0] + c * x[1])))


def fit_fragility_curve(x, y):
    """
    Fit logistic regression-based fragility curve using curve fitting.

    Args:
        X: Input variables
        y: Output variable
    """
    popt, _ = curve_fit(logistic_function, x.T, y, maxfev=5000)
    return popt


def compute_tree_failure_probability(height, wind_speed, fragility_params):
    """
    Compute probability of tree failure (stem breakage or uprooting).

    Args:
        height: Tree height
        wind_speed: Wind speed
        fragility_params: Fragility parameters
    """
    a, b, c = fragility_params
    return logistic_function((height, wind_speed), a, b, c)


def compute_pds_component_failure(wind_speed, wind_angle, fragility_params):
    """
    Compute probability of PDS component failure due to wind load.

    Args:
        wind_speed: Wind speed
        wind_angle: Wind angle
        fragility_params: Fragility parameters
    """
    a, b, c = fragility_params
    return logistic_function((wind_speed, wind_angle), a, b, c)


def compute_tree_fall_load(tree_height):
    """
    Estimate tree fall load using power-law function from the paper.

    Args:
        tree_height: Tree height
    """
    return 2.008 * tree_height**3.076  # Example for stem breakage


def estimate_resilience_index(performance_curve, time_horizon):
    """
    Estimate RI.

    Compute resilience index based on system performance
    over time.

    Args:
        performance_curve: System performance curve
        time_horizon: Time horizon
    """
    return np.trapz(performance_curve, dx=1) / time_horizon
