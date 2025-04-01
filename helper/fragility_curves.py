"""
This script contains the code for the fragility curves in the paper.

Paper:Hou, G. and Muraleetharan, K.K., 2023.
Modeling the resilience of power distribution systems subjected to
extreme winds considering tree failures: An integrated framework.
International Journal of Disaster Risk Science, 14(2), pp.194-208.
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import lognorm

# pylint: disable=R1710, R0913, R0917, R0914, R0912, R0915
# pylint: disable=W0715, W0640, W0613, W0012, W0511, W0632


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def logistic_function(x, a, b, c):
    """
    Logistic function for fragility curves.

    Args:
        x: Input variable
        a: Intercept
        b: Slope
        c: Shape parameter

    Tree height and wind speed are the input variables.
    """
    logger.info("Applying logistic function: in array of shape %s", x.shape)
    return 1 / (1 + np.exp(-(a + b * x[0] + c * x[1])))


# TODO: expand to accomodate input vector. add data error handling
def logistic_function_vector(x, a, b, c):
    """
    Logistic function for fragility curves.

    Accomodates more than wind speed and tree height
    (for example: wind angle, geometrical data, intensity measures).

    Args:
        x: Input array of shape (n_samples, 2) where
           each row is a pair of input variables
        a: Intercept
        b: Slope
        c: Shape parameter

    Returns:
        np.ndarray: Array of logistic function results for each input pair
    """
    # Ensure x is a numpy array for vectorized operations
    x = np.asarray(x)

    # Check if x has the correct shape
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("Input x must be a 2D array shape: (n_samples, 2)")

    # Apply the logistic function to each row in the input array
    logger.info(
        "Applying logistic function: in array of shape %s",
        x.shape,
    )
    return 1 / (1 + np.exp(-(a + b * x[:, 0] + c * x[:, 1])))


def fit_fragility_curve(x, y):
    """
    Fit logistic regression-based fragility curve using curve fitting.

    Args:
        X: Input variables
        y: Output variable
    """
    popt, _ = curve_fit(logistic_function, x.T, y, maxfev=5000)
    logger.info("Fitted parameters: %s", popt)
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
    logger.info(
        "Computing tree failure probability for height %s and wind speed %s",
        height,
        wind_speed,
    )
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
    logger.info(
        "Computing PDS component failure probability for "
        "w. speed %s and w. angle %s",
        wind_speed,
        wind_angle,
    )
    return logistic_function((wind_speed, wind_angle), a, b, c)


def compute_tree_fall_load(tree_height):
    """
    Estimate tree fall load using power-law function from the paper.

    Args:
        tree_height: Tree height
    """
    logger.info("Estimating tree fall load for tree height %s", tree_height)
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
    logger.info(
        "Estimating resilience index for performance curve "
        "of shape %s and time horizon %s",
        performance_curve.shape,
        time_horizon,
    )
    return np.trapz(performance_curve, dx=1) / time_horizon


def analyze_wind_fragility(
    wind_data,
    fragility_params,
    components=None,
    output_file=None,
    angle_sensitivity=False,
    plot_raw=True,
    confidence_interval=False,
    storm_threshold=50,
):
    """
    Analyze wind fragility.

    Analyze wind data and generate fragility curves for
    power infrastructure components.

    Parameters:
    -----------
    wind_data : pandas.DataFrame
        DataFrame containing 'Timestamp', 'Wind_speed',
        and 'Wind_angle' columns

    fragility_params : dict
        Dictionary with component types as keys and their
        fragility parameters as values. Each component should
        contain parameters for a lognormal distribution
        (e.g., {'wooden_poles': {'median': 70, 'beta': 0.3},
        'substations': {'median': 85, 'beta': 0.25}})
        OR a custom function that takes wind speed and angle
        and returns probability

    components : list, optional
        List of component types to analyze.
        If None, all components in fragility_params are used.

    output_file : str, optional
        If provided, saves the plot to this file path

    angle_sensitivity : bool, default=False
        If True, consider wind angle in the damage probability calculation

    plot_raw : bool, default=True
        If True, plot raw data points in addition to fitted curves

    confidence_interval : bool, default=False
        If True, plot 95% confidence intervals around the fitted curves

    storm_threshold : int, default=50
        Wind speed threshold (mph) for highlighting storm conditions

    Returns:
    --------
    tuple
        (figure, damage_probabilities)
        figure: matplotlib Figure object
        damage_probabilities: DataFrame with calculated damage
        probabilities for each timestamp
    """
    # Ensure column names are standardized
    wind_data = wind_data.copy()
    wind_data.columns = [
        col.strip().lower().replace(" ", "_") for col in wind_data.columns
    ]
    logger.info(
        "Analyzing wind fragility: Standardized column names: %s",
        wind_data.columns,
    )

    # Validate input data
    required_cols = ["timestamp", "wind_speed_mph", "wind_angle_deg"]
    for col in required_cols:
        if col not in wind_data.columns:
            raise ValueError(
                "Required column '%s' not found in wind_data",
                col,
            )

    # Use all components if none specified
    if components is None:
        components = list(fragility_params.keys())

    logger.info("Analyzing wind fragility: Using components: %s", components)
    # Initialize figure
    plt.figure(figsize=(12, 8))

    # Colors for different components
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
    ]

    # Initialize damage probability dataframe
    vars_to_keep = ["timestamp", "wind_speed_mph", "wind_angle_deg"]
    damage_probs = wind_data[vars_to_keep].copy()

    # Wind speed range for plotting curves
    wind_speeds = np.linspace(0, max(wind_data["wind_speed_mph"]) * 1.2, 100)

    # Process each component
    for i, component in enumerate(components):
        color = colors[i % len(colors)]

        if component not in fragility_params:
            logger.warning(
                "Warning: No fragility parameters found for %s",
                component,
            )
            continue

        params = fragility_params[component]

        # Calculate damage probabilities based on wind data - speed and angle.
        if callable(params):
            # Custom probability function
            if angle_sensitivity:
                target_col = f"{component}_prob"
                damage_probs[target_col] = wind_data.apply(
                    lambda row: params(
                        row["wind_speed_mph"], row["wind_angle_deg"]
                    ),  # pylint: disable=E501
                    axis=1,
                )
                logger.info(
                    "Damage probabilities for %s calculated using "
                    "custom function with angle sensitivity.",
                    component,
                )
            else:
                target_col = f"{component}_prob"
                damage_probs[target_col] = wind_data["wind_speed_mph"].apply(
                    params
                )  # pylint: disable=E501
                logger.info(
                    "Damage probabilities for %s calculated using "
                    "custom function without angle sensitivity.",
                    component,
                )

        else:
            # Use lognormal distribution
            median = params.get("median", 70)
            beta = params.get("beta", 0.3)

            # Convert from median to mu for lognormal
            mu = np.log(median)
            sigma = beta
            logger.info(
                "Using lognormal distribution with mu %s and sigma %s.",
                mu,
                sigma,
            )

            if angle_sensitivity:
                # Test: adjust median based on angle (can be customized)
                def angle_factor(angle):
                    """
                    Adjust median based on angle.

                    Case: structures more vulnerable to perpendicular winds
                    Adjust based on your specific infrastructure
                    vulnerabilities.
                    """
                    return 1 - 0.2 * np.abs(np.sin(np.radians(angle)))

                damage_probs[f"{component}_prob"] = wind_data.apply(
                    lambda row: lognorm.cdf(
                        row["wind_speed_mph"],
                        sigma,
                        scale=np.exp(mu) * angle_factor(row["wind_angle_deg"]),
                    ),
                    axis=1,
                )
                logger.info(
                    "Damage probabilities for %s calculated using lognormal "
                    "distribution with angle sensitivity.",
                    component,
                )
            else:
                damage_probs[f"{component}_prob"] = lognorm.cdf(
                    wind_data["wind_speed_mph"], sigma, scale=np.exp(mu)
                )
                logger.info(
                    "Damage probabilities for %s calculated using lognormal "
                    "distribution without angle sensitivity.",
                    component,
                )
        # Plot fitted curve
        if callable(params):
            if angle_sensitivity:
                # Use average angle effect for plotting
                avg_angle = wind_data["wind_angle_deg"].mean()
                curve_probs = [params(ws, avg_angle) for ws in wind_speeds]
            else:
                curve_probs = [params(ws) for ws in wind_speeds]
            plt.plot(
                wind_speeds,
                curve_probs,
                "-",
                color=color,
                linewidth=2.5,
                label=f"{component.replace('_', ' ').title()}",
            )
        else:
            # Plot lognormal curve
            plt.plot(
                wind_speeds,
                lognorm.cdf(wind_speeds, sigma, scale=np.exp(mu)),
                "-",
                color=color,
                linewidth=2.5,
                label=f"{component.replace('_', ' ').title()}",
            )

        #  TO-DO: fix this

        # # Plot raw data points if requested
        # if plot_raw:
        #     # Get unique wind speeds and their corresponding
        #     # avg probabilities
        #     unique_speeds = (
        #         wind_data.groupby("wind_speed_mph")[f"{component}_prob"]
        #         .mean()
        #         .reset_index()
        #     )
        #     plt.scatter(
        #         unique_speeds["wind_speed_mph"],
        #         unique_speeds[f"{component}_prob"],
        #         color=color,
        #         alpha=0.6,
        #         s=30,
        #     )

        # Add confidence intervals if requested
        if confidence_interval and not callable(params):
            # Calculate 95% confidence interval
            lower = lognorm.cdf(
                wind_speeds,
                sigma * 1.1,
                scale=np.exp(mu) * 0.9,
            )
            upper = lognorm.cdf(
                wind_speeds,
                sigma * 0.9,
                scale=np.exp(mu) * 1.1,
            )
            plt.fill_between(wind_speeds, lower, upper, color=color, alpha=0.2)

    # Add storm threshold line
    plt.axvline(
        x=storm_threshold,
        color="red",
        linestyle="--",
        label=f"Storm Threshold ({storm_threshold} mph)",
    )

    # Add grid, legend, and labels
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc="best", fontsize=12)
    plt.xlabel("Wind Speed (mph)", fontsize=14)
    plt.ylabel("Probability of Damage/Failure", fontsize=14)
    plt.title(
        "Fragility Curves for Power Infrastructure Components",
        fontsize=16,
    )
    plt.xlim(0, max(wind_data["wind_speed_mph"]) * 1.2)
    plt.ylim(0, 1.05)

    # TO-DO: Add annotations for key points

    # # Add annotations for key points
    # # Find the wind speed at which each component has 50% failure probability
    # for i, component in enumerate(components):
    #     if f"{component}_prob" in damage_probs.columns:
    #         # Interpolate to find wind speed at 50% probability
    #         wind_50p = np.interp(
    #             0.5,
    #             sorted(damage_probs[f"{component}_prob"].unique()),
    #             sorted(
    #                 damage_probs.loc[
    #                     damage_probs[f"{component}_prob"].isin(
    #                         sorted(damage_probs[f"{component}_prob"].unique())
    #                     ),
    #                     "wind_speed",
    #                 ].unique()
    #             ),
    #         )

    #         if not np.isnan(wind_50p):
    #             plt.annotate(
    #                 f"{wind_50p:.1f} mph",
    #                 xy=(wind_50p, 0.5),
    #                 xytext=(wind_50p + 5, 0.55),
    #                 arrowprops=dict(
    #                     arrowstyle="->",
    #                     connectionstyle="arc3",
    #                     color=colors[i % len(colors)],
    #                 ),
    #                 fontsize=10,
    #                 color=colors[i % len(colors)],
    #             )

    # Tight layout
    plt.tight_layout()

    # Save if requested
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")

    # Calculate additional statistics
    for component in components:
        if f"{component}_prob" in damage_probs.columns:
            # Flag timestamps with high damage probability
            damage_probs[f"{component}_high_risk"] = (
                damage_probs[f"{component}_prob"] > 0.5
            )

            # Calculate maximum probability during the period
            max_prob = damage_probs[f"{component}_prob"].max()
            max_prob_time = damage_probs.loc[
                damage_probs[f"{component}_prob"] == max_prob, "timestamp"
            ].iloc[0]
            print(
                f"{component.replace('_', ' ').title()}: Max probability"
                " {max_prob:.2f} at {max_prob_time}"
            )
            logger.info(
                "Max prob time for %s: %s",
                component,
                max_prob_time,
            )

    return plt.gcf(), damage_probs
