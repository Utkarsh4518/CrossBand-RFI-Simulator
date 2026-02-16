"""
EPFD regulatory compliance evaluation utilities for time-varying interference analysis.
"""

import numpy as np


def compute_epfd_time_series_dbw_m2_mhz(
    eirp_int_dbw: float,
    slant_range_km_array,
    off_axis_angle_deg_array,
    g_rx_max_db: float,
    theta_3db_deg: float,
    f_ghz: float,
    bandwidth_mhz: float,
) -> np.ndarray:
    """
    Compute EPFD time series using geometry-driven slant range and off-axis angle.

    For each time step:
        - Compute free-space path loss
        - Compute off-axis gain using S.1528 model
        - Compute EPFD using compute_epfd_dbw_m2_mhz()

    Returns:
        epfd_time_series_db (np.ndarray)
    """
    from rfi.equations_itu import (
        free_space_path_loss_db,
        compute_epfd_dbw_m2_mhz,
    )

    slant_range_km_array = np.asarray(slant_range_km_array)
    off_axis_angle_deg_array = np.asarray(off_axis_angle_deg_array)

    l_fs_int_db = free_space_path_loss_db(f_ghz, slant_range_km_array)

    theta_edge = 2.5 * theta_3db_deg
    if theta_3db_deg == 0:
        g_rx_off_axis_db = np.full_like(off_axis_angle_deg_array, g_rx_max_db, dtype=float)
    else:
        g_rx_off_axis_db = np.where(
            off_axis_angle_deg_array < theta_edge,
            g_rx_max_db - 12.0 * (off_axis_angle_deg_array / theta_3db_deg) ** 2,
            g_rx_max_db - 30.0,
        )

    epfd_time_series_db = compute_epfd_dbw_m2_mhz(
        eirp_int_dbw=eirp_int_dbw,
        g_rx_off_axis_db=g_rx_off_axis_db,
        l_fs_int_db=l_fs_int_db,
        bandwidth_mhz=bandwidth_mhz,
    )

    return np.asarray(epfd_time_series_db, dtype=float)


def compute_epfd_exceedance_probability(
    epfd_time_series_db,
    epfd_limit_db: float,
) -> float:
    """
    Returns percentage of time EPFD exceeds regulatory limit.
    """
    arr = np.array(epfd_time_series_db)
    if arr.size == 0:
        return 0.0

    return float(np.mean(arr > epfd_limit_db) * 100.0)


def classify_epfd_compliance(
    exceedance_percent: float,
    allowed_percent_threshold: float = 0.1,
) -> str:
    """
    Classify compliance status.

    If exceedance_percent <= allowed_percent_threshold:
        return "Compliant"
    elif exceedance_percent <= 5:
        return "Marginal"
    else:
        return "Non-Compliant"
    """
    if exceedance_percent <= allowed_percent_threshold:
        return "Compliant"
    if exceedance_percent <= 5.0:
        return "Marginal"
    return "Non-Compliant"
