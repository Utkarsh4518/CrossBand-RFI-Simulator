"""
Deterministic circular LEO satellite geometry model for RFI analysis.
Generates time-varying distance and off-axis angle between an interfering
LEO satellite and a GEO ground station victim link.
"""

import numpy as np
from typing import Tuple

# Earth and orbit constants (km, km^3/s^2)
RE_KM = 6378.0
GEO_ALTITUDE_KM = 35786.0
MU_KM3_S2 = 398600.0


def generate_leo_interference_geometry(
    leo_altitude_km: float,
    duration_sec: float,
    dt_sec: float,
    initial_phase_rad: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute LEO–GEO geometry: time, slant range, and off-axis angle at GEO.

    Assumes equatorial circular LEO orbit and GEO fixed above longitude 0.
    GEO antenna boresight points toward Earth center.

    Args:
        leo_altitude_km: LEO orbit altitude in km.
        duration_sec: Simulation duration in seconds.
        dt_sec: Time step in seconds.
        initial_phase_rad: Initial orbital phase of LEO in radians.

    Returns:
        time_array_sec: Time samples (s).
        slant_range_km: LEO–GEO slant range (km).
        off_axis_angle_deg: Off-axis angle at GEO receiver (degrees).
    """
    time_array_sec = np.arange(0.0, duration_sec, dt_sec)

    r_leo_km = RE_KM + leo_altitude_km
    omega_rad_s = np.sqrt(MU_KM3_S2 / (r_leo_km ** 3))

    phase = omega_rad_s * time_array_sec + initial_phase_rad
    x_leo = r_leo_km * np.cos(phase)
    y_leo = r_leo_km * np.sin(phase)

    x_geo = RE_KM + GEO_ALTITUDE_KM
    y_geo = 0.0

    dx = x_leo - x_geo
    dy = y_leo - y_geo
    slant_range_km = np.sqrt(dx * dx + dy * dy)

    # GEO boresight: from GEO to Earth center = (-x_geo, -y_geo)
    # Vector from GEO to LEO = (dx, dy)
    # cos(off_axis) = dot / (|boresight| * |to_leo|) = (-x_geo*dx - y_geo*dy) / (x_geo * d)
    # With y_geo = 0: cos(off_axis) = -(x_leo - x_geo) / d = -dx / d
    d_safe = np.maximum(slant_range_km, 1e-12)
    cos_off_axis = np.clip(-dx / d_safe, -1.0, 1.0)
    off_axis_angle_rad = np.arccos(cos_off_axis)
    off_axis_angle_deg = np.degrees(off_axis_angle_rad)

    return time_array_sec, slant_range_km, off_axis_angle_deg
