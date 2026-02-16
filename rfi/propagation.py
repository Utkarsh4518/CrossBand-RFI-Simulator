"""
Simplified rain attenuation model for Earth-to-space links.
"""

import numpy as np


def compute_specific_attenuation_db_per_km(f_ghz, rain_rate_mm_per_hr):
    """
    Simplified frequency-dependent specific attenuation model.

    gamma = k * (rain_rate ^ alpha)

    Bands:
        f < 5 GHz:   k=0.0001, alpha=1.0
        5–15 GHz:    k=0.01,   alpha=1.1
        15–25 GHz:   k=0.05,   alpha=1.2
        >25 GHz:     k=0.15,   alpha=1.3

    Returns gamma in dB/km.
    """
    f_ghz = np.asarray(f_ghz, dtype=float)
    rain_rate_mm_per_hr = np.asarray(rain_rate_mm_per_hr, dtype=float)
    rain_rate = np.maximum(rain_rate_mm_per_hr, 0.0)

    k = np.where(
        f_ghz < 5,
        0.0001,
        np.where(
            f_ghz < 15,
            0.01,
            np.where(f_ghz < 25, 0.05, 0.15),
        ),
    )
    alpha = np.where(
        f_ghz < 5,
        1.0,
        np.where(
            f_ghz < 15,
            1.1,
            np.where(f_ghz < 25, 1.2, 1.3),
        ),
    )
    gamma = k * (rain_rate ** alpha)
    return gamma


def compute_rain_attenuation_db(
    f_ghz,
    rain_rate_mm_per_hr,
    effective_path_km,
):
    """
    Rain attenuation: A_rain = gamma * effective_path

    Returns attenuation in dB.
    """
    gamma = compute_specific_attenuation_db_per_km(f_ghz, rain_rate_mm_per_hr)
    effective_path_km = np.asarray(effective_path_km, dtype=float)
    return gamma * effective_path_km


def generate_rain_time_series(
    duration_sec,
    dt_sec,
    rain_rate_mm_per_hr,
    rain_probability=0.2,
):
    """
    Generate deterministic pseudo-random rain occurrence.

    With probability rain_probability, rain occurs (constant rate); else no rain (0).
    One draw for the whole duration; time series is constant.

    Returns:
        rain_rate_time_series (np.ndarray), in mm/hr.
    """
    time_array = np.arange(0.0, duration_sec, dt_sec)
    n = len(time_array)
    rng = np.random.default_rng(seed=42)
    rain_occurs = rng.random() < rain_probability
    rate = float(rain_rate_mm_per_hr) if rain_occurs else 0.0
    return np.full(n, rate, dtype=float)
