"""
Map SNR to modulation scheme and compute spectral efficiency and throughput.
"""

import numpy as np
from typing import List, Dict, Any


def get_modulation_table() -> List[Dict[str, Any]]:
    """
    Returns a list of dict entries describing a simple AMC table.

    Each entry contains:
        - "name": modulation name
        - "snr_min_db": minimum required SNR
        - "spectral_efficiency_bpshz": bits per second per Hz
    """
    return [
        {"name": "QPSK", "snr_min_db": 0.0, "spectral_efficiency_bpshz": 2.0},
        {"name": "8PSK", "snr_min_db": 5.0, "spectral_efficiency_bpshz": 3.0},
        {"name": "16APSK", "snr_min_db": 10.0, "spectral_efficiency_bpshz": 4.0},
        {"name": "32APSK", "snr_min_db": 15.0, "spectral_efficiency_bpshz": 5.0},
    ]


def map_snr_to_spectral_efficiency(snr_db_array) -> np.ndarray:
    """
    For each SNR value:
        - Select highest modulation whose snr_min_db <= SNR
        - If SNR < lowest threshold, spectral efficiency = 0

    Returns:
        spectral_efficiency_array (np.ndarray)
    """
    snr_db_array = np.asarray(snr_db_array, dtype=float)
    table = get_modulation_table()
    out = np.zeros_like(snr_db_array, dtype=float)

    for i in range(len(table) - 1, -1, -1):
        mask = snr_db_array >= table[i]["snr_min_db"]
        out[mask] = table[i]["spectral_efficiency_bpshz"]

    return out


def compute_throughput_bps(
    spectral_efficiency_array,
    bandwidth_hz: float,
) -> np.ndarray:
    """
    Throughput = spectral_efficiency * bandwidth

    Returns throughput array in bits per second.
    """
    spectral_efficiency_array = np.asarray(spectral_efficiency_array, dtype=float)
    return spectral_efficiency_array * bandwidth_hz


def compute_throughput_degradation_percent(
    baseline_throughput_array,
    degraded_throughput_array,
) -> float:
    """
    Returns average percent degradation:

    100 * (mean(baseline - degraded) / mean(baseline))
    """
    baseline_throughput_array = np.asarray(baseline_throughput_array, dtype=float)
    degraded_throughput_array = np.asarray(degraded_throughput_array, dtype=float)
    mean_baseline = np.mean(baseline_throughput_array)
    if mean_baseline <= 0:
        return 0.0
    return float(100.0 * np.mean(baseline_throughput_array - degraded_throughput_array) / mean_baseline)
