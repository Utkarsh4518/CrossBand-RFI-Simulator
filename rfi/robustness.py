"""
Compute cross-band RFI robustness index.
"""

from typing import Dict, List, Tuple, Optional


def compute_rfi_robustness_index(
    throughput_degradation_percent: float,
    link_availability_percent: float,
    epfd_exceed_percent: float,
    joint_outage_percent: float,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Compute RFI Robustness Index (RRI).

    Higher RRI = more robust.

    Default equal weights.
    """
    if weights is None:
        weights = {
            "throughput": 0.25,
            "availability": 0.25,
            "epfd": 0.25,
            "outage": 0.25,
        }

    t_term = 1.0 - throughput_degradation_percent / 100.0
    a_term = link_availability_percent / 100.0
    e_term = 1.0 - epfd_exceed_percent / 100.0
    o_term = 1.0 - joint_outage_percent / 100.0

    rri = (
        weights["throughput"] * t_term
        + weights["availability"] * a_term
        + weights["epfd"] * e_term
        + weights["outage"] * o_term
    )

    return rri


def rank_bands_by_robustness(
    band_rri_dict: Dict[str, float],
) -> List[Tuple[str, float]]:
    """
    Sort bands by descending RRI.

    Input:
        {"S-band": 0.82, "X-band": 0.76, ...}

    Return:
        sorted list of tuples
    """
    return sorted(
        band_rri_dict.items(),
        key=lambda x: x[1],
        reverse=True,
    )
