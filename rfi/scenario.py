import numpy as np
from typing import Dict, Any, List

# Import equation-level models (pure physics, no scenarios)
from rfi.itu_models import (
    compute_thermal_noise_dbw,
    free_space_path_loss_db,
    compute_interference_power_dbw,
    compute_aggregate_interference_dbw,
    compute_epfd_dbw_m2_mhz,
    compute_carrier_to_interference_db,
    compute_snr_with_interference_db,
    compute_off_axis_gain_s1528_db,
    generate_log_normal_interference_samples_dbw,
    compute_time_fraction_exceeded,
)

# ---------------------------------------------------------------------
# Default constants
# These act as reference values if not explicitly specified
# ---------------------------------------------------------------------

DEFAULT_T_SYS_K = 250.0     # System noise temperature [K]
DEFAULT_BW_HZ = 1e6         # Receiver bandwidth [Hz]
DEFAULT_L_ATM_DB = 0.0      # Atmospheric attenuation [dB] (fixed to zero)
DEFAULT_L_OTHER_DB = 0.0    # Miscellaneous losses [dB]


# ---------------------------------------------------------------------
# Main scenario engine
# ---------------------------------------------------------------------

def run_multi_entry_rfi_scenario(
    band_params: Dict[str, Any],            # Victim link definition (one band)
    interferer_list: List[Dict[str, Any]],  # List of independent interferers
    time_sim_samples: int = 1000,
) -> Dict[str, Any]:
    """
    Runs a single aggregate RFI scenario for one victim link.

    A scenario is defined by:
    - one victim link (band-specific parameters)
    - one set of interfering transmitters

    The function computes:
    - baseline (interference-free) SNR
    - aggregate interference power
    - SNR degradation (ΔSNR)
    - optional statistical characterization of variability
    """

    # =============================================================
    # 1. Baseline carrier and noise computation
    #    (Interference-free reference state)
    # =============================================================

    # Extract basic victim-link parameters
    f_ghz = band_params["f_ghz"]   # Operating frequency
    d_km = band_params["d_km"]     # Link distance

    # Optional parameters with defaults
    T_sys_k = band_params.get("T_sys_k", DEFAULT_T_SYS_K)
    BW_hz = band_params.get("BW_Hz", DEFAULT_BW_HZ)
    L_atm_db = band_params.get("L_atm_db", DEFAULT_L_ATM_DB)
    L_other_db = band_params.get("L_other_db", DEFAULT_L_OTHER_DB)

    # Thermal noise power N = kTB
    N_dbw = compute_thermal_noise_dbw(T_sys_k, BW_hz)

    # Free-space path loss between desired transmitter and receiver
    L_fs_db = free_space_path_loss_db(f_ghz, d_km)

    # Received carrier power (link budget form)
    C_dbw = (
        band_params["EIRP_dbw"]
        + band_params["G_rx_db"]
        - L_fs_db
        - L_atm_db
        - L_other_db
    )

    # Baseline SNR (no interference)
    baseline_snr_db = C_dbw - N_dbw
    # This is the reference performance against which attenuation is measured

    # =============================================================
    # 2. Deterministic aggregate interference computation
    # =============================================================

    # Store individual interference contributions
    I_single_powers_dbw = []
    epfd_single_values_db = []

    # Process each interferer independently
    for i_params in interferer_list:

        # Path loss from interferer to victim receiver
        L_fs_int_db = free_space_path_loss_db(
            f_ghz, i_params["d_km"]
        )

        # Receive antenna discrimination for off-axis arrival
        g_rx_off_axis_db = compute_off_axis_gain_s1528_db(
            g_max=band_params["G_rx_db"],
            theta_deg=i_params["theta_off_axis_deg"],
            theta_3db=band_params["theta_3db"],
        )

        # Interference power from this interferer alone
        # (link-budget-style computation in dB)
        I_single_dbw = compute_interference_power_dbw(
            eirp_int_dbw=i_params["EIRP_int_dbw"],
            l_fs_int_db=L_fs_int_db,
            l_atm_db=L_atm_db,
            g_rx_off_axis_db=g_rx_off_axis_db,
            l_misc_db=i_params.get("L_misc_db", 0.0),
        )

        # Store single-entry interference contribution
        I_single_powers_dbw.append(I_single_dbw)

        # Compute EPFD (receiver-independent descriptor)
        epfd_single_db = compute_epfd_dbw_m2_mhz(
            eirp_int_dbw=i_params["EIRP_int_dbw"],
            g_rx_off_axis_db=g_rx_off_axis_db,
            l_fs_int_db=L_fs_int_db,
            bandwidth_mhz=BW_hz / 1e6,
        )
        epfd_single_values_db.append(epfd_single_db)

    # Aggregate all interferers by summing linear powers
    I_aggregate_dbw = compute_aggregate_interference_dbw(
        I_single_powers_dbw
    )

    # Aggregate EPFD by summing linear EPFD contributions
    epfd_aggregate_db = compute_aggregate_interference_dbw(
        epfd_single_values_db
    )

    # Compute degraded link performance
    C_I_db = compute_carrier_to_interference_db(C_dbw, I_aggregate_dbw)
    SNR_with_I_db = compute_snr_with_interference_db(
        C_dbw, N_dbw, I_aggregate_dbw
    )

    # RFI-induced attenuation
    SNR_loss_db = baseline_snr_db - SNR_with_I_db

    # =============================================================
    # 3. Statistical characterization (optional)
    # =============================================================

    # Average statistical parameters across interferers
    sigma_db = np.mean([
        i.get("sigma_db", 4.0) for i in interferer_list
    ])

    duty_cycle = np.mean([
        i.get("duty_cycle", 1.0) for i in interferer_list
    ])

    # Generate time-series samples of aggregate interference
    i_samples_dbw = generate_log_normal_interference_samples_dbw(
        mean_dbw=I_aggregate_dbw,
        std_dev_db=sigma_db,
        num_samples=time_sim_samples,
        duty_cycle=duty_cycle,
    )

    # Compute SNR degradation for each time sample
    snr_with_i_samples_db = np.array([
        compute_snr_with_interference_db(C_dbw, N_dbw, i_dbw)
        for i_dbw in i_samples_dbw
    ])

    snr_loss_samples_db = baseline_snr_db - snr_with_i_samples_db

    # =============================================================
    # 4. Assemble and return results
    # =============================================================

    return {
        # Deterministic metrics
        "Baseline SNR (dB)": baseline_snr_db,
        "I_Aggregate (dBW)": I_aggregate_dbw,
        "C/I_Aggregate (dB)": C_I_db,
        "SNR with I_Agg (dB)": SNR_with_I_db,
        "SNR Loss (dB)": SNR_loss_db,
        "epfd_Aggregate (dBW/m2/MHz)": epfd_aggregate_db,

        # Probabilistic descriptors
        "P(SNR Loss > 1 dB) (%)": compute_time_fraction_exceeded(
            snr_loss_samples_db, 1.0
        ),
        "P(SNR Loss > 3 dB) (%)": compute_time_fraction_exceeded(
            snr_loss_samples_db, 3.0
        ),
        "P(SNR Loss > 6 dB) (%)": compute_time_fraction_exceeded(
            snr_loss_samples_db, 6.0
        ),

        # Raw samples (used for CCDFs and heat maps)
        "SNR_Loss_Samples_dB": snr_loss_samples_db,
    }

# ---------------------------------------------------------------------
# Victim band scenario definitions
# ---------------------------------------------------------------------

DKM_GEO = 35786.0  # km

VICTIM_BANDS = {
    "S-band": {
        "f_ghz": 2.25,
        "d_km": DKM_GEO,
        "EIRP_dbw": 26.0,
        "G_rx_db": 30.0,
        "theta_3db": 2.0,
    },
    "X-band": {
        "f_ghz": 8.0,
        "d_km": DKM_GEO,
        "EIRP_dbw": 30.0,
        "G_rx_db": 35.0,
        "theta_3db": 1.5,
    },
    "Ku-band": {
        "f_ghz": 14.25,
        "d_km": DKM_GEO,
        "EIRP_dbw": 45.0,
        "G_rx_db": 40.0,
        "theta_3db": 1.0,
    },
    "K-band": {
        "f_ghz": 20.0,
        "d_km": DKM_GEO,
        "EIRP_dbw": 50.0,
        "G_rx_db": 45.0,
        "theta_3db": 0.8,
    },
    "Ka-band": {
        "f_ghz": 30.0,
        "d_km": DKM_GEO,
        "EIRP_dbw": 55.0,
        "G_rx_db": 50.0,
        "theta_3db": 0.6,
    },
}

# ---------------------------------------------------------------------
# Interference scenario definitions
# ---------------------------------------------------------------------

INTERFERENCE_SCENARIOS = {
    "Weak": [
        {
            "EIRP_int_dbw": 5.0,
            "d_km": 2000.0,
            "theta_off_axis_deg": 15.0,
            "sigma_db": 3.0,
            "duty_cycle": 1.0,
        }
    ],
    "Moderate": [
        {
            "EIRP_int_dbw": 15.0,
            "d_km": 1000.0,
            "theta_off_axis_deg": 7.5,
            "sigma_db": 4.0,
            "duty_cycle": 1.0,
        }
    ],
    "Strong": [
        {
            "EIRP_int_dbw": 25.0,
            "d_km": 500.0,
            "theta_off_axis_deg": 3.0,
            "sigma_db": 5.0,
            "duty_cycle": 1.0,
        }
    ],
}

ALL_SCENARIOS = {
    "bands": VICTIM_BANDS,
    "interference": INTERFERENCE_SCENARIOS,
}