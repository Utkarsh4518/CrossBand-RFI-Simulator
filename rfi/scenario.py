import numpy as np
from typing import Dict, Any, List

# Import equation-level models (pure physics, no scenarios)
from rfi.equations_itu import (
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
from rfi.geometry import generate_leo_interference_geometry
from rfi.compliance import (
    compute_epfd_time_series_dbw_m2_mhz,
    compute_epfd_exceedance_probability,
    classify_epfd_compliance,
)
from rfi.link_adaptation import (
    get_modulation_table,
    map_snr_to_spectral_efficiency,
    compute_throughput_bps,
    compute_throughput_degradation_percent,
)
from rfi.propagation import (
    compute_rain_attenuation_db,
    generate_rain_time_series,
)

# Default constants (fixed parameters)

DEFAULT_BW_HZ = 1e6          # 1 MHz receiver bandwidth (all bands)
DEFAULT_L_ATM_DB = 0.0       # Atmospheric attenuation fixed to 0 dB
DEFAULT_L_OTHER_DB = 0.0     # Miscellaneous / polarization losses fixed to 0 dB

# GEO reference distance (victim link and interferer geometry)
DKM_GEO = 36000.0  # km


# Main scenario engine

def run_multi_entry_rfi_scenario(
    band_params: Dict[str, Any],
    interferer_list: List[Dict[str, Any]],
    time_sim_samples: int = 1000,
) -> Dict[str, Any]:

    # 1. Baseline carrier and noise computation

    f_ghz = band_params["f_ghz"]
    d_km = band_params["d_km"]

    T_sys_k = band_params["T_sys_k"]
    BW_hz = band_params.get("BW_Hz", DEFAULT_BW_HZ) #Use the band’s bandwidth if defined, otherwise default to 1 MHz

    # Thermal noise
    N_dbw = compute_thermal_noise_dbw(T_sys_k, BW_hz)

    # Free-space path loss (victim link)
    L_fs_db = free_space_path_loss_db(f_ghz, d_km)

    # Received carrier power
    C_dbw = (
        band_params["EIRP_dbw"]
        + band_params["G_rx_db"]
        - L_fs_db
        - DEFAULT_L_ATM_DB
        - DEFAULT_L_OTHER_DB
    )

    baseline_snr_db = C_dbw - N_dbw

    # 2. Deterministic aggregate interference

    I_single_powers_dbw = []
    epfd_single_values_db = []

    for i_params in interferer_list:

	# free-space path loss from interferer to victim
        L_fs_int_db = free_space_path_loss_db(
            f_ghz, i_params["d_km"]
        )
	
	# how much the victim antenna hears the interferer
        g_rx_off_axis_db = compute_off_axis_gain_s1528_db(
            g_max=band_params["G_rx_db"],
            theta_deg=i_params["theta_off_axis_deg"],
            theta_3db=band_params["theta_3db"],
        )

	# EIRP Computation
        I_single_dbw = compute_interference_power_dbw(
            eirp_int_dbw=i_params["EIRP_int_dbw"],
            l_fs_int_db=L_fs_int_db,
            l_atm_db=DEFAULT_L_ATM_DB,
            g_rx_off_axis_db=g_rx_off_axis_db,
            l_misc_db=DEFAULT_L_OTHER_DB,
        )

	# Add it to the list
        I_single_powers_dbw.append(I_single_dbw)
	
	# equivalent power flux density computation
        epfd_single_db = compute_epfd_dbw_m2_mhz(
            eirp_int_dbw=i_params["EIRP_int_dbw"],
            g_rx_off_axis_db=g_rx_off_axis_db,
            l_fs_int_db=L_fs_int_db,
            bandwidth_mhz=BW_hz / 1e6,
        )

        epfd_single_values_db.append(epfd_single_db)
    
    # total interference power (dbW)
    I_aggregate_dbw = compute_aggregate_interference_dbw(
        I_single_powers_dbw
    )

    epfd_aggregate_db = compute_aggregate_interference_dbw(
        epfd_single_values_db
    )

    C_I_db = compute_carrier_to_interference_db(
        C_dbw, I_aggregate_dbw
    )

    # computation of SNR with Interference 
    SNR_with_I_db = compute_snr_with_interference_db(
        C_dbw, N_dbw, I_aggregate_dbw
    )

    #degraded SNR
    SNR_loss_db = baseline_snr_db - SNR_with_I_db 

    # =============================================================
    # 3. Statistical characterization
    # =============================================================

    # standard deviation(dB) for interference fluctutation
    sigma_db = (
        np.mean([i.get("sigma_db", 4.0) for i in interferer_list])
        if interferer_list else 4.0
    )
    duty_cycle = (
        np.mean([i.get("duty_cycle", 1.0) for i in interferer_list])
        if interferer_list else 1.0
    )

    i_samples_dbw = generate_log_normal_interference_samples_dbw(
        mean_dbw=I_aggregate_dbw,
        std_dev_db=sigma_db,
        num_samples=time_sim_samples,
        duty_cycle=duty_cycle,
    )

    snr_with_i_samples_db = np.array([
        compute_snr_with_interference_db(C_dbw, N_dbw, i_dbw)
        for i_dbw in i_samples_dbw
    ])

    snr_loss_samples_db = baseline_snr_db - snr_with_i_samples_db

    # 4. Results

    return {
        "Baseline SNR (dB)": baseline_snr_db,
        "I_Aggregate (dBW)": I_aggregate_dbw,
        "C/I_Aggregate (dB)": C_I_db,
        "SNR with I_Agg (dB)": SNR_with_I_db,
        "SNR Loss (dB)": SNR_loss_db,
        "epfd_Aggregate (dBW/m2/MHz)": epfd_aggregate_db,

        "P(SNR Loss > 1 dB) (%)": compute_time_fraction_exceeded(
            snr_loss_samples_db, 1.0
        ),
        "P(SNR Loss > 3 dB) (%)": compute_time_fraction_exceeded(
            snr_loss_samples_db, 3.0
        ),
        "P(SNR Loss > 6 dB) (%)": compute_time_fraction_exceeded(
            snr_loss_samples_db, 6.0
        ),

        "SNR_Loss_Samples_dB": snr_loss_samples_db,
    }


def run_dynamic_geometry_rfi_scenario(
    band_params: Dict[str, Any],
    leo_altitude_km: float,
    duration_sec: float,
    dt_sec: float,
    interferer_eirp_dbw: float,
    rain_rate_mm_per_hr: float = 0.0,
    rain_probability: float = 0.0,
    effective_rain_path_km: float = 5.0,
) -> Dict[str, Any]:
    """
    Compute time-varying interference using geometry-based slant range
    and off-axis angle from LEO orbit. Purely deterministic, no log-normal statistics.
    """
    time_sec, d_km_array, theta_deg_array = generate_leo_interference_geometry(
        leo_altitude_km=leo_altitude_km,
        duration_sec=duration_sec,
        dt_sec=dt_sec,
    )

    f_ghz = band_params["f_ghz"]
    d_km = band_params["d_km"]
    T_sys_k = band_params["T_sys_k"]
    BW_hz = band_params.get("BW_Hz", DEFAULT_BW_HZ)

    N_dbw = compute_thermal_noise_dbw(T_sys_k, BW_hz)
    L_fs_db = free_space_path_loss_db(f_ghz, d_km)
    C_dbw = (
        band_params["EIRP_dbw"]
        + band_params["G_rx_db"]
        - L_fs_db
        - DEFAULT_L_ATM_DB
        - DEFAULT_L_OTHER_DB
    )
    baseline_snr_db = C_dbw - N_dbw

    rain_rate_series = generate_rain_time_series(
        duration_sec,
        dt_sec,
        rain_rate_mm_per_hr,
        rain_probability,
    )

    A_rain_db = compute_rain_attenuation_db(
        f_ghz,
        rain_rate_series,
        effective_rain_path_km,
    )
    C_time_step_dbw = C_dbw - A_rain_db

    L_fs_int_db = free_space_path_loss_db(f_ghz, d_km_array)

    theta_3db = band_params["theta_3db"]
    g_max = band_params["G_rx_db"]
    theta_edge = 2.5 * theta_3db
    if theta_3db == 0:
        g_rx_off_axis_db = np.full_like(theta_deg_array, g_max, dtype=float)
    else:
        g_rx_off_axis_db = np.where(
            theta_deg_array < theta_edge,
            g_max - 12.0 * (theta_deg_array / theta_3db) ** 2,
            g_max - 30.0,
        )

    I_time_series_dbw = compute_interference_power_dbw(
        eirp_int_dbw=interferer_eirp_dbw,
        l_fs_int_db=L_fs_int_db,
        l_atm_db=DEFAULT_L_ATM_DB,
        g_rx_off_axis_db=g_rx_off_axis_db,
        l_misc_db=DEFAULT_L_OTHER_DB,
    )

    c_lin = 10 ** (C_dbw / 10.0)
    n_lin = 10 ** (N_dbw / 10.0)
    i_lin = 10 ** (I_time_series_dbw / 10.0)
    denom = n_lin + i_lin
    snr_lin = np.where(denom > 0, c_lin / denom, 1e-30)
    SNR_time_series_db = 10.0 * np.log10(snr_lin)

    SNR_loss_time_series_db = baseline_snr_db - SNR_time_series_db

    c_joint_lin = 10 ** (C_time_step_dbw / 10.0)
    denom_joint = n_lin + i_lin
    snr_joint_lin = np.where(denom_joint > 0, c_joint_lin / denom_joint, 1e-30)
    SNR_joint_time_series_db = 10.0 * np.log10(snr_joint_lin)
    SNR_joint_loss_time_series_db = baseline_snr_db - SNR_joint_time_series_db

    spectral_efficiency_joint = map_snr_to_spectral_efficiency(SNR_joint_time_series_db)
    outage_joint_percent = (
        np.sum(spectral_efficiency_joint == 0) / len(spectral_efficiency_joint) * 100.0
    )

    spectral_efficiency_degraded = map_snr_to_spectral_efficiency(SNR_time_series_db)
    throughput_degraded_bps = compute_throughput_bps(
        spectral_efficiency_degraded,
        BW_hz,
    )

    baseline_snr_array = np.full_like(SNR_time_series_db, baseline_snr_db)
    spectral_efficiency_baseline = map_snr_to_spectral_efficiency(baseline_snr_array)
    throughput_baseline_bps = compute_throughput_bps(
        spectral_efficiency_baseline,
        BW_hz,
    )

    throughput_degradation_percent = compute_throughput_degradation_percent(
        throughput_baseline_bps,
        throughput_degraded_bps,
    )

    modulation_table = get_modulation_table()
    modulation_distribution = {}
    total_samples = len(spectral_efficiency_degraded)
    for entry in modulation_table:
        eff = entry["spectral_efficiency_bpshz"]
        name = entry["name"]
        percent = (
            np.sum(spectral_efficiency_degraded == eff)
            / total_samples
            * 100.0
        )
        modulation_distribution[name] = percent
    # Outage condition (no modulation selected)
    outage_percent = (
        np.sum(spectral_efficiency_degraded == 0)
        / total_samples
        * 100.0
    )
    modulation_distribution["Outage"] = outage_percent

    link_availability_percent = 100.0 - outage_percent

    epfd_time_series_db = compute_epfd_time_series_dbw_m2_mhz(
        eirp_int_dbw=interferer_eirp_dbw,
        slant_range_km_array=d_km_array,
        off_axis_angle_deg_array=theta_deg_array,
        g_rx_max_db=band_params["G_rx_db"],
        theta_3db_deg=band_params["theta_3db"],
        f_ghz=band_params["f_ghz"],
        bandwidth_mhz=BW_hz / 1e6,
    )

    epfd_limit_db = -150.0  # placeholder regulatory threshold
    epfd_exceed_percent = compute_epfd_exceedance_probability(
        epfd_time_series_db,
        epfd_limit_db,
    )
    compliance_status = classify_epfd_compliance(epfd_exceed_percent)

    return {
        "Time (s)": time_sec,
        "Slant Range (km)": d_km_array,
        "Off-axis Angle (deg)": theta_deg_array,
        "I_time_series (dBW)": I_time_series_dbw,
        "SNR_time_series (dB)": SNR_time_series_db,
        "SNR_loss_time_series (dB)": SNR_loss_time_series_db,
        "P(SNR Loss > 1 dB) (%)": compute_time_fraction_exceeded(
            SNR_loss_time_series_db, 1.0
        ),
        "P(SNR Loss > 3 dB) (%)": compute_time_fraction_exceeded(
            SNR_loss_time_series_db, 3.0
        ),
        "P(SNR Loss > 6 dB) (%)": compute_time_fraction_exceeded(
            SNR_loss_time_series_db, 6.0
        ),
        "EPFD_time_series (dBW/m2/MHz)": epfd_time_series_db,
        "EPFD_limit (dBW/m2/MHz)": epfd_limit_db,
        "EPFD_exceedance (%)": epfd_exceed_percent,
        "EPFD_compliance_status": compliance_status,
        "Spectral_efficiency_degraded (bps/Hz)": spectral_efficiency_degraded,
        "Spectral_efficiency_baseline (bps/Hz)": spectral_efficiency_baseline,
        "Throughput_degraded (bps)": throughput_degraded_bps,
        "Throughput_baseline (bps)": throughput_baseline_bps,
        "Throughput_degradation (%)": throughput_degradation_percent,
        "Modulation_distribution (%)": modulation_distribution,
        "Link_availability (%)": link_availability_percent,
        "Rain_rate_time_series (mm/hr)": rain_rate_series,
        "SNR_joint_time_series (dB)": SNR_joint_time_series_db,
        "SNR_joint_loss_time_series (dB)": SNR_joint_loss_time_series_db,
        "Joint_outage (%)": outage_joint_percent,
    }


# Victim band scenario definitions (Assumptions)

VICTIM_BANDS = {
    "S-band": {
        "f_ghz": 3.0,
        "d_km": DKM_GEO,
        "EIRP_dbw": 26.0,
        "G_rx_db": 32.0,
        "theta_3db": 2.5,
        "T_sys_k": 250.0,
        "BW_Hz": DEFAULT_BW_HZ,
    },
    "X-band": {
        "f_ghz": 8.4,
        "d_km": DKM_GEO,
        "EIRP_dbw": 30.0,
        "G_rx_db": 38.0,
        "theta_3db": 1.5,
        "T_sys_k": 300.0,
        "BW_Hz": DEFAULT_BW_HZ,
    },
    "Ku-band": {
        "f_ghz": 14.0,
        "d_km": DKM_GEO,
        "EIRP_dbw": 45.0,
        "G_rx_db": 42.0,
        "theta_3db": 1.2,
        "T_sys_k": 350.0,
        "BW_Hz": DEFAULT_BW_HZ,
    },
    "K-band": {
        "f_ghz": 22.0,
        "d_km": DKM_GEO,
        "EIRP_dbw": 50.0,
        "G_rx_db": 45.0,
        "theta_3db": 1.0,
        "T_sys_k": 400.0,
        "BW_Hz": DEFAULT_BW_HZ,
    },
    "Ka-band": {
        "f_ghz": 32.0,
        "d_km": DKM_GEO,
        "EIRP_dbw": 55.0,
        "G_rx_db": 48.0,
        "theta_3db": 0.8,
        "T_sys_k": 450.0,
        "BW_Hz": DEFAULT_BW_HZ,
    },
}



# Interference strength scenarios

INTERFERENCE_SCENARIOS = {
    "Weak": [
        {
            "EIRP_int_dbw": 10.0, #10 Watts
            "d_km": DKM_GEO,
            "theta_off_axis_deg": 10.0,
            "sigma_db": 6.0,
            "duty_cycle": 1,
        }
    ],
    "Moderate": [
        {
            "EIRP_int_dbw": 20.0, #100 Watts
            "d_km": DKM_GEO,
            "theta_off_axis_deg": 10,
            "sigma_db": 6.0,
            "duty_cycle": 1,
        }
    ],
    "Strong": [
        {
            "EIRP_int_dbw": 30.0, #1000 Watts
            "d_km": DKM_GEO,
            "theta_off_axis_deg": 10,
            "sigma_db": 6.0,
            "duty_cycle": 1.0,
        }
    ],
}

ALL_SCENARIOS = {
    "bands": VICTIM_BANDS,
    "interference": INTERFERENCE_SCENARIOS,
}
