import numpy as np
from typing import List

# ---------------------------------------------------------------------
# 1. Link Budget & Noise Helpers (ITU-R P.525)
# ---------------------------------------------------------------------

def compute_thermal_noise_dbw(T_sys_k: float, BW_hz: float) -> float:
    """
    Thermal noise power:
        N = 10*log10(k_B * T_sys * B)
    """
    k_boltzmann = 1.380649e-23  # J/K
    n_watts = k_boltzmann * T_sys_k * BW_hz
    return 10 * np.log10(n_watts)

#T_sys_K = system noise temperature of the receiver, in Kelvin. It represents how noisy the receiver is.

def free_space_path_loss_db(f_ghz: float, d_km: float) -> float:
    """
    ITU-R P.525 free-space path loss:
        L_fs = 32.45 + 20*log10(f_MHz) + 20*log10(d_km)
    """
    f_mhz = f_ghz * 1000.0
    return 32.45 + 20 * np.log10(f_mhz) + 20 * np.log10(d_km)


# ---------------------------------------------------------------------
# 2. Receive Antenna Pattern (ITU-R S.1528-type)
# ---------------------------------------------------------------------

def compute_off_axis_gain_s1528_db(
    g_max: float,
    theta_deg: float,
    theta_3db: float,
) -> float:
    """
    Simplified ITU-R S.1528 receive antenna off-axis gain model.

    Main lobe:
        G(θ) = G_max - 12 * (θ / θ_3dB)^2
    Side-lobe floor:
        G(θ) = G_max - 30 dB
    """
    theta_edge = 2.5 * theta_3db

    if theta_deg < theta_edge:
        return g_max - 12.0 * (theta_deg / theta_3db) ** 2
    else:
        return g_max - 30.0


# ---------------------------------------------------------------------
# 3. Interference Modeling (ITU-R S.1325 methodology)
# ---------------------------------------------------------------------

def compute_interference_power_dbw(
    eirp_int_dbw: float,
    l_fs_int_db: float,
    l_atm_db: float,
    g_rx_off_axis_db: float,
    l_misc_db: float = 0.0,
) -> float:
    """
    Single-entry interference power:
        I_i = EIRP_i - L_fs - L_atm + G_rx(θ_i) - L_misc
    """
    return (
        eirp_int_dbw
        - l_fs_int_db
        - l_atm_db
        + g_rx_off_axis_db
        - l_misc_db
    )


def compute_aggregate_interference_dbw(i_powers_dbw: List[float]) -> float:
    """
    Aggregate interference power:
        I_agg = 10*log10( sum_i 10^(I_i/10) )
    """
    i_lin = [10 ** (i_dbw / 10) for i_dbw in i_powers_dbw]        #list of interference (linear) power in Watts
    i_sum = np.sum(i_lin)

    if i_sum <= 1e-30:
        return -300.0

    return 10 * np.log10(i_sum)


def compute_carrier_to_interference_db(c_dbw: float, i_dbw: float) -> float:
    """
    Carrier-to-interference ratio:
        C/I = C - I_agg
    """
    return c_dbw - i_dbw


def compute_snr_with_interference_db(
    c_dbw: float,
    n_dbw: float,
    i_dbw: float,
) -> float:
    """
    Signal-to-noise ratio with interference:
        SNRI = 10*log10( C / (N + I_agg) )
    """
    c_lin = 10 ** (c_dbw / 10) #linear power
    n_lin = 10 ** (n_dbw / 10) #in watts
    i_lin = 10 ** (i_dbw / 10) #

    snr_lin = c_lin / (n_lin + i_lin)
    return 10 * np.log10(snr_lin)


# ---------------------------------------------------------------------
# 4. Equivalent Power Flux Density (EPFD) — PDF Eq. (12)
# ---------------------------------------------------------------------

def compute_epfd_dbw_m2_mhz(
    eirp_int_dbw: float,
    g_rx_off_axis_db: float,
    l_fs_int_db: float,
    bandwidth_mhz: float,
) -> float:
    """
    Equivalent Power Flux Density (EPFD), per Eq. (12):

        EPFD = EIRP_i - L_fs(f,d) + G_rx(θ) - 10*log10(B_MHz)

    Note:
    Geometric spreading is fully captured by L_fs (ITU-R P.525).
    """
    return (
        eirp_int_dbw
        - l_fs_int_db
        + g_rx_off_axis_db
        - 10 * np.log10(bandwidth_mhz)
    )


# ---------------------------------------------------------------------
# 5. Statistical / Time-Fraction Metrics (SA.1157-style)
# ---------------------------------------------------------------------

def compute_time_fraction_exceeded(              #fraction of samples exceeding the threshold
    data_samples_db: np.ndarray,
    threshold_db: float,
) -> float:
    """
    Percentage of samples exceeding a given threshold.
    """
    return 100.0 * np.mean(data_samples_db > threshold_db)  #Tells the %age of exceedence


def generate_log_normal_interference_samples_dbw(       #to introduce variability
    mean_dbw: float,
    std_dev_db: float,
    num_samples: int,
    duty_cycle: float = 1.0,
) -> np.ndarray:
    """
    Log-normal interference model (Gaussian in dB).
    """
    samples = np.random.normal(mean_dbw, std_dev_db, num_samples) #Generates a time series of interference levels around the mean

    if duty_cycle < 1.0:
        mask = np.random.rand(num_samples) < duty_cycle
        samples = np.where(mask, samples, -300.0)

    return samples


# ---------------------------------------------------------------------
# 6. Geometric Sweep Helper (Dynamic Off-Axis Geometry)
# ---------------------------------------------------------------------

def generate_geometric_sweep(
    max_theta_deg: float,
    min_theta_deg: float,
    num_steps: int = 100,
) -> np.ndarray:
    """
    Smooth off-axis angle sweep for dynamic pass simulations.
    """
    t = np.linspace(-1.0, 1.0, num_steps)
    sweep = 0.5 * (1.0 + np.cos(np.pi * t))
    return min_theta_deg + sweep * (max_theta_deg - min_theta_deg)
