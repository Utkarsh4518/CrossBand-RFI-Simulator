import numpy as np


# 1) noise ------------------------------------------------------------

def compute_thermal_noise_dbw(T_sys_k, BW_hz):
    # TODO: check units again...
    kb = 1.380649e-23  # J/K
    n = kb * T_sys_k * BW_hz
    if n <= 0:
        return -300
    x = 10 * np.log10(n)
    return x


def free_space_path_loss_db(f_ghz, d_km):
    # ITU-R P.525 FSPL
    # f in GHz, d in km
    f_mhz = f_ghz * 1e3
    # probably ok like this
    fspl = 32.45 + 20 * np.log10(f_mhz) + 20 * np.log10(d_km)
    return fspl


# 2) antenna pattern ---------------------------------------------------

# Speed of light (m/s) for parabolic antenna physics
C_LIGHT_M_S = 2.998e8


def compute_parabolic_antenna_params(
    f_ghz,
    diameter_m,
    efficiency=0.65,
):
    """
    Compute gain and 3-dB beamwidth for a parabolic aperture.

    Physics:
        λ = c / (f * 1e9)
        G = η * (π D / λ)^2
        G_dBi = 10 * log10(G)
        theta_3db_deg = 70 * λ / D

    Args:
        f_ghz: frequency in GHz
        diameter_m: aperture diameter in meters
        efficiency: aperture efficiency η (default 0.65)

    Returns:
        G_rx_db (dBi), theta_3db (degrees).
        Returns (None, None) if inputs are invalid (D <= 0, f <= 0, η <= 0)
        so caller can fall back to band-specific constants.
    """
    try:
        f_ghz = float(f_ghz)
        diameter_m = float(diameter_m)
        efficiency = float(efficiency)
    except (TypeError, ValueError):
        return (None, None)

    if diameter_m <= 0 or f_ghz <= 0 or efficiency <= 0:
        return (None, None)

    # Wavelength in m: λ = c / (f_Hz) with f_Hz = f_ghz * 1e9
    lam_m = C_LIGHT_M_S / (f_ghz * 1e9)
    # Gain (linear): G = η * (π D / λ)^2
    g_lin = efficiency * (np.pi * diameter_m / lam_m) ** 2
    if g_lin <= 0:
        return (None, None)
    G_rx_db = 10.0 * np.log10(g_lin)
    # 3-dB beamwidth (deg): theta_3db ≈ 70 * λ / D
    theta_3db = 70.0 * lam_m / diameter_m
    return (float(G_rx_db), float(theta_3db))


def compute_off_axis_gain_s1528_db(g_max, theta_deg, theta_3db):
    # quick&dirty S.1528 kind of pattern
    if theta_3db == 0:
        # avoid div by zero
        return g_max

    theta_edge = 2.5 * theta_3db  # might tweak

    if theta_deg < theta_edge:
        # main lobe
        tmp = (theta_deg / theta_3db) ** 2
        g = g_max - 12.0 * tmp
    else:
        # side lobe floor
        g = g_max - 30.0

    return g


# 3) interference stuff ------------------------------------------------

def compute_interference_power_dbw(
        eirp_int_dbw,
        l_fs_int_db,
        l_atm_db,
        g_rx_off_axis_db,
        l_misc_db=0.0):

    # I = EIRP - Lfs - Latm + G_rx - Lmisc
    i = eirp_int_dbw - l_fs_int_db - l_atm_db + g_rx_off_axis_db - l_misc_db
    return i


def compute_aggregate_interference_dbw(i_powers_dbw):
    # sum interference in lin
    if i_powers_dbw is None or len(i_powers_dbw) == 0:
        return -300.0

    # make sure it's iterable
    vals = []
    for x in i_powers_dbw:
        vals.append(10 ** (x / 10.0))

    s = float(np.sum(vals))
    if s <= 1e-30:
        return -300.0

    return 10.0 * np.log10(s)


def compute_carrier_to_interference_db(c_dbw, i_dbw):
    # C/I
    return c_dbw - i_dbw


def compute_snr_with_interference_db(c_dbw, n_dbw, i_dbw):
    # SNRI = 10log10( C / (N + I) )
    c_lin = 10 ** (c_dbw / 10.0)
    n_lin = 10 ** (n_dbw / 10.0)
    i_lin = 10 ** (i_dbw / 10.0)

    denom = n_lin + i_lin
    if denom <= 0:
        # just in case
        return 999

    snr_lin = c_lin / denom
    snr_db = 10.0 * np.log10(snr_lin)
    return snr_db


# 4) EPFD --------------------------------------------------------------

def compute_epfd_dbw_m2_mhz(
        eirp_int_dbw,
        g_rx_off_axis_db,
        l_fs_int_db,
        bandwidth_mhz):

    if bandwidth_mhz <= 0:
        # arbitrary floor
        return -999.0

    # EPFD = EIRP - Lfs + G_rx - 10log10(B_MHz)
    epfd = eirp_int_dbw - l_fs_int_db + g_rx_off_axis_db \
           - 10.0 * np.log10(bandwidth_mhz)
    return epfd


# 5) stats / time-fraction ---------------------------------------------

def compute_time_fraction_exceeded(data_samples_db, threshold_db):
    # data_samples_db can be list, np array, whatever
    arr = np.array(data_samples_db)
    if arr.size == 0:
        return 0.0
    # fraction above threshold in %
    frac = np.mean(arr > threshold_db) * 100.0
    return frac


def generate_log_normal_interference_samples_dbw(
        mean_dbw,
        std_dev_db,
        num_samples,
        duty_cycle=1.0):

    # Gaussian in dB, then optionally puncture with -300 dBW
    if num_samples <= 0:
        return np.array([])

    s = np.random.normal(mean_dbw, std_dev_db, int(num_samples))

    if duty_cycle < 1.0:
        # crude duty-cycle model
        m = np.random.rand(s.size) < duty_cycle
        s = np.where(m, s, -300.0)

    return s


# 6) geometric sweep ---------------------------------------------------

def generate_geometric_sweep(max_theta_deg, min_theta_deg, num_steps=100):
    # some kind of smooth sweep, cosine type
    if num_steps <= 1:
        return np.array([min_theta_deg])

    t = np.linspace(-1.0, 1.0, num_steps)
    # 0..1..0 shape
    sweep = 0.5 * (1.0 + np.cos(np.pi * t))
    out = min_theta_deg + sweep * (max_theta_deg - min_theta_deg)
    return out
