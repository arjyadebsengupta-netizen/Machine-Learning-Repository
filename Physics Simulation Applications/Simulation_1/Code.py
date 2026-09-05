# ============================================================
# CELL 1 — IMPORTS
# ============================================================

import os
import random
import time
import copy

import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import expm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

# ============================================================
# CELL 2 — GLOBAL CONFIGURATION
# ============================================================

CONFIG = {
    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------
    "train_samples": 40,
    "test_samples": 10,

    # Five independent random seeds
    "seeds": [
        1234,
        2345,
        3456,
        4567,
        5678,
    ],

    # --------------------------------------------------------
    # Transmon
    # --------------------------------------------------------
    "EJ_GHz": 20.0,
    "EC_GHz": 0.30,
    "ng0": 0.0,

    "n_cut": 50,
    "n_levels": 10,

    # --------------------------------------------------------
    # Physical simulation
    # --------------------------------------------------------
    "dt_env_ns": 2.0,
    "dt_internal_ns": 0.05,
    "total_time_ns": 200.0,

    # --------------------------------------------------------
    # Neural-operator temporal grid
    # --------------------------------------------------------
    "operator_dt_ns": 2.0,

    # --------------------------------------------------------
    # Neural-operator tensors
    # --------------------------------------------------------
    # Filled after the effective Hamiltonian representation
    # has been defined.
    "input_channels": None,
    "output_channels": 10,

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------
    "batch_size": 8,
    "learning_rate": 1e-3,
    "epochs": 100,

    # --------------------------------------------------------
    # Numerical precision
    # --------------------------------------------------------
    "dtype": torch.float32,

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------
    "device": (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    ),
}


DEVICE = torch.device(CONFIG["device"])


# ============================================================
# TEMPORAL GRID
# ============================================================

TIME_GRID = np.arange(
    0.0,
    CONFIG["total_time_ns"] + CONFIG["operator_dt_ns"],
    CONFIG["operator_dt_ns"],
    dtype=np.float64,
)

# Remove possible floating-point endpoint overshoot
TIME_GRID = TIME_GRID[
    TIME_GRID <= CONFIG["total_time_ns"] + 1e-12
]


N_TIME = len(TIME_GRID)


# ============================================================
# CONFIGURATION CHECK
# ============================================================

assert CONFIG["train_samples"] == 40
assert CONFIG["test_samples"] == 10

assert len(CONFIG["seeds"]) == 5

assert CONFIG["n_levels"] == 10

assert CONFIG["dt_env_ns"] == 2.0
assert CONFIG["operator_dt_ns"] == 2.0

assert CONFIG["batch_size"] == 8
assert CONFIG["learning_rate"] == 1e-3

assert N_TIME == 101


# ============================================================
# DISPLAY
# ============================================================

print("=" * 60)
print("CONFIGURATION")
print("=" * 60)

print(f"Device              : {DEVICE}")
print(f"Train samples       : {CONFIG['train_samples']}")
print(f"Test samples        : {CONFIG['test_samples']}")
print(f"Seeds               : {CONFIG['seeds']}")
print(f"Transmon levels     : {CONFIG['n_levels']}")
print(f"Charge cutoff       : {CONFIG['n_cut']}")
print(f"EJ                  : {CONFIG['EJ_GHz']} GHz")
print(f"EC                  : {CONFIG['EC_GHz']} GHz")
print(f"Environment dt      : {CONFIG['dt_env_ns']} ns")
print(f"Internal dt         : {CONFIG['dt_internal_ns']} ns")
print(f"Total time          : {CONFIG['total_time_ns']} ns")
print(f"Operator dt         : {CONFIG['operator_dt_ns']} ns")
print(f"Time points         : {N_TIME}")
print(f"Batch size          : {CONFIG['batch_size']}")
print(f"Learning rate       : {CONFIG['learning_rate']}")
print(f"Epochs              : {CONFIG['epochs']}")

# ============================================================
# CELL 3 — TRANSMON HAMILTONIAN AND EIGENBASIS
# ============================================================

# ------------------------------------------------------------
# Charge basis
# ------------------------------------------------------------

N_CHARGE = 2 * CONFIG["n_cut"] + 1

n_charge = np.arange(
    -CONFIG["n_cut"],
    CONFIG["n_cut"] + 1,
    dtype=np.float64
)

print("=" * 60)
print("CHARGE-BASIS TRANSMON MODEL")
print("=" * 60)

print(f"Charge cutoff      : {CONFIG['n_cut']}")
print(f"Charge basis size  : {N_CHARGE}")


# ------------------------------------------------------------
# Charge operator n
# ------------------------------------------------------------

n_operator = np.diag(n_charge)


# ------------------------------------------------------------
# Shift operators
#
# exp(+/- i phi) shifts the charge state by +/- 1.
# ------------------------------------------------------------

shift_plus = np.zeros(
    (N_CHARGE, N_CHARGE),
    dtype=np.complex128
)

shift_minus = np.zeros(
    (N_CHARGE, N_CHARGE),
    dtype=np.complex128
)

for i in range(N_CHARGE - 1):
    shift_plus[i + 1, i] = 1.0
    shift_minus[i, i + 1] = 1.0


# ------------------------------------------------------------
# Cos(phi)
#
# cos(phi) = [exp(i phi) + exp(-i phi)] / 2
# ------------------------------------------------------------

cos_phi = 0.5 * (
    shift_plus + shift_minus
)


# ------------------------------------------------------------
# Full cosine Hamiltonian
#
# H = 4 E_C (n - n_g)^2 - E_J cos(phi)
#
# Energies are represented in GHz.
# ------------------------------------------------------------

charging_hamiltonian = (
    4.0
    * CONFIG["EC_GHz"]
    * np.diag(
        (n_charge - CONFIG["ng0"]) ** 2
    )
)

josephson_hamiltonian = (
    -CONFIG["EJ_GHz"]
    * cos_phi
)

H_charge = (
    charging_hamiltonian
    + josephson_hamiltonian
)


# ------------------------------------------------------------
# Enforce Hermiticity numerically
# ------------------------------------------------------------

H_charge = 0.5 * (
    H_charge + H_charge.conj().T
)


# ------------------------------------------------------------
# Diagonalize the full cosine Hamiltonian
# ------------------------------------------------------------

eigenvalues, eigenvectors = np.linalg.eigh(
    H_charge
)


# ------------------------------------------------------------
# Retain the lowest 10 eigenstates
# ------------------------------------------------------------

N_LEVELS = CONFIG["n_levels"]

energies = eigenvalues[:N_LEVELS]

V = eigenvectors[:, :N_LEVELS]


# ------------------------------------------------------------
# Shift the energy reference so E0 = 0
# ------------------------------------------------------------

energies = energies - energies[0]


# ------------------------------------------------------------
# Transform charge operator into the eigenbasis
# ------------------------------------------------------------

n_eigenbasis = (
    V.conj().T
    @ n_operator
    @ V
)


# ------------------------------------------------------------
# Transform cos(phi) into the eigenbasis
# ------------------------------------------------------------

cos_phi_eigenbasis = (
    V.conj().T
    @ cos_phi
    @ V
)


# ------------------------------------------------------------
# Transition frequencies
#
# f_ij = E_j - E_i
#
# Since the Hamiltonian is expressed in GHz,
# the energy differences are directly frequencies in GHz.
# ------------------------------------------------------------

transition_frequencies_GHz = (
    energies[1:] - energies[0]
)


# ------------------------------------------------------------
# Fundamental transition frequency
# ------------------------------------------------------------

f01_GHz = (
    energies[1] - energies[0]
)


# ------------------------------------------------------------
# Anharmonicity
#
# alpha = f12 - f01
# ------------------------------------------------------------

f12_GHz = (
    energies[2] - energies[1]
)

anharmonicity_GHz = (
    f12_GHz - f01_GHz
)


# ------------------------------------------------------------
# Store quantities in CONFIG
# ------------------------------------------------------------

CONFIG["input_channels"] = 4

CONFIG["f01_GHz"] = f01_GHz
CONFIG["anharmonicity_GHz"] = anharmonicity_GHz


# ------------------------------------------------------------
# Diagnostics
# ------------------------------------------------------------

print()
print("Hamiltonian")
print("-" * 60)

print(f"Hamiltonian shape  : {H_charge.shape}")
print(f"Retained levels    : {N_LEVELS}")

print()
print("Lowest energy levels")
print("-" * 60)

for j, E in enumerate(energies):
    print(f"E_{j} = {E:.9f} GHz")

print()
print("Spectral properties")
print("-" * 60)

print(f"f01                : {f01_GHz:.9f} GHz")
print(f"f12                : {f12_GHz:.9f} GHz")
print(f"Anharmonicity      : {anharmonicity_GHz:.9f} GHz")


# ------------------------------------------------------------
# Numerical checks
# ------------------------------------------------------------

assert H_charge.shape == (
    N_CHARGE,
    N_CHARGE
)

assert V.shape == (
    N_CHARGE,
    N_LEVELS
)

assert energies.shape == (
    N_LEVELS,
)

assert np.all(
    np.diff(eigenvalues) >= -1e-12
)

assert np.allclose(
    H_charge,
    H_charge.conj().T,
    atol=1e-12
)

assert np.allclose(
    V.conj().T @ V,
    np.eye(N_LEVELS),
    atol=1e-10
)

assert np.allclose(
    energies[0],
    0.0,
    atol=1e-12
)

print()
print("All Cell 3 checks passed.")

# ============================================================
# CELL 4 — CONTROL OPERATORS IN THE 10-LEVEL EIGENBASIS
# ============================================================

# ------------------------------------------------------------
# Charge operator matrix elements
# ------------------------------------------------------------

n_control = n_eigenbasis.copy()


# ------------------------------------------------------------
# Control Hamiltonian
#
# The microwave drive couples through the charge operator:
#
# H_drive(t) = u_I(t) H_I + u_Q(t) H_Q
#
# We use the two quadratures I and Q as the control channels.
# ------------------------------------------------------------

H_I = n_control.copy()

H_Q = 1j * n_control


# ------------------------------------------------------------
# Hermitian control operators
# ------------------------------------------------------------

H_I = 0.5 * (
    H_I + H_I.conj().T
)

H_Q = 0.5 * (
    H_Q + H_Q.conj().T
)


# ------------------------------------------------------------
# Drift Hamiltonian
#
# H0 = diag(E0, E1, ..., E9)
# ------------------------------------------------------------

H0 = np.diag(
    energies
).astype(np.complex128)


# ------------------------------------------------------------
# Store the operators
# ------------------------------------------------------------

CONTROL_OPERATORS = {
    "H0": H0,
    "H_I": H_I,
    "H_Q": H_Q,
}


# ------------------------------------------------------------
# Basic diagnostics
# ------------------------------------------------------------

print("=" * 60)
print("10-LEVEL CONTROL HAMILTONIAN")
print("=" * 60)

print(f"H0 shape            : {H0.shape}")
print(f"H_I shape           : {H_I.shape}")
print(f"H_Q shape           : {H_Q.shape}")

print()
print("Hermiticity checks")
print("-" * 60)

print(
    f"H0 Hermitian        : "
    f"{np.allclose(H0, H0.conj().T, atol=1e-12)}"
)

print(
    f"H_I Hermitian       : "
    f"{np.allclose(H_I, H_I.conj().T, atol=1e-12)}"
)

print(
    f"H_Q Hermitian       : "
    f"{np.allclose(H_Q, H_Q.conj().T, atol=1e-12)}"
)


# ------------------------------------------------------------
# Numerical assertions
# ------------------------------------------------------------

assert H0.shape == (N_LEVELS, N_LEVELS)
assert H_I.shape == (N_LEVELS, N_LEVELS)
assert H_Q.shape == (N_LEVELS, N_LEVELS)

assert np.allclose(
    H0,
    H0.conj().T,
    atol=1e-12
)

assert np.allclose(
    H_I,
    H_I.conj().T,
    atol=1e-12
)

assert np.allclose(
    H_Q,
    H_Q.conj().T,
    atol=1e-12
)

print()
print("All Cell 4 checks passed.")

# ============================================================
# CELL 5 — NOISE AND CONTROL PARAMETERS
# ============================================================

# ------------------------------------------------------------
# Control amplitude
# ------------------------------------------------------------

MAX_IQ = 0.100                  # GHz
CONTROL_BANDWIDTH = 0.250       # GHz

CONTROL_TAU_NS = (
    1.0 /
    (
        2.0
        * np.pi
        * CONTROL_BANDWIDTH
    )
)


# ------------------------------------------------------------
# Frequency noise
# ------------------------------------------------------------

FREQUENCY_NOISE_RMS = 1.0e-4    # GHz

NOISE_F_MIN_HZ = 1.0
NOISE_F_KNEE_HZ = 1.0e7
NOISE_F_MAX_HZ = 2.0e9


# ------------------------------------------------------------
# Charge noise
# ------------------------------------------------------------

CHARGE_NOISE_AMPLITUDE = 2.0e-3


# ------------------------------------------------------------
# Microwave amplitude and phase noise
# ------------------------------------------------------------

AMPLITUDE_NOISE_SIGMA = 1.0e-4
PHASE_NOISE_SIGMA_RAD = 1.0e-3


# ------------------------------------------------------------
# Slow frequency drift
# ------------------------------------------------------------

DRIFT_SIGMA_GHZ = 50.0e-6


# ------------------------------------------------------------
# TLS fluctuations
# ------------------------------------------------------------

TLS_SHIFT_GHZ = 500.0e-6


# ------------------------------------------------------------
# Dissipation parameters
# ------------------------------------------------------------

T1_US = 30.0
T2_US = 20.0


# ------------------------------------------------------------
# Temperature
# ------------------------------------------------------------

TEMPERATURE_K = 0.020


# ------------------------------------------------------------
# Store physical parameters in CONFIG
# ------------------------------------------------------------

CONFIG["max_IQ_GHz"] = MAX_IQ
CONFIG["control_bandwidth_GHz"] = CONTROL_BANDWIDTH
CONFIG["frequency_noise_rms_GHz"] = FREQUENCY_NOISE_RMS
CONFIG["charge_noise_amplitude"] = CHARGE_NOISE_AMPLITUDE
CONFIG["amplitude_noise_sigma"] = AMPLITUDE_NOISE_SIGMA
CONFIG["phase_noise_sigma_rad"] = PHASE_NOISE_SIGMA_RAD
CONFIG["drift_sigma_GHz"] = DRIFT_SIGMA_GHZ
CONFIG["tls_shift_GHz"] = TLS_SHIFT_GHZ
CONFIG["T1_us"] = T1_US
CONFIG["T2_us"] = T2_US
CONFIG["temperature_K"] = TEMPERATURE_K


# ------------------------------------------------------------
# Diagnostics
# ------------------------------------------------------------

print("=" * 60)
print("CONTROL AND NOISE PARAMETERS")
print("=" * 60)

print(f"Maximum I/Q amplitude       : {MAX_IQ} GHz")
print(f"Control bandwidth           : {CONTROL_BANDWIDTH} GHz")
print(f"Control time constant       : {CONTROL_TAU_NS:.6f} ns")

print()
print(f"Frequency noise RMS         : {FREQUENCY_NOISE_RMS:.2e} GHz")
print(f"Charge noise amplitude      : {CHARGE_NOISE_AMPLITUDE:.2e}")
print(f"Amplitude noise sigma       : {AMPLITUDE_NOISE_SIGMA:.2e}")
print(f"Phase noise sigma           : {PHASE_NOISE_SIGMA_RAD:.2e} rad")
print(f"Frequency drift sigma       : {DRIFT_SIGMA_GHZ:.2e} GHz")
print(f"TLS frequency shift         : {TLS_SHIFT_GHZ:.2e} GHz")

print()
print(f"T1                          : {T1_US} us")
print(f"T2                          : {T2_US} us")
print(f"Temperature                 : {TEMPERATURE_K} K")


# ------------------------------------------------------------
# Numerical checks
# ------------------------------------------------------------

assert MAX_IQ > 0.0
assert CONTROL_BANDWIDTH > 0.0
assert CONTROL_TAU_NS > 0.0

assert FREQUENCY_NOISE_RMS >= 0.0
assert CHARGE_NOISE_AMPLITUDE >= 0.0
assert AMPLITUDE_NOISE_SIGMA >= 0.0
assert PHASE_NOISE_SIGMA_RAD >= 0.0
assert DRIFT_SIGMA_GHZ >= 0.0
assert TLS_SHIFT_GHZ >= 0.0

assert T1_US > 0.0
assert T2_US > 0.0
assert TEMPERATURE_K > 0.0

print()
print("All Cell 5 checks passed.")

# ============================================================
# CELL 6 — CONTROL BANDWIDTH STATE AND NOISE STATE
# ============================================================

# ------------------------------------------------------------
# Random generator
# ------------------------------------------------------------

RNG = np.random.default_rng(
    CONFIG["seeds"][0]
)


# ------------------------------------------------------------
# Control state
# ------------------------------------------------------------

control_I = 0.0
control_Q = 0.0


# ------------------------------------------------------------
# Slow frequency drift
# ------------------------------------------------------------

frequency_drift_GHz = 0.0


# ------------------------------------------------------------
# TLS state
#
# Two-state telegraph process:
#     0 -> no TLS shift
#     1 -> TLS frequency shift
# ------------------------------------------------------------

tls_state = 0


# ------------------------------------------------------------
# Noise state
# ------------------------------------------------------------

frequency_noise_GHz = 0.0
charge_noise = 0.0

amplitude_noise = 0.0
phase_noise_rad = 0.0


# ------------------------------------------------------------
# Time-dependent Hamiltonian coefficient container
#
# These are the quantities that will eventually form the
# operator-learning input u(t).
# ------------------------------------------------------------

effective_hamiltonian_coefficients = np.zeros(
    (
        CONFIG["input_channels"],
        N_TIME
    ),
    dtype=np.float64
)


# ------------------------------------------------------------
# Initial-state density matrix
#
# The system starts in the ground state |0>.
# ------------------------------------------------------------

rho0 = np.zeros(
    (N_LEVELS, N_LEVELS),
    dtype=np.complex128
)

rho0[0, 0] = 1.0


# ------------------------------------------------------------
# Diagnostics
# ------------------------------------------------------------

print("=" * 60)
print("CONTROL AND NOISE STATE")
print("=" * 60)

print(f"Initial I control       : {control_I}")
print(f"Initial Q control       : {control_Q}")
print(f"Frequency drift         : {frequency_drift_GHz}")
print(f"TLS state               : {tls_state}")
print(f"Initial frequency noise : {frequency_noise_GHz}")
print(f"Initial charge noise    : {charge_noise}")
print(f"Initial amplitude noise : {amplitude_noise}")
print(f"Initial phase noise     : {phase_noise_rad}")

print()
print(f"Coefficient tensor      : "
      f"{effective_hamiltonian_coefficients.shape}")

print(f"Initial density matrix  : {rho0.shape}")


# ------------------------------------------------------------
# Numerical checks
# ------------------------------------------------------------

assert effective_hamiltonian_coefficients.shape == (
    CONFIG["input_channels"],
    N_TIME
)

assert rho0.shape == (
    N_LEVELS,
    N_LEVELS
)

assert np.isclose(
    np.trace(rho0),
    1.0
)

assert np.allclose(
    rho0,
    rho0.conj().T,
    atol=1e-12
)

assert np.all(
    np.linalg.eigvalsh(rho0) >= -1e-12
)

print()
print("All Cell 6 checks passed.")

# ============================================================
# CELL 7 — DISSIPATION AND LINDBLAD OPERATORS
# ============================================================

# ------------------------------------------------------------
# Convert relaxation and dephasing times to ns
# ------------------------------------------------------------

T1_NS = T1_US * 1.0e3
T2_NS = T2_US * 1.0e3


# ------------------------------------------------------------
# Dissipation rates
#
# Gamma_1 : energy relaxation rate
# Gamma_phi : pure dephasing rate
#
# 1/T2 = 1/(2T1) + Gamma_phi
# ------------------------------------------------------------

GAMMA_1 = 1.0 / T1_NS

GAMMA_2 = 1.0 / T2_NS

GAMMA_PHI = (
    GAMMA_2
    - 0.5 * GAMMA_1
)


# ------------------------------------------------------------
# Numerical consistency check
# ------------------------------------------------------------

if GAMMA_PHI < 0.0:
    raise ValueError(
        "The supplied T1 and T2 values are physically inconsistent: "
        "T2 must satisfy T2 <= 2*T1."
    )


# ------------------------------------------------------------
# Energy relaxation operator
#
# We use adjacent-level relaxation channels:
#
# |j-1><j|
#
# for j = 1,...,9.
# ------------------------------------------------------------

relaxation_operators = []

for j in range(1, N_LEVELS):

    L_relax = np.zeros(
        (N_LEVELS, N_LEVELS),
        dtype=np.complex128
    )

    L_relax[j - 1, j] = np.sqrt(
        GAMMA_1
    )

    relaxation_operators.append(
        L_relax
    )


# ------------------------------------------------------------
# Pure dephasing operator
#
# A diagonal operator produces phase damping without
# directly changing the populations.
# ------------------------------------------------------------

L_dephasing = np.diag(
    np.arange(N_LEVELS, dtype=np.float64)
).astype(np.complex128)

L_dephasing *= np.sqrt(
    GAMMA_PHI
)


# ------------------------------------------------------------
# Complete Lindblad operator list
# ------------------------------------------------------------

LINDBLAD_OPERATORS = (
    relaxation_operators
    + [L_dephasing]
)


# ------------------------------------------------------------
# Store rates
# ------------------------------------------------------------

CONFIG["T1_ns"] = T1_NS
CONFIG["T2_ns"] = T2_NS
CONFIG["gamma_1"] = GAMMA_1
CONFIG["gamma_2"] = GAMMA_2
CONFIG["gamma_phi"] = GAMMA_PHI


# ------------------------------------------------------------
# Diagnostics
# ------------------------------------------------------------

print("=" * 60)
print("DISSIPATION MODEL")
print("=" * 60)

print(f"T1                     : {T1_NS:.3f} ns")
print(f"T2                     : {T2_NS:.3f} ns")
print(f"Gamma_1                : {GAMMA_1:.6e} ns^-1")
print(f"Gamma_2                : {GAMMA_2:.6e} ns^-1")
print(f"Gamma_phi              : {GAMMA_PHI:.6e} ns^-1")

print()
print(f"Relaxation channels    : {len(relaxation_operators)}")
print(f"Total Lindblad channels: {len(LINDBLAD_OPERATORS)}")


# ------------------------------------------------------------
# Numerical checks
# ------------------------------------------------------------

assert T1_NS > 0.0
assert T2_NS > 0.0

assert GAMMA_1 > 0.0
assert GAMMA_2 > 0.0
assert GAMMA_PHI >= 0.0

assert len(relaxation_operators) == N_LEVELS - 1

for L in LINDBLAD_OPERATORS:
    assert L.shape == (
        N_LEVELS,
        N_LEVELS
    )

print()
print("All Cell 7 checks passed.")

# ============================================================
# CELL 8 — EFFECTIVE HAMILTONIAN COEFFICIENT REPRESENTATION
# ============================================================

# ------------------------------------------------------------
# IMPORTANT
#
# The neural operator will learn
#
#     G : u(t) -> P(t)
#
# where u(t) contains the time-dependent coefficients that
# determine the effective Hamiltonian.
#
# We use four real-valued coefficients:
#
#     u_0(t) = frequency-detuning coefficient
#     u_1(t) = I-quadrature coefficient
#     u_2(t) = Q-quadrature coefficient
#     u_3(t) = charge-noise coefficient
#
# Therefore:
#
#     u(t) in R^4
#
# and the input tensor will eventually have shape
#
#     [B, 4, N]
# ------------------------------------------------------------


# ------------------------------------------------------------
# Drift operator
#
# H0 = diag(E0, ..., E9)
#
# The frequency-noise / drift contribution is represented by
# the diagonal spectral operator.
# ------------------------------------------------------------

H_frequency = np.diag(
    np.arange(N_LEVELS, dtype=np.float64)
).astype(np.complex128)


# ------------------------------------------------------------
# Charge operator in the eigenbasis
#
# The charge operator provides the microwave coupling
# structure of the transmon.
# ------------------------------------------------------------

H_charge_control = (
    n_eigenbasis.copy()
)


# ------------------------------------------------------------
# Positive-frequency part of the charge operator
#
# n_+ contains transitions |i> -> |j> with j > i.
#
# This allows us to construct two independent Hermitian
# microwave quadratures.
# ------------------------------------------------------------

n_plus = np.zeros(
    (N_LEVELS, N_LEVELS),
    dtype=np.complex128
)

for i in range(N_LEVELS):
    for j in range(i + 1, N_LEVELS):
        n_plus[i, j] = (
            H_charge_control[i, j]
        )


# ------------------------------------------------------------
# I quadrature
#
# H_I = n_+ + n_+^\dagger
# ------------------------------------------------------------

H_I = (
    n_plus
    + n_plus.conj().T
)

H_I = 0.5 * (
    H_I + H_I.conj().T
)


# ------------------------------------------------------------
# Q quadrature
#
# H_Q = i(n_+ - n_+^\dagger)
#
# This is Hermitian and is independent of H_I.
# ------------------------------------------------------------

H_Q = (
    1j
    * (
        n_plus
        - n_plus.conj().T
    )
)

H_Q = 0.5 * (
    H_Q + H_Q.conj().T
)


# ------------------------------------------------------------
# Charge-noise operator
#
# The Hamiltonian dependence on offset charge is
#
#     H = 4 E_C (n - n_g)^2 - E_J cos(phi)
#
# and therefore
#
#     dH/dn_g = -8 E_C (n - n_g).
#
# At the nominal operating point n_g = ng0:
# ------------------------------------------------------------

H_charge_noise = (
    -8.0
    * CONFIG["EC_GHz"]
    * (
        n_eigenbasis
        - CONFIG["ng0"]
        * np.eye(N_LEVELS)
    )
)

H_charge_noise = 0.5 * (
    H_charge_noise
    + H_charge_noise.conj().T
)


# ------------------------------------------------------------
# Store effective Hamiltonian basis
# ------------------------------------------------------------

EFFECTIVE_HAMILTONIAN_BASIS = {

    "frequency": H_frequency,

    "I": H_I,

    "Q": H_Q,

    "charge": H_charge_noise,
}


# ------------------------------------------------------------
# Number of input channels
# ------------------------------------------------------------

INPUT_CHANNELS = len(
    EFFECTIVE_HAMILTONIAN_BASIS
)

CONFIG["input_channels"] = INPUT_CHANNELS


# ------------------------------------------------------------
# Store the ordered channel names
# ------------------------------------------------------------

INPUT_CHANNEL_NAMES = [
    "frequency",
    "I",
    "Q",
    "charge",
]


# ------------------------------------------------------------
# Diagnostics
# ------------------------------------------------------------

print("=" * 60)
print("EFFECTIVE HAMILTONIAN REPRESENTATION")
print("=" * 60)

print()
print("Input channels")
print("-" * 60)

for k, name in enumerate(INPUT_CHANNEL_NAMES):
    print(
        f"Channel {k}: {name}"
    )

print()
print(
    f"Number of input channels : "
    f"{INPUT_CHANNELS}"
)

print()
print("Operator shapes")
print("-" * 60)

for name, operator in EFFECTIVE_HAMILTONIAN_BASIS.items():
    print(
        f"{name:12s}: {operator.shape}"
    )


# ------------------------------------------------------------
# Hermiticity checks
# ------------------------------------------------------------

for name, operator in EFFECTIVE_HAMILTONIAN_BASIS.items():

    assert np.allclose(
        operator,
        operator.conj().T,
        atol=1e-12
    ), (
        f"{name} operator is not Hermitian."
    )


# ------------------------------------------------------------
# Independence check for the I/Q operators
# ------------------------------------------------------------

assert np.linalg.norm(H_I) > 0.0
assert np.linalg.norm(H_Q) > 0.0

assert np.linalg.norm(
    H_I - H_Q
) > 1e-12


# ------------------------------------------------------------
# Configuration checks
# ------------------------------------------------------------

assert INPUT_CHANNELS == 4

assert CONFIG["input_channels"] == 4

assert CONFIG["output_channels"] == 10

assert len(INPUT_CHANNEL_NAMES) == 4


print()
print("All Cell 8 checks passed.")

# ============================================================
# CELL 9 — TIME-DEPENDENT HAMILTONIAN AND LINDBLAD EVOLUTION
# ============================================================

# ------------------------------------------------------------
# Hamiltonian from effective coefficients
#
#     H(t) = H0
#          + u_frequency(t) H_frequency
#          + u_I(t)         H_I
#          + u_Q(t)         H_Q
#          + u_charge(t)    H_charge
#
# The coefficients are real-valued.
# ------------------------------------------------------------

def build_effective_hamiltonian(
    u_frequency,
    u_I,
    u_Q,
    u_charge,
):
    """
    Construct the 10-level effective Hamiltonian.

    Parameters
    ----------
    u_frequency : float
        Frequency-detuning coefficient in GHz.

    u_I : float
        In-phase microwave coefficient.

    u_Q : float
        Quadrature microwave coefficient.

    u_charge : float
        Effective charge-noise coefficient.

    Returns
    -------
    H : ndarray
        Complex Hermitian Hamiltonian of shape
        (N_LEVELS, N_LEVELS).
    """

    H = (
        H0
        + u_frequency * H_frequency
        + u_I * H_I
        + u_Q * H_Q
        + u_charge * H_charge_noise
    )

    H = 0.5 * (
        H + H.conj().T
    )

    return H


# ------------------------------------------------------------
# Lindblad dissipator
#
# D[L](rho)
# =
# L rho L^\dagger
# -
# 1/2 {L^\dagger L, rho}
# ------------------------------------------------------------

def lindblad_dissipator(
    rho,
    L,
):
    """
    Calculate one Lindblad dissipator.
    """

    L_dagger = L.conj().T

    return (
        L @ rho @ L_dagger
        - 0.5
        * (
            L_dagger @ L @ rho
            + rho @ L_dagger @ L
        )
    )


# ------------------------------------------------------------
# Full master-equation right-hand side
#
# d rho / dt =
#
#     -i [H,rho]
#     + sum_k D[L_k](rho)
#
# H is expressed in GHz.
#
# Since:
#
#     hbar = 1 / (2 pi) GHz ns
#
# the Hamiltonian contribution contains 2 pi.
# ------------------------------------------------------------

def lindblad_rhs(
    rho,
    H,
):
    """
    Evaluate the Lindblad master equation.
    """

    commutator = (
        H @ rho
        - rho @ H
    )

    hamiltonian_term = (
        -2.0
        * np.pi
        * 1j
        * commutator
    )

    dissipative_term = np.zeros_like(
        rho,
        dtype=np.complex128
    )

    for L in LINDBLAD_OPERATORS:

        dissipative_term += (
            lindblad_dissipator(
                rho,
                L
            )
        )

    return (
        hamiltonian_term
        + dissipative_term
    )


# ------------------------------------------------------------
# One RK4 integration step
# ------------------------------------------------------------

def rk4_density_step(
    rho,
    H,
    dt_ns,
):
    """
    Advance the density matrix by one RK4 step.
    """

    k1 = lindblad_rhs(
        rho,
        H
    )

    k2 = lindblad_rhs(
        rho + 0.5 * dt_ns * k1,
        H
    )

    k3 = lindblad_rhs(
        rho + 0.5 * dt_ns * k2,
        H
    )

    k4 = lindblad_rhs(
        rho + dt_ns * k3,
        H
    )

    rho_next = (
        rho
        + (
            dt_ns
            / 6.0
        )
        * (
            k1
            + 2.0 * k2
            + 2.0 * k3
            + k4
        )
    )

    # Numerical Hermitization
    rho_next = 0.5 * (
        rho_next
        + rho_next.conj().T
    )

    # Numerical trace normalization
    trace = np.trace(rho_next)

    if abs(trace) > 1e-14:
        rho_next /= trace

    return rho_next


# ------------------------------------------------------------
# Population extraction
# ------------------------------------------------------------

def extract_populations(
    rho,
):
    """
    Extract the ten level populations.

    P_j = <j|rho|j>
    """

    populations = np.real(
        np.diag(rho)
    )

    # Remove tiny numerical errors
    populations = np.maximum(
        populations,
        0.0
    )

    population_sum = np.sum(
        populations
    )

    if population_sum > 0.0:
        populations /= population_sum

    return populations


# ------------------------------------------------------------
# Basic test Hamiltonian
# ------------------------------------------------------------

H_test = build_effective_hamiltonian(
    u_frequency=0.0,
    u_I=0.0,
    u_Q=0.0,
    u_charge=0.0,
)


# ------------------------------------------------------------
# Test evolution
# ------------------------------------------------------------

rho_test = rho0.copy()

rho_test_next = rk4_density_step(
    rho=rho_test,
    H=H_test,
    dt_ns=CONFIG["dt_internal_ns"],
)

P_test = extract_populations(
    rho_test_next
)


# ------------------------------------------------------------
# Diagnostics
# ------------------------------------------------------------

print("=" * 60)
print("HAMILTONIAN AND LINDBLAD EVOLUTION")
print("=" * 60)

print(
    f"Hamiltonian shape     : {H_test.shape}"
)

print(
    f"Density matrix shape  : {rho_test_next.shape}"
)

print(
    f"Population shape      : {P_test.shape}"
)

print(
    f"Population sum        : "
    f"{np.sum(P_test):.12f}"
)

print(
    f"Minimum population    : "
    f"{np.min(P_test):.12e}"
)

print(
    f"Maximum population    : "
    f"{np.max(P_test):.12e}"
)


# ------------------------------------------------------------
# Numerical checks
# ------------------------------------------------------------

assert H_test.shape == (
    N_LEVELS,
    N_LEVELS
)

assert np.allclose(
    H_test,
    H_test.conj().T,
    atol=1e-12
)

assert rho_test_next.shape == (
    N_LEVELS,
    N_LEVELS
)

assert np.allclose(
    rho_test_next,
    rho_test_next.conj().T,
    atol=1e-10
)

assert np.isclose(
    np.trace(rho_test_next),
    1.0,
    atol=1e-10
)

assert P_test.shape == (
    N_LEVELS,
)

assert np.isclose(
    np.sum(P_test),
    1.0,
    atol=1e-10
)

assert np.all(
    P_test >= -1e-12
)

print()
print("All Cell 9 checks passed.")

# ============================================================
# CELL 10 — SINGLE TRAJECTORY GENERATION
# ============================================================

def generate_control_sequence(
    rng,
    n_time=N_TIME,
):
    """
    Generate one smooth I/Q control trajectory.

    Returns
    -------
    I_control : ndarray
        In-phase control, shape (n_time,).

    Q_control : ndarray
        Quadrature control, shape (n_time,).
    """

    I_control = np.zeros(
        n_time,
        dtype=np.float64
    )

    Q_control = np.zeros(
        n_time,
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Piecewise-random control values
    # --------------------------------------------------------

    coarse_points = max(
        5,
        n_time // 10
    )

    coarse_I = rng.uniform(
        -MAX_IQ,
        MAX_IQ,
        size=coarse_points
    )

    coarse_Q = rng.uniform(
        -MAX_IQ,
        MAX_IQ,
        size=coarse_points
    )

    coarse_time = np.linspace(
        0.0,
        n_time - 1,
        coarse_points
    )

    fine_time = np.arange(
        n_time,
        dtype=np.float64
    )

    I_control = np.interp(
        fine_time,
        coarse_time,
        coarse_I
    )

    Q_control = np.interp(
        fine_time,
        coarse_time,
        coarse_Q
    )

    # --------------------------------------------------------
    # Enforce amplitude limit
    # --------------------------------------------------------

    I_control = np.clip(
        I_control,
        -MAX_IQ,
        MAX_IQ
    )

    Q_control = np.clip(
        Q_control,
        -MAX_IQ,
        MAX_IQ
    )

    return I_control, Q_control


def generate_noise_trajectories(
    rng,
    n_time=N_TIME,
):
    """
    Generate slowly varying stochastic trajectories.

    Returns
    -------
    frequency_noise : ndarray
    charge_noise : ndarray
    amplitude_noise : ndarray
    phase_noise : ndarray
    frequency_drift : ndarray
    tls_state : ndarray
    """

    # --------------------------------------------------------
    # Frequency noise
    # --------------------------------------------------------

    frequency_noise = np.zeros(
        n_time,
        dtype=np.float64
    )

    for k in range(1, n_time):

        frequency_noise[k] = (
            0.995 * frequency_noise[k - 1]
            + 0.005
            * FREQUENCY_NOISE_RMS
            * rng.normal()
        )

    # --------------------------------------------------------
    # Charge noise
    # --------------------------------------------------------

    charge_noise = np.zeros(
        n_time,
        dtype=np.float64
    )

    for k in range(1, n_time):

        charge_noise[k] = (
            0.995 * charge_noise[k - 1]
            + 0.005
            * CHARGE_NOISE_AMPLITUDE
            * rng.normal()
        )

    # --------------------------------------------------------
    # Microwave amplitude noise
    # --------------------------------------------------------

    amplitude_noise = (
        AMPLITUDE_NOISE_SIGMA
        * rng.normal(
            size=n_time
        )
    )

    # --------------------------------------------------------
    # Microwave phase noise
    # --------------------------------------------------------

    phase_noise = (
        PHASE_NOISE_SIGMA_RAD
        * rng.normal(
            size=n_time
        )
    )

    # --------------------------------------------------------
    # Slow frequency drift
    # --------------------------------------------------------

    frequency_drift = np.zeros(
        n_time,
        dtype=np.float64
    )

    for k in range(1, n_time):

        frequency_drift[k] = (
            0.999 * frequency_drift[k - 1]
            + 0.001
            * DRIFT_SIGMA_GHZ
            * rng.normal()
        )

    # --------------------------------------------------------
    # TLS telegraph process
    # --------------------------------------------------------

    tls_state = np.zeros(
        n_time,
        dtype=np.int64
    )

    for k in range(1, n_time):

        tls_state[k] = tls_state[k - 1]

        # Small switching probability per operator step
        if rng.random() < 0.01:
            tls_state[k] = 1 - tls_state[k]

    return (
        frequency_noise,
        charge_noise,
        amplitude_noise,
        phase_noise,
        frequency_drift,
        tls_state,
    )


def apply_control_noise(
    I_control,
    Q_control,
    amplitude_noise,
    phase_noise,
):
    """
    Apply microwave amplitude and phase noise.

    The noisy complex envelope is

        (I + iQ)
        (1 + delta_A)
        exp(i delta_phi)

    """

    complex_control = (
        I_control
        + 1j * Q_control
    )

    noisy_control = (
        complex_control
        * (
            1.0
            + amplitude_noise
        )
        * np.exp(
            1j * phase_noise
        )
    )

    I_noisy = np.real(
        noisy_control
    )

    Q_noisy = np.imag(
        noisy_control
    )

    return I_noisy, Q_noisy


def generate_single_trajectory(
    seed,
):
    """
    Generate one complete input/output trajectory.

    Returns
    -------
    U : ndarray
        Effective Hamiltonian coefficients,
        shape (4, N_TIME).

    Y : ndarray
        Ten level populations,
        shape (10, N_TIME).
    """

    rng = np.random.default_rng(
        seed
    )

    # --------------------------------------------------------
    # Generate control
    # --------------------------------------------------------

    I_control, Q_control = (
        generate_control_sequence(
            rng
        )
    )

    # --------------------------------------------------------
    # Generate stochastic processes
    # --------------------------------------------------------

    (
        frequency_noise,
        charge_noise,
        amplitude_noise,
        phase_noise,
        frequency_drift,
        tls_state,
    ) = generate_noise_trajectories(
        rng
    )

    # --------------------------------------------------------
    # Apply amplitude and phase noise
    # --------------------------------------------------------

    I_noisy, Q_noisy = (
        apply_control_noise(
            I_control,
            Q_control,
            amplitude_noise,
            phase_noise,
        )
    )

    # --------------------------------------------------------
    # Allocate tensors
    # --------------------------------------------------------

    U = np.zeros(
        (
            INPUT_CHANNELS,
            N_TIME
        ),
        dtype=np.float64
    )

    Y = np.zeros(
        (
            N_LEVELS,
            N_TIME
        ),
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Initial density matrix
    # --------------------------------------------------------

    rho = rho0.copy()

    # --------------------------------------------------------
    # Time evolution
    #
    # The operator grid spacing is 2 ns.
    #
    # Each 2 ns interval is internally integrated using
    # dt_internal_ns = 0.05 ns.
    # --------------------------------------------------------

    internal_dt = CONFIG[
        "dt_internal_ns"
    ]

    operator_dt = CONFIG[
        "operator_dt_ns"
    ]

    n_substeps = int(
        round(
            operator_dt
            / internal_dt
        )
    )

    assert np.isclose(
        n_substeps * internal_dt,
        operator_dt
    )

    # --------------------------------------------------------
    # Time loop
    # --------------------------------------------------------

    for k in range(N_TIME):

        # ----------------------------------------------------
        # Effective frequency coefficient
        # ----------------------------------------------------

        u_frequency = (
            frequency_noise[k]
            + frequency_drift[k]
            + tls_state[k]
            * TLS_SHIFT_GHZ
        )

        # ----------------------------------------------------
        # Effective microwave coefficients
        # ----------------------------------------------------

        u_I = I_noisy[k]

        u_Q = Q_noisy[k]

        # ----------------------------------------------------
        # Effective charge-noise coefficient
        # ----------------------------------------------------

        u_charge = charge_noise[k]

        # ----------------------------------------------------
        # Store input coefficients
        # ----------------------------------------------------

        U[0, k] = u_frequency
        U[1, k] = u_I
        U[2, k] = u_Q
        U[3, k] = u_charge

        # ----------------------------------------------------
        # Store current population
        # ----------------------------------------------------

        Y[:, k] = extract_populations(
            rho
        )

        # ----------------------------------------------------
        # No propagation after final time point
        # ----------------------------------------------------

        if k == N_TIME - 1:
            break

        # ----------------------------------------------------
        # Hamiltonian for this interval
        # ----------------------------------------------------

        H = build_effective_hamiltonian(
            u_frequency=u_frequency,
            u_I=u_I,
            u_Q=u_Q,
            u_charge=u_charge,
        )

        # ----------------------------------------------------
        # Internal RK4 integration
        # ----------------------------------------------------

        for _ in range(n_substeps):

            rho = rk4_density_step(
                rho=rho,
                H=H,
                dt_ns=internal_dt,
            )

    return U, Y


# ============================================================
# TEST ONE TRAJECTORY
# ============================================================

U_test, Y_test = (
    generate_single_trajectory(
        seed=CONFIG["seeds"][0]
    )
)


# ============================================================
# DIAGNOSTICS
# ============================================================

print("=" * 60)
print("SINGLE TRAJECTORY")
print("=" * 60)

print(
    f"Input shape           : {U_test.shape}"
)

print(
    f"Output shape          : {Y_test.shape}"
)

print(
    f"Input minimum         : "
    f"{np.min(U_test):.6e}"
)

print(
    f"Input maximum         : "
    f"{np.max(U_test):.6e}"
)

print(
    f"Output minimum        : "
    f"{np.min(Y_test):.6e}"
)

print(
    f"Output maximum        : "
    f"{np.max(Y_test):.6e}"
)

print(
    f"Population sum range  : "
    f"{np.min(np.sum(Y_test, axis=0)):.12f}"
    f" to "
    f"{np.max(np.sum(Y_test, axis=0)):.12f}"
)


# ============================================================
# NUMERICAL CHECKS
# ============================================================

assert U_test.shape == (
    INPUT_CHANNELS,
    N_TIME
)

assert Y_test.shape == (
    N_LEVELS,
    N_TIME
)

assert np.all(
    np.isfinite(U_test)
)

assert np.all(
    np.isfinite(Y_test)
)

assert np.all(
    Y_test >= -1e-10
)

assert np.allclose(
    np.sum(Y_test, axis=0),
    1.0,
    atol=1e-8
)

print()
print("All Cell 10 checks passed.")

# ============================================================
# CELL 11 — DATASET GENERATION
# ============================================================

def generate_dataset(
    n_samples,
    seed,
):
    """
    Generate a complete operator-learning dataset.

    Parameters
    ----------
    n_samples : int
        Number of trajectories.

    seed : int
        Base random seed.

    Returns
    -------
    U_dataset : ndarray
        Input coefficient trajectories,
        shape (n_samples, INPUT_CHANNELS, N_TIME).

    Y_dataset : ndarray
        Population trajectories,
        shape (n_samples, N_LEVELS, N_TIME).
    """

    # --------------------------------------------------------
    # Allocate dataset arrays
    # --------------------------------------------------------

    U_dataset = np.zeros(
        (
            n_samples,
            INPUT_CHANNELS,
            N_TIME,
        ),
        dtype=np.float64,
    )

    Y_dataset = np.zeros(
        (
            n_samples,
            N_LEVELS,
            N_TIME,
        ),
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Independent trajectory seeds
    # --------------------------------------------------------

    seed_sequence = np.random.SeedSequence(
        seed
    )

    child_sequences = (
        seed_sequence.spawn(
            n_samples
        )
    )

    # --------------------------------------------------------
    # Generate trajectories
    # --------------------------------------------------------

    start_time = time.time()

    for sample_idx in range(
        n_samples
    ):

        trajectory_seed = (
            child_sequences[
                sample_idx
            ].generate_state(
                1
            )[0]
        )

        U, Y = (
            generate_single_trajectory(
                seed=int(
                    trajectory_seed
                )
            )
        )

        U_dataset[
            sample_idx
        ] = U

        Y_dataset[
            sample_idx
        ] = Y

        # ----------------------------------------------------
        # Progress display
        # ----------------------------------------------------

        if (
            (sample_idx + 1) % 25 == 0
            or sample_idx == 0
            or sample_idx == n_samples - 1
        ):
            print(
                f"Generated "
                f"{sample_idx + 1:4d}"
                f" / "
                f"{n_samples:4d}"
            )

    elapsed = (
        time.time()
        - start_time
    )

    print(
        f"Generation time: "
        f"{elapsed:.2f} s"
    )

    return (
        U_dataset,
        Y_dataset,
    )


# ============================================================
# GENERATE TRAINING DATA
# ============================================================

print("=" * 60)
print("GENERATING TRAINING DATA")
print("=" * 60)

U_train, Y_train = (
    generate_dataset(
        n_samples=CONFIG["train_samples"],
        seed=CONFIG["seeds"][0],
    )
)


# ============================================================
# GENERATE TEST DATA
# ============================================================

print()
print("=" * 60)
print("GENERATING TEST DATA")
print("=" * 60)

U_test_dataset, Y_test_dataset = (
    generate_dataset(
        n_samples=CONFIG["test_samples"],
        seed=CONFIG["seeds"][0] + 100000,
    )
)


# ============================================================
# DATASET SHAPES
# ============================================================

print()
print("=" * 60)
print("DATASET SHAPES")
print("=" * 60)

print(
    f"U_train            : "
    f"{U_train.shape}"
)

print(
    f"Y_train            : "
    f"{Y_train.shape}"
)

print(
    f"U_test             : "
    f"{U_test_dataset.shape}"
)

print(
    f"Y_test             : "
    f"{Y_test_dataset.shape}"
)


# ============================================================
# DATASET STATISTICS
# ============================================================

print()
print("=" * 60)
print("DATASET STATISTICS")
print("=" * 60)

print(
    f"Training samples   : "
    f"{len(U_train)}"
)

print(
    f"Test samples       : "
    f"{len(U_test_dataset)}"
)

print(
    f"Input channels     : "
    f"{U_train.shape[1]}"
)

print(
    f"Output channels    : "
    f"{Y_train.shape[1]}"
)

print(
    f"Temporal points    : "
    f"{U_train.shape[2]}"
)


# ============================================================
# NUMERICAL CHECKS
# ============================================================

assert U_train.shape == (
    CONFIG["train_samples"],
    INPUT_CHANNELS,
    N_TIME,
)

assert Y_train.shape == (
    CONFIG["train_samples"],
    N_LEVELS,
    N_TIME,
)

assert U_test_dataset.shape == (
    CONFIG["test_samples"],
    INPUT_CHANNELS,
    N_TIME,
)

assert Y_test_dataset.shape == (
    CONFIG["test_samples"],
    N_LEVELS,
    N_TIME,
)


# ------------------------------------------------------------
# Finite-value checks
# ------------------------------------------------------------

assert np.all(
    np.isfinite(U_train)
)

assert np.all(
    np.isfinite(Y_train)
)

assert np.all(
    np.isfinite(U_test_dataset)
)

assert np.all(
    np.isfinite(Y_test_dataset)
)


# ------------------------------------------------------------
# Population constraints
# ------------------------------------------------------------

assert np.all(
    Y_train >= -1e-10
)

assert np.all(
    Y_test_dataset >= -1e-10
)

assert np.allclose(
    np.sum(
        Y_train,
        axis=1,
    ),
    1.0,
    atol=1e-8,
)

assert np.allclose(
    np.sum(
        Y_test_dataset,
        axis=1,
    ),
    1.0,
    atol=1e-8,
)


# ------------------------------------------------------------
# Dataset independence check
# ------------------------------------------------------------

assert not np.array_equal(
    U_train[0],
    U_test_dataset[0],
)


print()
print("All Cell 11 checks passed.")

# ============================================================
# CELL 12 — CHANNELWISE STANDARDIZATION
# ============================================================

# ------------------------------------------------------------
# Compute statistics ONLY from the training set.
#
# Each input channel is standardized independently:
#
#     U_standardized = (U - mean) / std
#
# The training statistics are also used for the test set.
# ------------------------------------------------------------


# ============================================================
# INPUT STANDARDIZATION
# ============================================================

U_train_mean = np.mean(
    U_train,
    axis=(0, 2),
)

U_train_std = np.std(
    U_train,
    axis=(0, 2),
)


# ------------------------------------------------------------
# Protect against zero-variance channels
# ------------------------------------------------------------

U_train_std = np.where(
    U_train_std < 1e-12,
    1.0,
    U_train_std,
)


# ============================================================
# STANDARDIZE TRAINING INPUT
# ============================================================

U_train_standardized = (
    U_train
    - U_train_mean[None, :, None]
) / (
    U_train_std[None, :, None]
)


# ============================================================
# STANDARDIZE TEST INPUT
#
# IMPORTANT:
# Use training mean and standard deviation.
# Do NOT calculate statistics from the test set.
# ============================================================

U_test_standardized = (
    U_test_dataset
    - U_train_mean[None, :, None]
) / (
    U_train_std[None, :, None]
)


# ============================================================
# OUTPUT DATA
#
# Population trajectories remain in their physical form.
#
#     0 <= P_j(t) <= 1
#
# No output standardization is applied.
# ============================================================

Y_train_processed = Y_train.copy()

Y_test_processed = Y_test_dataset.copy()


# ============================================================
# DIAGNOSTICS
# ============================================================

print("=" * 60)
print("CHANNELWISE STANDARDIZATION")
print("=" * 60)

print()
print("Training input statistics")
print("-" * 60)

for c, name in enumerate(
    INPUT_CHANNEL_NAMES
):

    print(
        f"Channel {c} | "
        f"{name:10s} | "
        f"mean = {U_train_mean[c]: .6e} | "
        f"std = {U_train_std[c]: .6e}"
    )


print()
print("Standardized training statistics")
print("-" * 60)

for c, name in enumerate(
    INPUT_CHANNEL_NAMES
):

    channel = U_train_standardized[
        :, c, :
    ]

    print(
        f"Channel {c} | "
        f"{name:10s} | "
        f"mean = {np.mean(channel): .6e} | "
        f"std = {np.std(channel): .6e}"
    )


# ============================================================
# SHAPE CHECKS
# ============================================================

assert U_train_standardized.shape == (
    CONFIG["train_samples"],
    INPUT_CHANNELS,
    N_TIME,
)

assert U_test_standardized.shape == (
    CONFIG["test_samples"],
    INPUT_CHANNELS,
    N_TIME,
)

assert Y_train_processed.shape == (
    CONFIG["train_samples"],
    N_LEVELS,
    N_TIME,
)

assert Y_test_processed.shape == (
    CONFIG["test_samples"],
    N_LEVELS,
    N_TIME,
)


# ============================================================
# FINITE-VALUE CHECKS
# ============================================================

assert np.all(
    np.isfinite(
        U_train_standardized
    )
)

assert np.all(
    np.isfinite(
        U_test_standardized
    )
)


# ============================================================
# TRAINING STANDARDIZATION CHECK
# ============================================================

for c in range(
    INPUT_CHANNELS
):

    channel = U_train_standardized[
        :, c, :
    ]

    assert np.isclose(
        np.mean(channel),
        0.0,
        atol=1e-10,
    )

    assert np.isclose(
        np.std(channel),
        1.0,
        atol=1e-10,
    )


# ============================================================
# OUTPUT PHYSICAL CHECKS
# ============================================================

assert np.all(
    Y_train_processed >= -1e-10
)

assert np.all(
    Y_test_processed >= -1e-10
)

assert np.allclose(
    np.sum(
        Y_train_processed,
        axis=1,
    ),
    1.0,
    atol=1e-8,
)

assert np.allclose(
    np.sum(
        Y_test_processed,
        axis=1,
    ),
    1.0,
    atol=1e-8,
)


# ============================================================
# FINAL INFORMATION
# ============================================================

print()
print(
    f"Training samples : "
    f"{U_train_standardized.shape[0]}"
)

print(
    f"Test samples     : "
    f"{U_test_standardized.shape[0]}"
)

print(
    f"Input shape       : "
    f"{U_train_standardized.shape[1:]}"
)

print(
    f"Output shape      : "
    f"{Y_train_processed.shape[1:]}"
)

print()
print("Input standardization : PASSED")
print("Output probabilities   : UNCHANGED")
print("Training/test leakage  : NONE")
print()
print("All Cell 12 checks passed.")

# ============================================================
# CELL 13 — PYTORCH DATASET AND DATALOADERS
# ============================================================


# ============================================================
# PYTORCH DATASET
# ============================================================

class TransmonOperatorDataset(Dataset):

    def __init__(
        self,
        U,
        Y,
    ):
        """
        Parameters
        ----------
        U : ndarray
            Input trajectories.

            Shape:
                (N_samples, 4, N_time)

        Y : ndarray
            Output population trajectories.

            Shape:
                (N_samples, 10, N_time)
        """

        # ----------------------------------------------------
        # Convert NumPy arrays to PyTorch tensors
        # ----------------------------------------------------

        self.U = torch.tensor(
            U,
            dtype=CONFIG["dtype"],
        )

        self.Y = torch.tensor(
            Y,
            dtype=CONFIG["dtype"],
        )

        # ----------------------------------------------------
        # Basic shape checks
        # ----------------------------------------------------

        assert self.U.ndim == 3
        assert self.Y.ndim == 3

        assert self.U.shape[1] == INPUT_CHANNELS
        assert self.U.shape[2] == N_TIME

        assert self.Y.shape[1] == N_LEVELS
        assert self.Y.shape[2] == N_TIME

        assert self.U.shape[0] == self.Y.shape[0]


    def __len__(self):

        return self.U.shape[0]


    def __getitem__(
        self,
        index,
    ):

        return (
            self.U[index],
            self.Y[index],
        )


# ============================================================
# CREATE DATASETS
# ============================================================

train_dataset = TransmonOperatorDataset(
    U_train_standardized,
    Y_train_processed,
)

test_dataset = TransmonOperatorDataset(
    U_test_standardized,
    Y_test_processed,
)


# ============================================================
# CREATE DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=True,
    drop_last=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=False,
    drop_last=False,
)


# ============================================================
# INSPECT ONE BATCH
# ============================================================

U_batch, Y_batch = next(
    iter(train_loader)
)


# ============================================================
# DIAGNOSTICS
# ============================================================

print("=" * 60)
print("PYTORCH DATASET AND DATALOADERS")
print("=" * 60)

print()
print("Training dataset")
print("-" * 60)

print(
    f"Number of samples : "
    f"{len(train_dataset)}"
)

print(
    f"Input shape       : "
    f"{train_dataset.U.shape}"
)

print(
    f"Output shape      : "
    f"{train_dataset.Y.shape}"
)


print()
print("Test dataset")
print("-" * 60)

print(
    f"Number of samples : "
    f"{len(test_dataset)}"
)

print(
    f"Input shape       : "
    f"{test_dataset.U.shape}"
)

print(
    f"Output shape      : "
    f"{test_dataset.Y.shape}"
)


print()
print("Training batch")
print("-" * 60)

print(
    f"U batch shape     : "
    f"{U_batch.shape}"
)

print(
    f"Y batch shape     : "
    f"{Y_batch.shape}"
)

print(
    f"U dtype           : "
    f"{U_batch.dtype}"
)

print(
    f"Y dtype           : "
    f"{Y_batch.dtype}"
)


print()
print("Device")
print("-" * 60)

print(
    f"Training device   : "
    f"{DEVICE}"
)


# ============================================================
# EXPECTED SHAPES
# ============================================================

assert train_dataset.U.shape == (
    CONFIG["train_samples"],
    INPUT_CHANNELS,
    N_TIME,
)

assert train_dataset.Y.shape == (
    CONFIG["train_samples"],
    N_LEVELS,
    N_TIME,
)

assert test_dataset.U.shape == (
    CONFIG["test_samples"],
    INPUT_CHANNELS,
    N_TIME,
)

assert test_dataset.Y.shape == (
    CONFIG["test_samples"],
    N_LEVELS,
    N_TIME,
)


# ============================================================
# BATCH SHAPE CHECKS
# ============================================================

assert U_batch.ndim == 3
assert Y_batch.ndim == 3

assert U_batch.shape[1] == INPUT_CHANNELS
assert U_batch.shape[2] == N_TIME

assert Y_batch.shape[1] == N_LEVELS
assert Y_batch.shape[2] == N_TIME

assert U_batch.shape[0] <= CONFIG["batch_size"]
assert Y_batch.shape[0] <= CONFIG["batch_size"]


# ============================================================
# DTYPE CHECKS
# ============================================================

assert U_batch.dtype == CONFIG["dtype"]
assert Y_batch.dtype == CONFIG["dtype"]


# ============================================================
# FINITE-VALUE CHECKS
# ============================================================

assert torch.isfinite(
    U_batch
).all()

assert torch.isfinite(
    Y_batch
).all()


# ============================================================
# POPULATION CHECK
# ============================================================

assert torch.all(
    Y_batch >= -1e-10
)

population_sum = torch.sum(
    Y_batch,
    dim=1,
)

assert torch.allclose(
    population_sum,
    torch.ones_like(
        population_sum
    ),
    atol=1e-6,
)


print()
print("Tensor format:")
print("U = [B, 4, N]")
print("Y = [B, 10, N]")

print()
print("All Cell 13 checks passed.")

# ============================================================
# CELL 14 — FOURIER NEURAL OPERATOR: SPECTRAL CONVOLUTION
# ============================================================


class SpectralConv1d(nn.Module):
    """
    One-dimensional Fourier spectral convolution.

    Input:
        x : [B, C_in, N]

    Output:
        y : [B, C_out, N]

    Only the lowest `modes` Fourier modes are learned.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        modes,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        # ----------------------------------------------------
        # Complex Fourier weights
        #
        # Shape:
        #     [C_in, C_out, modes]
        # ----------------------------------------------------

        scale = (
            1.0
            / (
                in_channels
                * out_channels
            )
        )

        self.weights = nn.Parameter(
            scale
            * torch.randn(
                in_channels,
                out_channels,
                modes,
                dtype=torch.cfloat,
            )
        )


    def forward(
        self,
        x,
    ):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape [B, C_in, N]

        Returns
        -------
        torch.Tensor
            Shape [B, C_out, N]
        """

        batch_size = x.shape[0]
        n_points = x.shape[-1]

        # ----------------------------------------------------
        # Fourier transform
        # ----------------------------------------------------

        x_ft = torch.fft.rfft(
            x,
            dim=-1,
        )

        # ----------------------------------------------------
        # Allocate Fourier-space output
        # ----------------------------------------------------

        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            x_ft.shape[-1],
            dtype=torch.cfloat,
            device=x.device,
        )

        # ----------------------------------------------------
        # Number of usable Fourier modes
        # ----------------------------------------------------

        usable_modes = min(
            self.modes,
            x_ft.shape[-1],
        )

        # ----------------------------------------------------
        # Spectral multiplication
        #
        # out_ft[b,o,k]
        #
        # = sum_i
        #     x_ft[b,i,k]
        #     W[i,o,k]
        # ----------------------------------------------------

        out_ft[
            :,
            :,
            :usable_modes
        ] = torch.einsum(
            "bik,iok->bok",
            x_ft[
                :,
                :,
                :usable_modes
            ],
            self.weights[
                :,
                :,
                :usable_modes
            ],
        )

        # ----------------------------------------------------
        # Inverse Fourier transform
        # ----------------------------------------------------

        x = torch.fft.irfft(
            out_ft,
            n=n_points,
            dim=-1,
        )

        return x


# ============================================================
# TEST SPECTRAL CONVOLUTION
# ============================================================

FNO_MODES = min(
    16,
    N_TIME // 2 + 1,
)

FNO_WIDTH = 32


spectral_test = SpectralConv1d(
    in_channels=FNO_WIDTH,
    out_channels=FNO_WIDTH,
    modes=FNO_MODES,
).to(DEVICE)


x_test = torch.randn(
    2,
    FNO_WIDTH,
    N_TIME,
    dtype=CONFIG["dtype"],
    device=DEVICE,
)


y_test = spectral_test(
    x_test
)


# ============================================================
# DIAGNOSTICS
# ============================================================

print("=" * 60)
print("FOURIER SPECTRAL CONVOLUTION")
print("=" * 60)

print(
    f"Input shape          : "
    f"{x_test.shape}"
)

print(
    f"Output shape         : "
    f"{y_test.shape}"
)

print(
    f"Fourier modes        : "
    f"{FNO_MODES}"
)

print(
    f"Hidden width         : "
    f"{FNO_WIDTH}"
)

print(
    f"Device               : "
    f"{DEVICE}"
)


# ============================================================
# NUMERICAL CHECKS
# ============================================================

assert y_test.shape == (
    2,
    FNO_WIDTH,
    N_TIME,
)

assert torch.isfinite(
    y_test
).all()

assert y_test.dtype == CONFIG["dtype"]


print()
print("All Cell 14 checks passed.")

# ============================================================
# CELL 15 — FOURIER NEURAL OPERATOR MODEL
# ============================================================


class FNO1d(nn.Module):
    """
    One-dimensional Fourier Neural Operator.

    Input:
        [B, C_in, N]

    Output:
        [B, C_out, N]
    """

    def __init__(
        self,
        input_channels,
        output_channels,
        modes,
        width,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.modes = modes
        self.width = width

        # ----------------------------------------------------
        # Input projection
        #
        # Maps the physical input channels to the latent
        # feature dimension.
        # ----------------------------------------------------

        self.input_projection = nn.Conv1d(
            input_channels,
            width,
            kernel_size=1,
        )

        # ----------------------------------------------------
        # Fourier layers
        # ----------------------------------------------------

        self.spectral_layers = nn.ModuleList(
            [
                SpectralConv1d(
                    width,
                    width,
                    modes,
                )
                for _ in range(4)
            ]
        )

        # ----------------------------------------------------
        # Local pointwise transformations
        # ----------------------------------------------------

        self.pointwise_layers = nn.ModuleList(
            [
                nn.Conv1d(
                    width,
                    width,
                    kernel_size=1,
                )
                for _ in range(4)
            ]
        )

        # ----------------------------------------------------
        # Output projection
        # ----------------------------------------------------

        self.output_projection_1 = nn.Conv1d(
            width,
            width,
            kernel_size=1,
        )

        self.output_projection_2 = nn.Conv1d(
            width,
            output_channels,
            kernel_size=1,
        )


    def forward(
        self,
        x,
    ):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape [B, C_in, N]

        Returns
        -------
        torch.Tensor
            Shape [B, C_out, N]
        """

        # ----------------------------------------------------
        # Input lifting
        # ----------------------------------------------------

        x = self.input_projection(
            x
        )

        # ----------------------------------------------------
        # Fourier operator blocks
        # ----------------------------------------------------

        for spectral_layer, pointwise_layer in zip(
            self.spectral_layers,
            self.pointwise_layers,
        ):

            spectral_output = spectral_layer(
                x
            )

            pointwise_output = pointwise_layer(
                x
            )

            x = (
                spectral_output
                + pointwise_output
            )

            x = F.gelu(
                x
            )

        # ----------------------------------------------------
        # Output projection
        # ----------------------------------------------------

        x = self.output_projection_1(
            x
        )

        x = F.gelu(
            x
        )

        x = self.output_projection_2(
            x
        )

        return x


# ============================================================
# FNO CONFIGURATION
# ============================================================

FNO_MODES = min(
    16,
    N_TIME // 2 + 1,
)

FNO_WIDTH = 32


# ============================================================
# CREATE FNO
# ============================================================

fno_model = FNO1d(
    input_channels=INPUT_CHANNELS,
    output_channels=N_LEVELS,
    modes=FNO_MODES,
    width=FNO_WIDTH,
).to(DEVICE)


# ============================================================
# TEST FORWARD PASS
# ============================================================

with torch.no_grad():

    fno_test_output = fno_model(
        U_batch.to(DEVICE)
    )


# ============================================================
# DIAGNOSTICS
# ============================================================

print("=" * 60)
print("FOURIER NEURAL OPERATOR")
print("=" * 60)

print()
print(f"Input channels      : {INPUT_CHANNELS}")
print(f"Output channels     : {N_LEVELS}")
print(f"Fourier modes       : {FNO_MODES}")
print(f"Hidden width        : {FNO_WIDTH}")
print(f"Fourier layers      : {len(fno_model.spectral_layers)}")
print(f"Device              : {DEVICE}")

print()
print(
    f"Input shape         : "
    f"{U_batch.shape}"
)

print(
    f"Output shape        : "
    f"{fno_test_output.shape}"
)


# ============================================================
# PARAMETER COUNT
# ============================================================

number_of_parameters = sum(
    parameter.numel()
    for parameter in fno_model.parameters()
)

print()
print(
    f"Trainable parameters: "
    f"{number_of_parameters:,}"
)


# ============================================================
# NUMERICAL CHECKS
# ============================================================

assert fno_test_output.shape == (
    U_batch.shape[0],
    N_LEVELS,
    N_TIME,
)

assert torch.isfinite(
    fno_test_output
).all()

assert fno_test_output.dtype == (
    CONFIG["dtype"]
)


# ============================================================
# MODEL CHECK
# ============================================================

assert (
    len(fno_model.spectral_layers)
    == 4
)

assert (
    len(fno_model.pointwise_layers)
    == 4
)

assert (
    number_of_parameters > 0
)


print()
print("All Cell 15 checks passed.")

# ============================================================
# CELL 16 — FNO TRAINING SETUP
# ============================================================


# ============================================================
# LOSS FUNCTION
# ============================================================

criterion = nn.MSELoss()


# ============================================================
# OPTIMIZER
# ============================================================

fno_optimizer = torch.optim.Adam(
    fno_model.parameters(),
    lr=CONFIG["learning_rate"],
)


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

FNO_EPOCHS = CONFIG["epochs"]


# ============================================================
# TRAINING HISTORY
# ============================================================

fno_training_loss = []


# ============================================================
# DIAGNOSTICS
# ============================================================

print("=" * 60)
print("FNO TRAINING SETUP")
print("=" * 60)

print()
print(
    f"Model               : FNO1d"
)

print(
    f"Trainable parameters: "
    f"{sum(p.numel() for p in fno_model.parameters()):,}"
)

print(
    f"Loss                : MSE"
)

print(
    f"Optimizer           : Adam"
)

print(
    f"Learning rate       : "
    f"{CONFIG['learning_rate']}"
)

print(
    f"Batch size          : "
    f"{CONFIG['batch_size']}"
)

print(
    f"Epochs              : "
    f"{FNO_EPOCHS}"
)

print(
    f"Device              : "
    f"{DEVICE}"
)


# ============================================================
# LOSS TEST
# ============================================================

loss_test = criterion(
    fno_test_output,
    Y_batch.to(DEVICE),
)


print()
print(
    f"Initial test MSE    : "
    f"{loss_test.item():.6e}"
)


# ============================================================
# NUMERICAL CHECK
# ============================================================

assert isinstance(
    criterion,
    nn.MSELoss,
)

assert isinstance(
    fno_optimizer,
    torch.optim.Adam,
)

assert (
    fno_optimizer.param_groups[0]["lr"]
    == CONFIG["learning_rate"]
)

assert FNO_EPOCHS == CONFIG["epochs"]

assert torch.isfinite(
    loss_test
)

assert loss_test.item() >= 0.0


print()
print("All Cell 16 checks passed.")

# ============================================================
# CELL 17 — FNO TRAINING LOOP
# ============================================================

print("=" * 60)
print("TRAINING FNO")
print("=" * 60)

fno_model.train()

fno_training_loss = []

training_start_time = time.time()

for epoch in range(
    1,
    FNO_EPOCHS + 1,
):

    epoch_start_time = time.time()

    running_loss = 0.0
    n_batches = 0

    for U_batch, Y_batch in train_loader:

        U_batch = U_batch.to(
            DEVICE
        )

        Y_batch = Y_batch.to(
            DEVICE
        )

        # ----------------------------------------------------
        # Clear gradients
        # ----------------------------------------------------

        fno_optimizer.zero_grad()

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        prediction = fno_model(
            U_batch
        )

        # ----------------------------------------------------
        # MSE loss
        # ----------------------------------------------------

        loss = criterion(
            prediction,
            Y_batch
        )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()

        # ----------------------------------------------------
        # Parameter update
        # ----------------------------------------------------

        fno_optimizer.step()

        running_loss += loss.item()
        n_batches += 1

    epoch_loss = (
        running_loss
        / n_batches
    )

    fno_training_loss.append(
        epoch_loss
    )

    epoch_time = (
        time.time()
        - epoch_start_time
    )

    print(
        f"Epoch "
        f"{epoch:4d}/{FNO_EPOCHS:4d} | "
        f"MSE = {epoch_loss:.6e} | "
        f"Time = {epoch_time:.2f} s"
    )


total_training_time = (
    time.time()
    - training_start_time
)

print()
print(
    f"Total training time: "
    f"{total_training_time:.2f} s"
)


# ============================================================
# TRAINING CHECKS
# ============================================================

assert len(
    fno_training_loss
) == FNO_EPOCHS

assert np.all(
    np.isfinite(
        fno_training_loss
    )
)

assert np.all(
    np.asarray(
        fno_training_loss
    ) >= 0.0
)

assert (
    fno_training_loss[-1]
    <= fno_training_loss[0]
    or FNO_EPOCHS == 1
)


print()
print("Final training MSE:")
print(
    f"{fno_training_loss[-1]:.6e}"
)

print()
print("All Cell 17 checks passed.")

# ============================================================
# CELL 18 — FNO TEST EVALUATION
# ============================================================

fno_model.eval()

fno_test_mse = 0.0
n_test_batches = 0

all_fno_predictions = []
all_fno_targets = []


with torch.no_grad():

    for U_batch, Y_batch in test_loader:

        U_batch = U_batch.to(
            DEVICE
        )

        Y_batch = Y_batch.to(
            DEVICE
        )

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        prediction = fno_model(
            U_batch
        )

        # ----------------------------------------------------
        # MSE
        # ----------------------------------------------------

        batch_mse = criterion(
            prediction,
            Y_batch
        )

        fno_test_mse += (
            batch_mse.item()
        )

        n_test_batches += 1

        # ----------------------------------------------------
        # Store predictions and targets
        # ----------------------------------------------------

        all_fno_predictions.append(
            prediction.cpu()
        )

        all_fno_targets.append(
            Y_batch.cpu()
        )


fno_test_mse /= n_test_batches


# ============================================================
# COMBINE ALL TEST SAMPLES
# ============================================================

fno_predictions = torch.cat(
    all_fno_predictions,
    dim=0
)

fno_targets = torch.cat(
    all_fno_targets,
    dim=0
)


# ============================================================
# RELATIVE L2 ERROR
# ============================================================

prediction_error = (
    fno_predictions
    - fno_targets
)

numerator = torch.linalg.vector_norm(
    prediction_error.reshape(
        prediction_error.shape[0],
        -1
    ),
    dim=1,
)

denominator = torch.linalg.vector_norm(
    fno_targets.reshape(
        fno_targets.shape[0],
        -1
    ),
    dim=1,
)

relative_l2_per_sample = (
    numerator
    / torch.clamp(
        denominator,
        min=1e-12,
    )
)

fno_relative_l2 = (
    torch.mean(
        relative_l2_per_sample
    ).item()
)


# ============================================================
# POPULATION NORMALIZATION CHECK
# ============================================================

fno_prediction_population_sum = (
    torch.sum(
        fno_predictions,
        dim=1,
    )
)

fno_target_population_sum = (
    torch.sum(
        fno_targets,
        dim=1,
    )
)


# ============================================================
# DIAGNOSTICS
# ============================================================

print("=" * 60)
print("FNO TEST EVALUATION")
print("=" * 60)

print()
print(
    f"Test samples        : "
    f"{len(test_dataset)}"
)

print(
    f"Test MSE            : "
    f"{fno_test_mse:.6e}"
)

print(
    f"Relative L2 error   : "
    f"{fno_relative_l2:.6e}"
)

print()
print("Prediction tensor")
print("-" * 60)
print(
    f"Shape               : "
    f"{fno_predictions.shape}"
)

print(
    f"Minimum             : "
    f"{fno_predictions.min().item():.6e}"
)

print(
    f"Maximum             : "
    f"{fno_predictions.max().item():.6e}"
)

print()
print("Target tensor")
print("-" * 60)
print(
    f"Shape               : "
    f"{fno_targets.shape}"
)


# ============================================================
# NUMERICAL CHECKS
# ============================================================

assert fno_predictions.shape == (
    CONFIG["test_samples"],
    N_LEVELS,
    N_TIME,
)

assert fno_targets.shape == (
    CONFIG["test_samples"],
    N_LEVELS,
    N_TIME,
)

assert torch.isfinite(
    fno_predictions
).all()

assert torch.isfinite(
    fno_targets
).all()

assert np.isfinite(
    fno_test_mse
)

assert np.isfinite(
    fno_relative_l2
)

assert fno_test_mse >= 0.0
assert fno_relative_l2 >= 0.0

assert torch.allclose(
    fno_target_population_sum,
    torch.ones_like(
        fno_target_population_sum
    ),
    atol=1e-6,
)


print()
print("FNO test MSE          : PASSED")
print("FNO Relative L2       : PASSED")
print("Prediction shape      : PASSED")
print("Target normalization  : PASSED")
print()
print("All Cell 18 checks passed.")

# ============================================================
# CELL 19 — FNO RELATIVE TEMPORAL H1 ERROR
# ============================================================


# ============================================================
# TEMPORAL GRID
# ============================================================

dt_operator = CONFIG[
    "operator_dt_ns"
]


# ============================================================
# FINITE-DIFFERENCE TEMPORAL DERIVATIVE
# ============================================================

def temporal_derivative(
    tensor,
    dt,
):
    """
    Compute the temporal derivative using
    second-order finite differences.

    Input:
        tensor : [B, C, N]

    Output:
        derivative : [B, C, N]
    """

    derivative = torch.zeros_like(
        tensor
    )

    # --------------------------------------------------------
    # Interior points:
    # centered finite difference
    # --------------------------------------------------------

    derivative[
        :,
        :,
        1:-1
    ] = (
        tensor[
            :,
            :,
            2:
        ]
        -
        tensor[
            :,
            :,
            :-2
        ]
    ) / (
        2.0 * dt
    )

    # --------------------------------------------------------
    # Initial point:
    # forward difference
    # --------------------------------------------------------

    derivative[
        :,
        :,
        0
    ] = (
        tensor[
            :,
            :,
            1
        ]
        -
        tensor[
            :,
            :,
            0
        ]
    ) / dt

    # --------------------------------------------------------
    # Final point:
    # backward difference
    # --------------------------------------------------------

    derivative[
        :,
        :,
        -1
    ] = (
        tensor[
            :,
            :,
            -1
        ]
        -
        tensor[
            :,
            :,
            -2
        ]
    ) / dt

    return derivative


# ============================================================
# COMPUTE TEMPORAL DERIVATIVES
# ============================================================

fno_prediction_dt = temporal_derivative(
    fno_predictions,
    dt_operator,
)

fno_target_dt = temporal_derivative(
    fno_targets,
    dt_operator,
)


# ============================================================
# TEMPORAL H1 ERROR
# ============================================================

function_difference = (
    fno_predictions
    - fno_targets
)

derivative_difference = (
    fno_prediction_dt
    - fno_target_dt
)


function_error_squared = (
    torch.sum(
        function_difference ** 2,
        dim=(1, 2),
    )
)

derivative_error_squared = (
    torch.sum(
        derivative_difference ** 2,
        dim=(1, 2),
    )
)

target_function_squared = (
    torch.sum(
        fno_targets ** 2,
        dim=(1, 2),
    )
)

target_derivative_squared = (
    torch.sum(
        fno_target_dt ** 2,
        dim=(1, 2),
    )
)


# ------------------------------------------------------------
# H1 seminorm + L2 norm
# ------------------------------------------------------------

h1_numerator = torch.sqrt(
    function_error_squared
    + derivative_error_squared
)

h1_denominator = torch.sqrt(
    target_function_squared
    + target_derivative_squared
)

relative_temporal_h1_per_sample = (
    h1_numerator
    / torch.clamp(
        h1_denominator,
        min=1e-12,
    )
)

fno_relative_temporal_h1 = (
    torch.mean(
        relative_temporal_h1_per_sample
    ).item()
)


# ============================================================
# DIAGNOSTICS
# ============================================================

print("=" * 60)
print("FNO RELATIVE TEMPORAL H1 ERROR")
print("=" * 60)

print()
print(
    f"Operator dt         : "
    f"{dt_operator} ns"
)

print(
    f"Test samples        : "
    f"{fno_predictions.shape[0]}"
)

print(
    f"Temporal points     : "
    f"{fno_predictions.shape[2]}"
)

print()
print(
    f"Relative L2         : "
    f"{fno_relative_l2:.6e}"
)

print(
    f"Relative temporal H1: "
    f"{fno_relative_temporal_h1:.6e}"
)


# ============================================================
# NUMERICAL CHECKS
# ============================================================

assert fno_prediction_dt.shape == (
    CONFIG["test_samples"],
    N_LEVELS,
    N_TIME,
)

assert fno_target_dt.shape == (
    CONFIG["test_samples"],
    N_LEVELS,
    N_TIME,
)

assert torch.isfinite(
    fno_prediction_dt
).all()

assert torch.isfinite(
    fno_target_dt
).all()

assert np.isfinite(
    fno_relative_temporal_h1
)

assert (
    fno_relative_temporal_h1 >= 0.0
)

print()
print(
    "Temporal derivative     : PASSED"
)

print(
    "Relative temporal H1    : PASSED"
)

print()
print("All Cell 19 checks passed.")

# ============================================================
# CELL 20 — FNO TRAINING CURVE
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    range(
        1,
        FNO_EPOCHS + 1
    ),
    fno_training_loss,
    marker="o",
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Training MSE"
)

plt.title(
    "FNO Training Loss"
)

plt.grid(
    True,
    alpha=0.3,
)

plt.tight_layout()

plt.show()


# ============================================================
# TRAINING SUMMARY
# ============================================================

print("=" * 60)
print("FNO TRAINING SUMMARY")
print("=" * 60)

print()
print(
    f"Initial training MSE : "
    f"{fno_training_loss[0]:.6e}"
)

print(
    f"Final training MSE   : "
    f"{fno_training_loss[-1]:.6e}"
)

print(
    f"Test MSE              : "
    f"{fno_test_mse:.6e}"
)

print(
    f"Relative L2           : "
    f"{fno_relative_l2:.6e}"
)

print(
    f"Relative temporal H1  : "
    f"{fno_relative_temporal_h1:.6e}"
)

print(
    f"Total training time   : "
    f"{total_training_time:.2f} s"
)


# ============================================================
# CHECKS
# ============================================================

assert len(
    fno_training_loss
) == FNO_EPOCHS

assert np.all(
    np.isfinite(
        fno_training_loss
    )
)

assert np.isfinite(
    fno_test_mse
)

assert np.isfinite(
    fno_relative_l2
)

assert np.isfinite(
    fno_relative_temporal_h1
)

print()
print("Training history : PASSED")
print("Evaluation metrics: PASSED")
print()
print("All Cell 20 checks passed.")


# ============================================================
# CELL 21 — FNO QUALITATIVE TRAJECTORY COMPARISON
# ============================================================

# Select one test trajectory and compare the predicted
# and true populations for all ten transmon levels.

sample_index = 0

fno_sample_prediction = (
    fno_predictions[
        sample_index
    ]
    .numpy()
)

fno_sample_target = (
    fno_targets[
        sample_index
    ]
    .numpy()
)


# ============================================================
# PLOT POPULATION TRAJECTORIES
# ============================================================

plt.figure(
    figsize=(10, 6)
)

for level in range(
    N_LEVELS
):

    plt.plot(
        TIME_GRID,
        fno_sample_target[level],
        linestyle="-",
        label=f"Level {level} — True",
    )

    plt.plot(
        TIME_GRID,
        fno_sample_prediction[level],
        linestyle="--",
        label=f"Level {level} — FNO",
    )


plt.xlabel(
    "Time (ns)"
)

plt.ylabel(
    "Population"
)

plt.title(
    "FNO Prediction vs True Population Trajectories"
)

plt.legend(
    ncol=2,
    fontsize=8,
)

plt.grid(
    True,
    alpha=0.3,
)

plt.tight_layout()

plt.show()


# ============================================================
# POPULATION-SUM COMPARISON
# ============================================================

true_population_sum = np.sum(
    fno_sample_target,
    axis=0,
)

predicted_population_sum = np.sum(
    fno_sample_prediction,
    axis=0,
)


plt.figure(
    figsize=(10, 4)
)

plt.plot(
    TIME_GRID,
    true_population_sum,
    label="True",
)

plt.plot(
    TIME_GRID,
    predicted_population_sum,
    linestyle="--",
    label="FNO",
)

plt.xlabel(
    "Time (ns)"
)

plt.ylabel(
    "Total population"
)

plt.title(
    "Population Conservation: FNO vs True"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3,
)

plt.tight_layout()

plt.show()


# ============================================================
# SAMPLE-LEVEL METRICS
# ============================================================

sample_error = (
    fno_sample_prediction
    - fno_sample_target
)

sample_relative_l2 = (
    np.linalg.norm(
        sample_error.ravel()
    )
    /
    max(
        np.linalg.norm(
            fno_sample_target.ravel()
        ),
        1e-12,
    )
)

sample_population_sum_error = np.max(
    np.abs(
        predicted_population_sum
        - true_population_sum
    )
)


# ============================================================
# DIAGNOSTICS
# ============================================================

print("=" * 60)
print("FNO QUALITATIVE TRAJECTORY COMPARISON")
print("=" * 60)

print()
print(
    f"Test sample index          : "
    f"{sample_index}"
)

print(
    f"Population trajectory shape : "
    f"{fno_sample_target.shape}"
)

print(
    f"Sample Relative L2          : "
    f"{sample_relative_l2:.6e}"
)

print(
    f"Maximum population-sum error: "
    f"{sample_population_sum_error:.6e}"
)


# ============================================================
# CHECKS
# ============================================================

assert fno_sample_target.shape == (
    N_LEVELS,
    N_TIME,
)

assert fno_sample_prediction.shape == (
    N_LEVELS,
    N_TIME,
)

assert np.all(
    np.isfinite(
        fno_sample_target
    )
)

assert np.all(
    np.isfinite(
        fno_sample_prediction
    )
)

assert np.isfinite(
    sample_relative_l2
)

assert sample_relative_l2 >= 0.0

assert np.isfinite(
    sample_population_sum_error
)

print()
print("Prediction trajectory : PASSED")
print("Target trajectory     : PASSED")
print("Sample Relative L2    : PASSED")
print("Population comparison : PASSED")
print()
print("All Cell 21 checks passed.")
