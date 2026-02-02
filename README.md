# Analysis and Modeling of the Interferences Affecting Space Communication Links

A Python-based simulation framework for analyzing Radio Frequency Interference (RFI) effects on space communication systems.

## Overview

This project implements ITU-R recommendation-based models to quantify how interference from various sources affects victim satellite links across multiple frequency bands (S, X, Ku, K, and Ka bands).

## Key Features

- **Multi-band Analysis**: Evaluates interference across 5 frequency bands commonly used in space communications
- **ITU-R Compliant Models**: Implements standards including:
  - ITU-R P.525 (Free-space path loss)
  - ITU-R S.1528 (Antenna patterns)
  - ITU-R S.1325 (Interference methodology)
- **Statistical Characterization**: Log-normal interference modeling with configurable duty cycles
- **Metrics Computed**:
  - Baseline and degraded SNR
  - Carrier-to-Interference ratio (C/I)
  - Equivalent Power Flux Density (EPFD)
  - Time-fraction exceedance probabilities

## Project Structure

```
rfi-model/
├── rfi/
│   ├── scenario.py      # Main scenario engine and configurations
│   ├── itu_models.py    # Core ITU equation implementations
│   └── equations_itu.py # Alternative/legacy equations
├── notebooks/
│   └── Untitled.ipynb   # Analysis and visualization notebook
├── data/                # Input data files
└── requirements.txt     # Python dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from rfi.scenario import run_multi_entry_rfi_scenario, VICTIM_BANDS, INTERFERENCE_SCENARIOS

# Run a scenario
result = run_multi_entry_rfi_scenario(
    band_params=VICTIM_BANDS["Ka-band"],
    interferer_list=INTERFERENCE_SCENARIOS["Moderate"],
    time_sim_samples=5000,
)

print(f"SNR Loss: {result['SNR Loss (dB)']:.2f} dB")
```

## Interference Scenarios

Three pre-defined interference scenarios:
- **Weak**: Low EIRP, distant interferer, large off-axis angle
- **Moderate**: Medium parameters
- **Strong**: High EIRP, close interferer, small off-axis angle

## Author

Utkarsh Maurya
