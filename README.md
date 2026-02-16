# Analysis and Modeling of Interferences Affecting Space Communication Links

A Python-based simulation framework for evaluating Radio Frequency Interference (RFI) impacts on space communication systems using ITU-R aligned models.

---

## Overview

This project implements a modular simulation engine to analyze dynamic, cross-band interference effects on satellite communication links across:

- S-band
- X-band
- Ku-band
- K-band
- Ka-band

The framework integrates deterministic orbital geometry, regulatory EPFD evaluation, adaptive modulation, atmospheric attenuation, and cross-band robustness ranking.

All results are generated from a single final Jupyter notebook.

---

## Implemented Capabilities

### Geometry-Driven Interference
- Circular LEO propagation model
- Time-varying slant range and off-axis angle
- Deterministic interference power I(t)
- Time-varying SNR degradation

### ITU-R Based Link Modeling
- ITU-R P.525 free-space path loss
- ITU-R S.1528 antenna pattern
- Aggregate interference computation
- EPFD calculation

### Regulatory Compliance
- EPFD time-series evaluation
- Exceedance probability computation
- Compliance classification (Compliant / Marginal / Non-Compliant)

### Capacity-Level Impact
- Adaptive modulation (QPSK, 8PSK, 16APSK, 32APSK)
- Spectral efficiency mapping
- Throughput time-series
- Modulation distribution statistics
- Link availability computation

### Atmospheric + RFI Joint Modeling
- Frequency-dependent rain attenuation
- Rain occurrence modeling
- Joint SNR degradation
- Joint outage probability

### Cross-Band Robustness Index (RRI)
- Combines:
  - Throughput degradation
  - Availability
  - EPFD exceedance
  - Joint outage
- Cross-band ranking under identical stress conditions

### 6G X-Band What-If Scenario
- High-EIRP stress testing
- Baseline vs 6G comparison
- Robustness impact assessment

### Sensitivity Analysis
- Interferer EIRP sweep
- Robustness vs interference strength
- Stability check via repeated runs

---

## Project Structure

```
rfi-model/
├── Project_Equations.tex    # Equation reference (LaTeX)
├── rfi/
│   ├── equations_itu.py
│   ├── geometry.py
│   ├── propagation.py
│   ├── link_adaptation.py
│   ├── compliance.py
│   └── robustness.py
├── notebooks/
│   └── Final_RFI_Extended_Analysis.ipynb
├── scripts/
│   └── build_latex.bat      # Build PDF from Project_Equations.tex
├── requirements.txt
└── README.md
```

---

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```
notebooks/Final_RFI_Extended_Analysis.ipynb
```

All simulations, comparisons, and visualizations are executed from this notebook.

To build the equation reference PDF: open `Project_Equations.tex` in the repo (e.g. with LaTeX Workshop in VS Code/Cursor) or run `scripts\build_latex.bat`. Output: `build/Project_Equations.pdf`.

---

**Author**  
Utkarsh Maurya
