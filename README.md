# Trainable Optimal-Transport Ambiguity Sets in Distributionally Robust Optimization

Implementation accompanying:

> *Loss-aware distributionally robust optimization via trainable optimal transport ambiguity sets*, Preprint, 2025

<p align="center">
  <img src="https://github.com/user-attachments/assets/321c2670-0f3b-4f32-bb62-423a1790f13f" alt="Sketch of Shaping Ambiguity Sets" width="30%"/>
  <em>Figure 1: Sketch of Shaping Ambiguity Sets</em>
</p>

## Installation
```bash
git clone https://github.com/JonasOhn/trainable-ot-dro.git
cd trainable-ot-dro
pip install -e .
```

### Dependencies
Requires Python ≥ 3.10.
Core dependencies (via `pyproject.toml`): `numpy`, `scipy`, `POT`, `cvxpy`, `matplotlib==3.9.0`, `clarabel` (as the CP solver).

## Quick check
After installing:
```bash
ot-dro-demo
```

Expected:
```
trainable_ot_dro imported successfully.
Core modules available.
```

## Repository structure

```
trainable-ot-dro/
├─ examples/                                # Examples of the end-to-end pipeline
│  ├─ linreg_dro.py                         # Linear regression DRO demo
│  ├─ portfolioopt_discrete.py              # Portfolio DRO (discrete)
│  ├─ portfolioopt_gaussian.py              # Portfolio DRO (Gaussian)
│  └─ portfolioopt_gaussianmixture.py       # Portfolio DRO (Gaussian mixture)
├─ results/                                 # Outputs (arrays, logs, figures)
│  └─ README.md                             # Notes on result files
├─ src/
│  └─ trainable_ot_dro/
│     ├─ __init__.py
│     ├─ cli.py                             # Lightweight smoke-test CLI
│     ├─ bilevel_optimization.py            # Bilevel training loop
│     ├─ cones.py                           # Cone definitions
│     ├─ conic_problem.py                   # Generic conic problem class
│     ├─ reformulations.py                  # Reformulations: DRO --> Conic form
│     └─ utils/
│        ├─ __init__.py
│        ├─ ellipses.py                     # Helpers for plotting ellipses
│        ├─ gelbrich_distance.py            # Gelbrich distance + gradients
│        ├─ generating_distributions.py     # Distribution generators
│        ├─ numerical_utilities.py          # Numeric stabilization methods
│        ├─ risk_measures.py                # CVaR, VaR calculation
│        ├─ sampling_from_distributions.py  # Methods to sample from distributions
│        └─ wasserstein_distance.py         # Wasserstein distance + gradients
├─ pyproject.toml                           # Python project setup (dependencies etc.)
├─ README.md
└─ .gitignore
```

## Example scripts
All examples assume the package is installed.

```bash
python examples/linreg_dro.py
python examples/portfolioopt_gaussian.py
python examples/portfolioopt_gaussianmixture.py
python examples/portfolioopt_discrete.py
```
Each script writes a timestamped `.npy` result to `results/...` and saves figures under `results/` with a compact, informative, timestamped filename.

## License
MIT.
