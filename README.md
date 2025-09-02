# Trainable Optimal-Transport Ambiguity Sets in Distributionally Robust Optimization

Implementation accompanying:

> *Loss-aware distributionally robust optimization via trainable optimal transport ambiguity sets*, Preprint, 2025

## Installation
```bash
git clone https://github.com/JonasOhn/trainable-ot-dro.git
cd trainable-ot-dro
pip install -e .
```

### Dependencies
Requires Python ≥ 3.10.
Core dependencies (via `pyproject.toml`): `numpy`, `scipy`, `cvxpy`, `matplotlib==3.9.0`, `clarabel` (as the CP solver).

## 🗂️ Repository structure

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

## Running the examples

All examples assume the package is installed.

```bash
python examples/linreg_dro.py
python examples/portfolioopt_gaussian.py
# …and so on
```

Each script writes a timestamped `.npy` result to `results/...` and saves figures under `results/figures/` with a compact, informative filename (problem, Wasserstein type, sample/boot counts, seed, timestamp).

## License

MIT.
