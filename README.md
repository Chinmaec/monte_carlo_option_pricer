# Monte Carlo Option Pricer

## Overview
Monte Carlo pricing engine for derivative pricing under risk-neutral Geometric Brownian Motion, with integrated variance reduction and Greeks estimation.

Supports European and Asian options, quantifies estimator uncertainty, and validates European pricing against Black–Scholes.

## Methodology
- Simulate under $\mathbb{Q}$:

$$
S_{t+\Delta t}=S_t\exp\left((r-\frac{1}{2}\sigma^2)\Delta t+\sigma\sqrt{\Delta t}\,Z\right),\quad Z\sim\mathcal{N}(0,1)
$$
- Price estimator:

$$
\hat V_0=e^{-rT}\frac{1}{N}\sum_{i=1}^N \mathrm{Payoff}\!\left(S^{(i)}\right)
$$
- Variance reduction via antithetic variates using shared path construction $(Z,-Z)$
- Inference metrics: Monte Carlo standard error and two-sided 95% confidence interval.
- Greeks (Delta, Gamma) estimated via central finite differences with common random numbers (CRN)
- Benchmarks European options against closed-form Black–Scholes for validation.
- Empirical convergence consistent with Monte Carlo rate ($O(N^{-1/2})$)

## Features
- European and Asian option pricing (arithmetic & geometric averaging)
- Variance-reduced estimators for improved efficiency
- Vectorized simulation and payoff evaluation
- Integrated diagnostics: price, SE, CI95, Delta, Gamma
- Reproducible results via controlled seeding and CRN-consistent Greek estimation

## Results (sample run, seed=42)
| Contract | Price | SE | CI95 | Delta | Gamma |
|---|---:|---:|---|---:|---:|
| European Call (MC, antithetic) | 10.47 | 0.02 | (10.44, 10.50) | 0.64 | 0.02 |
| European Call (Black-Scholes) | 10.45 | - | - | - | - |
| Asian Call (MC, arithmetic, control_variate) | 5.78 | 0.00 | (5.78, 5.79) | 0.59 | 0.03 |

## Requirements 
```
numpy==2.3.3
scipy==1.16.2
pandas==2.3.2
matplotlib==3.10.6
```
## Installation
### From GitHub
```bash
pip install "https://github.com/Chinmaec/monte_carlo_option_pricer.git"

## Usage
```powershell
cd monte_carlo_simulation
pip install -r requirements.txt
python test.py
```

## Experiments & Diagnostics

Detailed convergence analysis, variance reduction comparisons, and pricing diagnostics are available in the `cookbook.ipynb`.

Includes:
- Convergence behaviour (error vs simulations)
- Plain vs antithetic (European) and plain vs antithetic vs control variate (Asian) variance comparison
- Greeks estimation (Delta, Gamma)

## Visualization 

<p align="center">
  <img src="./images/mc_european_asian_dark.png" alt="European and Asian Monte Carlo" width="900"/>
</p>

