# # Asian Monte Carlo (plain and antithetic methods)
import numpy as np
from scipy.stats import norm
from mc_pricer.core import payoff_vanilla, mc_stats


def _simulate_asian_terminal_matrix(S0, T, r, sigma, n_steps, Z):
    dt = T / n_steps
    drift = (r - 0.5 * sigma**2) * dt
    vol = sigma * np.sqrt(dt)
    log_paths = np.cumsum(drift + vol * Z, axis=1)
    return S0 * np.exp(log_paths)


def _asian_average(S, averaging):
    if averaging == "arithmetic":
        return S.mean(axis=1)
    if averaging == "geometric":
        return np.exp(np.log(S).mean(axis=1))
    raise ValueError("averaging must be 'arithmetic' or 'geometric'")


def _geom_asian_closed_form_discrete(
    S0, K, T, r, sigma, n_steps, option_type="call", include_S0=False
):
    # Exact price for discretely monitored geometric-average Asian option (GBM, risk-neutral)
    if T <= 0:
        return payoff_vanilla(np.array([S0]), K, option_type)[0]

    n = n_steps
    logS0 = np.log(S0)

    if include_S0:
        m = logS0 + (r - 0.5 * sigma**2) * (T / 2.0)
        v = sigma**2 * T * (2 * n + 1) / (6.0 * (n + 1))
    else:
        m = logS0 + (r - 0.5 * sigma**2) * T * (n + 1) / (2.0 * n)
        v = sigma**2 * T * (n + 1) * (2 * n + 1) / (6.0 * n * n)

    disc = np.exp(-r * T)

    # deterministic/near-deterministic safeguard
    if v < 1e-14:
        G_det = np.exp(m)
        return disc * payoff_vanilla(np.array([G_det]), K, option_type)[0]

    sv = np.sqrt(v)
    d1 = (m - np.log(K) + v) / sv
    d2 = d1 - sv
    EG = np.exp(m + 0.5 * v)

    if option_type == "call":
        return disc * (EG * norm.cdf(d1) - K * norm.cdf(d2))
    if option_type == "put":
        return disc * (K * norm.cdf(-d2) - EG * norm.cdf(-d1))
    raise ValueError("option_type must be 'call' or 'put'")


def mc_asian(
    S0, K, T, r, sigma,
    n_steps=252,
    n_samples=100_000,
    option_type="call",
    averaging="arithmetic",
    include_S0=False,
    method="plain",   # "plain", "antithetic", "control_variate"
    seed=42,
):
    rng = np.random.default_rng(seed)

    if method == "plain":
        Z = rng.standard_normal((n_samples, n_steps))
        S = _simulate_asian_terminal_matrix(S0, T, r, sigma, n_steps, Z)
        if include_S0:
            S = np.concatenate([np.full((n_samples, 1), S0), S], axis=1)
        A = _asian_average(S, averaging)
        disc = np.exp(-r * T) * payoff_vanilla(A, K, option_type)
        return mc_stats(disc)

    if method == "antithetic":
        Z = rng.standard_normal((n_samples, n_steps))
        S_plus = _simulate_asian_terminal_matrix(S0, T, r, sigma, n_steps, Z)
        S_minus = _simulate_asian_terminal_matrix(S0, T, r, sigma, n_steps, -Z)

        if include_S0:
            S_plus = np.concatenate([np.full((n_samples, 1), S0), S_plus], axis=1)
            S_minus = np.concatenate([np.full((n_samples, 1), S0), S_minus], axis=1)

        A_plus = _asian_average(S_plus, averaging)
        A_minus = _asian_average(S_minus, averaging)
        pay_plus = payoff_vanilla(A_plus, K, option_type)
        pay_minus = payoff_vanilla(A_minus, K, option_type)
        disc_pair = np.exp(-r * T) * 0.5 * (pay_plus + pay_minus)
        return mc_stats(disc_pair)

    if method == "control_variate":
        if averaging != "arithmetic":
            raise ValueError("control_variate is intended for arithmetic averaging")
        if n_samples < 2:
            raise ValueError("n_samples must be >= 2 for control variate")

        Z = rng.standard_normal((n_samples, n_steps))
        S = _simulate_asian_terminal_matrix(S0, T, r, sigma, n_steps, Z)
        if include_S0:
            S = np.concatenate([np.full((n_samples, 1), S0), S], axis=1)

        A = _asian_average(S, "arithmetic")
        G = _asian_average(S, "geometric")

        disc = np.exp(-r * T)
        X = disc * payoff_vanilla(A, K, option_type)  # target
        Y = disc * payoff_vanilla(G, K, option_type)  # control sample

        theta = _geom_asian_closed_form_discrete(
            S0, K, T, r, sigma, n_steps, option_type=option_type, include_S0=include_S0
        )  # control exact expectation

        Xc = X - X.mean()
        Yc = Y - Y.mean()
        varY = (Yc @ Yc) / (n_samples - 1)
        beta = 0.0 if varY < 1e-14 else ((Xc @ Yc) / (n_samples - 1)) / varY

        X_cv = X - beta * (Y - theta)
        return mc_stats(X_cv)

    raise ValueError("method must be 'plain', 'antithetic' or 'control_variate'")
