# european options(plain and antithetic)
import numpy as np 
from scipy.stats import norm 

from mc_pricer.core import payoff_vanilla, mc_stats, mt19937_rng, box_muller_normals

def black_scholes_price(S0, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    if option_type == "call":
        return S0 * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r*T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    raise ValueError("option_type must be 'call' or 'put'")

def mc_european(
    S0, K, T, r, sigma,
    n_samples=100_000,
    option_type="call",
    method="plain",   # "plain" or "antithetic"
    seed=42
):
    
    rng = mt19937_rng(seed=seed)
    drift = (r - 0.5 * sigma**2) * T
    vol = sigma * np.sqrt(T)
    disc_factor = np.exp(-r * T)

    if method == "plain":
        Z = box_muller_normals(n_samples, rng)
        ST = S0 * np.exp(drift + vol * Z)
        disc = disc_factor * payoff_vanilla(ST, K, option_type)

    elif method == "antithetic":
        Z = box_muller_normals(n_samples, rng)
        ST_plus = S0 * np.exp(drift + vol * Z)
        ST_minus = S0 * np.exp(drift - vol * Z)
        pay_plus = payoff_vanilla(ST_plus, K, option_type)
        pay_minus = payoff_vanilla(ST_minus, K, option_type)
        disc = disc_factor * 0.5 * (pay_plus + pay_minus)

    else:
        raise ValueError("method must be 'plain' or 'antithetic'")

    return mc_stats(disc)
