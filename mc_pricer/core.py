import numpy as np

def payoff_vanilla(ST, K, option_type="call"):
    if option_type == "call":
        return np.maximum(ST - K, 0.0)
    elif option_type == "put":
        return np.maximum(K - ST, 0.0)
    raise ValueError("option_type must be 'call' or 'put'")

def mc_stats(discounted_payoffs):
    price = discounted_payoffs.mean()
    se = discounted_payoffs.std(ddof=1) / np.sqrt(discounted_payoffs.size)
    ci95 = (price - 1.96 * se, price + 1.96 * se)
    return price, se, ci95