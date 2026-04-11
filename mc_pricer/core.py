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

def mt19937_rng(seed):
    return np.random.Generator(np.random.MT19937(seed))

def box_muller_normals(n, rng):
    """
    Generate n ~ N(0,1) using Box-Muller from uniform(0,1)
    """
    m = (n + 1) // 2 
    u1 = rng.random(m)
    u2 = rng.random(m)

    # avoid log(0) 
    u1 = np.clip(u1, 1e-12, 1.0)

    r = np.sqrt(-2.0 * np.log(u1))
    theta = 2.0 * np.pi * u2 

    z0 = r * np.cos(theta)
    z1 = r * np.sin(theta)

    z = np.empty(2 * m, dtype= float)
    z[0::2] = z0 
    z[1::2] = z1 

    return z[:n]

def box_muller_matrix(shape, rng):
    n = int(np.prod(shape))
    return box_muller_normals(n, rng).reshape(shape)