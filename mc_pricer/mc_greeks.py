import numpy as np

def _price_only(result):
    # result is usually (price, se, ci95)
    return result[0] if isinstance(result, tuple) else result

def _summary(x):
    x = np.asarray(x, dtype=float)
    mean = x.mean()
    if x.size > 1:
        se = x.std(ddof=1) / np.sqrt(x.size)
    else:
        se = 0.0
    ci95 = (mean - 1.96 * se, mean + 1.96 * se)
    return mean, se, ci95

def mc_delta(pricer, S0, K, T, r, sigma, bump_rel=1e-2, n_rep=20, seed=42, **kwargs):
    h = bump_rel * S0
    vals = []
    for i in range(n_rep):
        rep_seed = None if seed is None else seed + i
        # CRN inside each replication
        p_up = _price_only(pricer(S0 + h, K, T, r, sigma, seed=rep_seed, **kwargs))
        p_dn = _price_only(pricer(S0 - h, K, T, r, sigma, seed=rep_seed, **kwargs))
        vals.append((p_up - p_dn) / (2 * h))
    return _summary(vals)

def mc_gamma(pricer, S0, K, T, r, sigma, bump_rel=1e-2, n_rep=20, seed=42, **kwargs):
    h = bump_rel * S0
    vals = []
    for i in range(n_rep):
        rep_seed = None if seed is None else seed + i
        # CRN inside each replication
        p_up = _price_only(pricer(S0 + h, K, T, r, sigma, seed=rep_seed, **kwargs))
        p_md = _price_only(pricer(S0,     K, T, r, sigma, seed=rep_seed, **kwargs))
        p_dn = _price_only(pricer(S0 - h, K, T, r, sigma, seed=rep_seed, **kwargs))
        vals.append((p_up - 2 * p_md + p_dn) / (h * h))
    return _summary(vals)
