"""
IV smile + delta-hedging experiment for a Heston + Merton (Bates) Monte Carlo.

Swap `simulate_bates` for your own simulator if you like: all you need is a
function that returns a (n_paths, n_steps+1) array of risk-neutral prices.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq

# ---------------------------------------------------------------- simulator
def simulate_bates(S0, v0, r, kappa, theta, xi, rho, lam, muJ, sigJ,
                   T, n_steps, n_paths, seed=0):
    """Risk-neutral Heston + Merton jumps. Full-truncation Euler for variance."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    k = np.exp(muJ + 0.5 * sigJ**2) - 1          # jump compensator E[e^J]-1
    S = np.empty((n_paths, n_steps + 1))
    S[:, 0] = S0
    v = np.full(n_paths, v0)
    for t in range(n_steps):
        z1 = rng.standard_normal(n_paths)
        z2 = rho * z1 + np.sqrt(1 - rho**2) * rng.standard_normal(n_paths)
        vp = np.maximum(v, 0.0)
        sv = np.sqrt(vp)
        N = rng.poisson(lam * dt, n_paths)
        J = muJ * N + sigJ * np.sqrt(N) * rng.standard_normal(n_paths)
        S[:, t + 1] = S[:, t] * np.exp((r - lam * k - 0.5 * vp) * dt
                                       + sv * np.sqrt(dt) * z1 + J)
        v = v + kappa * (theta - vp) * dt + xi * sv * np.sqrt(dt) * z2
    return S

# ------------------------------------------------------------ Black-Scholes
def bs_price(S, K, T, r, sig, call=True):
    d1 = (np.log(S / K) + (r + 0.5 * sig**2) * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    if call:
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def bs_delta(S, K, tau, r, sig):
    d1 = (np.log(S / K) + (r + 0.5 * sig**2) * tau) / (sig * np.sqrt(tau))
    return norm.cdf(d1)

def implied_vol(price, S, K, T, r, call=True):
    intrinsic = max((S - K * np.exp(-r * T)) if call else (K * np.exp(-r * T) - S), 0)
    if price <= intrinsic + 1e-10:
        return np.nan
    try:
        return brentq(lambda s: bs_price(S, K, T, r, s, call) - price, 1e-4, 5.0)
    except ValueError:
        return np.nan

# ------------------------------------------------------------- IV smile
def mc_smile(S_paths, S0, r, T, step_idx, strikes):
    """IV for each strike at one maturity. Uses OTM options (puts below the
    forward, calls above) because they have much lower MC noise."""
    ST = S_paths[:, step_idx]
    disc = np.exp(-r * T)
    F = S0 * np.exp(r * T)
    ivs = []
    for K in strikes:
        call = K >= F
        payoff = np.maximum(ST - K, 0) if call else np.maximum(K - ST, 0)
        ivs.append(implied_vol(disc * payoff.mean(), S0, K, T, r, call))
    return np.array(ivs)

# ---------------------------------------------------------- delta hedging
def delta_hedge_pnl(S, K, T, r, sigma, premium, every=1):
    """P&L at expiry of: sell a call at `premium`, delta-hedge with
    Black-Scholes delta (vol = sigma), rebalancing every `every` steps."""
    n = S.shape[1] - 1
    dt = T / n
    delta = bs_delta(S[:, 0], K, T, r, sigma)
    cash = premium - delta * S[:, 0]
    for i in range(1, n + 1):
        cash *= np.exp(r * dt)
        if i % every == 0 and i < n:
            new = bs_delta(S[:, i], K, T - i * dt, r, sigma)
            cash -= (new - delta) * S[:, i]
            delta = new
    return cash + delta * S[:, -1] - np.maximum(S[:, -1] - K, 0)

def summarize(name, pnl, premium):
    cvar = pnl[pnl <= np.quantile(pnl, 0.05)].mean()
    print(f"{name:<26} mean {pnl.mean():7.3f}  std {pnl.std():6.3f} "
          f"({100 * pnl.std() / premium:5.1f}% of premium)  CVaR5% {cvar:7.3f}")

# -------------------------------------------------------------------- main
if __name__ == "__main__":
    P = dict(S0=100.0, v0=0.04, r=0.03, kappa=2.0, theta=0.04, xi=0.3,
             rho=-0.7, lam=0.5, muJ=-0.10, sigJ=0.15)
    T, n_steps = 1.0, 252
    S0, r = P["S0"], P["r"]

    # ---- 1. IV smile at three maturities
    S = simulate_bates(**P, T=T, n_steps=n_steps, n_paths=50_000, seed=1)
    strikes = np.linspace(70, 130, 25)
    plt.figure(figsize=(7, 4.5))
    for T_m, idx in [(0.25, 63), (0.5, 126), (1.0, 252)]:
        iv = mc_smile(S, S0, r, T_m, idx, strikes)
        plt.plot(np.log(strikes / (S0 * np.exp(r * T_m))), iv, marker="o",
                 ms=3, label=f"T = {T_m}")
    plt.xlabel("log-moneyness  ln(K/F)")
    plt.ylabel("implied vol")
    plt.title("Bates model implied vol smile (Monte Carlo)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("iv_smile.png", dpi=150)

    # ---- 2. Delta hedging a 1y ATM call
    K = 100.0
    prem = np.exp(-r * T) * np.maximum(S[:, -1] - K, 0).mean()
    iv_atm = implied_vol(prem, S0, K, T, r, True)
    print(f"\n1y ATM call: MC price {prem:.3f}, implied vol {iv_atm:.3%}\n")

    S_h = simulate_bates(**P, T=T, n_steps=n_steps, n_paths=20_000, seed=2)
    res = {
        "Bates, daily hedge": delta_hedge_pnl(S_h, K, T, r, iv_atm, prem, 1),
        "Bates, weekly hedge": delta_hedge_pnl(S_h, K, T, r, iv_atm, prem, 5),
        "Bates, monthly hedge": delta_hedge_pnl(S_h, K, T, r, iv_atm, prem, 21),
    }
    # baseline: pure GBM at the same implied vol (BS world)
    gbm = dict(P, v0=iv_atm**2, theta=iv_atm**2, xi=1e-8, lam=0.0, kappa=1.0)
    S_g = simulate_bates(**gbm, T=T, n_steps=n_steps, n_paths=20_000, seed=3)
    prem_g = bs_price(S0, K, T, r, iv_atm, True)
    res["GBM baseline, daily"] = delta_hedge_pnl(S_g, K, T, r, iv_atm, prem_g, 1)
    for name, pnl in res.items():
        summarize(name, pnl, prem if "Bates" in name else prem_g)

    plt.figure(figsize=(7, 4.5))
    for name, pnl in res.items():
        plt.hist(pnl, bins=120, alpha=0.5, density=True, label=name)
    plt.xlim(-25, 25)
    plt.xlabel("hedging P&L at expiry")
    plt.ylabel("density")
    plt.title("Delta-hedge P&L: Black-Scholes delta vs Bates world")
    plt.legend()
    plt.tight_layout()
    plt.savefig("hedge_pnl.png", dpi=150)
    plt.show()
