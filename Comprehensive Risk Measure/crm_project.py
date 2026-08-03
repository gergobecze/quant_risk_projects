# %% Step 1: build a synthetic credit basket
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)  # fixes randomness so results are reproducible

n_names = 100          # number of companies in the basket (like a CDX/iTraxx index)
recovery_rate = 0.40   # assumed % of exposure recovered if a company defaults

# Give each company a credit spread in basis points (bps) - wider spread = riskier.
# We're drawing these from a realistic range for investment-grade/crossover names.
spreads_bps = rng.uniform(50, 400, n_names)

# Convert spread to an approximate 1-year default probability.
# Rule of thumb: default probability ≈ spread / (1 - recovery rate)
default_prob_1y = (spreads_bps / 10000) / (1 - recovery_rate)

basket = pd.DataFrame({
    "name": [f"Company_{i+1}" for i in range(n_names)],
    "spread_bps": spreads_bps.round(1),
    "default_prob_1y": default_prob_1y.round(4),
})

print(basket.head(10))
print("\nAverage 1-year default probability across the basket:", basket["default_prob_1y"].mean().round(4))


# %% Step 2: simulate correlated defaults with a one-factor Gaussian copula
from scipy.stats import norm

n_sims = 50000       # number of Monte Carlo scenarios
rho = 0.15            # asset correlation - how much companies move together (0=independent, 1=perfectly correlated)

# Default threshold for each company: the "asset value" level below which it defaults,
# calibrated so each company's own probability of crossing it matches its real default_prob_1y
thresholds = norm.ppf(basket["default_prob_1y"].to_numpy())

# M = one shared "market" factor per scenario (same value for every company in that scenario)
M = rng.standard_normal(n_sims)

# Z = one independent, company-specific factor per company per scenario
Z = rng.standard_normal((n_sims, n_names))

# Each company's simulated asset value in each scenario:
# a blend of the shared market factor and its own idiosyncratic noise
X = np.sqrt(rho) * M[:, None] + np.sqrt(1 - rho) * Z

# A company defaults in a scenario if its asset value falls below its threshold
defaults = X < thresholds[None, :]   # shape: (n_sims, n_names), True/False grid

print("Simulated average default rate:", defaults.mean().round(4))
print("Target average default rate (from basket):", basket['default_prob_1y'].mean().round(4))
print("Defaults per scenario - min/mean/max:", defaults.sum(axis=1).min(), defaults.sum(axis=1).mean().round(2), defaults.sum(axis=1).max())


# %% Step 3: turn simulated defaults into tranche losses, and find the 99.9% CRM figure

loss_given_default = 1 - recovery_rate     # % of exposure lost per default, e.g. 0.60
weight_per_name = 1 / n_names               # equal notional weighting across the basket

# Portfolio loss (%) in each scenario = (number of defaults / n_names) * loss given default
n_defaults = defaults.sum(axis=1)                       # defaults per scenario
portfolio_loss_pct = n_defaults * weight_per_name * loss_given_default

# Define a mezzanine tranche: it absorbs losses between the attachment and detachment points
attachment = 0.03    # tranche starts taking losses once basket losses exceed 3%
detachment = 0.15    # tranche is wiped out once basket losses reach 15%
tranche_width = detachment - attachment

# Tranche loss (%) in each scenario: 0 below attachment, 100% above detachment,
# and proportional in between
tranche_loss_pct = np.clip(portfolio_loss_pct - attachment, 0, tranche_width) / tranche_width

tranche_notional = 10_000_000   # $10m - the size of the position we're measuring risk on
tranche_loss_usd = tranche_loss_pct * tranche_notional

print(f"Mean tranche loss: ${tranche_loss_usd.mean():,.0f}")
print(f"Share of scenarios with zero tranche loss: {(tranche_loss_pct==0).mean():.1%}")
for p in [90, 95, 99, 99.5, 99.9]:
    print(f"{p}% loss level: ${np.percentile(tranche_loss_usd, p):,.0f}")
print(f"\nCRM (99.9% / 1-year loss): ${np.percentile(tranche_loss_usd, 99.9):,.0f}")


# %% Step 4: sensitivity to the correlation assumption
import matplotlib.pyplot as plt

def simulate_crm(rho, n_sims=50000, seed=42):
    rng_local = np.random.default_rng(seed)
    thresholds = norm.ppf(basket["default_prob_1y"].to_numpy())
    M = rng_local.standard_normal(n_sims)
    Z = rng_local.standard_normal((n_sims, n_names))
    X = np.sqrt(rho) * M[:, None] + np.sqrt(1 - rho) * Z
    defaults_local = X < thresholds[None, :]
    n_defaults_local = defaults_local.sum(axis=1)
    portfolio_loss = n_defaults_local * (1 / n_names) * (1 - recovery_rate)
    tranche_loss = np.clip(portfolio_loss - attachment, 0, tranche_width) / tranche_width
    tranche_loss_usd_local = tranche_loss * tranche_notional
    return tranche_loss_usd_local.mean(), np.percentile(tranche_loss_usd_local, 99.9)
rho_scenarios = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
results = [(r,) + simulate_crm(r) for r in rho_scenarios]

print(f"{'rho':>6} {'mean loss':>15} {'CRM (99.9%)':>15}")
for r, mean_loss, crm_999 in results:
    print(f"{r:6.2f} {mean_loss:15,.0f} {crm_999:15,.0f}")

plt.figure(figsize=(7,4))
plt.plot([r[0] for r in results], [r[2] for r in results], marker="o")
plt.xlabel("Assumed asset correlation (rho)")
plt.ylabel("CRM - 99.9% / 1yr tranche loss ($)")
plt.title("CRM sensitivity to the correlation assumption")
plt.tight_layout()
plt.savefig("/Users/beczegergo/Documents/crm_correlation_sensitivity.png", dpi=120)
plt.show()