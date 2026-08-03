# %% Step 1: build the credit trading book
# IRC covers unsecuritized credit products (bonds, loans) in the trading book -
# NOT the correlation/tranche products CRM covers. So instead of a basket priced
# for tranche losses, we build a book of individual rated bonds.
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)

ratings = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]   # "Default" is an 8th, implicit state
n_ratings = len(ratings)
rating_probs = [0.03, 0.07, 0.20, 0.30, 0.22, 0.12, 0.06]  # typical trading-book mix, skewed IG/crossover

n_bonds = 60
initial_rating_idx = rng.choice(n_ratings, size=n_bonds, p=rating_probs)
exposure = rng.uniform(500_000, 2_000_000, n_bonds).round(0)

# credit spread by rating (bps) - used to reprice a bond when its rating migrates
spread_by_rating_bps = np.array([30, 50, 80, 150, 300, 500, 900])  # AAA..CCC
duration = 5.0          # assumed modified duration (years), same for every bond - a simplification
recovery_rate = 0.40    # same convention used in the CRM project

book = pd.DataFrame({
    "bond": [f"Bond_{i+1}" for i in range(n_bonds)],
    "rating": [ratings[r] for r in initial_rating_idx],
    "rating_idx": initial_rating_idx,
    "exposure": exposure,
})

print(book.head(10))
print("\nTotal exposure: ${:,.0f}".format(book["exposure"].sum()))
print("\nRating distribution:\n", book["rating"].value_counts())

# %% Step 2: quarterly rating transition matrix -> migration/default thresholds
# This is the standard "CreditMetrics" technique: take a transition matrix (probability
# of migrating from rating i to rating j over a quarter, including default), and turn
# each row into a set of thresholds on a standard normal variable. It's the same idea as
# the single default threshold in the CRM project, just with multiple cutoffs instead of one.

# Rows = current rating (AAA..CCC), columns = AAA,AA,A,BBB,BB,B,CCC,Default (quarterly, illustrative)
raw_matrix = np.array([
    [97.50, 2.00, 0.40, 0.08, 0.01, 0.005, 0.003, 0.002],
    [1.00, 97.00, 1.70, 0.25, 0.04, 0.01, 0.005, 0.005],
    [0.15, 2.30, 96.00, 1.30, 0.18, 0.04, 0.01, 0.02],
    [0.04, 0.30, 3.50, 94.00, 1.70, 0.30, 0.10, 0.06],
    [0.02, 0.06, 0.50, 5.00, 90.00, 3.50, 0.60, 0.32],
    [0.01, 0.02, 0.10, 0.80, 6.00, 87.00, 4.50, 1.57],
    [0.00, 0.01, 0.03, 0.20, 1.50, 8.00, 77.00, 13.26],
])
transition_matrix = raw_matrix / raw_matrix.sum(axis=1, keepdims=True)  # force each row to sum to exactly 1

# For each starting rating, build cumulative probability from the WORST outcome (Default)
# upward, then convert to normal thresholds via the inverse CDF.
thresholds_by_rating = {}
for i, r in enumerate(ratings):
    row = transition_matrix[i]
    worst_to_best = row[::-1]                          # [P(Default), P(CCC), ..., P(AA), P(AAA)]
    cum = np.clip(np.cumsum(worst_to_best)[:-1], 1e-12, 1 - 1e-12)   # 7 cutoffs for 8 states
    thresholds_by_rating[r] = norm.ppf(cum)

print("Example thresholds for a BBB bond (Default | CCC | B | BB | A | AA cutoffs):")
print(thresholds_by_rating["BBB"].round(3))


def migrate(current_rating_idx, x):
    """Given a bond's current rating index (0=AAA..6=CCC) and a simulated asset value x,
    return (new_rating_idx or None-if-default, is_default)."""
    r_name = ratings[current_rating_idx]
    thr = thresholds_by_rating[r_name]
    state_idx = np.searchsorted(thr, x)      # 0 = Default, 1 = CCC, ..., 7 = AAA
    is_default = state_idx == 0
    mig_from_worst = np.clip(state_idx - 1, 0, 6)   # 0=CCC .. 6=AAA
    new_rating_idx = 6 - mig_from_worst              # convert back to AAA=0..CCC=6 indexing
    return new_rating_idx, is_default


# %% Step 3: Monte Carlo simulation with quarterly "constant level of risk" rebalancing
# This is IRC's signature mechanic: over the 1-year horizon, the book is assumed to be
# rebalanced back to its ORIGINAL risk profile at each "liquidity horizon" (here, quarterly).
# So instead of one rating drifting for a full year, we simulate 4 independent quarters,
# each one starting fresh from the bond's ORIGINAL rating, and sum the P&L across all 4.
n_sims = 20000
rho = 0.15          # asset correlation, same assumption used in the CRM project
n_quarters = 4
liquidity_horizon_seeds = [100, 101, 102, 103]


def simulate_quarter_pnl(rho, seed):
    rl = np.random.default_rng(seed)
    M = rl.standard_normal(n_sims)                       # shared systematic factor
    Z = rl.standard_normal((n_sims, n_bonds))            # idiosyncratic factors
    X = np.sqrt(rho) * M[:, None] + np.sqrt(1 - rho) * Z  # correlated asset values

    pnl = np.zeros((n_sims, n_bonds))
    for i, row in book.iterrows():
        new_rating_idx, is_default = migrate(row["rating_idx"], X[:, i])
        new_spread = spread_by_rating_bps[new_rating_idx]
        old_spread = spread_by_rating_bps[row["rating_idx"]]
        price_change_pct = -duration * (new_spread - old_spread) / 10000.0
        default_loss = -(1 - recovery_rate) * row["exposure"]
        pnl[:, i] = np.where(is_default, default_loss, price_change_pct * row["exposure"])
    return pnl.sum(axis=1)


quarterly_pnls = [simulate_quarter_pnl(rho, seed=s) for s in liquidity_horizon_seeds]
annual_pnl = np.sum(quarterly_pnls, axis=0)
annual_loss = -annual_pnl

irc_999 = np.percentile(annual_loss, 99.9)

print(f"\nMean 1-year P&L: ${annual_pnl.mean():,.0f}")
for p in [90, 95, 99, 99.5, 99.9]:
    print(f"{p}% loss level: ${np.percentile(annual_loss, p):,.0f}")
print(f"IRC (99.9% / 1yr): ${irc_999:,.0f}")

plt.figure(figsize=(9, 5))
plt.hist(annual_loss, bins=80, color="#2a78d6", alpha=0.8)
plt.axvline(irc_999, color="#e34948", linestyle="--", linewidth=2, label=f"IRC 99.9% = ${irc_999:,.0f}")
plt.title("Simulated 1-year portfolio loss distribution (quarterly rebalanced)")
plt.xlabel("Loss ($)")
plt.ylabel("Scenario count")
plt.legend()
plt.tight_layout()
plt.savefig("/Users/beczegergo/Documents/irc_loss_distribution.png", dpi=120)
plt.close()

# %% Step 4: liquidity horizon sensitivity - why "constant level of risk" matters
# Compare the quarterly-rebalanced IRC above against a "no rebalancing" version, where
# each bond's rating is allowed to drift (compound) across all 4 quarters instead of being
# reset. This isolates exactly how much the liquidity-horizon assumption is worth - the
# same role that the correlation sensitivity study played for CRM.
cur_rating_idx = np.tile(book["rating_idx"].values, (n_sims, 1))
defaulted_mask = np.zeros((n_sims, n_bonds), dtype=bool)
total_pnl_drift = np.zeros(n_sims)

for q, seed in enumerate([200, 201, 202, 203]):
    rl = np.random.default_rng(seed)
    M = rl.standard_normal(n_sims)
    Z = rl.standard_normal((n_sims, n_bonds))
    X = np.sqrt(rho) * M[:, None] + np.sqrt(1 - rho) * Z

    quarter_pnl = np.zeros((n_sims, n_bonds))
    for i in range(n_bonds):
        exposure_i = book["exposure"].iloc[i]
        for r_idx in range(n_ratings):
            active = (cur_rating_idx[:, i] == r_idx) & (~defaulted_mask[:, i])
            if not active.any():
                continue
            new_rating_idx, is_default = migrate(r_idx, X[active, i])
            new_spread = spread_by_rating_bps[new_rating_idx]
            old_spread = spread_by_rating_bps[r_idx]
            price_change_pct = -duration * (new_spread - old_spread) / 10000.0
            default_loss = -(1 - recovery_rate) * exposure_i
            q_pnl = np.where(is_default, default_loss, price_change_pct * exposure_i)

            idxs = np.where(active)[0]
            quarter_pnl[idxs, i] = q_pnl
            cur_rating_idx[idxs, i] = new_rating_idx
            defaulted_mask[idxs, i] = is_default
    total_pnl_drift += quarter_pnl.sum(axis=1)

annual_loss_drift = -total_pnl_drift
irc_999_drift = np.percentile(annual_loss_drift, 99.9)

print("\n=== Liquidity horizon sensitivity ===")
print(f"Quarterly rebalanced ('constant level of risk') IRC 99.9%: ${irc_999:,.0f}")
print(f"No rebalancing (rating drifts for the full year)  IRC 99.9%: ${irc_999_drift:,.0f}")
print(f"Difference: ${irc_999_drift - irc_999:,.0f}  ({(irc_999_drift/irc_999-1)*100:.0f}% higher)")

plt.figure(figsize=(7, 5))
bars = plt.bar(["Quarterly rebalanced\n(constant level of risk)", "No rebalancing\n(full-year drift)"],
               [irc_999, irc_999_drift], color=["#2a78d6", "#e34948"])
for b in bars:
    plt.text(b.get_x() + b.get_width() / 2, b.get_height(), f"${b.get_height():,.0f}",
              ha="center", va="bottom", fontweight="bold")
plt.title("IRC (99.9% / 1yr) - effect of the liquidity horizon assumption")
plt.ylabel("IRC ($)")
plt.tight_layout()
plt.savefig("/Users/beczegergo/Documents/irc_liquidity_horizon_sensitivity.png", dpi=120)
plt.close()

print("\nSaved: irc_loss_distribution.png, irc_liquidity_horizon_sensitivity.png")