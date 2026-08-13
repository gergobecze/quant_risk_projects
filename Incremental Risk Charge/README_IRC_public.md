
## What IRC is, and how it differs from CRM

IRC is also a 99.9%/1-year capital charge, but it covers **unsecuritized credit products** — plain bonds and loans in the trading book — not correlation/tranche products (that's CRM's job). It captures two things CRM doesn't need to: **default risk** and **credit migration risk** (a bond that gets downgraded loses value even if it never defaults). It also has a mechanic CRM doesn't have: the **"constant level of risk"** assumption — regulators assume the bank rebalances its book back to its original risk profile at each **liquidity horizon** (commonly quarterly) rather than holding the same position drifting for the full year.

## Methodology (4 steps, mirroring the CRM project's structure)

**Step 1 — the trading book.** 60 bonds, ratings AAA down to CCC assigned with a realistic skew toward BBB/BB, random exposures $0.5M–$2M, a credit spread per rating (30bps for AAA up to 900bps for CCC), and an assumed 5-year duration used to reprice bonds when their rating migrates.

**Step 2 — transition matrix → migration thresholds.** A quarterly rating transition matrix (probability of moving from any rating to any other, including default). This is the "CreditMetrics" technique: each row of the matrix becomes a set of thresholds on a normal variable — the same idea as the single default threshold in CRM, just with 7 cutoffs instead of 1, since a bond can land in any of 8 states (7 ratings + default).

**Step 3 — Monte Carlo simulation with quarterly rebalancing.** Same one-factor Gaussian copula as CRM (shared factor `M` + idiosyncratic `Z`, correlation `rho=0.15`), but now used to decide *which rating a bond migrates to*, not just default/no-default. The year is split into 4 quarters; each quarter starts fresh from the bond's **original** rating (constant level of risk), and the P&L from all 4 quarters is summed to get the 1-year result. **IRC = the 99.9th percentile of that 1-year loss distribution.**

**Step 4 — liquidity horizon sensitivity.** Reruns the simulation letting ratings **drift** for the full year instead of resetting quarterly, to isolate how much the rebalancing assumption is worth.

## Results

On a $76.0M book:

| Metric | Quarterly rebalanced (constant level of risk) | No rebalancing (full-year drift) |
|---|---|---|
| Mean 1-year loss | $1,606,848 | $4,745,665 |
| 99% loss | $6,912,279 | $14,429,667 |
| **IRC (99.9% / 1yr)** | **$9,448,651** | **$19,330,896** |

The liquidity horizon assumption roughly doubles or halves IRC: removing the quarterly rebalancing (i.e., assuming the bank can't trade out of deteriorating positions for a full year) pushes the 99.9% figure from $9.4M to $19.3M — a bigger swing than most single input assumptions in VaR or CRM produce. The "constant level of risk" assumption is a modeling convenience, not a law of physics — real desks can't always rebalance a distressed bond position on schedule, especially exactly when they'd want to (in a downturn, liquidity dries up first).

## Files

- `irc_project.py` — the full, runnable script (four `# %%` sections matching the steps above)
- `irc_loss_distribution.png` — Step 3's simulated loss histogram with the IRC line marked
- `irc_liquidity_horizon_sensitivity.png` — Step 4's rebalanced-vs-drift comparison chart

## Requirements to run

`numpy`, `pandas`, `scipy`, `matplotlib`. Run with `python3 irc_project.py`, or execute the `# %%` sections one at a time in VS Code.
