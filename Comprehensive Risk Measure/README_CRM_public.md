
## What CRM is, briefly

CRM is a regulatory capital charge (introduced under Basel II.5) for a bank's **correlation trading portfolio** — positions like CDO tranches and credit index tranches (CDX/iTraxx) whose risk depends not just on individual default probabilities, but on how correlated defaults are across a basket of names. It's a Value-at-Risk-type measure, but projected over a **1-year horizon at 99.9% confidence**, and it has to capture credit spread risk, correlation risk, basis risk, recovery risk, and default risk together.

## Methodology

**Step 1 — synthetic credit basket.** 100 companies, each assigned a random credit spread (50–400 bps), converted to an implied 1-year default probability via the standard market approximation `default probability ≈ spread / (1 - recovery rate)`.

**Step 2 — one-factor Gaussian copula.** The industry-standard technique for simulating *correlated* defaults. Each company gets a simulated "asset value" built from a shared market factor (`M`) and an idiosyncratic factor (`Z`), blended by an assumed asset correlation `rho`: `X = sqrt(rho)*M + sqrt(1-rho)*Z`. A company defaults in a scenario if `X` falls below a threshold calibrated to its real default probability. Run over 50,000 Monte Carlo scenarios.

**Step 3 — tranche loss and the CRM figure.** A tranche absorbs portfolio losses between an attachment point (3%) and detachment point (15%) — below attachment it's untouched, above detachment it's a total loss, in between it's proportional. CRM is the **99.9th percentile of the simulated tranche loss distribution.**

**Step 4 — correlation sensitivity.** `rho` is the hardest input in this model to observe directly — in practice it's often backed out from index tranche market prices. This step re-runs the full simulation across a range of `rho` assumptions to show how much the CRM figure depends on getting that one assumption right.

## Key results

With a $10M tranche notional, 3%–15% attachment/detachment, and 40% recovery:

| rho | Mean tranche loss | CRM (99.9% / 1yr) |
|---|---|---|
| 0.05 | $278,780 | $5,000,500 |
| 0.10 | $393,720 | $8,000,500 |
| 0.15 | $493,330 | $10,000,000 (full loss) |
| 0.20 | $577,630 | $10,000,000 |
| 0.30 | $701,930 | $10,000,000 |
| 0.40 | $779,580 | $10,000,000 |

The tail is far more sensitive to correlation than the average outcome is: mean loss grows steadily and modestly as `rho` rises, but CRM (the tail figure) nearly doubles between `rho`=0.05 and 0.10, then hits a hard ceiling — full tranche loss — by `rho`=0.15 and stays there no matter how much higher correlation goes. Once correlation crosses a threshold, this tranche is already maxed out — additional correlation can't make the worst case worse, only more likely.

## Files

- `crm_project.py` — the full, runnable script (four `# %%` sections matching the steps above)
- `crm_correlation_sensitivity.png` — the sensitivity chart

## Requirements to run

`numpy`, `pandas`, `scipy`, `matplotlib`. Run with `python3 crm_project.py`, or execute the `# %%` sections one at a time in VS Code.
