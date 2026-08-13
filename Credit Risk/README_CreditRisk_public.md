
## Why synthetic data

Real bank default data is proprietary. Instead, this project simulates a 6,000-borrower portfolio with named, realistic risk drivers (income, debt-to-income, utilization, late payments, credit history, employment status) and a known underlying default process — a logistic function of those drivers. That means the "true" answer is known, so the model's output can be checked against it directly.

## Methodology

**Step 1 — simulate the portfolio.** Generates borrower-level features and a binary `default` outcome from a known logistic data-generating process. A `macro_stress` flag splits the book into a "normal" and a "downturn" vintage, which is what makes the backtest in Step 8 possible.

**Step 2 — EDA.** Compares feature distributions between defaulters and non-defaulters, and plots default rate by employment status.

**Step 3 — hypothesis testing.** Welch's two-sample t-test (implemented from scratch, no scipy) on each risk driver, testing whether its mean differs significantly between defaulters and non-defaulters.

**Step 4 — feature engineering.** Standardizes numeric features, encodes employment status, and splits the data into a normal-vintage train/test set plus a held-out stress vintage.

**Step 5 — logistic regression from scratch.** Implements Newton-Raphson (IRLS) — the same algorithm behind `statsmodels`/`glm` — by hand with numpy, including standard errors, z-scores, and p-values from the inverse Fisher information matrix.

**Step 6 — Fisher's Linear Discriminant Analysis.** A from-scratch two-class LDA (pooled covariance, prior-adjusted threshold).

**Step 7 — evaluation.** Confusion matrix, precision/recall, and ROC/AUC (AUC computed via the Mann-Whitney U statistic — exact, no numerical integration needed). Logistic regression and LDA land at a similar AUC, which is expected since both are near-optimal under the underlying Gaussian/linear-logit assumptions used to generate the data.

**Step 8 — backtesting / model stability.** The model is trained only on the "normal" vintage, then tested against the held-out "stress" vintage. The decile calibration table shows predicted PD vs. realized default rate under stress — a check for model drift.

**Step 9 — sensitivity study.** Shocks debt-to-income and utilization upward by 10/20/30% and reports the resulting change in portfolio expected loss.

**Step 10 — credit limit setting.** An illustrative rule: cap each borrower's limit so expected loss (PD × LGD × exposure) never exceeds 2% of income.

## Files

- `credit_risk_project.py` — the full, runnable script (ten `# %%` sections matching the steps above)
- `credit_risk_data.csv` — the simulated portfolio
- `credit_limit_suggestions.csv` — risk-based limit suggestions
- `fig1_eda_distributions.png`, `fig2_default_rate_by_employment.png`, `fig3_roc_curve.png`, `fig4_backtest_calibration.png`

## Requirements to run

`pandas`, `numpy`, `matplotlib` — no scikit-learn, scipy, or statsmodels needed. Run with `python3 credit_risk_project.py`.
