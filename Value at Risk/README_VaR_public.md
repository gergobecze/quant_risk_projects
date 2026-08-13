
## What VaR is, briefly

VaR answers a simple question: over a given holding period and confidence level, how much could this position lose? It's the most widely used market risk measure at banks, and it's the natural entry point into the rest of the risk-model toolkit — CRM and IRC (see the other two projects in this repo) are both built on the same tail-risk logic, just for different product types and at a longer horizon.

## Methodology

**Data.** Five years of daily closing prices, pulled from Stooq, with daily log returns computed from the price series.

**Historical Simulation VaR.** Takes the empirical distribution of actual historical returns and reads off the 5th/1st percentile directly — no distributional assumption required. Simple, robust to fat tails, but entirely dependent on the historical window actually containing a representative bad day.

**Parametric (variance-covariance) VaR.** Assumes returns are normally distributed, then derives VaR analytically from the mean and standard deviation of returns: `VaR% = z × σ − μ`, where `z` is the normal-distribution quantile for the chosen confidence level (`NORMSINV`). Faster to compute and easy to shock/stress, but understates risk whenever real returns have fatter tails than the normal distribution assumes.

**Backtesting — Kupiec Proportion-of-Failures test.** Counts how many days the actual loss exceeded the model's VaR estimate ("breaches"), then runs a formal likelihood-ratio test comparing the observed breach rate to the expected rate at the chosen confidence level, checked against a chi-square critical value.

**Rolling VaR.** A 250-day rolling window (the Basel-standard lookback) recomputes VaR every day rather than once for the whole sample, showing how the risk estimate adapts as market conditions change.

**Volatility stress test.** Shocks the standard deviation input upward (+10%/+20%/+30%) to see how sensitive the VaR figure is to a volatility spike.

## Files

the full workbook: Data sheet (prices, returns, rolling stats), VaR Summary sheet (Historical + Parametric VaR, backtest + Kupiec test, stress test), and a Chart sheet (daily return vs. rolling VaR bands)

## Requirements to open

Microsoft Excel (uses `PERCENTILE`, `STDEV`, `NORMSINV`, `CHIINV`, `CHIDIST`, `COUNTIF` — no add-ins required).
