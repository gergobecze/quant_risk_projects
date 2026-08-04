"""
Credit Risk Project - Probability of Default (PD) Modeling
=============================================================
Everything below (logistic regression, LDA, ROC/AUC, hypothesis tests) is
implemented from first principles with numpy/pandas only - no scikit-learn or
statsmodels - so you can explain every line of math behind it in an interview.
"""

# %% Step 1: imports and portfolio simulation
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt

RNG = np.random.default_rng(42)


def simulate_portfolio(n=6000, seed=42):
    rng = np.random.default_rng(seed)

    age = rng.uniform(21, 70, n)
    income = rng.lognormal(mean=10.8, sigma=0.45, size=n)          # annual income, $
    credit_history_years = np.clip((age - 21) * rng.uniform(0.3, 0.9, n), 0, 40)
    dti = np.clip(rng.beta(2, 5, n) * 1.4, 0.01, 1.3)               # debt-to-income
    utilization = np.clip(rng.beta(2, 3, n), 0.01, 0.99)            # credit utilization
    late_payments = rng.poisson(lam=0.4 + 2.5 * utilization**2)     # late pmts / 12m
    employment = rng.choice(
        ["employed", "self_employed", "unemployed"], size=n, p=[0.72, 0.19, 0.09]
    )
    loan_amount = income * rng.uniform(0.1, 0.6, n)

    # Vintage: half the book was originated in a benign period, half in a
    # downturn (macro_stress = 1). This is what lets us test model stability later.
    macro_stress = rng.binomial(1, 0.5, n)

    def z(x):
        return (x - x.mean()) / x.std()

    logit_true = (
        -2.35
        + 0.85 * z(dti)
        + 0.65 * z(utilization)
        + 0.55 * z(late_payments)
        - 0.45 * z(np.log(income))
        - 0.30 * z(credit_history_years)
        + 0.55 * (employment == "unemployed")
        + 0.25 * (employment == "self_employed")
        + 0.80 * macro_stress
        + rng.normal(0, 0.55, n)  # idiosyncratic noise
    )
    p_default = 1 / (1 + np.exp(-logit_true))
    default = rng.binomial(1, p_default)

    df = pd.DataFrame({
        "age": age,
        "income": income,
        "credit_history_years": credit_history_years,
        "dti": dti,
        "utilization": utilization,
        "late_payments_12m": late_payments,
        "employment": employment,
        "loan_amount": loan_amount,
        "macro_stress": macro_stress,
        "default": default,
    })
    return df


df = simulate_portfolio()
df.to_csv("/Users/beczegergo/Desktop/credit_risk_data.csv", index=False)
print("Simulated portfolio:", df.shape)
print(df["default"].value_counts(normalize=True).rename("share"))
# %% Step 2: exploratory data analysis
summary = df.groupby("default")[["age", "income", "dti", "utilization",
                                  "late_payments_12m", "credit_history_years"]].mean()
print("\nMeans by default status:\n", summary)

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, col in zip(axes.ravel(), ["dti", "utilization", "late_payments_12m",
                                   "income", "credit_history_years", "age"]):
    for d, color in [(0, "tab:blue"), (1, "tab:red")]:
        ax.hist(df.loc[df["default"] == d, col], bins=30, alpha=0.5,
                label=f"default={d}", color=color, density=True)
    ax.set_title(col)
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("/Users/beczegergo/Desktop/fig1_eda_distributions.png", dpi=110)
plt.close()

default_rate_by_emp = df.groupby("employment")["default"].mean().sort_values()
default_rate_by_emp.plot(kind="barh", figsize=(5, 3), color="tab:orange", title="Default rate by employment status")
plt.tight_layout()
plt.savefig("/Users/beczegergo/Desktop/fig2_default_rate_by_employment.png", dpi=110)
plt.close()
print("\nSaved fig1_eda_distributions.png, fig2_default_rate_by_employment.png")


# %% Step 3: hypothesis testing - Welch's two-sample t-test
# H0: mean value of a feature is the same for defaulters vs non-defaulters.
# Welch's t-test doesn't assume equal variances between the two groups,
# implemented from scratch here (no scipy needed).

def welch_t_test(x, y):
    nx, ny = len(x), len(y)
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    se = math.sqrt(vx / nx + vy / ny)
    t_stat = (mx - my) / se
    df_ = (vx / nx + vy / ny) ** 2 / ((vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1))
    z_eq = t_stat / math.sqrt(1 + t_stat**2 / df_)
    p_val = 2 * (1 - 0.5 * (1 + math.erf(abs(z_eq) / math.sqrt(2))))
    return t_stat, df_, p_val

for feature in ["dti", "utilization", "late_payments_12m", "income"]:
    g0 = df.loc[df.default == 0, feature].to_numpy()
    g1 = df.loc[df.default == 1, feature].to_numpy()
    t, dof, p = welch_t_test(g1, g0)
    print(f"{feature:20s}  t={t:7.3f}  df={dof:8.1f}  p-value={p:.4g}")
    # %% Step 4: feature engineering
df_model = df.copy()
df_model["log_income"] = np.log(df_model["income"])
df_model["emp_self_employed"] = (df_model["employment"] == "self_employed").astype(int)
df_model["emp_unemployed"] = (df_model["employment"] == "unemployed").astype(int)

feature_cols = ["dti", "utilization", "late_payments_12m", "log_income",
                 "credit_history_years", "emp_self_employed", "emp_unemployed"]

# Standardize numeric features (mean 0, sd 1)
scaler_mean = df_model[feature_cols].mean()
scaler_std = df_model[feature_cols].std()
X_full = (df_model[feature_cols] - scaler_mean) / scaler_std
y_full = df_model["default"].to_numpy()

# Train on the NORMAL vintage only, hold out the STRESS vintage entirely for
# backtesting later - fit on data you have, then check whether it still holds
# under stress, exactly like a real risk team would.
train_mask = df_model["macro_stress"] == 0
X_train, y_train = X_full[train_mask].to_numpy(), y_full[train_mask]
X_stress, y_stress = X_full[~train_mask].to_numpy(), y_full[~train_mask]

# Further split the NORMAL vintage into train/test
rng2 = np.random.default_rng(1)
idx = rng2.permutation(len(X_train))
n_test = int(0.25 * len(idx))
test_idx, tr_idx = idx[:n_test], idx[n_test:]
X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
X_te, y_te = X_train[test_idx], y_train[test_idx]

# add an intercept column (a column of 1s) so the model can fit a baseline level
X_tr_i = np.column_stack([np.ones(len(X_tr)), X_tr])
X_te_i = np.column_stack([np.ones(len(X_te)), X_te])
X_stress_i = np.column_stack([np.ones(len(X_stress)), X_stress])


# %% Step 5: logistic regression via Newton-Raphson (from scratch)
# This is the same IRLS algorithm used under the hood by statsmodels/glm -
# implemented by hand so you can defend the mechanics in an interview.

def fit_logistic_newton_raphson(X, y, n_iter=50, tol=1e-8):
    n, k = X.shape
    w = np.zeros(k)
    for it in range(n_iter):
        eta = X @ w
        p = 1 / (1 + np.exp(-eta))
        W = p * (1 - p)
        grad = X.T @ (y - p)
        WX = X * W[:, None]
        H = X.T @ WX
        delta = np.linalg.solve(H + 1e-8 * np.eye(k), grad)
        w = w + delta
        if np.max(np.abs(delta)) < tol:
            break
    eta = X @ w
    p = 1 / (1 + np.exp(-eta))
    W = p * (1 - p)
    H = X.T @ (X * W[:, None])
    cov = np.linalg.inv(H + 1e-8 * np.eye(k))
    se = np.sqrt(np.diag(cov))
    z_scores = w / se
    p_values = 2 * (1 - 0.5 * (1 + np.vectorize(math.erf)(np.abs(z_scores) / math.sqrt(2))))
    return w, se, z_scores, p_values, it + 1


coef, se, z_scores, p_values, n_iters = fit_logistic_newton_raphson(X_tr_i, y_tr)
coef_names = ["intercept"] + feature_cols
print(f"\nLogistic regression converged in {n_iters} Newton-Raphson iterations")
print(f"{'variable':22s} {'coef':>8s} {'se':>8s} {'z':>8s} {'p-value':>10s}")
for name, c, s, z, p in zip(coef_names, coef, se, z_scores, p_values):
    print(f"{name:22s} {c:8.3f} {s:8.3f} {z:8.3f} {p:10.4g}")


def predict_proba(X_i, w):
    return 1 / (1 + np.exp(-(X_i @ w)))


proba_te = predict_proba(X_te_i, coef)


# %% Step 6: Fisher's Linear Discriminant Analysis (from scratch)
# Classic two-class Fisher LDA: project onto w = pooled_covariance^-1 (mu1 - mu0),
# classify by comparing the projected score against a prior-adjusted threshold.

def fit_lda(X, y):
    X0, X1 = X[y == 0], X[y == 1]
    mu0, mu1 = X0.mean(axis=0), X1.mean(axis=0)
    cov0 = np.cov(X0, rowvar=False)
    cov1 = np.cov(X1, rowvar=False)
    n0, n1 = len(X0), len(X1)
    pooled_cov = ((n0 - 1) * cov0 + (n1 - 1) * cov1) / (n0 + n1 - 2)
    w = np.linalg.solve(pooled_cov + 1e-6 * np.eye(pooled_cov.shape[0]), mu1 - mu0)
    midpoint = 0.5 * w @ (mu0 + mu1)
    log_prior_ratio = math.log(n1 / n0)
    threshold = midpoint - log_prior_ratio
    return w, threshold


w_lda, thr_lda = fit_lda(X_tr, y_tr)
score_te_lda = X_te @ w_lda
pred_lda = (score_te_lda > thr_lda).astype(int)
lda_accuracy = (pred_lda == y_te).mean()
print(f"\nFisher LDA holdout accuracy: {lda_accuracy:.3f}")
# %% Step 7: model evaluation - confusion matrix, ROC, AUC

def confusion_matrix(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, tn, fp, fn


def roc_curve_manual(y_true, scores):
    order = np.argsort(-scores)
    y_sorted = y_true[order]
    P = y_sorted.sum()
    N = len(y_sorted) - P
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    tpr = tps / P
    fpr = fps / N
    tpr = np.concatenate([[0], tpr])
    fpr = np.concatenate([[0], fpr])
    return fpr, tpr


def auc_mann_whitney(y_true, scores):
    # AUC == P(score of random positive > score of random negative)
    ranks = pd.Series(scores).rank().to_numpy()
    n1 = y_true.sum()
    n0 = len(y_true) - n1
    r1 = ranks[y_true == 1].sum()
    return (r1 - n1 * (n1 + 1) / 2) / (n1 * n0)


pred_logit_05 = (proba_te > 0.5).astype(int)
tp, tn, fp, fn = confusion_matrix(y_te, pred_logit_05)
precision = tp / (tp + fp) if (tp + fp) else float("nan")
recall = tp / (tp + fn) if (tp + fn) else float("nan")
accuracy = (tp + tn) / len(y_te)
auc_logit = auc_mann_whitney(y_te, proba_te)
auc_lda = auc_mann_whitney(y_te, score_te_lda)

print(f"\nLogistic regression @0.5 threshold - accuracy: {accuracy:.3f}  "
      f"precision: {precision:.3f}  recall: {recall:.3f}")
print(f"AUC  -  Logistic regression: {auc_logit:.3f}   Fisher LDA: {auc_lda:.3f}")

fpr_l, tpr_l = roc_curve_manual(y_te, proba_te)
fpr_d, tpr_d = roc_curve_manual(y_te, score_te_lda)
plt.figure(figsize=(5, 5))
plt.plot(fpr_l, tpr_l, label=f"Logistic (AUC={auc_logit:.3f})")
plt.plot(fpr_d, tpr_d, label=f"Fisher LDA (AUC={auc_lda:.3f})")
plt.plot([0, 1], [0, 1], "k--", lw=0.8)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC curve - holdout set")
plt.legend()
plt.tight_layout()
plt.savefig("/Users/beczegergo/Desktop/fig3_roc_curve.png", dpi=110)
plt.close()
print("Saved fig3_roc_curve.png")


# %% Step 8: backtesting on the STRESS vintage (model stability)
# We take the model fit ONLY on the normal vintage and check whether its
# predicted probabilities still line up with realized defaults in the stress
# vintage - a decile calibration check.

proba_stress = predict_proba(X_stress_i, coef)
bt = pd.DataFrame({"pred": proba_stress, "actual": y_stress})
bt["decile"] = pd.qcut(bt["pred"], 10, labels=False, duplicates="drop")
calib = bt.groupby("decile").agg(predicted=("pred", "mean"), realized=("actual", "mean"),
                                  n=("actual", "size"))
print("\nBacktest on STRESS vintage (model trained on NORMAL vintage only):")
print(calib)

plt.figure(figsize=(6, 4))
x = np.arange(len(calib))
w = 0.35
plt.bar(x - w/2, calib["predicted"], width=w, label="Predicted PD")
plt.bar(x + w/2, calib["realized"], width=w, label="Realized default rate")
plt.xlabel("Decile of predicted PD")
plt.ylabel("Default rate")
plt.title("Backtest: predicted vs. realized default rate\n(trained on normal period, tested on stress period)")
plt.legend()
plt.tight_layout()
plt.savefig("/Users/beczegergo/Desktop/fig4_backtest_calibration.png", dpi=110)
plt.close()
print("Saved fig4_backtest_calibration.png")

overall_pred = bt["pred"].mean()
overall_real = bt["actual"].mean()
print(f"\nPortfolio-level: predicted avg PD={overall_pred:.3f} vs. realized default rate={overall_real:.3f}"
      f"  (model was NOT trained on this stressed period -> gap shows model drift under stress)")


# %% Step 9: sensitivity study
# Shock DTI and utilization upward (a plausible downturn scenario) and measure
# the resulting shift in average predicted PD and portfolio expected loss.

def portfolio_expected_loss(X_i, w, loan_amounts, lgd=0.45):
    pd_hat = predict_proba(X_i, w)
    el = pd_hat * lgd * loan_amounts
    return pd_hat, el

loan_amounts_test = df_model.loc[train_mask].iloc[test_idx]["loan_amount"].to_numpy()

base_pd, base_el = portfolio_expected_loss(X_te_i, coef, loan_amounts_test)
print(f"\nBaseline: mean predicted PD={base_pd.mean():.3f}, "
      f"portfolio expected loss=${base_el.sum():,.0f}")

shock_grid = [0.0, 0.10, 0.20, 0.30]
print("\nSensitivity of portfolio expected loss to a DTI + utilization shock:")
print(f"{'shock %':>8s} {'mean PD':>10s} {'expected loss ($)':>20s} {'%chg EL':>10s}")
for shock in shock_grid:
    X_shocked = X_te.copy()
    dti_col = feature_cols.index("dti")
    util_col = feature_cols.index("utilization")
    dti_raw_shocked = (df_model.loc[train_mask].iloc[test_idx]["dti"].to_numpy() * (1 + shock))
    util_raw_shocked = np.clip(df_model.loc[train_mask].iloc[test_idx]["utilization"].to_numpy() * (1 + shock), 0, 1)
    X_shocked[:, dti_col] = (dti_raw_shocked - scaler_mean["dti"]) / scaler_std["dti"]
    X_shocked[:, util_col] = (util_raw_shocked - scaler_mean["utilization"]) / scaler_std["utilization"]
    X_shocked_i = np.column_stack([np.ones(len(X_shocked)), X_shocked])
    pd_shocked, el_shocked = portfolio_expected_loss(X_shocked_i, coef, loan_amounts_test)
    pct_chg = (el_shocked.sum() - base_el.sum()) / base_el.sum() * 100
    print(f"{shock*100:7.0f}% {pd_shocked.mean():10.3f} {el_shocked.sum():20,.0f} {pct_chg:9.1f}%")


# %% Step 10: simple risk-based credit limit rule
# Cap each borrower's credit limit so expected loss never exceeds 2% of income.

income_test = df_model.loc[train_mask].iloc[test_idx]["income"].to_numpy()
lgd = 0.45
max_el_share_of_income = 0.02
suggested_limit = (max_el_share_of_income * income_test) / (base_pd * lgd + 1e-9)
suggested_limit = np.minimum(suggested_limit, income_test * 0.6)

limit_table = pd.DataFrame({
    "predicted_PD": base_pd,
    "income": income_test,
    "current_loan_amount": loan_amounts_test,
    "suggested_credit_limit": suggested_limit,
}).sort_values("predicted_PD").reset_index(drop=True)
print("\nSample of risk-based suggested credit limits (lowest-risk borrowers first):")
print(limit_table.head(10).round(2))
limit_table.to_csv("/Users/beczegergo/Desktop/credit_limit_suggestions.csv", index=False)
print("\nSaved credit_limit_suggestions.csv")

print("\n=== DONE ===")