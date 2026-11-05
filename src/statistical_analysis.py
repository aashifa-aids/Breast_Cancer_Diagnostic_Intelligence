from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

def run_statistical_analysis(frame, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    malignant = frame[frame.diagnosis == "M"]
    benign = frame[frame.diagnosis == "B"]
    rows = []

    for feature in frame.select_dtypes(include="number").columns:
        m, b = malignant[feature], benign[feature]
        t_stat, t_p = ttest_ind(m, b, equal_var=False)
        _, u_p = mannwhitneyu(m, b, alternative="two-sided")
        pooled_sd = np.sqrt(((len(m)-1)*m.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) /
                            (len(m)+len(b)-2))
        d = (m.mean()-b.mean())/pooled_sd if pooled_sd else np.nan
        rows.append({
            "feature": feature, "malignant_mean": m.mean(),
            "benign_mean": b.mean(), "welch_t": t_stat,
            "welch_p": t_p, "mann_whitney_p": u_p, "cohens_d": d
        })

    stats = pd.DataFrame(rows).sort_values("welch_p")
    stats.to_csv(output_dir / "statistical_tests.csv", index=False)

    X = frame.drop(columns="diagnosis").astype(float)
    X = (X-X.mean()) / X.std(ddof=0).replace(0, 1)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    Xc = sm.add_constant(X, has_constant="add")
    vif_rows = []
    for i, col in enumerate(Xc.columns):
        if col == "const":
            continue
        try:
            val = variance_inflation_factor(Xc.values, i)
        except Exception:
            val = np.inf
        vif_rows.append({"feature": col, "VIF": val})
    vif = pd.DataFrame(vif_rows).sort_values("VIF", ascending=False)
    vif.to_csv(output_dir / "vif_analysis.csv", index=False)

    # A small, interpretable logistic regression inference example.
    chosen = ["mean radius", "mean texture", "mean smoothness", "mean concavity"]
    Xlog = frame[chosen].astype(float)
    Xlog = (Xlog-Xlog.mean()) / Xlog.std(ddof=0)
    Xlog = sm.add_constant(Xlog)
    y = (frame.diagnosis == "M").astype(int)
    try:
        fit = sm.Logit(y, Xlog).fit(disp=False, maxiter=200)
        ci = fit.conf_int()
        infer = pd.DataFrame({
            "coefficient": fit.params, "p_value": fit.pvalues,
            "odds_ratio": np.exp(fit.params),
            "odds_ratio_ci_lower": np.exp(ci[0]),
            "odds_ratio_ci_upper": np.exp(ci[1])
        })
        infer.to_csv(output_dir / "logistic_regression_inference.csv")
    except Exception as exc:
        (output_dir / "logistic_regression_inference_error.txt").write_text(str(exc))

    return stats, vif
