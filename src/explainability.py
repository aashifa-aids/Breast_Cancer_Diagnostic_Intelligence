from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance

def create_permutation_importance(model, X_test, y_test, output_dir):
    output_dir = Path(output_dir)
    result = permutation_importance(
        model, X_test, y_test, n_repeats=15, random_state=42, scoring="recall"
    )
    table = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std
    }).sort_values("importance_mean", ascending=False)
    table.to_csv(output_dir / "permutation_importance.csv", index=False)
    top = table.head(15).sort_values("importance_mean")
    plt.figure(figsize=(9, 7))
    plt.barh(top.feature, top.importance_mean, xerr=top.importance_std)
    plt.xlabel("Decrease in recall after feature permutation")
    plt.title("Permutation Feature Importance")
    plt.tight_layout()
    plt.savefig(output_dir / "permutation_importance.png", dpi=160)
    plt.close()
    return table

def try_shap(model, X_train, X_test, output_dir):
    try:
        import shap
        background = X_train.sample(min(20, len(X_train)), random_state=42)
        sample = X_test.sample(min(10, len(X_test)), random_state=42)
        predict_fn = lambda values: model.predict_proba(
            pd.DataFrame(values, columns=X_train.columns)
        )[:, 1]
        explainer = shap.KernelExplainer(predict_fn, background)
        values = explainer.shap_values(sample, nsamples=50)
        shap.summary_plot(values, sample, show=False)
        plt.tight_layout()
        plt.savefig(Path(output_dir) / "shap_summary.png", dpi=160, bbox_inches="tight")
        plt.close()
        return "SHAP summary generated."
    except Exception as exc:
        return f"SHAP skipped (optional dependency or compatibility issue): {exc}"
