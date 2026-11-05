from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, RocCurveDisplay, PrecisionRecallDisplay
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

def evaluate_models(models, X_train, X_test, y_train, y_test, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows, fitted = [], {}

    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="recall")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]
        rows.append({
            "model": name, "cv_recall_mean": scores.mean(),
            "cv_recall_std": scores.std(),
            "test_accuracy": accuracy_score(y_test, pred),
            "test_precision": precision_score(y_test, pred, zero_division=0),
            "test_recall": recall_score(y_test, pred, zero_division=0),
            "test_f1": f1_score(y_test, pred, zero_division=0),
            "test_roc_auc": roc_auc_score(y_test, prob)
        })
        fitted[name] = model
        report = classification_report(
            y_test, pred, target_names=["Benign", "Malignant"],
            zero_division=0, output_dict=True
        )
        pd.DataFrame(report).to_csv(
            output_dir / f"{name.lower().replace(' ', '_')}_report.csv"
        )

    results = pd.DataFrame(rows).sort_values(
        ["cv_recall_mean", "test_roc_auc"], ascending=False
    )
    results.to_csv(output_dir / "model_comparison.csv", index=False)
    best_name = results.iloc[0]["model"]
    best_model = fitted[best_name]
    best_pred = best_model.predict(X_test)

    pd.DataFrame(
        confusion_matrix(y_test, best_pred),
        index=["Actual Benign", "Actual Malignant"],
        columns=["Predicted Benign", "Predicted Malignant"]
    ).to_csv(output_dir / "best_model_confusion_matrix.csv")

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, model in fitted.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "--")
    ax.set_title("ROC Curve Comparison")
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curve_comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    PrecisionRecallDisplay.from_estimator(best_model, X_test, y_test, ax=ax)
    ax.set_title(f"Precision-Recall Curve — {best_name}")
    fig.tight_layout()
    fig.savefig(output_dir / "precision_recall_curve.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    results.set_index("model")[[
        "cv_recall_mean", "test_precision", "test_recall", "test_f1", "test_roc_auc"
    ]].plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(output_dir / "model_comparison.png", dpi=160)
    plt.close(fig)

    (output_dir / "best_model_classification_report.txt").write_text(
        f"Selected model: {best_name}\n\n" +
        classification_report(y_test, best_pred, target_names=["Benign", "Malignant"])
    )
    return results, best_name, best_model
