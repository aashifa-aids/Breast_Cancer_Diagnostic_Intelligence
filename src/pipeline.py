from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from .data_loading import ROOT, load_dataset, save_raw_dataset
from .preprocessing import clean_dataset
from .eda import create_eda_outputs
from .statistical_analysis import run_statistical_analysis
from .model_training import build_models
from .model_evaluation import evaluate_models
from .explainability import create_permutation_importance, try_shap

def run_pipeline():
    dirs = {
        "raw": ROOT / "data" / "raw",
        "processed": ROOT / "data" / "processed",
        "viz": ROOT / "visualizations",
        "reports": ROOT / "reports",
        "models": ROOT / "models"
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    raw = load_dataset()
    save_raw_dataset(raw)
    frame, quality = clean_dataset(raw)
    frame.to_csv(dirs["processed"] / "wdbc_cleaned.csv", index=False)
    (dirs["reports"] / "data_quality_report.json").write_text(json.dumps(quality, indent=2))

    create_eda_outputs(frame, dirs["viz"])
    stats, vif = run_statistical_analysis(frame, dirs["reports"])

    X = frame.drop(columns="diagnosis")
    y = (frame.diagnosis == "M").astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    results, baseline_name, baseline_model = evaluate_models(
        build_models(), X_train, X_test, y_train, y_test, dirs["reports"]
    )

    # Tune Random Forest using training data only; test set remains untouched until evaluation.
    search = GridSearchCV(
        build_models()["Random Forest"],
        {
            "model__max_depth": [None, 4, 8],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", 0.5]
        },
        scoring="recall",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1
    )
    search.fit(X_train, y_train)
    tuned_model = search.best_estimator_
    tuned_recall = search.best_score_
    baseline_cv = float(results.loc[results.model == baseline_name, "cv_recall_mean"].iloc[0])

    best_name, best_model = baseline_name, baseline_model
    if tuned_recall > baseline_cv:
        best_name, best_model = "Tuned Random Forest", tuned_model
        pred = best_model.predict(X_test)
        prob = best_model.predict_proba(X_test)[:, 1]
        tuned_row = {
            "model": best_name, "cv_recall_mean": tuned_recall, "cv_recall_std": 0.0,
            "test_accuracy": accuracy_score(y_test, pred),
            "test_precision": precision_score(y_test, pred, zero_division=0),
            "test_recall": recall_score(y_test, pred, zero_division=0),
            "test_f1": f1_score(y_test, pred, zero_division=0),
            "test_roc_auc": roc_auc_score(y_test, prob)
        }
        results = pd.concat([results, pd.DataFrame([tuned_row])], ignore_index=True)
        results.to_csv(dirs["reports"] / "model_comparison_with_tuning.csv", index=False)

    joblib.dump(best_model, dirs["models"] / "best_model.joblib")
    metadata = {
        "selected_model": best_name,
        "feature_names": list(X.columns),
        "positive_class": "M (malignant)",
        "negative_class": "B (benign)",
        "default_threshold": 0.5,
        "disclaimer": "Educational demonstration only; not for clinical use."
    }
    (dirs["models"] / "model_metadata.json").write_text(json.dumps(metadata, indent=2))

    importance = create_permutation_importance(best_model, X_test, y_test, dirs["viz"])
    shap_status = try_shap(best_model, X_train, X_test, dirs["viz"])

    summary = [
        "# Week 6 Results Summary", "",
        f"- Rows after cleaning: {len(frame)}",
        f"- Input features: {X.shape[1]}",
        f"- Selected model: {best_name}",
        f"- Optional SHAP: {shap_status}", "",
        "## Model comparison", results.to_markdown(index=False), "",
        "## Top statistical results", stats.head(10).to_markdown(index=False), "",
        "## Highest VIF values", vif.head(10).to_markdown(index=False), "",
        "## Top permutation importance", importance.head(10).to_markdown(index=False), "",
        "Educational results only; not clinically validated."
    ]
    (dirs["reports"] / "results_summary.md").write_text("\n".join(summary))
    return results
