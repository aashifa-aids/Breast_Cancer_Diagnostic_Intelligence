# Breast Cancer Diagnostic Intelligence System — Yuvaintern Week 6

Educational, reproducible data-science project using the Wisconsin Diagnostic Breast Cancer dataset bundled with scikit-learn.

## Workflow
1. Dataset collection and validation
2. Cleaning and preprocessing
3. Exploratory data analysis
4. Statistical tests, effect size and VIF
5. Six machine-learning models
6. Stratified cross-validation and hyperparameter tuning
7. Model evaluation and visualizations
8. Permutation importance and optional SHAP
9. Model export and optional Streamlit demo

## Setup
Python 3.10+ recommended.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python run_project.py
```

Optional interface:
```bash
streamlit run app/app.py
```

## Dataset
The project loads 569 observations and 30 numeric features from `sklearn.datasets.load_breast_cancer`. Target labels are converted to M (malignant) and B (benign). No external download is required.

## Important
This is an educational machine-learning project, not a medical device or diagnostic system. Model probabilities are not clinically calibrated risk estimates. Do not use outputs for healthcare decisions. External validation and qualified clinical oversight would be required for any real-world clinical use.
