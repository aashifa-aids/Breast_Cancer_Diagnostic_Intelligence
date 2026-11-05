from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def build_models():
    models = {}
    scaled = {
        "Logistic Regression": LogisticRegression(max_iter=3000, class_weight="balanced"),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Support Vector Machine": SVC(probability=True, class_weight="balanced", random_state=42)
    }
    for name, estimator in scaled.items():
        models[name] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", estimator)
        ])

    trees = {
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=250, class_weight="balanced", random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }
    for name, estimator in trees.items():
        models[name] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", estimator)
        ])
    return models
