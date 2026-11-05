from pathlib import Path
import pandas as pd
from sklearn.datasets import load_breast_cancer

ROOT = Path(__file__).resolve().parents[1]

def load_dataset():
    data = load_breast_cancer()
    frame = pd.DataFrame(data.data, columns=data.feature_names)
    frame["diagnosis"] = ["M" if x == 0 else "B" for x in data.target]
    return frame

def save_raw_dataset(frame):
    path = ROOT / "data" / "raw" / "wdbc_raw.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path
