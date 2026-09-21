from pathlib import Path
import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

TARGET = "Attack Type"
BASE_DIR = Path(__file__).resolve().parent


def train(dataset_path: Path) -> None:
    df = pd.read_csv(dataset_path)
    if TARGET not in df.columns:
        raise ValueError(f"Dataset must contain {TARGET!r}")

    df[TARGET] = df[TARGET].astype(str).str.strip()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # Remove rare classes before fitting the encoder so model and encoder share one mapping.
    class_counts = df[TARGET].value_counts()
    df = df[df[TARGET].isin(class_counts[class_counts >= 2].index)].copy()

    X = df.drop(columns=[TARGET]).copy()
    y_text = df[TARGET].copy()
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_text)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(encoder.classes_),
        random_state=42,
        tree_method="hist",
        eval_metric="mlogloss",
    )
    model.fit(X_train, y_train)

    model_classes = np.asarray(model.classes_)
    expected_classes = np.arange(len(encoder.classes_))
    if not np.array_equal(model_classes, expected_classes):
        raise RuntimeError(f"Unexpected model classes: {model_classes.tolist()}")
    if model.n_classes_ != len(encoder.classes_):
        raise RuntimeError("Model and encoder class counts differ.")

    output_dir = BASE_DIR / "models"
    output_dir.mkdir(exist_ok=True)
    joblib.dump(model, output_dir / "network_intrusion_detector.pkl")
    joblib.dump(encoder, output_dir / "label_encoder.pkl")

    print("Saved model and encoder from the same training run.")
    print("Features:", len(model.feature_names_in_))
    print("Classes:", list(encoder.classes_))
    print("Validation rows:", len(X_test))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path, help="Full labeled training CSV")
    args = parser.parse_args()
    train(args.dataset)
