import argparse
import os
from typing import Dict, Tuple, List

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_MODEL_PATH = os.path.join("artifacts", "crop_pipeline.joblib")


FEATURES = [
    "Phosphorus",
    "Potassium",
    "Rainfall_mm",
    "Temperature_C",
    "Humidity_percent",
    "Previous_Crop",
    "State",
    "Season",
    "Soil_Type",
    "Soil_pH",
    "Nitrogen",
]
TARGET = "Crop"

NUMERIC_FEATURES = [
    "Phosphorus",
    "Potassium",
    "Rainfall_mm",
    "Temperature_C",
    "Humidity_percent",
    "Soil_pH",
    "Nitrogen",
]
CATEGORICAL_FEATURES = ["Previous_Crop", "State", "Season", "Soil_Type"]


def build_pipeline(
    n_estimators: int = 500,
    max_depth: int | None = None,
    min_samples_leaf: int = 2,
    min_samples_split: int = 4,
    max_features: str | float = "sqrt",
    random_state: int = 42,
) -> Pipeline:
    """
    Same overall shape as before: ColumnTransformer -> RandomForestClassifier
    inside a single Pipeline. Internals tuned for accuracy:

    - StandardScaler added on numeric features (harmless for RF, helps if the
      pipeline is ever reused with a different/linear model).
    - OneHotEncoder now groups rare categories instead of letting them create
      noisy, low-support splits.
    - RandomForest defaults changed from a hard depth cap to leaf/split based
      regularization, which tends to generalize better on tabular data with
      many one-hot columns. class_weight="balanced" handles crop imbalance.
    """
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=0.01,  # group categories seen in <1% of rows
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        max_features=max_features,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )

    return Pipeline(steps=[("preprocess", preprocessor), ("model", clf)])


def tune_hyperparameters(
    pipe: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, n_iter: int = 30, random_state: int = 42
) -> Pipeline:
    """
    RandomizedSearchCV over the model step only. Keeps the pipeline structure
    fixed and just searches for better RandomForest settings than the
    hand-picked defaults.
    """
    param_dist = {
        "model__n_estimators": [200, 300, 400, 500, 700, 900],
        "model__max_depth": [None, 10, 15, 20, 30, 40],
        "model__min_samples_leaf": [1, 2, 3, 4, 6],
        "model__min_samples_split": [2, 4, 6, 8, 10],
        "model__max_features": ["sqrt", "log2", 0.5, 0.7],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="accuracy",
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)

    print("Best CV accuracy:", search.best_score_)
    print("Best params:", search.best_params_)

    return search.best_estimator_


def train_and_save(
    csv_path: str,
    out_path: str = DEFAULT_MODEL_PATH,
    tune: bool = False,
    n_iter: int = 30,
) -> Tuple[Pipeline, Dict]:
    df = pd.read_csv(csv_path)

    # Keep only required columns
    needed = set(FEATURES + [TARGET])
    missing_cols = [c for c in needed if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset missing columns: {missing_cols}")

    df = df[FEATURES + [TARGET]].dropna(subset=[TARGET]).copy()

    X = df[FEATURES]
    y = df[TARGET].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None
    )

    pipe = build_pipeline()

    # Cross-validated score on the training split BEFORE fitting on the full
    # train set. This tells you whether your accuracy is stable or just a
    # lucky/unlucky single split.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"CV accuracy (5-fold, pre-tuning): {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    if tune:
        pipe = tune_hyperparameters(pipe, X_train, y_train, n_iter=n_iter)
    else:
        pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))

    meta = {
        "accuracy": acc,
        "cv_accuracy_mean": float(cv_scores.mean()),
        "cv_accuracy_std": float(cv_scores.std()),
        "n_rows": int(df.shape[0]),
        "n_classes": int(y.nunique()),
        "features": FEATURES,
        "target": TARGET,
        "tuned": tune,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump({"pipeline": pipe, "meta": meta}, out_path)

    print("Saved:", out_path)
    print("Test accuracy:", acc)
    print("\nClassification report:\n", classification_report(y_test, y_pred))
    return pipe, meta


def load_pipeline(path: str = DEFAULT_MODEL_PATH) -> Pipeline:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found at '{path}'. Train it first with "
            f"`python model.py --train --csv <path_to_csv>`."
        )
    obj = joblib.load(path)
    if isinstance(obj, dict) and "pipeline" in obj:
        return obj["pipeline"]
    # fallback if user saved pipeline directly
    return obj


def predict_crop(pipe: Pipeline, row: Dict, top_k: int = 3) -> Tuple[str, List[Dict]]:
    """
    row: dict with keys in FEATURES (numeric fields must be int/float,
    categorical fields must be strings).
    returns (predicted_crop, top_k_list)
    top_k_list is list of {crop, probability}
    """
    # Validate all required features are present
    missing = [f for f in FEATURES if f not in row]
    if missing:
        raise ValueError(f"predict_crop: missing required fields: {missing}")

    # Coerce numeric fields defensively (in case caller passed strings)
    clean_row = dict(row)
    for f in NUMERIC_FEATURES:
        try:
            clean_row[f] = float(clean_row[f])
        except (TypeError, ValueError):
            raise ValueError(f"predict_crop: field '{f}' must be numeric, got {clean_row[f]!r}")

    for f in CATEGORICAL_FEATURES:
        clean_row[f] = str(clean_row[f]) if clean_row[f] is not None else ""

    X = pd.DataFrame([clean_row], columns=FEATURES)
    pred = str(pipe.predict(X)[0])

    topk = []
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X)[0]
        classes = [str(c) for c in pipe.named_steps["model"].classes_]
        pairs = sorted(zip(classes, proba), key=lambda x: x[1], reverse=True)[:max(1, top_k)]
        topk = [{"crop": c, "probability": float(p)} for c, p in pairs]

    return pred, topk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train and save the pipeline")
    parser.add_argument("--csv", type=str, default="", help="Path to CSV dataset")
    parser.add_argument("--out", type=str, default=DEFAULT_MODEL_PATH, help="Output model path")
    parser.add_argument("--tune", action="store_true", help="Run RandomizedSearchCV before final fit")
    parser.add_argument("--n_iter", type=int, default=30, help="Number of RandomizedSearchCV iterations")
    args = parser.parse_args()

    if args.train:
        if not args.csv:
            raise SystemExit("Please provide --csv path/to/dataset.csv")
        train_and_save(args.csv, args.out, tune=args.tune, n_iter=args.n_iter)
    else:
        print("Nothing to do. Use --train --csv <path> to train.")


if __name__ == "__main__":
    main()