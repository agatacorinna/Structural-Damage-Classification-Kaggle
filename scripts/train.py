"""Latih & bandingkan model, simpan model final.

Kandidat:
  1. Logistic Regression + One-vs-All (OvR)
  2. Logistic Regression + One-vs-One (OvO)
  3. Gradient Boosting + fitur turunan + target encoding wilayah (+ bobot kelas untuk Macro F1)

Jalankan dari root proyek:  python -m scripts.train
"""
import warnings
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.pipeline import Pipeline

from scripts.feature_engineering import TARGET, FeatureBuilder, encode_target, prepare_features
from scripts.preprocess import create_preprocessing

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "train.csv"
TEST_PATH = ROOT / "data" / "test.csv"
MODEL_PATH = ROOT / "models" / "final_model.pkl"
RESULT_PATH = ROOT / "reports" / "model_comparison.csv"
FIGURE_PATH = ROOT / "reports" / "figures" / "feature_importance.png"


def new_gb():
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, max_leaf_nodes=63, l2_regularization=1.0,
        categorical_features="from_dtype", early_stopping=False, random_state=42,
    )


def tune_class_weights(y_true, proba):
    """Cari pengali per kelas yang memaksimalkan Macro F1 (coordinate ascent)."""
    w = np.ones(proba.shape[1])
    best = f1_score(y_true, proba.argmax(1) + 1, average="macro")
    for _ in range(3):
        for k in range(len(w)):
            for s in np.linspace(0.5, 2.5, 21):
                w2 = w.copy()
                w2[k] = s
                f = f1_score(y_true, (proba * w2).argmax(1) + 1, average="macro")
                if f > best:
                    best, w = f, w2
    return w, best


def score(y_true, pred):
    return f1_score(y_true, pred, average="macro"), accuracy_score(y_true, pred)


def main():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    y = encode_target(train[TARGET])
    X_raw = train.drop(columns=[TARGET])

    Xtr_raw, Xval_raw, ytr, yval = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )
    results = []

    # --- 1 & 2: baseline logistic regression, OvA vs OvO ---
    base = LogisticRegression(max_iter=500, class_weight="balanced")
    for name, strategy in {
        "LogReg One-vs-All": OneVsRestClassifier(base),
        "LogReg One-vs-One": OneVsOneClassifier(base),
    }.items():
        print(f"\nEvaluasi {name}...", flush=True)
        Xa, Xb = prepare_features(Xtr_raw), prepare_features(Xval_raw)
        pipe = Pipeline([("prep", create_preprocessing(Xa)), ("clf", strategy)])
        pipe.fit(Xa, ytr)
        f1, acc = score(yval, pipe.predict(Xb))
        results.append({"model": name, "macro_f1": f1, "accuracy": acc})
        print(f"Macro F1: {f1:.4f} | Accuracy: {acc:.4f}", flush=True)

    # --- 3: gradient boosting + fitur lokasi ---
    print("\nEvaluasi Gradient Boosting + fitur lokasi...", flush=True)
    fb = FeatureBuilder()
    Xt = fb.fit_transform(Xtr_raw, ytr, ref_raw=test)
    Xv = fb.transform(Xval_raw)
    gb = new_gb().fit(Xt, ytr)
    proba = gb.predict_proba(Xv)
    f1, acc = score(yval, proba.argmax(1) + 1)
    results.append({"model": "Gradient Boosting + fitur lokasi", "macro_f1": f1, "accuracy": acc})
    print(f"Macro F1: {f1:.4f} | Accuracy: {acc:.4f}", flush=True)

    weights, f1w = tune_class_weights(yval, proba)
    acc_w = accuracy_score(yval, (proba * weights).argmax(1) + 1)
    results.append({"model": "Gradient Boosting + fitur lokasi + bobot kelas",
                    "macro_f1": f1w, "accuracy": acc_w})
    print(f"Bobot kelas {weights.round(2)} -> Macro F1: {f1w:.4f}", flush=True)

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).sort_values("macro_f1", ascending=False).to_csv(RESULT_PATH, index=False)

    # --- feature importance (permutation, pada sebagian data validasi) ---
    print("\nMenghitung feature importance...", flush=True)
    idx = Xv.sample(15000, random_state=0).index
    imp = permutation_importance(
        gb, Xv.loc[idx], yval.loc[idx], scoring="f1_macro", n_repeats=3, random_state=0
    )
    imp_s = pd.Series(imp.importances_mean, index=Xv.columns).sort_values().tail(15)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    imp_s.plot(kind="barh")
    plt.xlabel("Penurunan Macro F1 saat fitur diacak")
    plt.title("15 Fitur Terpenting")
    plt.tight_layout()
    plt.savefig(FIGURE_PATH, dpi=150)

    # --- model final: latih ulang di seluruh data train ---
    print("\nMelatih ulang model final dengan seluruh data train...", flush=True)
    fb_full = FeatureBuilder()
    X_full = fb_full.fit_transform(X_raw, y, ref_raw=test)
    final = new_gb().fit(X_full, y)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"builder": fb_full, "model": final, "weights": weights}, MODEL_PATH)
    print(f"Model final disimpan di: {MODEL_PATH}", flush=True)


if __name__ == "__main__":
    main()
