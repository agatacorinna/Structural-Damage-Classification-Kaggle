"""Membuat file submission Kaggle dari model final.
Jalankan dari root proyek:  python -m scripts.predict
"""
from pathlib import Path

import joblib
import pandas as pd

from scripts.feature_engineering import ID_COLUMN, TARGET

ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = ROOT / "data" / "test.csv"
SAMPLE_PATH = ROOT / "data" / "sample_submission.csv"
MODEL_PATH = ROOT / "models" / "final_model.pkl"
OUTPUT_PATH = ROOT / "submissions" / "submission.csv"


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model belum ada. Jalankan: python -m scripts.train")

    test = pd.read_csv(TEST_PATH, dtype={ID_COLUMN: str})
    sample = pd.read_csv(SAMPLE_PATH, dtype={ID_COLUMN: str})

    bundle = joblib.load(MODEL_PATH)
    X_test = bundle["builder"].transform(test)
    proba = bundle["model"].predict_proba(X_test)
    labels = (proba * bundle["weights"]).argmax(axis=1) + 1   # kelas 1..5

    submission = sample[[ID_COLUMN]].merge(
        pd.DataFrame({ID_COLUMN: test[ID_COLUMN], TARGET: labels}),
        on=ID_COLUMN, how="left", validate="one_to_one",
    )
    if submission[TARGET].isna().any():
        raise ValueError("Ada building_id tanpa prediksi.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(OUTPUT_PATH, index=False)
    print(submission.head().to_string(index=False))
    print("\nUkuran:", submission.shape, "| tersimpan di:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
