# Structural Damage Classification (Kaggle)

Klasifikasi tingkat kerusakan bangunan (`damage_grade`, Grade 1-5) akibat gempa.
Metrik evaluasi: **Macro F1-Score**.

## Pendekatan

1. **Baseline multi-kelas**: Logistic Regression dengan strategi One-vs-All dan One-vs-One.
2. **Model final**: Gradient Boosting (`HistGradientBoostingClassifier`) dengan
   - fitur turunan (umur tidak diketahui, luas/tinggi per lantai, rasio ramping),
   - target encoding wilayah (`ward_id`, `vdcmun_id`) secara out-of-fold,
   - pengali bobot kelas yang dituning untuk memaksimalkan Macro F1.

## Hasil (validasi 20%)

| Model | Macro F1 | Accuracy |
|---|---|---|
| LogReg One-vs-All | 0.5044 | 0.5417 |
| LogReg One-vs-One | 0.5075 | 0.5390 |
| Gradient Boosting + fitur lokasi | 0.5397 | 0.5824 |
| Gradient Boosting + fitur lokasi + bobot kelas | **0.5502** | 0.5741 |

Detail ada di `reports/model_comparison.csv`, fitur terpenting di `reports/figures/feature_importance.png`.

## Struktur

```
data/            train.csv, test.csv, sample_submission.csv
notebooks/       01_eda, 02_modeling, 03_submission
scripts/         preprocess, feature_engineering, train, predict
models/          final_model.pkl
submissions/     submission.csv
reports/         model_comparison.csv, figures/
```

## Cara menjalankan

```bash
pip install -r requirements.txt
python -m scripts.train      # latih, bandingkan model, simpan model final
python -m scripts.predict    # buat submissions/submission.csv
```

Jalankan dari folder root proyek (bukan dari dalam `scripts/`).
