"""Fitur untuk klasifikasi kerusakan bangunan (damage_grade, 5 kelas)."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

TARGET = "damage_grade"
ID_COLUMN = "building_id"
CATEGORICAL = [
    "land_surface_condition", "foundation_type", "roof_type", "ground_floor_type",
    "other_floor_type", "position", "plan_configuration", "superstructure",
]
LOCATION = ("ward_id", "vdcmun_id")


def prepare_features(data):
    """Versi dasar (dipakai baseline logistic regression): cukup buang kolom ID."""
    return data.copy().drop(columns=[ID_COLUMN], errors="ignore")


def encode_target(series):
    """'Grade 3' -> 3."""
    return series.astype(str).str.extract(r"(\d+)", expand=False).astype(int)


class FeatureBuilder:
    """Fitur untuk gradient boosting: fitur turunan + target encoding wilayah.

    Alur: fit_transform(train) dipakai saat training (target encoding dihitung
    out-of-fold supaya tidak bocor), lalu transform(data_lain) untuk val/test.
    """

    def __init__(self, smooth=20, n_splits=5, seed=1):
        self.smooth, self.n_splits, self.seed = smooth, n_splits, seed

    # ---------- fitur dasar ----------
    def _base(self, data):
        d = data.drop(columns=[ID_COLUMN, TARGET], errors="ignore").copy()
        d["age_unknown"] = (d.age_building >= 999).astype(int)       # 999 = kode 'tidak diketahui'
        d["age_building"] = d.age_building.where(d.age_building < 999, np.nan)
        d["area_per_floor"] = d.plinth_area_sq_ft / d.count_floors_pre_eq
        d["height_per_floor"] = d.height_ft_pre_eq / d.count_floors_pre_eq
        d["slender"] = d.height_ft_pre_eq / np.sqrt(d.plinth_area_sq_ft)
        for c in LOCATION:
            d[f"{c}_freq"] = d[c].map(self.freq_[c]).fillna(0)
        for c in CATEGORICAL:
            d[c] = pd.Categorical(d[c], categories=self.categories_[c])
        return d

    # ---------- target encoding ----------
    def _tables(self, X, y):
        dummies = pd.get_dummies(y).reindex(columns=self.classes_, fill_value=0).astype(float)
        tables = {}
        for c in LOCATION:
            g = dummies.groupby(X[c].values)
            tables[c] = (g.sum() + self.smooth * self.prior_).div(g.size() + self.smooth, axis=0)
        return tables

    def _apply(self, X, tables):
        out = {}
        fallback = dict(zip(self.classes_, self.prior_))
        for c in LOCATION:
            mapped = tables[c].reindex(X[c].values).fillna(fallback)
            for k in self.classes_:
                out[f"{c}_te{k}"] = mapped[k].values
        return pd.DataFrame(out, index=X.index)

    # ---------- API ----------
    def fit_transform(self, train_raw, y, ref_raw=None):
        """train_raw: data train (tanpa kolom target). ref_raw: data tambahan (mis. test)
        yang hanya dipakai untuk menghitung frekuensi & daftar kategori (tanpa label)."""
        ref = train_raw if ref_raw is None else pd.concat([train_raw, ref_raw.drop(columns=[TARGET], errors="ignore")])
        self.categories_ = {c: sorted(ref[c].unique()) for c in CATEGORICAL}
        self.freq_ = {c: ref[c].value_counts() for c in LOCATION}
        self.classes_ = sorted(y.unique())
        self.prior_ = y.value_counts(normalize=True).reindex(self.classes_).values

        X = self._base(train_raw)
        y = y.set_axis(X.index)
        oof = []
        for a, b in StratifiedKFold(self.n_splits, shuffle=True, random_state=self.seed).split(X, y):
            oof.append(self._apply(X.iloc[b], self._tables(X.iloc[a], y.iloc[a])))
        oof = pd.concat(oof).loc[X.index]
        self.tables_ = self._tables(X, y)          # tabel final (seluruh train) untuk data lain
        return pd.concat([X, oof], axis=1)

    def transform(self, data_raw):
        X = self._base(data_raw)
        return pd.concat([X, self._apply(X, self.tables_)], axis=1)
