# untuk klasifikasi kerusakan bangunan (dipakai baseline logistic regression)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def create_preprocessing(X):
    categorical_columns = X.select_dtypes(
        include=["object", "category", "bool", "string"]
    ).columns.tolist()

    # Kolom kode wilayah berakhiran _id diperlakukan sebagai kategori.
    for col in X.columns:
        if col.lower().endswith("_id") and col not in categorical_columns:
            categorical_columns.append(col)

    numerical_columns = [col for col in X.columns if col not in categorical_columns]

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", numerical_pipeline, numerical_columns),
        ("cat", categorical_pipeline, categorical_columns),
    ])
