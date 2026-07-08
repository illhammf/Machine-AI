"""
=========================================================
STACK OVERFLOW SURVEY 2025
DATA PREPROCESSING MODULE
=========================================================

Tujuan:
1. Load dataset survey
2. Membersihkan data mentah
3. Membuat target AI usage
4. Mencegah data leakage
5. Membangun preprocessing pipeline ML
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple, Dict

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.feature_extraction.text import CountVectorizer


# =========================================================
# CONFIGURATION
# =========================================================

RANDOM_STATE = 42

TARGET_SOURCE_COLUMN = "AISelect"
TARGET_COLUMN = "AI_Usage"


# =========================================================
# FEATURE DEFINITION
# =========================================================

CANDIDATE_FEATURES = [
    "Age",
    "Country",
    "EdLevel",
    "DevType",
    "Employment",
    "RemoteWork",
    "YearsCode",
    "YearsCodePro",
    "LanguageHaveWorkedWith",
    "DatabaseHaveWorkedWith",
    "PlatformHaveWorkedWith",
    "WebframeHaveWorkedWith",
    "ToolsTechHaveWorkedWith",
]

NUMERIC_FEATURES = ["YearsCode", "YearsCodePro"]

MULTI_SELECT_FEATURES = [
    "LanguageHaveWorkedWith",
    "DatabaseHaveWorkedWith",
    "PlatformHaveWorkedWith",
    "WebframeHaveWorkedWith",
    "ToolsTechHaveWorkedWith",
]

AI_LEAKAGE_PREFIXES = ("AI", "AIAgent", "AIModel", "AITool")

AI_LEAKAGE_COLUMNS = {
    "AISelect",
    "AISent",
    "AIAcc",
    "AIBen",
    "AIComplex",
    "AIThreat",
    "AIFrustration",
    "AIExplain",
    "AIAgents",
    "AIAgentChange",
    "AIAgent_Uses",
    "AIHuman",
    "AIOpen",
    "LearnCodeAI",
    "AILearnHow",
}


# =========================================================
# DATA LOADING
# =========================================================

def find_dataset_path(project_root: str | Path, year: int = 2025) -> Path:
    """
    Mencari lokasi dataset survey secara otomatis.
    """

    root = Path(project_root)

    candidates = [
        root / "Survey" / "packages" / "archive" / str(year) / "results.csv",
        root / "data" / f"survey_results_public_{year}.csv",
        root / f"survey_results_public_{year}.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError("Dataset tidak ditemukan untuk tahun " + str(year))


def load_survey_data(path: str | Path) -> pd.DataFrame:
    """
    Load dataset CSV dengan handling missing value standar.
    """
    return pd.read_csv(path, low_memory=False, na_values=["NA", ""])


# =========================================================
# TARGET CREATION
# =========================================================

def create_ai_usage_target(df: pd.DataFrame, source_col: str) -> pd.Series:
    """
    Membuat target AI Usage:
    - Yes → 1
    - No → 0
    """

    if source_col not in df.columns:
        raise KeyError(f"Kolom {source_col} tidak ditemukan")

    values = df[source_col].astype(str).str.lower().str.strip()

    target = pd.Series(np.nan, index=df.index)

    target[values.str.contains("yes")] = 1
    target[values.str.contains("no")] = 0

    return target


# =========================================================
# DATA CLEANING
# =========================================================

def clean_years_code(value):
    """
    Convert YearsCode ke numeric.
    """

    if pd.isna(value):
        return np.nan

    value = str(value).strip()

    if value == "Less than 1 year":
        return 0.0

    if value == "More than 50 years":
        return 51.0

    try:
        return float(value)
    except:
        return np.nan


# =========================================================
# FEATURE FILTERING
# =========================================================

def is_leakage_column(col: str) -> bool:
    """
    Deteksi kolom AI leakage (tidak boleh dipakai model).
    """

    return col in AI_LEAKAGE_COLUMNS or any(
        col.startswith(prefix) for prefix in AI_LEAKAGE_PREFIXES
    )


def select_features(df: pd.DataFrame, features: Iterable[str]):
    """
    Memilih fitur yang tersedia dan aman dari leakage.
    """

    available = []
    missing = []

    for col in features:
        if col not in df.columns:
            missing.append(col)
            continue

        if is_leakage_column(col):
            continue

        available.append(col)

    return available, missing


# =========================================================
# PREPROCESSING HELPERS
# =========================================================

def flatten_text(values):
    """
    Flatten array ColumnTransformer ke string.
    """
    return pd.Series(values.ravel()).fillna("Unknown").astype(str)


def tokenize(text):
    """
    Tokenizer untuk multi-select field.
    """
    if pd.isna(text):
        return ["Unknown"]

    return [t.strip() for t in str(text).split(";") if t.strip()]


def make_one_hot():
    """
    OneHotEncoder compatible semua versi sklearn.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


# =========================================================
# MAIN PREPROCESSOR
# =========================================================

def build_preprocessor(X: pd.DataFrame):
    """
    Membuat pipeline preprocessing lengkap:
    numeric + categorical + multi-select
    """

    numeric = [c for c in NUMERIC_FEATURES if c in X.columns]

    multi = [c for c in MULTI_SELECT_FEATURES if c in X.columns]

    categorical = [
        c for c in X.columns
        if c not in numeric and c not in multi
    ]

    transformers = []

    # =========================
    # NUMERIC PIPELINE
    # =========================
    if numeric:
        transformers.append((
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            numeric
        ))

    # =========================
    # CATEGORICAL PIPELINE
    # =========================
    if categorical:
        transformers.append((
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", make_one_hot())
            ]),
            categorical
        ))

    # =========================
    # MULTI-SELECT PIPELINE
    # =========================
    for col in multi:
        transformers.append((
            f"multi_{col}",
            Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                ("flatten", FunctionTransformer(flatten_text, validate=False)),
                ("vectorizer", CountVectorizer(
                    tokenizer=tokenize,
                    binary=True,
                    min_df=10
                ))
            ]),
            [col]
        ))

    return ColumnTransformer(transformers, remainder="drop")


# =========================================================
# FINAL DATA PREPARATION
# =========================================================

def prepare_model_data(df: pd.DataFrame):
    """
    Pipeline utama:
    - create target
    - clean data
    - select features
    - return X, y
    """

    data = df.copy()

    # target
    data[TARGET_COLUMN] = create_ai_usage_target(data, TARGET_SOURCE_COLUMN)

    # drop missing target
    data = data.dropna(subset=[TARGET_COLUMN])

    data[TARGET_COLUMN] = data[TARGET_COLUMN].astype(int)

    # feature selection
    features, missing = select_features(data, CANDIDATE_FEATURES)

    # clean numeric
    for col in NUMERIC_FEATURES:
        if col in data.columns:
            data[col] = data[col].apply(clean_years_code)

    X = data[features]
    y = data[TARGET_COLUMN]

    metadata = {
        "features_used": features,
        "missing_features": missing,
        "target_distribution": y.value_counts().to_dict(),
        "rows": len(data)
    }

    return X, y, metadata