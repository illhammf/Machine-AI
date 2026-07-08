"""
=========================================================
STACK OVERFLOW AI TOOLS ML PROJECT
MODEL TRAINING MODULE
=========================================================

Isi:
1. Supervised models (RF, SVM, XGBoost, MLP)
2. Train-test split pipeline
3. KMeans clustering analysis
4. GMM clustering analysis
5. Cluster summarization
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier

from data_preprocessing import RANDOM_STATE


# =========================================================
# 1. CLASSIFICATION MODELS
# =========================================================

def build_classification_models(preprocessor, y_train=None):

    models = {}

    # -----------------------------
    # Random Forest (baseline kuat)
    # -----------------------------
    models["Random Forest"] = Pipeline([
        ("preprocess", clone(preprocessor)),
        ("model", RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE
        ))
    ])

    # -----------------------------
    # Linear SVM (baseline linear)
    # -----------------------------
    models["Linear SVM"] = Pipeline([
        ("preprocess", clone(preprocessor)),
        ("model", LinearSVC(
            class_weight="balanced",
            max_iter=5000,
            random_state=RANDOM_STATE
        ))
    ])

    # -----------------------------
    # XGBoost (boosting model)
    # -----------------------------
    scale_pos_weight = 1.0
    if y_train is not None:
        counts = y_train.value_counts().sort_index()
        if len(counts) == 2:
            neg, pos = counts.values
            scale_pos_weight = neg / max(pos, 1)

    models["XGBoost"] = Pipeline([
        ("preprocess", clone(preprocessor)),
        ("model", XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=RANDOM_STATE
        ))
    ])

    # -----------------------------
    # MLP Neural Network
    # -----------------------------
    models["MLP"] = Pipeline([
        ("preprocess", clone(preprocessor)),
        ("model", MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=500,
            early_stopping=True,
            random_state=RANDOM_STATE
        ))
    ])

    return models


# =========================================================
# 2. TRAIN CLASSIFICATION PIPELINE
# =========================================================

def train_classification_models(X, y, preprocessor, test_size=0.2):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=RANDOM_STATE
    )

    models = build_classification_models(preprocessor, y_train)

    fitted_models = {}

    for name, model in models.items():
        print(f"Training model: {name}")
        model.fit(X_train, y_train)
        fitted_models[name] = model

    return fitted_models, {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }


# =========================================================
# 3. KMEANS CLUSTERING ANALYSIS
# =========================================================

def run_kmeans_analysis(X, preprocessor, k_range=range(2, 11)):

    X_processed = clone(preprocessor).fit_transform(X)

    # dimensionality reduction
    svd = TruncatedSVD(
        n_components=min(50, X_processed.shape[1] - 1),
        random_state=RANDOM_STATE
    )

    X_reduced = svd.fit_transform(X_processed)

    # scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reduced)

    results = []
    labels_store = {}

    for k in k_range:

        model = KMeans(
            n_clusters=k,
            n_init=10,
            random_state=RANDOM_STATE
        )

        labels = model.fit_predict(X_scaled)

        silhouette = silhouette_score(X_scaled, labels)

        results.append({
            "k": k,
            "inertia": model.inertia_,
            "silhouette": silhouette
        })

        labels_store[k] = labels

    metrics_df = pd.DataFrame(results)

    best_k = int(metrics_df.sort_values("silhouette", ascending=False)["k"].iloc[0])

    final_model = KMeans(
        n_clusters=best_k,
        n_init=10,
        random_state=RANDOM_STATE
    )

    final_labels = final_model.fit_predict(X_scaled)

    return {
        "model": final_model,
        "labels": final_labels,
        "best_k": best_k,
        "metrics": metrics_df,
        "X_cluster": X_scaled,
        "svd": svd,
        "scaler": scaler
    }


# =========================================================
# 4. GMM CLUSTERING ANALYSIS
# =========================================================

def run_gmm_analysis(X, preprocessor, k_range=range(2, 9)):

    X_processed = clone(preprocessor).fit_transform(X)

    svd = TruncatedSVD(
        n_components=min(30, X_processed.shape[1] - 1),
        random_state=RANDOM_STATE
    )

    X_reduced = svd.fit_transform(X_processed)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reduced)

    results = []

    for k in k_range:

        model = GaussianMixture(
            n_components=k,
            random_state=RANDOM_STATE,
            n_init=3
        )

        model.fit(X_scaled)
        labels = model.predict(X_scaled)

        silhouette = silhouette_score(X_scaled, labels)

        results.append({
            "k": k,
            "bic": model.bic(X_scaled),
            "aic": model.aic(X_scaled),
            "silhouette": silhouette
        })

    metrics_df = pd.DataFrame(results)

    best_k = int(metrics_df.sort_values("silhouette", ascending=False)["k"].iloc[0])

    final_model = GaussianMixture(
        n_components=best_k,
        random_state=RANDOM_STATE,
        n_init=3
    )

    final_labels = final_model.fit_predict(X_scaled)

    return {
        "model": final_model,
        "labels": final_labels,
        "best_k": best_k,
        "metrics": metrics_df,
        "X_cluster": X_scaled,
        "svd": svd,
        "scaler": scaler,
        "probabilities": final_model.predict_proba(X_scaled)
    }


# =========================================================
# 5. CLUSTER SUMMARY FUNCTION
# =========================================================

def summarize_clusters(X, labels):

    data = X.copy()
    data["Cluster"] = labels

    summary = []

    for cluster_id, group in data.groupby("Cluster"):

        row = {
            "Cluster": cluster_id,
            "Size": len(group),
            "Percentage": len(group) / len(data) * 100
        }

        # categorical summary
        for col in ["Age", "Country", "EdLevel", "DevType", "Employment", "RemoteWork"]:
            if col in group.columns:
                row[col] = group[col].mode().iloc[0]

        # numeric summary
        for col in ["YearsCode", "YearsCodePro"]:
            if col in group.columns:
                row[col] = pd.to_numeric(group[col], errors="coerce").mean()

        summary.append(row)

    return pd.DataFrame(summary).sort_values("Cluster").reset_index(drop=True)