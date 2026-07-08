"""
=========================================================
STACK OVERFLOW AI TOOLS ML PROJECT
EVALUATION & VISUALIZATION MODULE
=========================================================

Isi:
1. Evaluasi model klasifikasi
2. Confusion matrix analysis
3. Model comparison table
4. Plot evaluasi model
5. Evaluasi clustering (KMeans & GMM)
6. Save semua output ke folder struktur project
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# =========================================================
# 1. MODEL EVALUATION
# =========================================================

def evaluate_classification_models(models: dict, X_test, y_test):
    """
    Evaluasi semua model klasifikasi dengan metric standar.
    """

    results = {}

    for name, model in models.items():

        y_pred = model.predict(X_test)

        results[name] = {
            "metrics": {
                "model": name,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1_score": f1_score(y_test, y_pred, zero_division=0),
            },

            "confusion_matrix": confusion_matrix(y_test, y_pred),

            "classification_report_text": classification_report(
                y_test,
                y_pred,
                target_names=["Tidak Menggunakan AI", "Menggunakan AI"],
                zero_division=0
            ),

            "classification_report_dict": classification_report(
                y_test,
                y_pred,
                target_names=["Tidak Menggunakan AI", "Menggunakan AI"],
                zero_division=0,
                output_dict=True
            )
        }

    return results


# =========================================================
# 2. MODEL COMPARISON TABLE
# =========================================================

def make_comparison_table(results: dict) -> pd.DataFrame:
    """
    Membuat tabel perbandingan performa model.
    """

    df = pd.DataFrame([r["metrics"] for r in results.values()])
    return df.sort_values("f1_score", ascending=False)


# =========================================================
# 3. SAVE OUTPUTS
# =========================================================

def save_classification_outputs(results: dict, output_root: str | Path):

    output_root = Path(output_root)

    tables_dir = output_root / "tables"
    reports_dir = output_root / "reports"

    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    comparison = make_comparison_table(results)

    # save comparison table
    comparison.to_csv(tables_dir / "model_comparison.csv", index=False)

    for model_name, result in results.items():

        safe_name = model_name.lower().replace(" ", "_")

        # confusion matrix
        cm_df = pd.DataFrame(
            result["confusion_matrix"],
            index=["Actual_0", "Actual_1"],
            columns=["Pred_0", "Pred_1"]
        )

        cm_df.to_csv(tables_dir / f"confusion_matrix_{safe_name}.csv")

        # classification report table
        report_df = pd.DataFrame(result["classification_report_dict"]).transpose()
        report_df.to_csv(tables_dir / f"classification_report_{safe_name}.csv")

        # text report
        with open(reports_dir / f"classification_report_{safe_name}.txt", "w", encoding="utf-8") as f:
            f.write(result["classification_report_text"])

    return comparison


# =========================================================
# 4. CONFUSION MATRIX PLOT
# =========================================================

def plot_confusion_matrices(results: dict, output_root: str | Path):

    figures_dir = Path(output_root) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    for model_name, result in results.items():

        safe_name = model_name.lower().replace(" ", "_")

        plt.figure(figsize=(5, 4))

        sns.heatmap(
            result["confusion_matrix"],
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["Actual 0", "Actual 1"]
        )

        plt.title(f"Confusion Matrix - {model_name}")
        plt.tight_layout()

        plt.savefig(figures_dir / f"confusion_matrix_{safe_name}.png", dpi=150)
        plt.close()


# =========================================================
# 5. MODEL COMPARISON PLOT
# =========================================================

def plot_model_comparison(comparison_df: pd.DataFrame, output_root: str | Path):

    figures_dir = Path(output_root) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_df = comparison_df.melt(
        id_vars="model",
        value_vars=["accuracy", "precision", "recall", "f1_score"],
        var_name="metric",
        value_name="score"
    )

    plt.figure(figsize=(9, 5))

    sns.barplot(
        data=plot_df,
        x="metric",
        y="score",
        hue="model"
    )

    plt.ylim(0, 1)
    plt.title("Perbandingan Performa Model Machine Learning")

    plt.tight_layout()
    plt.savefig(figures_dir / "model_comparison.png", dpi=150)
    plt.close()


# =========================================================
# 6. KMEANS EVALUATION PLOT
# =========================================================

def plot_kmeans_metrics(metrics_df: pd.DataFrame, output_root: str | Path):

    figures_dir = Path(output_root) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    sns.lineplot(data=metrics_df, x="k", y="inertia", marker="o", ax=ax[0])
    ax[0].set_title("Elbow Method (KMeans)")
    ax[0].set_xlabel("K")
    ax[0].set_ylabel("Inertia")

    sns.lineplot(data=metrics_df, x="k", y="silhouette", marker="o", ax=ax[1])
    ax[1].set_title("Silhouette Score (KMeans)")
    ax[1].set_xlabel("K")
    ax[1].set_ylabel("Score")

    plt.tight_layout()
    plt.savefig(figures_dir / "kmeans_metrics.png", dpi=150)
    plt.close()


# =========================================================
# 7. CLUSTER DISTRIBUTION
# =========================================================

def plot_cluster_distribution(cluster_summary: pd.DataFrame, output_root: str | Path):

    figures_dir = Path(output_root) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 4))

    sns.barplot(
        data=cluster_summary,
        x="Cluster",
        y="Size"
    )

    plt.title("Distribusi Data per Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Jumlah Data")

    plt.tight_layout()
    plt.savefig(figures_dir / "cluster_distribution.png", dpi=150)
    plt.close()


# =========================================================
# 8. GMM METRICS PLOT
# =========================================================

def plot_gmm_metrics(metrics_df: pd.DataFrame, output_root: str | Path):

    figures_dir = Path(output_root) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    sns.lineplot(data=metrics_df, x="k", y="bic", marker="o", ax=ax[0], label="BIC")
    sns.lineplot(data=metrics_df, x="k", y="aic", marker="o", ax=ax[0], label="AIC")

    ax[0].set_title("GMM Model Selection")
    ax[0].set_xlabel("K")

    sns.lineplot(data=metrics_df, x="k", y="silhouette", marker="o", ax=ax[1])
    ax[1].set_title("Silhouette Score (GMM)")
    ax[1].set_xlabel("K")

    plt.tight_layout()
    plt.savefig(figures_dir / "gmm_metrics.png", dpi=150)
    plt.close()