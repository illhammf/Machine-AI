"""Evaluation and output-saving helpers for the Stack Overflow ML project."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_classification_models(models: dict, X_test, y_test) -> dict:
    """Evaluate all fitted classifiers with standard binary metrics."""
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
                zero_division=0,
            ),
            "classification_report_dict": classification_report(
                y_test,
                y_pred,
                target_names=["Tidak Menggunakan AI", "Menggunakan AI"],
                zero_division=0,
                output_dict=True,
            ),
        }
    return results


def make_comparison_table(results: dict) -> pd.DataFrame:
    """Create a compact performance comparison table."""
    return pd.DataFrame([item["metrics"] for item in results.values()]).sort_values(
        "f1_score", ascending=False
    )


def save_classification_outputs(results: dict, output_root: str | Path) -> pd.DataFrame:
    """Save classification metrics, confusion matrices, and reports."""
    output_root = Path(output_root)
    tables_dir = output_root / "tables"
    reports_dir = output_root / "reports"
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    comparison = make_comparison_table(results)
    comparison.to_csv(tables_dir / "model_performance_comparison.csv", index=False)

    for model_name, result in results.items():
        safe_name = model_name.lower().replace(" ", "_")
        cm = pd.DataFrame(
            result["confusion_matrix"],
            index=["Actual_0", "Actual_1"],
            columns=["Predicted_0", "Predicted_1"],
        )
        cm.to_csv(tables_dir / f"confusion_matrix_{safe_name}.csv")

        report_df = pd.DataFrame(result["classification_report_dict"]).transpose()
        report_df.to_csv(tables_dir / f"classification_report_{safe_name}.csv")

        with open(reports_dir / f"classification_report_{safe_name}.txt", "w", encoding="utf-8") as file:
            file.write(result["classification_report_text"])

    return comparison


def plot_confusion_matrices(results: dict, output_root: str | Path) -> None:
    """Save confusion matrix heatmaps for every classifier."""
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
            xticklabels=["Prediksi 0", "Prediksi 1"],
            yticklabels=["Aktual 0", "Aktual 1"],
        )
        plt.title(f"Confusion Matrix - {model_name}")
        plt.tight_layout()
        plt.savefig(figures_dir / f"confusion_matrix_{safe_name}.png", dpi=150)
        plt.close()


def plot_model_comparison(comparison_df: pd.DataFrame, output_root: str | Path) -> None:
    """Save a bar chart comparing classifier metrics."""
    figures_dir = Path(output_root) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_df = comparison_df.melt(
        id_vars="model",
        value_vars=["accuracy", "precision", "recall", "f1_score"],
        var_name="metric",
        value_name="score",
    )

    plt.figure(figsize=(8, 5))
    sns.barplot(data=plot_df, x="metric", y="score", hue="model")
    plt.ylim(0, 1)
    plt.title("Perbandingan Performa Model Klasifikasi")
    plt.tight_layout()
    plt.savefig(figures_dir / "model_performance_comparison.png", dpi=150)
    plt.close()


def plot_kmeans_metrics(metrics_df: pd.DataFrame, output_root: str | Path) -> None:
    """Save elbow and silhouette plots."""
    figures_dir = Path(output_root) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.lineplot(data=metrics_df, x="k", y="inertia", marker="o", ax=axes[0])
    axes[0].set_title("Elbow Method")
    axes[0].set_xlabel("Jumlah Cluster (k)")
    axes[0].set_ylabel("Inertia")

    sns.lineplot(data=metrics_df, x="k", y="silhouette_score", marker="o", ax=axes[1])
    axes[1].set_title("Silhouette Score")
    axes[1].set_xlabel("Jumlah Cluster (k)")
    axes[1].set_ylabel("Silhouette Score")

    plt.tight_layout()
    plt.savefig(figures_dir / "kmeans_elbow_silhouette.png", dpi=150)
    plt.close()


def plot_cluster_distribution(
    cluster_summary: pd.DataFrame, output_root: str | Path, prefix: str = ""
) -> None:
    """Save cluster-size distribution chart."""
    figures_dir = Path(output_root) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{prefix}cluster_distribution.png" if prefix else "cluster_distribution.png"

    plt.figure(figsize=(7, 4))
    sns.barplot(data=cluster_summary, x="Cluster", y="Jumlah_Data")
    plt.title("Distribusi Data per Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Jumlah Data")
    plt.tight_layout()
    plt.savefig(figures_dir / filename, dpi=150)
    plt.close()


def plot_gmm_metrics(metrics_df: pd.DataFrame, output_root: str | Path) -> None:
    """Save GMM BIC/AIC and silhouette plots."""
    figures_dir = Path(output_root) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.lineplot(data=metrics_df, x="k", y="bic", marker="o", ax=axes[0], label="BIC")
    sns.lineplot(data=metrics_df, x="k", y="aic", marker="o", ax=axes[0], label="AIC")
    axes[0].set_title("GMM: BIC / AIC")
    axes[0].set_xlabel("Jumlah Komponen (k)")
    axes[0].set_ylabel("Nilai")

    sns.lineplot(data=metrics_df, x="k", y="silhouette_score", marker="o", ax=axes[1])
    axes[1].set_title("GMM: Silhouette Score")
    axes[1].set_xlabel("Jumlah Komponen (k)")
    axes[1].set_ylabel("Silhouette Score")

    plt.tight_layout()
    plt.savefig(figures_dir / "gmm_bic_silhouette.png", dpi=150)
    plt.close()
