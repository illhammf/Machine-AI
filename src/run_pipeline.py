"""Run the complete Stack Overflow AI tools ML pipeline from the terminal."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_preprocessing import (
    CANDIDATE_FEATURES,
    build_preprocessor,
    find_dataset_path,
    inspect_candidate_columns,
    load_survey_data,
    prepare_model_data,
)
from evaluate_models import (
    evaluate_classification_models,
    plot_cluster_distribution,
    plot_confusion_matrices,
    plot_gmm_metrics,
    plot_kmeans_metrics,
    plot_model_comparison,
    save_classification_outputs,
)
from train_models import (
    run_gmm_analysis,
    run_kmeans_analysis,
    summarize_clusters,
    train_classification_models,
)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_root = project_root / "outputs"
    figures_dir = output_root / "figures"
    tables_dir = output_root / "tables"
    reports_dir = output_root / "reports"

    for folder in [figures_dir, tables_dir, reports_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    data_path = find_dataset_path(project_root, year=2025)
    print(f"Dataset digunakan: {data_path}")

    df = load_survey_data(data_path)
    print(f"Ukuran dataset: {df.shape[0]:,} baris x {df.shape[1]:,} kolom")

    pd.DataFrame({"column": df.columns}).to_csv(
        tables_dir / "available_columns_2025.csv", index=False
    )
    inspect_candidate_columns(df, CANDIDATE_FEATURES).to_csv(
        tables_dir / "candidate_feature_availability_2025.csv", index=False
    )

    X, y, metadata = prepare_model_data(df)
    print(f"Jumlah data setelah target kosong dibuang: {len(X):,}")
    print(f"Fitur digunakan: {metadata['available_features']}")
    print(f"Fitur tidak tersedia: {metadata['missing_features']}")
    print(f"Distribusi target: {metadata['target_distribution']}")

    target_distribution = y.value_counts().sort_index().rename_axis("AI_Usage").reset_index(name="count")
    target_distribution["label"] = target_distribution["AI_Usage"].map(
        {0: "Tidak menggunakan AI", 1: "Menggunakan AI"}
    )
    target_distribution.to_csv(tables_dir / "target_distribution_2025.csv", index=False)

    preprocessor, feature_groups = build_preprocessor(X)
    pd.DataFrame(
        {"feature_group": list(feature_groups.keys()), "columns": list(feature_groups.values())}
    ).to_csv(tables_dir / "preprocessing_feature_groups.csv", index=False)

    # ---- Unsupervised: K-Means (Traditional) ----
    print("\n=== K-Means Clustering (Traditional) ===")
    kmeans_result = run_kmeans_analysis(X, preprocessor, k_values=range(2, 11))
    kmeans_metrics = kmeans_result["metrics"]
    best_k = kmeans_result["best_k"]
    cluster_labels = kmeans_result["labels"]

    kmeans_metrics.to_csv(tables_dir / "kmeans_elbow_silhouette.csv", index=False)
    plot_kmeans_metrics(kmeans_metrics, output_root)

    cluster_summary = summarize_clusters(X, cluster_labels)
    cluster_summary.to_csv(tables_dir / "kmeans_cluster_summary.csv", index=False)
    plot_cluster_distribution(cluster_summary, output_root, prefix="kmeans_")
    print(f"K-Means selesai. k terpilih: {best_k}")

    # ---- Unsupervised: GMM (Modern Probabilistic) ----
    print("\n=== GMM Clustering (Modern — Probabilistic) ===")
    gmm_result = run_gmm_analysis(X, preprocessor, k_values=range(2, 9))
    gmm_metrics = gmm_result["metrics"]
    gmm_best_k = gmm_result["best_k"]

    gmm_metrics.to_csv(tables_dir / "gmm_bic_silhouette.csv", index=False)
    plot_gmm_metrics(gmm_metrics, output_root)

    gmm_cluster_summary = summarize_clusters(X, gmm_result["labels"])
    gmm_cluster_summary.to_csv(tables_dir / "gmm_cluster_summary.csv", index=False)
    plot_cluster_distribution(gmm_cluster_summary, output_root, prefix="gmm_")
    print(f"GMM selesai. k terpilih: {gmm_best_k}")

    # ---- Supervised: Classification Models ----
    print("\n=== Melatih Model Klasifikasi ===")
    print("Model: Random Forest (Traditional), Linear SVM (Traditional), XGBoost (Modern), MLP (Modern)")
    models, split_data = train_classification_models(X, y, preprocessor, test_size=0.2)
    classification_results = evaluate_classification_models(
        models,
        split_data["X_test"],
        split_data["y_test"],
    )

    comparison_df = save_classification_outputs(classification_results, output_root)
    plot_confusion_matrices(classification_results, output_root)
    plot_model_comparison(comparison_df, output_root)

    best_model_row = comparison_df.sort_values("f1_score", ascending=False).iloc[0]
    summary_text = f"""# Ringkasan Awal BAB IV

## Deskripsi Dataset
- Dataset: Stack Overflow Developer Survey 2025.
- Jumlah baris awal: {df.shape[0]:,}.
- Jumlah kolom awal: {df.shape[1]:,}.
- Jumlah data setelah target kosong dibuang: {metadata['n_rows_after_target_drop']:,}.

## Target Prediksi
- Target: AI_Usage.
- Sumber target: {metadata['target_source_col']}.
- Mapping: 1 = menggunakan AI tools, 0 = tidak menggunakan AI tools.
- Distribusi target: {metadata['target_distribution']}.

## Fitur
- Fitur digunakan: {', '.join(metadata['available_features'])}.
- Fitur kandidat yang tidak tersedia: {', '.join(metadata['missing_features']) if metadata['missing_features'] else '-'}.
- Kolom AI tidak dipakai sebagai input untuk menghindari data leakage.

## Unsupervised Learning — Segmentasi Developer

### K-Means (Traditional — Hard Clustering)
- Rentang k diuji: 2 sampai 10.
- Nilai k terpilih berdasarkan silhouette score tertinggi: {best_k}.
- Tabel interpretasi cluster: outputs/tables/cluster_summary.csv.

### GMM (Modern — Probabilistic Soft Clustering)
- Rentang k diuji: 2 sampai 10.
- Nilai k terpilih berdasarkan silhouette score tertinggi: {gmm_best_k}.
- Tabel interpretasi cluster: outputs/tables/gmm_cluster_summary.csv.

## Supervised Learning — Klasifikasi AI Usage
Model dilatih: Random Forest (Traditional), Linear SVM (Traditional), XGBoost (Modern), MLP Neural Network (Modern).

### Evaluasi Model Klasifikasi
- Model terbaik berdasarkan F1-score: {best_model_row['model']}.
- Accuracy: {best_model_row['accuracy']:.4f}.
- Precision: {best_model_row['precision']:.4f}.
- Recall: {best_model_row['recall']:.4f}.
- F1-score: {best_model_row['f1_score']:.4f}.
- Tabel perbandingan model: outputs/tables/model_performance_comparison.csv.
"""

    (reports_dir / "bab_iv_ringkasan_awal.md").write_text(summary_text, encoding="utf-8")
    print(f"\n{comparison_df}")
    print(f"\nRingkasan BAB IV disimpan ke: {reports_dir / 'bab_iv_ringkasan_awal.md'}")


if __name__ == "__main__":
    main()
