"""
=========================================================
STACK OVERFLOW AI TOOLS ML PROJECT
FULL PIPELINE RUNNER
=========================================================

Menjalankan seluruh workflow:
1. Load dataset
2. Preprocessing
3. Clustering (KMeans + GMM)
4. Classification models
5. Evaluation + saving outputs
6. Generate BAB IV summary
"""

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

from train_models import (
    run_gmm_analysis,
    run_kmeans_analysis,
    summarize_clusters,
    train_classification_models,
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


# =========================================================
# MAIN PIPELINE
# =========================================================

def main():

    # -----------------------------------------------------
    # 1. SETUP PATH
    # -----------------------------------------------------
    project_root = Path(__file__).resolve().parents[1]

    output_root = project_root / "outputs"
    tables_dir = output_root / "tables"
    reports_dir = output_root / "reports"
    figures_dir = output_root / "figures"

    for d in [tables_dir, reports_dir, figures_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------
    # 2. LOAD DATASET
    # -----------------------------------------------------
    data_path = find_dataset_path(project_root, year=2025)
    print(f"[INFO] Dataset: {data_path}")

    df = load_survey_data(data_path)
    print(f"[INFO] Shape: {df.shape}")

    # save columns info
    pd.DataFrame(df.columns, columns=["column"]).to_csv(
        tables_dir / "columns_2025.csv", index=False
    )

    inspect_candidate_columns(df, CANDIDATE_FEATURES).to_csv(
        tables_dir / "feature_availability.csv",
        index=False
    )

    # -----------------------------------------------------
    # 3. PREPARE DATA
    # -----------------------------------------------------
    X, y, metadata = prepare_model_data(df)

    print("\n[INFO] Dataset ready")
    print(f"Features used: {len(metadata['available_features'])}")
    print(f"Missing features: {metadata['missing_features']}")

    # save target distribution
    y.value_counts().to_csv(tables_dir / "target_distribution.csv")

    # -----------------------------------------------------
    # 4. BUILD PREPROCESSOR
    # -----------------------------------------------------
    preprocessor = build_preprocessor(X)

    # -----------------------------------------------------
    # 5. UNSUPERVISED - KMEANS
    # -----------------------------------------------------
    print("\n[RUN] KMeans Clustering")

    kmeans_result = run_kmeans_analysis(X, preprocessor, k_values=range(2, 11))

    kmeans_metrics = kmeans_result["metrics"]
    kmeans_metrics.to_csv(tables_dir / "kmeans_metrics.csv", index=False)

    plot_kmeans_metrics(kmeans_metrics, output_root)

    kmeans_summary = summarize_clusters(X, kmeans_result["labels"])
    kmeans_summary.to_csv(tables_dir / "kmeans_summary.csv", index=False)

    plot_cluster_distribution(kmeans_summary, output_root)

    print(f"KMeans best k: {kmeans_result['best_k']}")

    # -----------------------------------------------------
    # 6. UNSUPERVISED - GMM
    # -----------------------------------------------------
    print("\n[RUN] GMM Clustering")

    gmm_result = run_gmm_analysis(X, preprocessor, k_values=range(2, 9))

    gmm_metrics = gmm_result["metrics"]
    gmm_metrics.to_csv(tables_dir / "gmm_metrics.csv", index=False)

    plot_gmm_metrics(gmm_metrics, output_root)

    gmm_summary = summarize_clusters(X, gmm_result["labels"])
    gmm_summary.to_csv(tables_dir / "gmm_summary.csv", index=False)

    plot_cluster_distribution(gmm_summary, output_root)

    print(f"GMM best k: {gmm_result['best_k']}")

    # -----------------------------------------------------
    # 7. SUPERVISED LEARNING
    # -----------------------------------------------------
    print("\n[RUN] Classification Models")

    models, split_data = train_classification_models(
        X, y, preprocessor, test_size=0.2
    )

    results = evaluate_classification_models(
        models,
        split_data["X_test"],
        split_data["y_test"]
    )

    comparison = save_classification_outputs(results, output_root)

    plot_confusion_matrices(results, output_root)
    plot_model_comparison(comparison, output_root)

    # -----------------------------------------------------
    # 8. BEST MODEL
    # -----------------------------------------------------
    best_model = comparison.sort_values("f1_score", ascending=False).iloc[0]

    print("\n[RESULT] BEST MODEL")
    print(best_model)

    # -----------------------------------------------------
    # 9. SAVE SUMMARY REPORT
    # -----------------------------------------------------
    summary_text = f"""
# BAB IV SUMMARY

Dataset:
- Rows: {df.shape[0]}
- Columns: {df.shape[1]}

Target:
- AI Usage distribution: {metadata['target_distribution']}

Best Models:
- Classification Best: {best_model['model']}
- F1-score: {best_model['f1_score']:.4f}

Clustering:
- KMeans best k: {kmeans_result['best_k']}
- GMM best k: {gmm_result['best_k']}
"""

    (reports_dir / "summary.md").write_text(summary_text, encoding="utf-8")

    print("\n[DONE] Pipeline finished successfully")


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()