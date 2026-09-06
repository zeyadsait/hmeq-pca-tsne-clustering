# ==============================================================================
# run_all.R
#
# Runs the full Week 7 (PCA + t-SNE) pipeline end-to-end, in order.
# Week 8 (06_cluster_analysis.R) is independent and reloads the data
# itself, so it can be sourced separately.
#
# Usage:
#   Rscript scripts/run_all.R
# ==============================================================================

message("Step 1/5: baseline models ...")
source("scripts/01_baseline_models.R")

message("Step 2/5: PCA analysis ...")
source("scripts/02_pca_analysis.R")

message("Step 3/5: t-SNE analysis ...")
source("scripts/03_tsne_analysis.R")

message("Step 4/5: models on original + PCA/t-SNE data ...")
source("scripts/04_models_original_data.R")

message("Step 5/5: models on PCA/t-SNE-only data ...")
source("scripts/05_models_pca_tsne_data.R")

message("Done. Run scripts/06_cluster_analysis.R separately for the ",
        "Week 8 clustering analysis.")
