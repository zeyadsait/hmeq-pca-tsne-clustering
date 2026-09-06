# Loan Default Prediction — PCA, t-SNE, and Cluster Analysis (HMEQ Dataset)

Dimensionality-reduction and clustering case study on the HMEQ home equity
loan dataset, comparing how well Decision Trees, Random Forests, GBM, and
Logistic Regression predict **loan default (`TARGET_BAD_FLAG`)** and
**loss amount (`TARGET_LOSS_AMT`)** when built on the original variables
versus on PCA/t-SNE–reduced features, and using PCA + K-Means to segment
borrowers into interpretable risk groups.

This repository packages two related course assignments into a single,
reproducible R project:

| | Focus | Scripts |
|---|---|---|
| **Part 1** | PCA & t-SNE for dimensionality reduction and predictive modeling | `scripts/01`–`05` |
| **Part 2** | PCA + K-Means clustering to segment borrowers | `scripts/06` |

## Reports

- **Live report:** https://zeyadsait.github.io/hmeq-pca-tsne-clustering/ — this README rendered as a standalone page with all figures embedded (served via GitHub Pages from [`index.html`](index.html)).
- [`report.Rmd`](report.Rmd) — the reproducible R Markdown source. Knit it yourself (open in RStudio and click "Knit", or run `rmarkdown::render("report.Rmd")`) any time you update the data or scripts; it's configured to regenerate `index.html` directly.

## Repository structure

```
.
├── README.md
├── LICENSE
├── .gitignore
├── report.Rmd                # R Markdown source -> knits to index.html
├── index.html                # pre-built HTML report, served live via GitHub Pages
├── data/
│   └── README.md            # where to put the HMEQ CSV (not included)
├── scripts/
│   ├── run_all.R             # runs 01-05 end to end
│   ├── 01_baseline_models.R  # Decision Tree / Random Forest / GBM, classification + regression
│   ├── 02_pca_analysis.R     # PCA on continuous predictors
│   ├── 03_tsne_analysis.R    # t-SNE at perplexity 5 / 30 / 80
│   ├── 04_models_original_data.R      # Tree + stepwise logistic regression, full feature set
│   ├── 05_models_pca_tsne_data.R      # Same models, PCA/t-SNE features only
│   └── 06_cluster_analysis.R # PCA + K-Means clustering + cluster-explaining tree
└── figures/                  # exported plots referenced below
```

## Data

A small synthetic dataset with the right schema is included at
`data/sample_HMEQ_Processed.csv` so the pipeline runs out of the box —
see `data/README.md` to use it (`cp` it to `data/HMEQ_Processed.csv`).
Its values are randomly generated and carry no real signal; swap in the
real HMEQ data for meaningful results.

The scripts expect a processed HMEQ file at `data/HMEQ_Processed.csv` —
the output of a prior cleaning step (imputed missing values with paired
`IMP_*` / `M_*` missing-flag columns, and one-hot encoded `JOB` / `REASON`
into `FLAG.Job.*` / `FLAG.Reason.*` columns). The original assignment
loaded this interactively with `file.choose()`; the scripts here use a
fixed relative path instead so the whole pipeline can be run
non-interactively. See `data/README.md` for details and swap in the raw
[HMEQ dataset](https://www.kaggle.com/datasets/ajay1735/hmeq-data) plus
your own imputation step if you don't already have a processed copy.

## Requirements

R (≥ 4.0) and the following packages:

```r
install.packages(c(
  "rpart", "rpart.plot", "randomForest", "gbm",
  "ggplot2", "Rtsne", "pROC",
  "dplyr", "tidyverse", "readr",
  "cluster", "factoextra", "flexclust"
))
```

## Running

```bash
# Part 1: baseline models -> PCA -> t-SNE -> modeling comparison
Rscript scripts/run_all.R

# Part 2: PCA + K-Means clustering (independent, reloads its own data)
Rscript scripts/06_cluster_analysis.R
```

Each script can also be run/sourced individually in an interactive R
session; scripts `02`–`05` assume `df` (with the columns added by the
previous step) is already in the environment.

---

## Part 1 — PCA & t-SNE for Predictive Modeling

### Step 1: Baseline models on the original data

Decision Tree, Random Forest, and GBM models were fit for both the
classification target (`TARGET_BAD_FLAG`) and the regression target
(`TARGET_LOSS_AMT`, restricted to defaulted loans).

| | |
|---|---|
| ![Decision Tree - Classification](figures/01_decision_tree_classification.png) | ![RF Variable Importance - Classification](figures/02_rf_varimp_classification.png) |
| ![GBM Variable Importance - Classification](figures/03_gbm_varimp_classification.png) | ![Regression Tree - Loss Amount](figures/04_regression_tree_loss.png) |
| ![RF Variable Importance - Loss](figures/05_rf_varimp_loss.png) | ![GBM Variable Importance - Loss](figures/06_gbm_varimp_loss.png) |

Across all three classification models, debt-to-income ratio
(`M_DEBTINC` / `IMP_DEBTINC`), delinquency history (`IMP_DELINQ`), and
credit age (`IMP_CLAGE`) consistently ranked as the strongest predictors
of default. For the loss-amount regression, `LOAN` (the loan amount
itself) dominated variable importance, followed by number of credit
lines and debt-to-income ratio.

### Step 2: PCA Analysis

PCA was applied to the continuous, non-flag predictors (after removing
the two target columns).

| Scree Plot | PC1 vs PC2 by Default Status |
|---|---|
| ![Scree Plot](figures/07_pca_scree_plot.png) | ![PC1 vs PC2](figures/08_pca_pc1_pc2_by_default.png) |

Variance explained drops off gradually rather than showing one or two
dominant components, meaning the underlying financial variables are only
moderately correlated. In the PC1–PC2 plane, defaulted and non-defaulted
borrowers overlap heavily, with only a loose tendency for defaults to
sit toward higher PC1 values — PCA alone does not cleanly separate the
two classes.

### Step 3: t-SNE Analysis

t-SNE was run at three perplexity settings on the same continuous
variables to check whether a nonlinear embedding reveals structure that
PCA misses.

| Perplexity = 5 | Perplexity = 30 | Perplexity = 80 |
|---|---|---|
| ![t-SNE perplexity 5](figures/09_tsne_perplexity5.png) | ![t-SNE perplexity 30](figures/11_tsne_perplexity30.png) | ![t-SNE perplexity 80](figures/10_tsne_perplexity80.png) |

At every perplexity, default and non-default points remain intermixed
rather than forming separate clusters. Lower perplexity (5) produces
many small, fragmented local groupings; higher perplexity (80) produces
a smoother, more global layout — but neither setting yields a boundary
that visually separates the two loan-status classes. A quick Random
Forest check confirmed the t-SNE coordinates are themselves predictable
from the original variables (i.e., they encode real structure, just not
one that aligns with the default label).

### Step 4: Modeling on the Original Data (+ PCA/t-SNE columns)

A classification tree and a stepwise logistic regression were fit using
the full feature set — original variables plus the PCA and t-SNE columns
generated in Steps 2–3.

| Decision Tree | ROC Curve |
|---|---|
| ![Decision Tree - Full Features](figures/12_decision_tree_full_features.png) | ![ROC Curve - Original Data](figures/13_roc_curve_original_data_auc0.914.png) |

The tree split primarily on `M_DEBTINC`, `IMP_DEBTINC`, credit-age, and
mortgage-related missing-value flags, plus PC1/PC2 in some minor
branches. The stepwise logistic regression's ROC curve gave an
**AUC = 0.914**, indicating strong discriminative power between default
and non-default borrowers.

### Step 5: Modeling on PCA/t-SNE Features Only

The same model types were refit after dropping every original continuous
variable, keeping only PC1, PC2, the three t-SNE embeddings, and the
categorical flag columns.

![ROC Curve - PCA/t-SNE Data](figures/14_roc_curve_pca_tsne_data_auc0.578.png)

Performance collapsed to **AUC = 0.578** — barely better than chance.
This is the central finding of Part 1: PCA and t-SNE compress the
continuous variables into a low-dimensional summary optimized for
*visualizing overall structure*, not for preserving the specific signal
that separates defaulters from non-defaulters. When the original
financial variables are removed entirely, most of the predictive
information is lost.

### Conclusion — Part 1

Dimensionality reduction is valuable here for exploration and
visualization, but not as a substitute for the original features in a
predictive model. The best-performing model by far used the full,
original feature set (AUC 0.914); reducing to two PCA dimensions and a
handful of t-SNE coordinates discarded the specific debt, delinquency,
and credit-history signals that actually drive default risk.

---

## Part 2 — PCA + K-Means Cluster Analysis

Four continuous variables sharing a common financial theme — mortgage
amount, home value, credit age, and debt-to-income ratio
(`IMP_MORTDUE`, `IMP_VALUE`, `IMP_CLAGE`, `IMP_DEBTINC`) — were reduced
via PCA and then clustered with K-Means to identify natural borrower
segments.

| Scree Plot | PC1 vs PC2 |
|---|---|
| ![Week 8 Scree Plot](figures/15_week8_pca_scree_plot.png) | ![Week 8 PCA Scatter](figures/16_week8_pca_pc1_pc2_scatter.png) |

The first two components explain about 73% of the variance (46.8% +
26.1%), a reasonable basis for clustering in 2D.

### Choosing k and fitting K-Means

| Elbow Plot | Cluster Sizes |
|---|---|
| ![Elbow Plot](figures/17_kmeans_elbow_plot.png) | ![Cluster Sizes](figures/18_cluster_sizes_barplot.png) |

The within-cluster sum of squares flattens out around **k = 3**. The
resulting clusters are uneven in size: Cluster 1 (~3,240 records) is the
largest and most typical group, Cluster 2 (~1,681 records) is a
moderate-size group, and Cluster 3 (~1,039 records) is the smallest and
most distinct.

![PC1 vs PC2 by Cluster](figures/19_pc1_pc2_by_cluster.png)

### Describing the clusters with a decision tree

A decision tree fit on the four original variables (predicting cluster
membership) makes the clusters interpretable in plain financial terms:

![Decision Tree Explaining Cluster Membership](figures/20_decision_tree_cluster_membership.png)

- **Cluster 1 — Higher-risk, high debt burden.** Lower mortgage amounts,
  higher debt-to-income ratios (`IMP_DEBTINC ≥ 31`), and shorter credit
  history. Financially more strained borrowers.
- **Cluster 2 — Moderate risk, average profile.** Moderate debt-to-income
  (below ~36), mid-range credit age, and modest mortgage values. A
  middle-risk group with more stable finances than Cluster 1.
- **Cluster 3 — Lower risk, strong financial profile.** High mortgage
  values, low debt-to-income ratios, and long credit history. The most
  financially secure segment.

### Conclusion — Part 2

The three clusters align with intuitive borrower-risk tiers, and the
splits used to describe them (debt-to-income ratio, mortgage value,
credit age) match well-known drivers of credit risk. In a corporate
lending setting, Cluster 1 could be flagged for closer monitoring or
stricter terms, Cluster 2 treated as a standard-risk segment, and
Cluster 3 targeted for premium products or higher credit limits.

---

## References

- Hlinka, J., Berk, M., Brosch, K., Demro, C., Fortea, L., Malt, U., & Meinert, S. (2024). Principal component analysis as an efficient method for capturing multivariate brain signatures of complex disorders — ENIGMA study in people with bipolar disorders and obesity. *Human Brain Mapping*, 45(8), 1–16. https://doi.org/10.1002/hbm.26682
- Junthopas, W., & Wongoutong, C. (2025). Pre-determining the optimal number of clusters for k-means clustering using the parameters package in R and distance metrics. *Applied Sciences*, 15(21), 11372. https://doi.org/10.3390/app152111372
- Ovchinnikova, S., & Anders, S. (2024). Simple but powerful interactive data analysis in R with R/LinkedCharts. *Genome Biology*, 25(1), 1–17. https://doi.org/10.1186/s13059-024-03164-3
- Zhang, W., Ge, Y., Liu, G., Qi, W., Xu, S., Peng, Z., & Li, Y. (2023). Clustering and decision tree-based analysis of typical operation modes of power systems. *Energy Reports*, 9, 60–69. https://doi.org/10.1016/j.egyr.2023.04.258

## License

MIT — see [LICENSE](LICENSE).
