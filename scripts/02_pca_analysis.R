# ==============================================================================
# 02_pca_analysis.R
#
# Week 7, Step 2 - Principal Component Analysis
#
# Reduces the continuous input variables to principal components and
# checks whether the top two components separate defaulted from
# non-defaulted loans.
#
# NOTE: run 01_baseline_models.R first (or otherwise load `df`) so that
# TARGET_BAD_FLAG / TARGET_LOSS_AMT and the raw predictor columns exist.
# ==============================================================================

library(ggplot2)

# Keep only continuous, non-flag, non-target predictors
df_inputs <- subset(df, select = -c(TARGET_BAD_FLAG, TARGET_LOSS_AMT))
df_inputs <- df_inputs[, !grepl("^FLAG", names(df_inputs))]
continuous_vars <- df_inputs[, sapply(df_inputs, is.numeric)]

pca_model <- prcomp(continuous_vars, scale. = TRUE)

# Figure 1: Scree plot
variance_explained <- pca_model$sdev^2 / sum(pca_model$sdev^2)
plot(variance_explained, type = "b",
     main = "Scree Plot of PCA",
     xlab = "Principal Component",
     ylab = "Variance Explained")

print(pca_model$rotation)   # loadings

df$PC1 <- pca_model$x[, 1]
df$PC2 <- pca_model$x[, 2]

# Figure 2: PC1 vs PC2 colored by default status
ggplot(df, aes(x = PC1, y = PC2, color = as.factor(TARGET_BAD_FLAG))) +
  geom_point(alpha = 0.6) +
  scale_color_manual(values = c("blue", "red"),
                      labels = c("Non-Default", "Default"),
                      name = "Loan Status") +
  ggtitle("PC1 vs PC2 Scatter Plot Colored by Default Flag") +
  theme_minimal()
