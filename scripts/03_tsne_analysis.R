# ==============================================================================
# 03_tsne_analysis.R
#
# Week 7, Step 3 - t-SNE Analysis
#
# Runs t-SNE at three perplexity settings (30, 80, 5) on the same
# continuous variables used for PCA, to see whether a nonlinear
# embedding separates defaulted from non-defaulted borrowers more
# clearly than PCA does.
#
# NOTE: run 02_pca_analysis.R first (or otherwise ensure `df` and
# `continuous_vars` exist).
# ==============================================================================

library(ggplot2)
library(randomForest)
library(Rtsne)

set.seed(123)

df_inputs <- subset(df, select = -c(TARGET_BAD_FLAG, TARGET_LOSS_AMT))
df_inputs <- df_inputs[, !grepl("^FLAG", names(df_inputs))]
continuous_vars <- df_inputs[, sapply(df_inputs, is.numeric)]

# ---- Perplexity = 30 --------------------------------------------------
tsne_30 <- Rtsne(continuous_vars, dims = 2, perplexity = 30,
                  verbose = TRUE, max_iter = 1000)
df$TSNE30_1 <- tsne_30$Y[, 1]
df$TSNE30_2 <- tsne_30$Y[, 2]

ggplot(df, aes(x = TSNE30_1, y = TSNE30_2, color = as.factor(TARGET_BAD_FLAG))) +
  geom_point(alpha = 0.6) +
  scale_color_manual(values = c("blue", "red"),
                      labels = c("Non-Default", "Default"), name = "Loan Status") +
  ggtitle("t-SNE (Perplexity = 30)") +
  theme_minimal()

# ---- Perplexity = 80 (less local, more global structure) --------------
tsne_high <- Rtsne(continuous_vars, dims = 2, perplexity = 80,
                    verbose = TRUE, max_iter = 1000)
df$TSNE80_1 <- tsne_high$Y[, 1]
df$TSNE80_2 <- tsne_high$Y[, 2]

ggplot(df, aes(x = TSNE80_1, y = TSNE80_2, color = as.factor(TARGET_BAD_FLAG))) +
  geom_point(alpha = 0.6) +
  scale_color_manual(values = c("blue", "red"),
                      labels = c("Non-Default", "Default"), name = "Loan Status") +
  ggtitle("t-SNE (Perplexity = 80)") +
  theme_minimal()

# ---- Perplexity = 5 (very local structure) -----------------------------
tsne_low <- Rtsne(continuous_vars, dims = 2, perplexity = 5,
                   verbose = TRUE, max_iter = 1000)
df$TSNE5_1 <- tsne_low$Y[, 1]
df$TSNE5_2 <- tsne_low$Y[, 2]

ggplot(df, aes(x = TSNE5_1, y = TSNE5_2, color = as.factor(TARGET_BAD_FLAG))) +
  geom_point(alpha = 0.6) +
  scale_color_manual(values = c("blue", "red"),
                      labels = c("Non-Default", "Default"), name = "Loan Status") +
  ggtitle("t-SNE (Perplexity = 5)") +
  theme_minimal()

# ---- Sanity check: can the t-SNE 30 coordinates be predicted from the --
# ---- original continuous variables? (confirms they encode real signal) -
rf_data <- data.frame(continuous_vars,
                       TSNE30_1 = df$TSNE30_1,
                       TSNE30_2 = df$TSNE30_2)

rf_tsne1 <- randomForest(TSNE30_1 ~ ., data = rf_data, importance = TRUE)
rf_tsne2 <- randomForest(TSNE30_2 ~ ., data = rf_data, importance = TRUE)

print(rf_tsne1)
print(rf_tsne2)
