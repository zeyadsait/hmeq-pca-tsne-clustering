# ==============================================================================
# 05_models_pca_tsne_data.R
#
# Week 7, Step 5 - Tree and Regression Analysis on the PCA/t-SNE Data
#
# Drops all original continuous predictors and refits the same tree +
# stepwise logistic regression using ONLY the dimensionality-reduction
# features (PC1, PC2, TSNE30, TSNE80, TSNE5) plus the categorical flag
# columns. This isolates how much predictive power PCA/t-SNE alone
# retain compared to the full feature set in Step 4.
#
# NOTE: run 01-03 first (or otherwise ensure `df` contains PC1, PC2 and
# the TSNE30/TSNE80/TSNE5 columns).
# ==============================================================================

library(rpart)
library(rpart.plot)
library(pROC)

df_step5 <- df
df_step5$TARGET_BAD_FLAG <- as.factor(df_step5$TARGET_BAD_FLAG)

# Drop every original continuous variable, keep only the
# dimensionality-reduction columns and the categorical flags.
continuous_cols <- names(df_step5)[sapply(df_step5, is.numeric)]
continuous_cols <- setdiff(continuous_cols,
                            c("PC1", "PC2",
                              "TSNE30_1", "TSNE30_2",
                              "TSNE80_1", "TSNE80_2",
                              "TSNE5_1", "TSNE5_2",
                              "TARGET_BAD_FLAG", "TARGET_LOSS_AMT"))

df_step5[continuous_cols] <- NULL
df_step5$TARGET_LOSS_AMT <- NULL

# ---- Decision tree ------------------------------------------------------
tree_pca_tsne <- rpart(TARGET_BAD_FLAG ~ ., data = df_step5, method = "class")
rpart.plot(tree_pca_tsne)

# ---- Stepwise logistic regression ---------------------------------------
df_step5$TARGET_BAD_FLAG <- as.numeric(as.character(df_step5$TARGET_BAD_FLAG))

log_full <- glm(TARGET_BAD_FLAG ~ ., data = df_step5, family = binomial)
log_step <- step(log_full, direction = "both")
summary(log_step)

# ---- ROC / AUC ------------------------------------------------------------
log_prob2 <- predict(log_step, type = "response")

roc_obj2 <- roc(df_step5$TARGET_BAD_FLAG, log_prob2)
plot(roc_obj2, col = "blue")
auc(roc_obj2)
# Result reported in the write-up: AUC = 0.578
# (much weaker than the 0.914 from Step 4 - PCA/t-SNE alone lose most of
#  the signal carried by the original financial variables.)
