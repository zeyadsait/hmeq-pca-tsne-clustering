# ==============================================================================
# 04_models_original_data.R
#
# Week 7, Step 4 - Tree and Regression Analysis on the Original Data
#
# Fits a classification tree and a stepwise logistic regression using the
# ORIGINAL variables plus the PCA components and t-SNE embeddings added in
# steps 2-3, then scores the logistic model with ROC/AUC.
#
# NOTE: run 01-03 first (or otherwise ensure `df` contains PC1, PC2 and
# the TSNE30/TSNE80/TSNE5 columns in addition to the original predictors).
# ==============================================================================

library(rpart)
library(rpart.plot)
library(pROC)

df_model <- df
df_model$TARGET_BAD_FLAG <- as.factor(df_model$TARGET_BAD_FLAG)
df_model <- subset(df_model, select = -c(TARGET_LOSS_AMT))

# ---- Decision tree ------------------------------------------------------
tree_model <- rpart(TARGET_BAD_FLAG ~ ., data = df_model, method = "class")
rpart.plot(tree_model)
tree_model$variable.importance

# ---- Stepwise logistic regression ---------------------------------------
df_model$TARGET_BAD_FLAG <- as.numeric(as.character(df_model$TARGET_BAD_FLAG))

full_log_model <- glm(TARGET_BAD_FLAG ~ ., data = df_model, family = binomial)
step_log_model <- step(full_log_model, direction = "both")
summary(step_log_model)

# ---- ROC / AUC ------------------------------------------------------------
log_prob <- predict(step_log_model, type = "response")

roc_obj <- roc(df_model$TARGET_BAD_FLAG, log_prob)
plot(roc_obj, col = "blue")
auc(roc_obj)
# Result reported in the write-up: AUC = 0.914
