# ==============================================================================
# 01_baseline_models.R
#
# Week 7, Step 1 - Baseline classification & regression models
#
# Carries the Decision Tree / Random Forest / GBM code forward from the
# Week 6 assignment as a starting point for the PCA / t-SNE comparison.
#
# Input : data/HMEQ_Processed.csv
#         (the cleaned/imputed HMEQ file produced in Week 6: missing values
#          imputed with IMP_ / M_ flag columns, and job/reason categories
#          one-hot encoded into FLAG.Job.* / FLAG.Reason.* columns)
#
# Targets:
#   TARGET_BAD_FLAG  - binary indicator of loan default (classification)
#   TARGET_LOSS_AMT  - dollar loss amount, populated only when a loan
#                      defaulted (regression, subset to bad loans)
# ==============================================================================

library(rpart)
library(rpart.plot)
library(randomForest)
library(gbm)

data_path <- "data/HMEQ_Processed.csv"
df <- read.csv(data_path, header = TRUE)

# ------------------------------------------------------------------------
# Classification: TARGET_BAD_FLAG
# ------------------------------------------------------------------------

tree_clf <- rpart(TARGET_BAD_FLAG ~ . - TARGET_LOSS_AMT,
                   data = df,
                   method = "class")

rpart.plot(tree_clf, main = "Decision Tree - Classification")

rf_clf <- randomForest(as.factor(TARGET_BAD_FLAG) ~ . - TARGET_LOSS_AMT,
                        data = df,
                        ntree = 300,
                        importance = TRUE)

varImpPlot(rf_clf, main = "Random Forest Variable Importance (Classification)")

gbm_clf <- gbm(TARGET_BAD_FLAG ~ . - TARGET_LOSS_AMT,
               data = df,
               distribution = "bernoulli",
               n.trees = 300,
               interaction.depth = 3,
               shrinkage = 0.05,
               verbose = FALSE)

summary(gbm_clf)   # relative influence bar chart

# ------------------------------------------------------------------------
# Regression: TARGET_LOSS_AMT (defaulted loans only)
# ------------------------------------------------------------------------

loss_df <- df[df$TARGET_BAD_FLAG == 1, ]

tree_reg <- rpart(TARGET_LOSS_AMT ~ . - TARGET_BAD_FLAG,
                   data = loss_df)

rpart.plot(tree_reg, main = "Regression Tree - Loss Amount")

rf_reg <- randomForest(TARGET_LOSS_AMT ~ . - TARGET_BAD_FLAG,
                        data = loss_df,
                        ntree = 300,
                        importance = TRUE)

varImpPlot(rf_reg, main = "Random Forest Variable Importance (Loss)")

gbm_reg <- gbm(TARGET_LOSS_AMT ~ . - TARGET_BAD_FLAG,
               data = loss_df,
               distribution = "gaussian",
               n.trees = 300,
               interaction.depth = 3,
               shrinkage = 0.05,
               verbose = FALSE)

summary(gbm_reg)   # relative influence bar chart

lm_reg <- lm(TARGET_LOSS_AMT ~ . - TARGET_BAD_FLAG, data = loss_df)
summary(lm_reg)
