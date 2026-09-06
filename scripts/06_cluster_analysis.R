# ==============================================================================
# 06_cluster_analysis.R
#
# Week 8 - PCA + K-Means Cluster Analysis
#
# Reduces four continuous HMEQ variables to principal components,
# k-means-clusters borrowers in that PCA space, and then uses a decision
# tree to explain what distinguishes each cluster in plain financial terms.
#
# Input: data/HMEQ_Processed.csv (same processed file as the Week 7 scripts)
# ==============================================================================

library(ggplot2)
library(cluster)
library(factoextra)
library(tidyverse)
library(readr)
library(rpart)
library(rpart.plot)

# ------------------------------------------------------------------------
# Step 1: Load data & libraries
# ------------------------------------------------------------------------
data_path <- "data/HMEQ_Processed.csv"
data <- read_csv(data_path)
data_clean <- na.omit(data)
numeric_data <- data_clean %>% select_if(is.numeric)

# ------------------------------------------------------------------------
# Step 2: PCA analysis
# ------------------------------------------------------------------------
# Four continuous variables sharing a common financial theme: mortgage
# amount, home value, credit age, and debt burden.
continuous_vars <- numeric_data %>%
  select(IMP_MORTDUE, IMP_VALUE, IMP_CLAGE, IMP_DEBTINC)

continuous_scaled <- scale(continuous_vars)
pca_result <- prcomp(continuous_scaled, center = TRUE, scale. = TRUE)

fviz_eig(pca_result, addlabels = TRUE, ylim = c(0, 100)) +
  ggtitle("Scree Plot of PCA")

print(pca_result$rotation)

fviz_pca_ind(pca_result,
             geom = "point",
             col.ind = "black",
             pointsize = 1,
             title = "PCA Scatter Plot (PC1 vs PC2)")

pca_scores <- as.data.frame(pca_result$x)
pc_data <- pca_scores[, 1:2]   # PC1 & PC2 used for clustering

# ------------------------------------------------------------------------
# Step 3: Find the number of clusters (elbow method)
# ------------------------------------------------------------------------
wss <- numeric(10)

for (k in 1:10) {
  set.seed(123)
  km <- kmeans(pc_data, centers = k, nstart = 25)
  wss[k] <- km$tot.withinss
}

plot(1:10, wss,
     type = "b",
     pch = 19,
     xlab = "Number of Clusters (k)",
     ylab = "Within-Cluster Sum of Squares (WSS)",
     main = "Elbow Plot for K-Means Clustering")

# The elbow flattens out around k = 3
k_opt <- 3

# ------------------------------------------------------------------------
# Step 4: Cluster analysis (final k-means fit)
# ------------------------------------------------------------------------
set.seed(123)
final_km <- kmeans(pc_data, centers = k_opt, nstart = 25)
final_km$size
final_km$centers

# as.kcca() wraps the kmeans result so predict() can be used on it
if (!requireNamespace("flexclust", quietly = TRUE)) install.packages("flexclust")
library(flexclust)
kcca_model <- as.kcca(final_km, data = pc_data)
cluster_membership <- predict(kcca_model)

barplot(final_km$size,
        main = "Cluster Sizes",
        xlab = "Cluster",
        ylab = "Number of Records",
        col = "skyblue")

pca_scores$cluster <- factor(cluster_membership)

plot(pca_scores$PC1, pca_scores$PC2,
     col = pca_scores$cluster,
     pch = 19,
     main = "PC1 vs PC2 Colored by Cluster",
     xlab = "PC1",
     ylab = "PC2")

legend("topright",
       legend = levels(pca_scores$cluster),
       col = 1:k_opt,
       pch = 19,
       title = "Cluster")

# ------------------------------------------------------------------------
# Step 5: Describe the clusters using a decision tree
# ------------------------------------------------------------------------
tree_data <- continuous_vars
tree_data$cluster <- pca_scores$cluster

dt_model <- rpart(cluster ~ ., data = tree_data, method = "class")

rpart.plot(dt_model,
           main = "Decision Tree Explaining Cluster Membership",
           cex = 0.8)
