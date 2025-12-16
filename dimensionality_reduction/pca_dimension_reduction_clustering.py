import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from util.filter_data import filtering

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

df=filtering('./src/claims_train.csv')

df_clean = df.drop(columns=["IDpol","ClaimNb", "Exposure","Region","Risk"])
numeric_cols = df_clean.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = df_clean.select_dtypes(include=["object", "category"]).columns

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean[numeric_cols])

pca = PCA(n_components=0.9, svd_solver='full')
X_pca = pca.fit_transform(X_scaled)

df_pca= pd.DataFrame(X_pca)

df_pca.to_csv("./src/pca_data.csv", index=False)

#print("Original shape:", X_scaled.shape)
#print("Reduced shape:", X_pca.shape)

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Scree Plot")
plt.grid(True)
plt.savefig("./plots/pca_scree_plot.png", dpi=300, bbox_inches='tight')
plt.close()

X_cat = pd.get_dummies(df_clean[categorical_cols], drop_first=True)
X_final = np.hstack([X_pca, X_cat.values])
X_final.shape

# scores = []
# for k in range(2, 10):
#     km = KMeans(n_clusters=k, random_state=42,)
#     labels = km.fit_predict(X_final)
#     score = silhouette_score(X_final, labels)
#     scores.append((k, score))

# best_k = max(scores, key=lambda x: x[1])[0]
# print("Best k by silhouette score:", best_k)

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_final)

df["Cluster"] = clusters

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df["Risk"], cmap="magma", alpha=0.6)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Node Risk in PCA-reduced Space")
plt.colorbar(label="Risk Value")
plt.grid(True)
plt.savefig("./plots/node_risk_in_pca_reduced_space.png", dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df["Cluster"], cmap="tab10", alpha=0.6)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Clusters in PCA-reduced Space")
plt.grid(True)
plt.savefig("./plots/clusters_in_pca_reduced_space.png", dpi=300, bbox_inches='tight')
plt.close()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter( X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=df["Cluster"], cmap="tab10", alpha=0.6)

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title("Clusters in 3D PCA space")

plt.legend(*scatter.legend_elements(), title="Cluster")
plt.savefig("./plots/clusters_in_3d_pca_space.png", dpi=300, bbox_inches='tight')
plt.close()

clusters = sorted(df["Cluster"].unique())
colors = ["blue", "brown", "cyan"]

plt.figure(figsize=(15, 4))
for i, cluster in enumerate(clusters):
    plt.subplot(1, len(clusters), i + 1)
    sns.histplot(
        df[df["Cluster"] == cluster]["Risk"],
        bins=30,
        kde=True,
        color=colors[i]
    )
    plt.title(f"Cluster {cluster} Risk Distribution")
    plt.xlabel("Risk_formula")
    plt.ylabel("Count (log scale)")
    #plt.yscale("log")
    plt.grid(True)

plt.tight_layout()
plt.savefig("./plots/risk_distribution.png", dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(12,7))

clusters = sorted(df["Cluster"].unique())

for c in clusters:
    sns.histplot(
        df.loc[df["Cluster"] == c, "Risk"],
        bins=40,
        kde=False,
        stat="count",
        label=f"Cluster {c}",
        element="step",
        fill=False,
        alpha=0.7
    )

#plt.yscale("log")
plt.xlabel("Risk")
plt.ylabel("Count (log scale)")
plt.title("Risk Distribution per Cluster (Histogram + Log Scale)")
plt.legend()
plt.tight_layout()
plt.savefig("./plots/risk_distribution_line.png", dpi=300, bbox_inches='tight')
plt.close()