import json
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    classification_report,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    normalized_mutual_info_score,
    precision_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, normalize
from sklearn.tree import DecisionTreeClassifier, export_text
from mlxtend.frequent_patterns import association_rules, fpgrowth

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#2c3e50"
plt.rcParams["axes.linewidth"] = 0.8


def analyze_similarity_and_distances(df_clean, k=10, random_state=42):
    features = [c for c in df_clean.columns if c != "Class"]
    X_raw = df_clean[features].values
    y = df_clean["Class"].values

    scaler_std = StandardScaler()
    X_std = scaler_std.fit_transform(X_raw)

    scaler_minmax = MinMaxScaler()
    X_minmax = scaler_minmax.fit_transform(X_raw)

    scaler_robust = RobustScaler()
    X_robust = scaler_robust.fit_transform(X_raw)

    nn_l2 = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(X_std)
    _, idx_l2 = nn_l2.kneighbors(X_std)

    nn_l1 = NearestNeighbors(n_neighbors=k + 1, metric="manhattan").fit(X_std)
    _, idx_l1 = nn_l1.kneighbors(X_std)

    X_unit = normalize(X_std, norm="l2")
    nn_cos = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(X_unit)
    _, idx_cos = nn_cos.kneighbors(X_unit)

    nn_raw = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(X_raw)
    _, idx_raw = nn_raw.kneighbors(X_raw)

    nn_mm = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(X_minmax)
    _, idx_mm = nn_mm.kneighbors(X_minmax)

    nn_rob = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(X_robust)
    _, idx_rob = nn_rob.kneighbors(X_robust)

    idx_l2_k = idx_l2[:, 1:]
    idx_l1_k = idx_l1[:, 1:]
    idx_cos_k = idx_cos[:, 1:]
    idx_raw_k = idx_raw[:, 1:]
    idx_mm_k = idx_mm[:, 1:]
    idx_rob_k = idx_rob[:, 1:]

    def calc_jaccard(a_mat, b_mat):
        n = a_mat.shape[0]
        jaccards = np.empty(n)
        for i in range(n):
            sa = set(a_mat[i])
            sb = set(b_mat[i])
            jaccards[i] = len(sa.intersection(sb)) / len(sa.union(sb))
        return float(np.mean(jaccards))

    def calc_purity(idx_mat):
        n = idx_mat.shape[0]
        purities = np.empty(n)
        for i in range(n):
            c_target = y[i]
            nbr_classes = y[idx_mat[i]]
            purities[i] = np.mean(nbr_classes == c_target)
        return float(np.mean(purities))

    jaccard_l2_l1 = calc_jaccard(idx_l2_k, idx_l1_k)
    jaccard_l2_cos = calc_jaccard(idx_l2_k, idx_cos_k)
    jaccard_l1_cos = calc_jaccard(idx_l1_k, idx_cos_k)

    agree_1nn_l2_l1 = float(np.mean(idx_l2[:, 1] == idx_l1[:, 1]))
    agree_1nn_l2_cos = float(np.mean(idx_l2[:, 1] == idx_cos[:, 1]))

    jaccard_raw_std = calc_jaccard(idx_raw_k, idx_l2_k)
    jaccard_mm_std = calc_jaccard(idx_mm_k, idx_l2_k)
    jaccard_rob_std = calc_jaccard(idx_rob_k, idx_l2_k)

    purity_l2 = calc_purity(idx_l2_k)
    purity_l1 = calc_purity(idx_l1_k)
    purity_cos = calc_purity(idx_cos_k)
    purity_raw = calc_purity(idx_raw_k)
    purity_mm = calc_purity(idx_mm_k)
    purity_rob = calc_purity(idx_rob_k)

    area_idx = features.index("Area")
    area_vals = X_raw[:, area_idx]
    area_1nn_raw_diff = np.abs(area_vals - area_vals[idx_raw[:, 1]])
    area_1nn_std_diff = np.abs(area_vals - area_vals[idx_l2[:, 1]])

    comparison_records = [
        {"Comparison_Type": "Distance Metric", "Metric_A": "Euclidean (L2, Std)", "Metric_B": "Manhattan (L1, Std)", "Jaccard_Overlap_k10": jaccard_l2_l1, "Exact_1NN_Agreement": agree_1nn_l2_l1, "Neighbor_Purity_A": purity_l2, "Neighbor_Purity_B": purity_l1},
        {"Comparison_Type": "Distance Metric", "Metric_A": "Euclidean (L2, Std)", "Metric_B": "Cosine Similarity (Std)", "Jaccard_Overlap_k10": jaccard_l2_cos, "Exact_1NN_Agreement": agree_1nn_l2_cos, "Neighbor_Purity_A": purity_l2, "Neighbor_Purity_B": purity_cos},
        {"Comparison_Type": "Distance Metric", "Metric_A": "Manhattan (L1, Std)", "Metric_B": "Cosine Similarity (Std)", "Jaccard_Overlap_k10": jaccard_l1_cos, "Exact_1NN_Agreement": float(np.mean(idx_l1[:, 1] == idx_cos[:, 1])), "Neighbor_Purity_A": purity_l1, "Neighbor_Purity_B": purity_cos},
        {"Comparison_Type": "Scaling Distortion", "Metric_A": "Unscaled (Raw Euclidean)", "Metric_B": "StandardScaler (L2)", "Jaccard_Overlap_k10": jaccard_raw_std, "Exact_1NN_Agreement": float(np.mean(idx_raw[:, 1] == idx_l2[:, 1])), "Neighbor_Purity_A": purity_raw, "Neighbor_Purity_B": purity_l2},
        {"Comparison_Type": "Scaling Distortion", "Metric_A": "MinMaxScaler (L2)", "Metric_B": "StandardScaler (L2)", "Jaccard_Overlap_k10": jaccard_mm_std, "Exact_1NN_Agreement": float(np.mean(idx_mm[:, 1] == idx_l2[:, 1])), "Neighbor_Purity_A": purity_mm, "Neighbor_Purity_B": purity_l2},
        {"Comparison_Type": "Scaling Distortion", "Metric_A": "RobustScaler (L2)", "Metric_B": "StandardScaler (L2)", "Jaccard_Overlap_k10": jaccard_rob_std, "Exact_1NN_Agreement": float(np.mean(idx_rob[:, 1] == idx_l2[:, 1])), "Neighbor_Purity_A": purity_rob, "Neighbor_Purity_B": purity_l2},
    ]
    comp_df = pd.DataFrame(comparison_records)
    comp_df.to_csv("results/tables/distance_similarity_comparison.csv", index=False)

    summary_metrics = {
        "jaccard_overlap_l2_l1": jaccard_l2_l1,
        "agreement_1nn_l2_l1": agree_1nn_l2_l1,
        "jaccard_overlap_l2_cos": jaccard_l2_cos,
        "agreement_1nn_l2_cos": agree_1nn_l2_cos,
        "jaccard_raw_to_standardized": jaccard_raw_std,
        "neighbor_purity_std_l2": purity_l2,
        "neighbor_purity_std_l1": purity_l1,
        "neighbor_purity_std_cosine": purity_cos,
        "neighbor_purity_raw_l2": purity_raw,
        "neighbor_purity_minmax_l2": purity_mm,
        "neighbor_purity_robust_l2": purity_rob,
        "purity_drop_raw_vs_std": purity_l2 - purity_raw,
        "mean_area_diff_1nn_raw": float(np.mean(area_1nn_raw_diff)),
        "mean_area_diff_1nn_std": float(np.mean(area_1nn_std_diff)),
    }
    return comp_df, summary_metrics


def plot_similarity_distance_comparison(comp_df, summary_metrics, output_prefix="results/figures/fig08_similarity_distance_comparison"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    metric_comparisons = comp_df[comp_df["Comparison_Type"] == "Distance Metric"]
    bar_width = 0.35
    x_pos = np.arange(len(metric_comparisons))
    labels = ["L2 vs L1", "L2 vs Cosine", "L1 vs Cosine"]

    axes[0].bar(x_pos - bar_width / 2, metric_comparisons["Jaccard_Overlap_k10"], width=bar_width, label="Jaccard Overlap @ 10", color="#2980b9", edgecolor="#1a5276", alpha=0.9)
    axes[0].bar(x_pos + bar_width / 2, metric_comparisons["Exact_1NN_Agreement"], width=bar_width, label="Exact 1-NN Agreement", color="#27ae60", edgecolor="#196f3d", alpha=0.9)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(labels, fontsize=10, fontweight="bold")
    axes[0].set_ylabel("Overlap / Agreement Proportion", fontsize=10, fontweight="bold")
    axes[0].set_title("Metric Concordance on Standardized Features", fontsize=11, fontweight="bold")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(frameon=True, facecolor="white", edgecolor="#bdc3c7")
    for i in x_pos:
        j_val = metric_comparisons.iloc[i]["Jaccard_Overlap_k10"]
        a_val = metric_comparisons.iloc[i]["Exact_1NN_Agreement"]
        axes[0].text(i - bar_width / 2, j_val + 0.02, f"{j_val:.3f}", ha="center", fontsize=8, fontweight="bold")
        axes[0].text(i + bar_width / 2, a_val + 0.02, f"{a_val:.3f}", ha="center", fontsize=8, fontweight="bold")

    scalers = ["Unscaled\n(Raw L2)", "MinMaxScaler\n(L2)", "RobustScaler\n(L2)", "StandardScaler\n(L2)"]
    purities = [
        summary_metrics["neighbor_purity_raw_l2"],
        summary_metrics["neighbor_purity_minmax_l2"],
        summary_metrics["neighbor_purity_robust_l2"],
        summary_metrics["neighbor_purity_std_l2"],
    ]
    jaccards_to_std = [
        summary_metrics["jaccard_raw_to_standardized"],
        comp_df[comp_df["Metric_A"] == "MinMaxScaler (L2)"]["Jaccard_Overlap_k10"].values[0],
        comp_df[comp_df["Metric_A"] == "RobustScaler (L2)"]["Jaccard_Overlap_k10"].values[0],
        1.0,
    ]
    x_sc = np.arange(len(scalers))
    axes[1].bar(x_sc - bar_width / 2, purities, width=bar_width, label="Neighbor Class Purity @ 10", color="#8e44ad", edgecolor="#5b2c6f", alpha=0.9)
    axes[1].bar(x_sc + bar_width / 2, jaccards_to_std, width=bar_width, label="Jaccard Overlap to StandardScaler", color="#e67e22", edgecolor="#a04000", alpha=0.9)
    axes[1].set_xticks(x_sc)
    axes[1].set_xticklabels(scalers, fontsize=9.5, fontweight="bold")
    axes[1].set_ylabel("Purity / Overlap Ratio", fontsize=10, fontweight="bold")
    axes[1].set_title("Scale Distortion & Nearest-Neighbor Integrity", fontsize=11, fontweight="bold")
    axes[1].set_ylim(0, 1.10)
    axes[1].legend(frameon=True, facecolor="white", edgecolor="#bdc3c7")
    for i in x_sc:
        axes[1].text(i - bar_width / 2, purities[i] + 0.02, f"{purities[i]:.3f}", ha="center", fontsize=8, fontweight="bold")
        axes[1].text(i + bar_width / 2, jaccards_to_std[i] + 0.02, f"{jaccards_to_std[i]:.3f}", ha="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(f"{output_prefix}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_prefix}.pdf", bbox_inches="tight")
    plt.close()


def run_kmeans_sweep(X_std, y_true, k_range=range(2, 11), random_state=42):
    records = []
    sample_size = min(3500, len(X_std))
    rng = np.random.RandomState(random_state)
    sample_indices = rng.choice(len(X_std), size=sample_size, replace=False)
    X_sub = X_std[sample_indices]

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_std)
        inertia = km.inertia_
        sil = silhouette_score(X_sub, labels[sample_indices], random_state=random_state)
        ch = calinski_harabasz_score(X_std, labels)
        db = davies_bouldin_score(X_std, labels)
        ari = adjusted_rand_score(y_true, labels)
        nmi = normalized_mutual_info_score(y_true, labels)

        records.append({
            "k": k,
            "Inertia_WCSS": inertia,
            "Silhouette_Score": sil,
            "Calinski_Harabasz": ch,
            "Davies_Bouldin": db,
            "Adjusted_Rand_Index": ari,
            "Normalized_Mutual_Info": nmi,
        })
    sweep_df = pd.DataFrame(records)
    return sweep_df


def plot_kmeans_elbow_silhouette(sweep_df, output_prefix="results/figures/fig09_kmeans_elbow_silhouette"):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=300)

    axes[0].plot(sweep_df["k"], sweep_df["Inertia_WCSS"], marker="o", color="#c0392b", linewidth=2, markersize=6)
    axes[0].axvline(x=7, color="#7f8c8d", linestyle="--", alpha=0.8, label="Domain k=7")
    axes[0].set_xlabel("Number of Clusters (k)", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("Inertia (Within-Cluster Sum of Squares)", fontsize=10, fontweight="bold")
    axes[0].set_title("k-Means Elbow Curve", fontsize=11, fontweight="bold")
    axes[0].legend(frameon=True)

    axes[1].plot(sweep_df["k"], sweep_df["Silhouette_Score"], marker="s", color="#2980b9", linewidth=2, markersize=6, label="Silhouette")
    axes[1].axvline(x=7, color="#7f8c8d", linestyle="--", alpha=0.8, label="Domain k=7")
    axes[1].set_xlabel("Number of Clusters (k)", fontsize=10, fontweight="bold")
    axes[1].set_ylabel("Silhouette Score", fontsize=10, fontweight="bold")
    axes[1].set_title("Silhouette Profile Across k", fontsize=11, fontweight="bold")
    axes[1].legend(frameon=True)

    axes[2].plot(sweep_df["k"], sweep_df["Adjusted_Rand_Index"], marker="^", color="#27ae60", linewidth=2, markersize=6, label="ARI (vs Ground Truth)")
    axes[2].plot(sweep_df["k"], sweep_df["Normalized_Mutual_Info"], marker="D", color="#8e44ad", linewidth=2, markersize=6, label="NMI (vs Ground Truth)")
    axes[2].axvline(x=7, color="#7f8c8d", linestyle="--", alpha=0.8, label="Domain k=7")
    axes[2].set_xlabel("Number of Clusters (k)", fontsize=10, fontweight="bold")
    axes[2].set_ylabel("Agreement with True Classes", fontsize=10, fontweight="bold")
    axes[2].set_title("External Cluster Validation (ARI & NMI)", fontsize=11, fontweight="bold")
    axes[2].legend(frameon=True)

    plt.tight_layout()
    plt.savefig(f"{output_prefix}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_prefix}.pdf", bbox_inches="tight")
    plt.close()


def run_clustering_comparison(X_std, y_true, df_clean, random_state=42):
    features = [c for c in df_clean.columns if c != "Class"]

    km7 = KMeans(n_clusters=7, random_state=random_state, n_init=10)
    km_labels = km7.fit_predict(X_std)

    sample_size = 2500
    stratified_sample = df_clean.groupby("Class", group_keys=False).apply(
        lambda g: g.sample(n=int(np.round(sample_size * len(g) / len(df_clean))), random_state=random_state)
    )
    hier_idx = stratified_sample.index.values
    X_hier_sub = X_std[hier_idx]
    y_hier_sub = y_true[hier_idx]

    hier = AgglomerativeClustering(n_clusters=7, linkage="ward")
    hier_labels = hier.fit_predict(X_hier_sub)

    db = DBSCAN(eps=1.5, min_samples=10)
    db_labels = db.fit_predict(X_std)

    sil_km = float(silhouette_score(X_std, km_labels, sample_size=3500, random_state=random_state))
    ari_km = float(adjusted_rand_score(y_true, km_labels))
    nmi_km = float(normalized_mutual_info_score(y_true, km_labels))
    ch_km = float(calinski_harabasz_score(X_std, km_labels))
    db_km = float(davies_bouldin_score(X_std, km_labels))

    sil_hier = float(silhouette_score(X_hier_sub, hier_labels))
    ari_hier = float(adjusted_rand_score(y_hier_sub, hier_labels))
    nmi_hier = float(normalized_mutual_info_score(y_hier_sub, hier_labels))
    ch_hier = float(calinski_harabasz_score(X_hier_sub, hier_labels))
    db_hier = float(davies_bouldin_score(X_hier_sub, hier_labels))

    valid_db_mask = db_labels != -1
    sil_db = float(silhouette_score(X_std[valid_db_mask], db_labels[valid_db_mask], sample_size=3500, random_state=random_state)) if np.sum(valid_db_mask) > 100 else 0.0
    ari_db = float(adjusted_rand_score(y_true, db_labels))
    nmi_db = float(normalized_mutual_info_score(y_true, db_labels))
    db_noise_count = int(np.sum(db_labels == -1))
    db_noise_pct = float(db_noise_count / len(X_std) * 100)
    db_n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)

    comparison_records = [
        {"Algorithm": "k-Means (k=7)", "Evaluated_Instances": len(X_std), "Clusters_Discovered": 7, "Noise_Records": 0, "Noise_Percentage": 0.0, "Silhouette_Score": sil_km, "Calinski_Harabasz": ch_km, "Davies_Bouldin": db_km, "Adjusted_Rand_Index": ari_km, "Normalized_Mutual_Info": nmi_km},
        {"Algorithm": "Hierarchical Ward (k=7, Subsampled)", "Evaluated_Instances": len(X_hier_sub), "Clusters_Discovered": 7, "Noise_Records": 0, "Noise_Percentage": 0.0, "Silhouette_Score": sil_hier, "Calinski_Harabasz": ch_hier, "Davies_Bouldin": db_hier, "Adjusted_Rand_Index": ari_hier, "Normalized_Mutual_Info": nmi_hier},
        {"Algorithm": "DBSCAN (eps=1.5, min_samples=10)", "Evaluated_Instances": len(X_std), "Clusters_Discovered": db_n_clusters, "Noise_Records": db_noise_count, "Noise_Percentage": db_noise_pct, "Silhouette_Score": sil_db, "Calinski_Harabasz": float(calinski_harabasz_score(X_std[valid_db_mask], db_labels[valid_db_mask])), "Davies_Bouldin": float(davies_bouldin_score(X_std[valid_db_mask], db_labels[valid_db_mask])), "Adjusted_Rand_Index": ari_db, "Normalized_Mutual_Info": nmi_db},
    ]
    comp_df = pd.DataFrame(comparison_records)
    comp_df.to_csv("results/tables/clustering_comparison.csv", index=False)

    df_profiles = df_clean.copy()
    df_profiles["kmeans_cluster"] = km_labels
    cluster_profiles = df_profiles.groupby("kmeans_cluster")[features].mean()
    cluster_profiles["record_count"] = df_profiles.groupby("kmeans_cluster").size()
    cluster_profiles.to_csv("results/tables/kmeans_cluster_profiles.csv")

    contingency = pd.crosstab(y_true, km_labels, rownames=["Cultivar"], colnames=["Cluster"])

    clustering_metrics = {
        "kmeans_k7_ari": ari_km,
        "kmeans_k7_nmi": nmi_km,
        "kmeans_k7_silhouette": sil_km,
        "hierarchical_k7_ari": ari_hier,
        "hierarchical_k7_silhouette": sil_hier,
        "dbscan_clusters": db_n_clusters,
        "dbscan_noise_count": db_noise_count,
        "dbscan_noise_pct": db_noise_pct,
        "dbscan_ari": ari_db,
    }
    return comp_df, km_labels, hier_labels, hier_idx, db_labels, contingency, clustering_metrics


def plot_clustering_comparison_pca(X_pca_2d, y_true, km_labels, hier_labels, hier_idx, db_labels, contingency, output_prefix="results/figures/fig10_clustering_comparison_pca"):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=300)

    classes = np.unique(y_true)
    class_palette = sns.color_palette("tab10", n_colors=len(classes))
    class_color_map = {cls: class_palette[i] for i, cls in enumerate(classes)}

    for cls in classes:
        mask = y_true == cls
        axes[0, 0].scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1], c=[class_color_map[cls]], label=cls, s=10, alpha=0.6, edgecolors="none")
    axes[0, 0].set_title("Ground Truth Cultivars (2D PCA)", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("Principal Component 1 (Size)", fontsize=10, fontweight="bold")
    axes[0, 0].set_ylabel("Principal Component 2 (Shape)", fontsize=10, fontweight="bold")
    axes[0, 0].legend(frameon=True, markerscale=2.5, fontsize=8.5, loc="upper right")

    km_palette = sns.color_palette("Set2", n_colors=7)
    for k in range(7):
        mask = km_labels == k
        axes[0, 1].scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1], color=km_palette[k], label=f"Cluster {k}", s=10, alpha=0.6, edgecolors="none")
    axes[0, 1].set_title("k-Means Clustering Solution (k=7)", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("Principal Component 1 (Size)", fontsize=10, fontweight="bold")
    axes[0, 1].set_ylabel("Principal Component 2 (Shape)", fontsize=10, fontweight="bold")
    axes[0, 1].legend(frameon=True, markerscale=2.5, fontsize=8.5, loc="upper right")

    db_unique = np.unique(db_labels)
    db_colors = { -1: "#7f8c8d", 0: "#2980b9", 1: "#e74c3c" }
    for db_c in db_unique:
        mask = db_labels == db_c
        lbl = "Noise / Outlier (-1)" if db_c == -1 else f"Dense Cluster {db_c}"
        clr = db_colors.get(db_c, "#27ae60")
        axes[1, 0].scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1], color=clr, label=lbl, s=12, alpha=0.7 if db_c != -1 else 0.4, edgecolors="none")
    axes[1, 0].set_title("DBSCAN Density Partition (eps=1.5, min_samples=10)", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("Principal Component 1 (Size)", fontsize=10, fontweight="bold")
    axes[1, 0].set_ylabel("Principal Component 2 (Shape)", fontsize=10, fontweight="bold")
    axes[1, 0].legend(frameon=True, markerscale=2.5, fontsize=8.5, loc="upper right")

    sns.heatmap(contingency, annot=True, fmt="d", cmap="Blues", cbar=True, ax=axes[1, 1])
    axes[1, 1].set_title("k-Means Cluster vs True Cultivar Contingency", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("k-Means Cluster Index", fontsize=10, fontweight="bold")
    axes[1, 1].set_ylabel("True Bean Cultivar", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(f"{output_prefix}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_prefix}.pdf", bbox_inches="tight")
    plt.close()


def train_and_evaluate_classifiers(X_train, X_test, y_train, y_test, random_state=42):
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    dt_cv_scores = []
    best_dt = None
    best_cv_f1 = -1.0
    best_params = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    for max_depth in [6, 8, 10, 12]:
        for min_samples_leaf in [2, 5, 10]:
            clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=random_state)
            scores = cross_val_score(clf, X_train_std, y_train, cv=cv, scoring="f1_macro")
            mean_score = float(np.mean(scores))
            if mean_score > best_cv_f1:
                best_cv_f1 = mean_score
                best_dt = clf
                best_params = {"max_depth": max_depth, "min_samples_leaf": min_samples_leaf}

    best_dt.fit(X_train_std, y_train)

    rule_clf = DecisionTreeClassifier(max_depth=4, random_state=random_state)
    rule_clf.fit(X_train_std, y_train)

    gnb = GaussianNB()
    gnb.fit(X_train_std, y_train)

    knn = KNeighborsClassifier(n_neighbors=7, metric="euclidean")
    knn.fit(X_train_std, y_train)

    models = {
        "Decision Tree (Tuned)": best_dt,
        "Interpretable Rule-Based (Depth 4)": rule_clf,
        "Gaussian Naive Bayes": gnb,
        "k-Nearest Neighbors (k=7)": knn,
    }

    eval_records = []
    per_class_dfs = []
    class_names = sorted(list(y_train.unique()))

    for name, model in models.items():
        y_pred = model.predict(X_test_std)
        acc = float(accuracy_score(y_test, y_pred))
        macro_p = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
        macro_r = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
        macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
        cv_f1 = float(np.mean(cross_val_score(model, X_train_std, y_train, cv=cv, scoring="f1_macro")))

        eval_records.append({
            "Classifier": name,
            "Accuracy": acc,
            "Macro_Precision": macro_p,
            "Macro_Recall": macro_r,
            "Macro_F1": macro_f1,
            "Weighted_F1": weighted_f1,
            "CV_5Fold_Macro_F1": cv_f1,
        })

        rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        for cls in class_names:
            per_class_dfs.append({
                "Classifier": name,
                "Cultivar": cls,
                "Precision": rep[cls]["precision"],
                "Recall": rep[cls]["recall"],
                "F1_Score": rep[cls]["f1-score"],
                "Support": rep[cls]["support"],
            })

    comparison_df = pd.DataFrame(eval_records)
    comparison_df.to_csv("results/tables/classification_comparison.csv", index=False)

    per_class_df = pd.DataFrame(per_class_dfs)
    per_class_df.to_csv("results/tables/per_class_classification_report.csv", index=False)

    rule_text = export_text(rule_clf, feature_names=list(X_train.columns))

    classification_metrics = {
        "dt_best_params": best_params,
        "dt_test_accuracy": float(comparison_df[comparison_df["Classifier"] == "Decision Tree (Tuned)"]["Accuracy"].values[0]),
        "dt_test_macro_f1": float(comparison_df[comparison_df["Classifier"] == "Decision Tree (Tuned)"]["Macro_F1"].values[0]),
        "rule_test_accuracy": float(comparison_df[comparison_df["Classifier"] == "Interpretable Rule-Based (Depth 4)"]["Accuracy"].values[0]),
        "rule_test_macro_f1": float(comparison_df[comparison_df["Classifier"] == "Interpretable Rule-Based (Depth 4)"]["Macro_F1"].values[0]),
        "gnb_test_accuracy": float(comparison_df[comparison_df["Classifier"] == "Gaussian Naive Bayes"]["Accuracy"].values[0]),
        "gnb_test_macro_f1": float(comparison_df[comparison_df["Classifier"] == "Gaussian Naive Bayes"]["Macro_F1"].values[0]),
        "knn_test_accuracy": float(comparison_df[comparison_df["Classifier"] == "k-Nearest Neighbors (k=7)"]["Accuracy"].values[0]),
        "knn_test_macro_f1": float(comparison_df[comparison_df["Classifier"] == "k-Nearest Neighbors (k=7)"]["Macro_F1"].values[0]),
    }

    return comparison_df, per_class_df, models, rule_text, scaler, X_test_std, classification_metrics


def plot_classification_results(models, X_test_std, y_test, class_names, per_class_df, output_prefix="results/figures/fig11_classification_confusion_matrices"):
    fig, axes = plt.subplots(2, 2, figsize=(15, 13), dpi=300)
    axes_flat = axes.flatten()

    for idx, (name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test_std)
        cm = confusion_matrix(y_test, y_pred, labels=class_names, normalize="true")
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", cbar=False,
                    xticklabels=class_names, yticklabels=class_names, ax=axes_flat[idx])
        axes_flat[idx].set_title(f"{name}\nNormalized Confusion Matrix", fontsize=10.5, fontweight="bold")
        axes_flat[idx].set_xlabel("Predicted Class", fontsize=9.5, fontweight="bold")
        axes_flat[idx].set_ylabel("True Class", fontsize=9.5, fontweight="bold")

    plt.tight_layout()
    plt.savefig(f"{output_prefix}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_prefix}.pdf", bbox_inches="tight")
    plt.close()


def mine_morphological_association_rules(df_clean, min_support=0.02, min_confidence=0.30):
    features_disc = ["Area", "AspectRation", "roundness", "Solidity", "Extent"]
    df_disc = pd.DataFrame()
    for col in features_disc:
        df_disc[col] = pd.qcut(df_clean[col], q=3, labels=["Low", "Medium", "High"]).astype(str)
    df_disc["Class"] = df_clean["Class"]

    df_ohe = pd.get_dummies(df_disc)
    frequent = fpgrowth(df_ohe, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent, metric="confidence", min_threshold=min_confidence)

    car_rules = rules[rules["consequents"].apply(lambda c: len(c) == 1 and list(c)[0].startswith("Class_"))].copy()
    car_rules["antecedents_str"] = car_rules["antecedents"].apply(lambda a: " + ".join(sorted(list(a))))
    car_rules["consequent_class"] = car_rules["consequents"].apply(lambda c: list(c)[0].replace("Class_", ""))
    car_rules = car_rules.sort_values(by=["lift", "confidence"], ascending=False)

    top_per_class = car_rules.groupby("consequent_class").head(3).copy()
    top_per_class_export = top_per_class[["consequent_class", "antecedents_str", "support", "confidence", "lift", "conviction"]].copy()
    top_per_class_export.columns = ["Target_Cultivar", "Morphological_Rule_Antecedent", "Support", "Confidence", "Lift", "Conviction"]
    top_per_class_export.to_csv("results/tables/association_rules_top.csv", index=False)

    top_lift_rule = car_rules.iloc[0]
    association_metrics = {
        "total_frequent_itemsets": len(frequent),
        "total_rules_min_conf_30": len(rules),
        "total_car_rules": len(car_rules),
        "max_lift_value": float(top_lift_rule["lift"]),
        "max_lift_cultivar": str(top_lift_rule["consequent_class"]),
        "max_lift_antecedent": str(top_lift_rule["antecedents_str"]),
        "max_lift_confidence": float(top_lift_rule["confidence"]),
        "max_lift_support": float(top_lift_rule["support"]),
    }

    return top_per_class_export, car_rules, association_metrics


def plot_association_rules(car_rules, output_prefix="results/figures/fig12_association_rules_scatter"):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    scatter = ax.scatter(car_rules["support"], car_rules["confidence"], c=car_rules["lift"], cmap="plasma", s=car_rules["lift"] * 15, alpha=0.75, edgecolors="#2c3e50", linewidths=0.5)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Rule Lift", fontsize=10, fontweight="bold")

    callout_specs = {
        "DERMASON": dict(xytext=(0.090, 1.030), rad=0.06,  shrinkB=12),
        "BARBUNYA": dict(xytext=(0.165, 0.965), rad=-0.04, shrinkB=8),
        "SIRA":     dict(xytext=(0.090, 0.915), rad=0.02,  shrinkB=8),
        "HOROZ":    dict(xytext=(0.165, 0.865), rad=-0.04, shrinkB=8),
        "SEKER":    dict(xytext=(0.090, 0.810), rad=-0.04, shrinkB=12),
        "CALI":     dict(xytext=(0.165, 0.745), rad=-0.04, shrinkB=8),
    }

    top_distinct = car_rules.groupby("consequent_class").head(1).sort_values(by="confidence", ascending=False)
    annotations = []
    for _, row in top_distinct.iterrows():
        c_name = row["consequent_class"]
        spec = callout_specs.get(c_name, dict(xytext=(row["support"] + 0.05, row["confidence"]), rad=0.0, shrinkB=5))
        label = f"{c_name} (Lift={row['lift']:.1f}, Conf={row['confidence']:.2f})"
        txt = ax.annotate(
            label,
            xy=(row["support"], row["confidence"]),
            xytext=spec["xytext"],
            fontsize=8.5,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.4", alpha=0.92),
            arrowprops=dict(
                arrowstyle="->",
                color="0.3",
                lw=0.9,
                connectionstyle=f"arc3,rad={spec['rad']}",
                shrinkB=spec["shrinkB"],
                mutation_scale=8
            )
        )
        annotations.append((c_name, txt))

    ax.set_xlabel("Rule Support", fontsize=10, fontweight="bold")
    ax.set_ylabel("Rule Confidence", fontsize=10, fontweight="bold")
    ax.set_title("Morphological Class Association Rules (Support vs Confidence vs Lift)", fontsize=11, fontweight="bold")
    ax.set_xlim(0.005, 0.255)
    ax.set_ylim(0.26, 1.07)

    plt.tight_layout()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_box = ax.get_window_extent(renderer=renderer)
    patch_boxes = [txt.get_bbox_patch().get_window_extent(renderer=renderer) for _, txt in annotations]
    for i in range(len(patch_boxes)):
        for j in range(i + 1, len(patch_boxes)):
            assert not patch_boxes[i].overlaps(patch_boxes[j]), f"Overlap detected between {annotations[i][0]} and {annotations[j][0]}"
    for idx, (name, _) in enumerate(annotations):
        b = patch_boxes[idx]
        assert ax_box.contains(b.x0, b.y0) and ax_box.contains(b.x1, b.y1), f"Annotation {name} box is outside plot axes!"

    plt.savefig(f"{output_prefix}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_prefix}.pdf", bbox_inches="tight")
    plt.close()


def run_stress_test_missing_data(X_train, X_test, y_train, y_test, X_full, y_full, fractions=[0.0, 0.05, 0.10, 0.20, 0.30], random_state=42):
    results = []

    for frac in fractions:
        rng = np.random.RandomState(random_state)

        for imputer_type in ["Median", "KNN"]:
            if frac == 0.0:
                X_tr_corrupt = X_train.values.copy()
                X_te_corrupt = X_test.values.copy()
                X_fl_corrupt = X_full.values.copy()
            else:
                mask_tr = rng.rand(*X_train.shape) < frac
                X_tr_corrupt = X_train.values.copy().astype(float)
                X_tr_corrupt[mask_tr] = np.nan

                mask_te = rng.rand(*X_test.shape) < frac
                X_te_corrupt = X_test.values.copy().astype(float)
                X_te_corrupt[mask_te] = np.nan

                mask_fl = rng.rand(*X_full.shape) < frac
                X_fl_corrupt = X_full.values.copy().astype(float)
                X_fl_corrupt[mask_fl] = np.nan

            if frac == 0.0:
                X_tr_imp, X_te_imp, X_fl_imp = X_tr_corrupt, X_te_corrupt, X_fl_corrupt
            elif imputer_type == "Median":
                imp = SimpleImputer(strategy="median")
                X_tr_imp = imp.fit_transform(X_tr_corrupt)
                X_te_imp = imp.transform(X_te_corrupt)
                X_fl_imp = imp.fit_transform(X_fl_corrupt)
            else:
                imp = KNNImputer(n_neighbors=5)
                imp.fit(X_tr_corrupt[:2500])
                X_tr_imp = imp.transform(X_tr_corrupt)
                X_te_imp = imp.transform(X_te_corrupt)
                imp_fl = KNNImputer(n_neighbors=5)
                imp_fl.fit(X_fl_corrupt[:2500])
                X_fl_imp = imp_fl.transform(X_fl_corrupt)

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr_imp)
            X_te_s = scaler.transform(X_te_imp)
            X_fl_s = scaler.fit_transform(X_fl_imp)

            dt = DecisionTreeClassifier(max_depth=10, min_samples_leaf=5, random_state=random_state)
            dt.fit(X_tr_s, y_train)
            dt_f1 = float(f1_score(y_test, dt.predict(X_te_s), average="macro", zero_division=0))

            rule = DecisionTreeClassifier(max_depth=4, random_state=random_state)
            rule.fit(X_tr_s, y_train)
            rule_f1 = float(f1_score(y_test, rule.predict(X_te_s), average="macro", zero_division=0))

            gnb = GaussianNB()
            gnb.fit(X_tr_s, y_train)
            gnb_f1 = float(f1_score(y_test, gnb.predict(X_te_s), average="macro", zero_division=0))

            knn = KNeighborsClassifier(n_neighbors=7)
            knn.fit(X_tr_s, y_train)
            knn_f1 = float(f1_score(y_test, knn.predict(X_te_s), average="macro", zero_division=0))

            km = KMeans(n_clusters=7, random_state=random_state, n_init=5)
            km_labels = km.fit_predict(X_fl_s)
            km_ari = float(adjusted_rand_score(y_full, km_labels))
            km_sil = float(silhouette_score(X_fl_s, km_labels, sample_size=3000, random_state=random_state))

            results.append({
                "Missing_Fraction": frac,
                "Data_Quality_Pct": int((1.0 - frac) * 100),
                "Imputer": imputer_type,
                "DT_Macro_F1": dt_f1,
                "Rule_Macro_F1": rule_f1,
                "GNB_Macro_F1": gnb_f1,
                "KNN_Macro_F1": knn_f1,
                "KMeans_ARI": km_ari,
                "KMeans_Silhouette": km_sil,
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv("results/tables/stress_test_missing_data.csv", index=False)
    return results_df


def plot_stress_test_missing_data(df_res, output_prefix="results/figures/fig13_robustness_curve_missing_data"):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), dpi=300)

    for imp, style in [("Median", "--"), ("KNN", "-")]:
        sub = df_res[df_res["Imputer"] == imp]
        axes[0].plot(sub["Data_Quality_Pct"], sub["DT_Macro_F1"], marker="o", linestyle=style, label=f"Decision Tree ({imp})", color="#2980b9", lw=1.8)
        axes[0].plot(sub["Data_Quality_Pct"], sub["KNN_Macro_F1"], marker="s", linestyle=style, label=f"k-NN ({imp})", color="#27ae60", lw=1.8)
        axes[0].plot(sub["Data_Quality_Pct"], sub["GNB_Macro_F1"], marker="^", linestyle=style, label=f"GaussianNB ({imp})", color="#e67e22", lw=1.8)
        axes[0].plot(sub["Data_Quality_Pct"], sub["Rule_Macro_F1"], marker="D", linestyle=style, label=f"Rule-Based ({imp})", color="#8e44ad", lw=1.8)

    axes[0].set_xlabel("Data Quality % (100% - MCAR %)", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("Classification Macro-F1", fontsize=10, fontweight="bold")
    axes[0].set_title("Classification Degradation Curves (SRQ1: Missingness Stress)", fontsize=11, fontweight="bold")
    axes[0].invert_xaxis()
    axes[0].legend(frameon=True, fontsize=8, ncol=2)

    for imp, style, clr in [("Median", "--", "#c0392b"), ("KNN", "-", "#16a085")]:
        sub = df_res[df_res["Imputer"] == imp]
        axes[1].plot(sub["Data_Quality_Pct"], sub["KMeans_ARI"], marker="o", linestyle=style, color=clr, label=f"k-Means ARI ({imp})", lw=2)
        axes[1].plot(sub["Data_Quality_Pct"], sub["KMeans_Silhouette"], marker="s", linestyle=style, color=clr, alpha=0.5, label=f"k-Means Silhouette ({imp})", lw=1.5)

    axes[1].set_xlabel("Data Quality % (100% - MCAR %)", fontsize=10, fontweight="bold")
    axes[1].set_ylabel("Clustering Validation Metric", fontsize=10, fontweight="bold")
    axes[1].set_title("Clustering Degradation Curves (k-Means ARI & Silhouette)", fontsize=11, fontweight="bold")
    axes[1].invert_xaxis()
    axes[1].legend(frameon=True, fontsize=8.5)

    plt.tight_layout()
    plt.savefig(f"{output_prefix}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_prefix}.pdf", bbox_inches="tight")
    plt.close()


def run_stress_test_outliers(X_train, X_test, y_train, y_test, X_full, y_full, rates=[0.0, 0.05, 0.10, 0.20], random_state=42):
    results = []

    for rate in rates:
        rng = np.random.RandomState(random_state)
        X_tr_corrupt = X_train.values.copy().astype(float)
        X_fl_corrupt = X_full.values.copy().astype(float)

        if rate > 0.0:
            n_tr_corrupt = int(len(X_tr_corrupt) * rate)
            idx_tr = rng.choice(len(X_tr_corrupt), size=n_tr_corrupt, replace=False)
            cols_tr = rng.choice(X_tr_corrupt.shape[1], size=(n_tr_corrupt, 3))
            tr_stds = np.std(X_tr_corrupt, axis=0)
            for i, r_idx in enumerate(idx_tr):
                for c_idx in cols_tr[i]:
                    X_tr_corrupt[r_idx, c_idx] += 8.0 * tr_stds[c_idx]

            n_fl_corrupt = int(len(X_fl_corrupt) * rate)
            idx_fl = rng.choice(len(X_fl_corrupt), size=n_fl_corrupt, replace=False)
            cols_fl = rng.choice(X_fl_corrupt.shape[1], size=(n_fl_corrupt, 3))
            fl_stds = np.std(X_fl_corrupt, axis=0)
            for i, r_idx in enumerate(idx_fl):
                for c_idx in cols_fl[i]:
                    X_fl_corrupt[r_idx, c_idx] += 8.0 * fl_stds[c_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr_corrupt)
        X_te_s = scaler.transform(X_test.values)
        X_fl_s = scaler.fit_transform(X_fl_corrupt)

        dt = DecisionTreeClassifier(max_depth=10, min_samples_leaf=5, random_state=random_state)
        dt.fit(X_tr_s, y_train)
        dt_f1 = float(f1_score(y_test, dt.predict(X_te_s), average="macro", zero_division=0))

        rule = DecisionTreeClassifier(max_depth=4, random_state=random_state)
        rule.fit(X_tr_s, y_train)
        rule_f1 = float(f1_score(y_test, rule.predict(X_te_s), average="macro", zero_division=0))

        gnb = GaussianNB()
        gnb.fit(X_tr_s, y_train)
        gnb_f1 = float(f1_score(y_test, gnb.predict(X_te_s), average="macro", zero_division=0))

        knn = KNeighborsClassifier(n_neighbors=7)
        knn.fit(X_tr_s, y_train)
        knn_f1 = float(f1_score(y_test, knn.predict(X_te_s), average="macro", zero_division=0))

        km = KMeans(n_clusters=7, random_state=random_state, n_init=5)
        km_labels = km.fit_predict(X_fl_s)
        km_ari = float(adjusted_rand_score(y_full, km_labels))
        km_sil = float(silhouette_score(X_fl_s, km_labels, sample_size=3000, random_state=random_state))

        db = DBSCAN(eps=1.5, min_samples=10)
        db_labels = db.fit_predict(X_fl_s)
        db_noise_pct = float(np.sum(db_labels == -1) / len(db_labels) * 100)
        db_ari = float(adjusted_rand_score(y_full, db_labels))

        results.append({
            "Outlier_Rate": rate,
            "Outlier_Pct": int(rate * 100),
            "DT_Macro_F1": dt_f1,
            "Rule_Macro_F1": rule_f1,
            "GNB_Macro_F1": gnb_f1,
            "KNN_Macro_F1": knn_f1,
            "KMeans_ARI": km_ari,
            "KMeans_Silhouette": km_sil,
            "DBSCAN_Noise_Pct": db_noise_pct,
            "DBSCAN_ARI": db_ari,
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv("results/tables/stress_test_outliers.csv", index=False)
    return results_df


def plot_stress_test_outliers(df_res, output_prefix="results/figures/fig14_robustness_curve_outliers"):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), dpi=300)

    axes[0].plot(df_res["Outlier_Pct"], df_res["DT_Macro_F1"], marker="o", label="Decision Tree", color="#2980b9", lw=2)
    axes[0].plot(df_res["Outlier_Pct"], df_res["KNN_Macro_F1"], marker="s", label="k-NN", color="#27ae60", lw=2)
    axes[0].plot(df_res["Outlier_Pct"], df_res["GNB_Macro_F1"], marker="^", label="GaussianNB", color="#e67e22", lw=2)
    axes[0].plot(df_res["Outlier_Pct"], df_res["Rule_Macro_F1"], marker="D", label="Rule-Based", color="#8e44ad", lw=2)
    axes[0].set_xlabel("Injected Outlier Percentage (%)", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("Classification Macro-F1", fontsize=10, fontweight="bold")
    axes[0].set_title("Classifier Resilience under Outlier Corruption (SRQ2)", fontsize=11, fontweight="bold")
    axes[0].legend(frameon=True, fontsize=9)

    axes[1].plot(df_res["Outlier_Pct"], df_res["KMeans_ARI"], marker="o", label="k-Means ARI", color="#c0392b", lw=2)
    axes[1].plot(df_res["Outlier_Pct"], df_res["DBSCAN_ARI"], marker="^", label="DBSCAN ARI", color="#16a085", lw=2)
    ax2_twin = axes[1].twinx()
    ax2_twin.plot(df_res["Outlier_Pct"], df_res["DBSCAN_Noise_Pct"], marker="x", color="#7f8c8d", linestyle="--", label="DBSCAN Noise %", lw=1.8)
    ax2_twin.set_ylabel("DBSCAN Flagged Noise %", fontsize=10, fontweight="bold", color="#7f8c8d")
    axes[1].set_xlabel("Injected Outlier Percentage (%)", fontsize=10, fontweight="bold")
    axes[1].set_ylabel("Clustering Adjusted Rand Index (ARI)", fontsize=10, fontweight="bold")
    axes[1].set_title("Clustering Distortion & Noise Absorption (SRQ2)", fontsize=11, fontweight="bold")
    axes[1].legend(loc="upper left", frameon=True, fontsize=9)
    ax2_twin.legend(loc="upper right", frameon=True, fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{output_prefix}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_prefix}.pdf", bbox_inches="tight")
    plt.close()


def run_stress_test_scaling(X_train, X_test, y_train, y_test, X_full, y_full, random_state=42):
    results = []

    scalers = {
        "Raw (Unscaled)": None,
        "StandardScaler": StandardScaler(),
        "MinMaxScaler": MinMaxScaler(),
        "RobustScaler": RobustScaler(),
    }

    for regime in ["Clean (0% Outliers)", "Corrupted (10% Outliers)"]:
        rng = np.random.RandomState(random_state)
        X_tr = X_train.values.copy().astype(float)
        X_te = X_test.values.copy().astype(float)
        X_fl = X_full.values.copy().astype(float)

        if "10%" in regime:
            n_tr_corrupt = int(len(X_tr) * 0.10)
            idx_tr = rng.choice(len(X_tr), size=n_tr_corrupt, replace=False)
            cols_tr = rng.choice(X_tr.shape[1], size=(n_tr_corrupt, 3))
            tr_stds = np.std(X_tr, axis=0)
            for i, r_idx in enumerate(idx_tr):
                for c_idx in cols_tr[i]:
                    X_tr[r_idx, c_idx] += 8.0 * tr_stds[c_idx]

            n_fl_corrupt = int(len(X_fl) * 0.10)
            idx_fl = rng.choice(len(X_fl), size=n_fl_corrupt, replace=False)
            cols_fl = rng.choice(X_fl.shape[1], size=(n_fl_corrupt, 3))
            fl_stds = np.std(X_fl, axis=0)
            for i, r_idx in enumerate(idx_fl):
                for c_idx in cols_fl[i]:
                    X_fl[r_idx, c_idx] += 8.0 * fl_stds[c_idx]

        for s_name, scaler_obj in scalers.items():
            if scaler_obj is None:
                X_tr_s = X_tr
                X_te_s = X_te
                X_fl_s = X_fl
            else:
                sc = scaler_obj.__class__()
                X_tr_s = sc.fit_transform(X_tr)
                X_te_s = sc.transform(X_te)
                sc_fl = scaler_obj.__class__()
                X_fl_s = sc_fl.fit_transform(X_fl)

            dt = DecisionTreeClassifier(max_depth=10, min_samples_leaf=5, random_state=random_state)
            dt.fit(X_tr_s, y_train)
            dt_f1 = float(f1_score(y_test, dt.predict(X_te_s), average="macro", zero_division=0))

            knn = KNeighborsClassifier(n_neighbors=7)
            knn.fit(X_tr_s, y_train)
            knn_f1 = float(f1_score(y_test, knn.predict(X_te_s), average="macro", zero_division=0))

            gnb = GaussianNB()
            gnb.fit(X_tr_s, y_train)
            gnb_f1 = float(f1_score(y_test, gnb.predict(X_te_s), average="macro", zero_division=0))

            km = KMeans(n_clusters=7, random_state=random_state, n_init=5)
            km_labels = km.fit_predict(X_fl_s)
            km_ari = float(adjusted_rand_score(y_full, km_labels))
            km_sil = float(silhouette_score(X_fl_s, km_labels, sample_size=3000, random_state=random_state))

            results.append({
                "Regime": regime,
                "Scaler": s_name,
                "DT_Macro_F1": dt_f1,
                "KNN_Macro_F1": knn_f1,
                "GNB_Macro_F1": gnb_f1,
                "KMeans_ARI": km_ari,
                "KMeans_Silhouette": km_sil,
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv("results/tables/stress_test_scaling.csv", index=False)
    return results_df


def plot_stress_test_scaling(df_res, output_prefix="results/figures/fig15_scaling_sensitivity_stress"):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), dpi=300)

    clean_sub = df_res[df_res["Regime"] == "Clean (0% Outliers)"]
    corrupt_sub = df_res[df_res["Regime"] == "Corrupted (10% Outliers)"]
    x_pos = np.arange(len(clean_sub))
    bar_width = 0.2

    scalers = clean_sub["Scaler"].values

    axes[0].bar(x_pos - 1.5 * bar_width, clean_sub["KNN_Macro_F1"], width=bar_width, label="k-NN (Clean)", color="#27ae60", alpha=0.9)
    axes[0].bar(x_pos - 0.5 * bar_width, corrupt_sub["KNN_Macro_F1"], width=bar_width, label="k-NN (10% Outliers)", color="#a9dfbf", alpha=0.9, hatch="//")
    axes[0].bar(x_pos + 0.5 * bar_width, clean_sub["DT_Macro_F1"], width=bar_width, label="Decision Tree (Clean)", color="#2980b9", alpha=0.9)
    axes[0].bar(x_pos + 1.5 * bar_width, corrupt_sub["DT_Macro_F1"], width=bar_width, label="Decision Tree (10% Outliers)", color="#aed6f1", alpha=0.9, hatch="//")
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(scalers, fontsize=9.5, fontweight="bold")
    axes[0].set_ylabel("Macro-F1 Score", fontsize=10, fontweight="bold")
    axes[0].set_title("Classifier Normalization & Outlier Stress (SRQ3)", fontsize=11, fontweight="bold")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(frameon=True, fontsize=8, ncol=2)

    axes[1].bar(x_pos - bar_width, clean_sub["KMeans_ARI"], width=bar_width * 0.9, label="k-Means ARI (Clean)", color="#c0392b", alpha=0.9)
    axes[1].bar(x_pos, corrupt_sub["KMeans_ARI"], width=bar_width * 0.9, label="k-Means ARI (10% Outliers)", color="#f5b7b1", alpha=0.9, hatch="//")
    axes[1].bar(x_pos + bar_width, clean_sub["KMeans_Silhouette"], width=bar_width * 0.9, label="k-Means Silhouette (Clean)", color="#8e44ad", alpha=0.9)
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(scalers, fontsize=9.5, fontweight="bold")
    axes[1].set_ylabel("Clustering Validation Score", fontsize=10, fontweight="bold")
    axes[1].set_title("k-Means Sensitivity to Normalization & Outliers (SRQ3)", fontsize=11, fontweight="bold")
    axes[1].legend(frameon=True, fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{output_prefix}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_prefix}.pdf", bbox_inches="tight")
    plt.close()


def run_stress_test_feature_reduction(X_train, X_test, y_train, y_test, X_full, y_full, selected_8, random_state=42):
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)
    X_fl_s = scaler.fit_transform(X_full)

    all_features = list(X_train.columns)
    idx_8 = [all_features.index(f) for f in selected_8]
    X_tr_8 = X_tr_s[:, idx_8]
    X_te_8 = X_te_s[:, idx_8]
    X_fl_8 = X_fl_s[:, idx_8]

    pca4 = PCA(n_components=4, random_state=random_state)
    X_tr_pca4 = pca4.fit_transform(X_tr_s)
    X_te_pca4 = pca4.transform(X_te_s)
    X_fl_pca4 = pca4.fit_transform(X_fl_s)

    pca2 = PCA(n_components=2, random_state=random_state)
    X_tr_pca2 = pca2.fit_transform(X_tr_s)
    X_te_pca2 = pca2.transform(X_te_s)
    X_fl_pca2 = pca2.fit_transform(X_fl_s)

    regimes = {
        "All 16 Features (Collinear)": (X_tr_s, X_te_s, X_fl_s, 16, 317671.59),
        "Pruned 8 Features": (X_tr_8, X_te_8, X_fl_8, 8, 18.44),
        "PCA — 4 Components (95.0%)": (X_tr_pca4, X_te_pca4, X_fl_pca4, 4, 1.0),
        "PCA — 2 Components (81.9%)": (X_tr_pca2, X_te_pca2, X_fl_pca2, 2, 1.0),
    }

    results = []
    for r_name, (tr_data, te_data, fl_data, n_feats, max_vif) in regimes.items():
        dt = DecisionTreeClassifier(max_depth=10, min_samples_leaf=5, random_state=random_state)
        dt.fit(tr_data, y_train)
        dt_f1 = float(f1_score(y_test, dt.predict(te_data), average="macro", zero_division=0))

        rule = DecisionTreeClassifier(max_depth=4, random_state=random_state)
        rule.fit(tr_data, y_train)
        rule_f1 = float(f1_score(y_test, rule.predict(te_data), average="macro", zero_division=0))

        gnb = GaussianNB()
        gnb.fit(tr_data, y_train)
        gnb_f1 = float(f1_score(y_test, gnb.predict(te_data), average="macro", zero_division=0))

        knn = KNeighborsClassifier(n_neighbors=7)
        knn.fit(tr_data, y_train)
        knn_f1 = float(f1_score(y_test, knn.predict(te_data), average="macro", zero_division=0))

        km = KMeans(n_clusters=7, random_state=random_state, n_init=5)
        km_labels = km.fit_predict(fl_data)
        km_ari = float(adjusted_rand_score(y_full, km_labels))
        km_sil = float(silhouette_score(fl_data, km_labels, sample_size=3000, random_state=random_state))

        results.append({
            "Feature_Regime": r_name,
            "Dimensionality": n_feats,
            "Max_VIF": max_vif,
            "DT_Macro_F1": dt_f1,
            "Rule_Macro_F1": rule_f1,
            "GNB_Macro_F1": gnb_f1,
            "KNN_Macro_F1": knn_f1,
            "KMeans_ARI": km_ari,
            "KMeans_Silhouette": km_sil,
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv("results/tables/stress_test_feature_reduction.csv", index=False)
    return results_df


def plot_stress_test_feature_reduction(df_res, output_prefix="results/figures/fig16_feature_reduction_stress"):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), dpi=300)

    regimes_short = ["All 16", "Pruned 8", "PCA-4 (95%)", "PCA-2 (82%)"]
    x_pos = np.arange(len(df_res))
    bar_width = 0.2

    axes[0].bar(x_pos - 1.5 * bar_width, df_res["DT_Macro_F1"], width=bar_width, label="Decision Tree", color="#2980b9", alpha=0.9)
    axes[0].bar(x_pos - 0.5 * bar_width, df_res["KNN_Macro_F1"], width=bar_width, label="k-NN (k=7)", color="#27ae60", alpha=0.9)
    axes[0].bar(x_pos + 0.5 * bar_width, df_res["GNB_Macro_F1"], width=bar_width, label="GaussianNB", color="#e67e22", alpha=0.9)
    axes[0].bar(x_pos + 1.5 * bar_width, df_res["Rule_Macro_F1"], width=bar_width, label="Rule-Based", color="#8e44ad", alpha=0.9)
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(regimes_short, fontsize=9.5, fontweight="bold")
    axes[0].set_ylabel("Classification Macro-F1", fontsize=10, fontweight="bold")
    axes[0].set_title("Classifier Performance vs Feature Representation (SRQ4)", fontsize=11, fontweight="bold")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(frameon=True, fontsize=8.5)

    axes[1].plot(x_pos, df_res["KMeans_ARI"], marker="o", color="#c0392b", linewidth=2.2, markersize=7, label="k-Means ARI")
    axes[1].plot(x_pos, df_res["KMeans_Silhouette"], marker="s", color="#16a085", linewidth=2.2, markersize=7, label="k-Means Silhouette")
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(regimes_short, fontsize=9.5, fontweight="bold")
    axes[1].set_ylabel("Clustering Validation Score", fontsize=10, fontweight="bold")
    axes[1].set_title("k-Means Clustering Geometry under Feature Compression (SRQ4)", fontsize=11, fontweight="bold")
    axes[1].legend(frameon=True, fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{output_prefix}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_prefix}.pdf", bbox_inches="tight")
    plt.close()


def build_master_comparative_table(clf_comp, miss_df, out_df, scal_df, feat_df):
    clean_dt = float(clf_comp[clf_comp["Classifier"] == "Decision Tree (Tuned)"]["Macro_F1"].values[0])
    clean_knn = float(clf_comp[clf_comp["Classifier"] == "k-Nearest Neighbors (k=7)"]["Macro_F1"].values[0])
    clean_gnb = float(clf_comp[clf_comp["Classifier"] == "Gaussian Naive Bayes"]["Macro_F1"].values[0])
    clean_rule = float(clf_comp[clf_comp["Classifier"] == "Interpretable Rule-Based (Depth 4)"]["Macro_F1"].values[0])

    miss_dt_drop = clean_dt - float(miss_df[(miss_df["Missing_Fraction"] == 0.30) & (miss_df["Imputer"] == "KNN")]["DT_Macro_F1"].values[0])
    miss_knn_drop = clean_knn - float(miss_df[(miss_df["Missing_Fraction"] == 0.30) & (miss_df["Imputer"] == "KNN")]["KNN_Macro_F1"].values[0])
    miss_gnb_drop = clean_gnb - float(miss_df[(miss_df["Missing_Fraction"] == 0.30) & (miss_df["Imputer"] == "KNN")]["GNB_Macro_F1"].values[0])
    miss_rule_drop = clean_rule - float(miss_df[(miss_df["Missing_Fraction"] == 0.30) & (miss_df["Imputer"] == "KNN")]["Rule_Macro_F1"].values[0])

    out_dt_drop = clean_dt - float(out_df[out_df["Outlier_Rate"] == 0.20]["DT_Macro_F1"].values[0])
    out_knn_drop = clean_knn - float(out_df[out_df["Outlier_Rate"] == 0.20]["KNN_Macro_F1"].values[0])
    out_gnb_drop = clean_gnb - float(out_df[out_df["Outlier_Rate"] == 0.20]["GNB_Macro_F1"].values[0])
    out_rule_drop = clean_rule - float(out_df[out_df["Outlier_Rate"] == 0.20]["Rule_Macro_F1"].values[0])

    clean_scal = scal_df[scal_df["Regime"] == "Clean (0% Outliers)"]
    scal_dt_raw_drop = float(clean_scal[clean_scal["Scaler"] == "StandardScaler"]["DT_Macro_F1"].values[0]) - float(clean_scal[clean_scal["Scaler"] == "Raw (Unscaled)"]["DT_Macro_F1"].values[0])
    scal_knn_raw_drop = float(clean_scal[clean_scal["Scaler"] == "StandardScaler"]["KNN_Macro_F1"].values[0]) - float(clean_scal[clean_scal["Scaler"] == "Raw (Unscaled)"]["KNN_Macro_F1"].values[0])
    scal_gnb_raw_drop = float(clean_scal[clean_scal["Scaler"] == "StandardScaler"]["GNB_Macro_F1"].values[0]) - float(clean_scal[clean_scal["Scaler"] == "Raw (Unscaled)"]["GNB_Macro_F1"].values[0])

    feat_dt_8_diff = float(feat_df[feat_df["Feature_Regime"] == "Pruned 8 Features"]["DT_Macro_F1"].values[0]) - clean_dt
    feat_knn_8_diff = float(feat_df[feat_df["Feature_Regime"] == "Pruned 8 Features"]["KNN_Macro_F1"].values[0]) - clean_knn
    feat_gnb_8_diff = float(feat_df[feat_df["Feature_Regime"] == "Pruned 8 Features"]["GNB_Macro_F1"].values[0]) - clean_gnb
    feat_rule_8_diff = float(feat_df[feat_df["Feature_Regime"] == "Pruned 8 Features"]["Rule_Macro_F1"].values[0]) - clean_rule

    master_records = [
        {"Model_Algorithm": "Decision Tree (Tuned)", "Paradigm": "Axis-Aligned Tree", "Clean_Baseline_Macro_F1": clean_dt, "Drop_at_30pct_MCAR": miss_dt_drop, "Drop_at_20pct_Outliers": out_dt_drop, "Raw_Scale_Distortion_Drop": scal_dt_raw_drop, "Pruned_8Feat_Diff": feat_dt_8_diff, "Primary_Vulnerability": "High variance under deep trees", "Optimal_Stress_Condition": "Scale-invariant, robust to monotonic noise"},
        {"Model_Algorithm": "k-Nearest Neighbors (k=7)", "Paradigm": "Instance-Based Distance", "Clean_Baseline_Macro_F1": clean_knn, "Drop_at_30pct_MCAR": miss_knn_drop, "Drop_at_20pct_Outliers": out_knn_drop, "Raw_Scale_Distortion_Drop": scal_knn_raw_drop, "Pruned_8Feat_Diff": feat_knn_8_diff, "Primary_Vulnerability": "Extreme raw scale disparity, distance hijacking", "Optimal_Stress_Condition": "Z-score or RobustScaler with low outliers"},
        {"Model_Algorithm": "Gaussian Naive Bayes", "Paradigm": "Probabilistic Density", "Clean_Baseline_Macro_F1": clean_gnb, "Drop_at_30pct_MCAR": miss_gnb_drop, "Drop_at_20pct_Outliers": out_gnb_drop, "Raw_Scale_Distortion_Drop": scal_gnb_raw_drop, "Pruned_8Feat_Diff": feat_gnb_8_diff, "Primary_Vulnerability": "Collinearity violating conditional independence", "Optimal_Stress_Condition": "De-correlated features (PCA or pruned 8-feature subset)"},
        {"Model_Algorithm": "Rule-Based Classifier", "Paradigm": "Interpretable IF-THEN Rules", "Clean_Baseline_Macro_F1": clean_rule, "Drop_at_30pct_MCAR": miss_rule_drop, "Drop_at_20pct_Outliers": out_rule_drop, "Raw_Scale_Distortion_Drop": 0.0, "Pruned_8Feat_Diff": feat_rule_8_diff, "Primary_Vulnerability": "Underfitting overlapping varieties", "Optimal_Stress_Condition": "High-interpretability human audit requirements"},
    ]
    master_df = pd.DataFrame(master_records)
    master_df.to_csv("results/tables/master_comparative_analysis.csv", index=False)
    return master_df


def merge_metrics_summary(existing_path, new_metrics_dict):
    with open(existing_path, "r", encoding="utf-8") as f:
        master_metrics = json.load(f)

    for k, v in new_metrics_dict.items():
        master_metrics[k] = v

    with open(existing_path, "w", encoding="utf-8") as f:
        json.dump(master_metrics, f, indent=2)

    return master_metrics
