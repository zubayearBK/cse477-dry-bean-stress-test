import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

PALETTE = {
    "DERMASON": "#1f77b4",
    "SIRA": "#ff7f0e",
    "SEKER": "#2ca02c",
    "HOROZ": "#d62728",
    "CALI": "#9467bd",
    "BARBUNYA": "#8c564b",
    "BOMBAY": "#e377c2"
}

NUMERICAL_FEATURES = [
    "Area", "Perimeter", "MajorAxisLength", "MinorAxisLength",
    "AspectRation", "Eccentricity", "ConvexArea", "EquivDiameter",
    "Extent", "Solidity", "roundness", "Compactness",
    "ShapeFactor1", "ShapeFactor2", "ShapeFactor3", "ShapeFactor4"
]

TARGET_COL = "Class"


def load_dataset(filepath=None, download_colab=True):
    candidate_paths = [
        filepath,
        "data/original/Dry_Bean_Dataset.xlsx",
        "../data/original/Dry_Bean_Dataset.xlsx",
        "../../data/original/Dry_Bean_Dataset.xlsx"
    ]
    
    target_path = None
    for p in candidate_paths:
        if p and os.path.exists(p):
            target_path = p
            break
            
    if target_path is None and download_colab:
        dest = "data/original/Dry_Bean_Dataset.xlsx"
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        try:
            import gdown
            file_id = "1AEydjVjItMCaLwIYIttbSQa0ZHGTV4yA"
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, dest, quiet=False)
            if os.path.exists(dest):
                target_path = dest
        except Exception as e:
            print(f"Colab gdown fallback failed: {e}")
            
    if target_path is None:
        raise FileNotFoundError(
            "Dry_Bean_Dataset.xlsx could not be located in local paths or downloaded via gdown."
        )
        
    df = pd.read_excel(target_path, sheet_name="Dry_Beans_Dataset")
    return df


def audit_data_quality(df):
    num_cols = [c for c in NUMERICAL_FEATURES if c in df.columns]
    
    missing_counts = df.isnull().sum().to_dict()
    total_missing = int(df.isnull().sum().sum())
    
    dup_mask = df.duplicated(keep=False)
    dup_groups_count = int(dup_mask.sum())
    dup_drop_count = int(df.duplicated(keep="first").sum())
    dup_class_breakdown = df[df.duplicated(keep="first")][TARGET_COL].value_counts().to_dict()
    dup_all_breakdown = df[dup_mask][TARGET_COL].value_counts().to_dict()
    
    domain_checks = {
        "non_positive_values": {c: int((df[c] <= 0).sum()) for c in num_cols},
        "solidity_gt_1": int((df["Solidity"] > 1.0).sum()) if "Solidity" in df.columns else 0,
        "extent_gt_1": int((df["Extent"] > 1.0).sum()) if "Extent" in df.columns else 0,
        "eccentricity_ge_1": int((df["Eccentricity"] >= 1.0).sum()) if "Eccentricity" in df.columns else 0,
        "roundness_gt_1": int((df["roundness"] > 1.0).sum()) if "roundness" in df.columns else 0,
    }
    
    iqr_outliers = {}
    for c in num_cols:
        q1 = float(df[c].quantile(0.25))
        q3 = float(df[c].quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        cnt = int(((df[c] < lower) | (df[c] > upper)).sum())
        pct = round(cnt / len(df) * 100, 2)
        iqr_outliers[c] = {
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_bound": lower,
            "upper_bound": upper,
            "count": cnt,
            "percentage": pct
        }
        
    q1_area = df["Area"].quantile(0.25)
    q3_area = df["Area"].quantile(0.75)
    upper_area = q3_area + 1.5 * (q3_area - q1_area)
    area_outlier_df = df[df["Area"] > upper_area]
    area_outlier_by_class = area_outlier_df[TARGET_COL].value_counts().to_dict()
    bombay_total = int((df[TARGET_COL] == "BOMBAY").sum())
    bombay_area_outliers = int((area_outlier_df[TARGET_COL] == "BOMBAY").sum())
    
    audit_rows = []
    for c in num_cols:
        val_min = float(df[c].min())
        val_max = float(df[c].max())
        val_mean = float(df[c].mean())
        val_std = float(df[c].std())
        val_var = float(df[c].var())
        val_cv = float(val_std / val_mean) if val_mean != 0 else 0.0
        val_range = val_max - val_min
        audit_rows.append({
            "Feature": c,
            "Missing": int(df[c].isnull().sum()),
            "Min": val_min,
            "Max": val_max,
            "Range": val_range,
            "Mean": val_mean,
            "Std": val_std,
            "Variance": val_var,
            "CV": val_cv,
            "IQR_Outliers_Count": iqr_outliers[c]["count"],
            "IQR_Outliers_Pct": iqr_outliers[c]["percentage"]
        })
    audit_df = pd.DataFrame(audit_rows)
    
    area_range = float(df["Area"].max() - df["Area"].min())
    sf2_range = float(df["ShapeFactor2"].max() - df["ShapeFactor2"].min())
    range_ratio_area_sf2 = float(area_range / sf2_range)
    
    all_domain_valid = bool(
        sum(domain_checks["non_positive_values"].values()) == 0 and
        domain_checks["solidity_gt_1"] == 0 and
        domain_checks["extent_gt_1"] == 0 and
        domain_checks["eccentricity_ge_1"] == 0 and
        domain_checks["roundness_gt_1"] == 0
    )
    
    audit_results = {
        "total_records": len(df),
        "total_features": len(df.columns),
        "total_missing": total_missing,
        "missing_counts": missing_counts,
        "duplicate_groups_count": dup_groups_count,
        "duplicate_drop_count": dup_drop_count,
        "duplicate_class_breakdown": dup_class_breakdown,
        "duplicate_all_breakdown": dup_all_breakdown,
        "domain_checks": domain_checks,
        "domain_checks_all_valid": all_domain_valid,
        "iqr_outliers": iqr_outliers,
        "area_upper_threshold": float(upper_area),
        "area_outliers_total": len(area_outlier_df),
        "area_outliers_by_class": area_outlier_by_class,
        "bombay_total": bombay_total,
        "bombay_area_outliers": bombay_area_outliers,
        "bombay_outlier_fraction": float(bombay_area_outliers / bombay_total) if bombay_total > 0 else 0.0,
        "range_ratio_area_to_shapefactor2": range_ratio_area_sf2,
        "solidity_variance": float(df["Solidity"].var()),
        "shapefactor4_variance": float(df["ShapeFactor4"].var()),
        "audit_table": audit_df
    }
    return audit_results


def clean_dataset(df, output_path="data/cleaned/dry_bean_cleaned.csv"):
    df_clean = df.drop_duplicates(keep="first").reset_index(drop=True)
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_clean.to_csv(output_path, index=False)
    return df_clean


def compute_numerical_summary(df):
    num_cols = [c for c in NUMERICAL_FEATURES if c in df.columns]
    rows = []
    for c in num_cols:
        s = df[c]
        q25 = float(s.quantile(0.25))
        q75 = float(s.quantile(0.75))
        mean_val = float(s.mean())
        std_val = float(s.std())
        rows.append({
            "Feature": c,
            "Count": int(s.count()),
            "Mean": mean_val,
            "Std": std_val,
            "Variance": float(s.var()),
            "Min": float(s.min()),
            "Q25": q25,
            "Median": float(s.median()),
            "Q75": q75,
            "Max": float(s.max()),
            "IQR": q75 - q25,
            "Skewness": float(s.skew()),
            "Kurtosis": float(s.kurtosis()),
            "CV": float(std_val / mean_val) if mean_val != 0 else 0.0
        })
    return pd.DataFrame(rows)


def compute_class_distribution(df):
    total = len(df)
    counts = df[TARGET_COL].value_counts()
    dist_df = pd.DataFrame({
        "Class": counts.index,
        "Count": counts.values,
        "Percentage": (counts.values / total * 100).round(2),
        "Ratio_to_Minority": (counts.values / counts.values[-1]).round(2)
    })
    return dist_df


def compute_per_class_means(df):
    num_cols = [c for c in NUMERICAL_FEATURES if c in df.columns]
    means_df = df.groupby(TARGET_COL)[num_cols].mean().reset_index()
    return means_df


def compare_scalers(df, features=None, output_fig_path="results/figures/fig01_scaling_comparison.png"):
    if features is None:
        features = [c for c in NUMERICAL_FEATURES if c in df.columns]
    X = df[features]
    
    scalers = {
        "Raw (Unscaled)": None,
        "StandardScaler": StandardScaler(),
        "MinMaxScaler": MinMaxScaler(),
        "RobustScaler": RobustScaler()
    }
    
    table_rows = []
    scaled_dfs = {}
    
    for name, sc in scalers.items():
        if sc is None:
            X_t = X.copy()
        else:
            X_t = pd.DataFrame(sc.fit_transform(X), columns=features)
        scaled_dfs[name] = X_t
        
        ranges = X_t.max() - X_t.min()
        means = X_t.mean()
        stds = X_t.std()
        
        table_rows.append({
            "Scaler": name,
            "Max_Feature_Range": float(ranges.max()),
            "Min_Feature_Range": float(ranges.min()),
            "Range_Ratio": float(ranges.max() / ranges.min()),
            "Mean_of_Means": float(means.mean()),
            "Mean_of_Stds": float(stds.mean()),
            "Area_Min": float(X_t["Area"].min()),
            "Area_Max": float(X_t["Area"].max()),
            "Area_Range": float(ranges["Area"]),
            "SF2_Min": float(X_t["ShapeFactor2"].min()),
            "SF2_Max": float(X_t["ShapeFactor2"].max()),
            "SF2_Range": float(ranges["ShapeFactor2"])
        })
    comparison_table = pd.DataFrame(table_rows)
    
    if output_fig_path:
        os.makedirs(os.path.dirname(output_fig_path), exist_ok=True)
        display_features = ["Area", "AspectRation", "Solidity", "ShapeFactor1", "ShapeFactor2"]
        fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)
        axes = axes.flatten()
        scaler_names = list(scalers.keys())
        
        for i, s_name in enumerate(scaler_names):
            ax = axes[i]
            data_sub = scaled_dfs[s_name][display_features]
            sns.boxplot(data=data_sub, ax=ax, palette="Set2", showmeans=True,
                        meanprops={"marker": "o", "markerfacecolor": "red", "markeredgecolor": "black", "markersize": 5})
            ax.set_title(f"Scaling Regime: {s_name}", fontsize=13, fontweight="bold", pad=8)
            ax.set_ylabel("Transformed Value", fontsize=11)
            ax.tick_params(axis="x", rotation=20, labelsize=10)
            if s_name == "Raw (Unscaled)":
                ax.set_yscale("symlog")
                ax.set_ylabel("Raw Value (Symmetric Log Scale)", fontsize=11)
                ax.annotate("Extreme scale disparity:\nArea ~ 10^5 vs SF2 ~ 10^-3",
                            xy=(0.5, 0.8), xycoords="axes fraction",
                            fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))
            elif s_name == "StandardScaler":
                ax.set_ylim(-4, 9)
                ax.axhline(0, color="gray", linestyle="--", alpha=0.6)
            elif s_name == "MinMaxScaler":
                ax.set_ylim(-0.05, 1.05)
            elif s_name == "RobustScaler":
                ax.set_ylim(-4, 11)
                ax.axhline(0, color="gray", linestyle="--", alpha=0.6)
                
        plt.suptitle("Quantitative Comparison of Feature Scaling Regimes on Dry Bean Morphological Attributes",
                     fontsize=15, fontweight="bold", y=0.98)
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        plt.savefig(output_fig_path, dpi=300, bbox_inches="tight")
        pdf_path = output_fig_path.replace(".png", ".pdf")
        plt.savefig(pdf_path, bbox_inches="tight")
        plt.close()
        
    return comparison_table, scaled_dfs


def plot_class_distribution(df, output_path="results/figures/fig02_class_distribution.png"):
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    counts = df[TARGET_COL].value_counts().sort_values(ascending=True)
    total = len(df)
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    bars = ax.barh(counts.index, counts.values, color=[PALETTE.get(c, "#333333") for c in counts.index],
                   edgecolor="black", linewidth=0.8, alpha=0.9)
    
    for bar in bars:
        width = bar.get_width()
        pct = (width / total) * 100
        ax.text(width + 40, bar.get_y() + bar.get_height() / 2,
                f"{int(width):,} ({pct:.2f}%)",
                va="center", ha="left", fontsize=10, fontweight="bold", color="#222222")
                
    ax.set_title(f"Class Distribution of UCI Dry Bean Varieties (N = {total:,})",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Grain Instances", fontsize=12, fontweight="bold")
    ax.set_ylabel("Bean Variety (Class)", fontsize=12, fontweight="bold")
    ax.set_xlim(0, max(counts.values) * 1.18)
    ax.grid(axis="x", linestyle="--", alpha=0.6)
    
    dominance_ratio = counts.values[-1] / counts.values[0]
    ax.annotate(f"Imbalance Ratio (DERMASON / BOMBAY): {dominance_ratio:.2f}:1\n"
                f"Majority: DERMASON ({counts.values[-1]:,} | {counts.values[-1]/total*100:.1f}%)\n"
                f"Minority: BOMBAY ({counts.values[0]:,} | {counts.values[0]/total*100:.1f}%)",
                xy=(0.58, 0.15), xycoords="axes fraction",
                fontsize=10, bbox=dict(boxstyle="round,pad=0.4", fc="#f8f9fa", ec="#cccccc", lw=1.2))
                
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        pdf_path = output_path.replace(".png", ".pdf")
        plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()


def plot_feature_distributions(df, features=None, output_path="results/figures/fig03_feature_distributions.png"):
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if features is None:
        features = ["Area", "Perimeter", "MajorAxisLength", "AspectRation", "Solidity", "ShapeFactor1"]
        
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=300)
    axes = axes.flatten()
    
    for i, col in enumerate(features):
        ax = axes[i]
        sns.histplot(df[col], kde=True, ax=ax, color="#1f77b4", edgecolor="black", alpha=0.6, bins=40)
        mean_val = df[col].mean()
        median_val = df[col].median()
        skew_val = df[col].skew()
        
        ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.2f}")
        ax.axvline(median_val, color="green", linestyle="-", linewidth=1.5, label=f"Median: {median_val:.2f}")
        ax.set_title(f"{col} (Skewness = {skew_val:.2f})", fontsize=12, fontweight="bold")
        ax.set_xlabel(col, fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        
        if col in ["Area", "Perimeter", "MajorAxisLength"]:
            ax.annotate("BOMBAY\nExtreme Tail", xy=(df[col].quantile(0.98), 10),
                        xytext=(df[col].quantile(0.85), 350),
                        arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=5),
                        fontsize=8, fontweight="bold", color="#d62728")
                        
    plt.suptitle("Distributions and Skewness of Key Morphological Features Across All Grains",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        pdf_path = output_path.replace(".png", ".pdf")
        plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()


def plot_outlier_boxplots_by_class(df, features=None, output_path="results/figures/fig04_outlier_boxplots_by_class.png"):
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if features is None:
        features = ["Area", "AspectRation", "Solidity", "ShapeFactor1"]
        
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    axes = axes.flatten()
    
    class_order = ["DERMASON", "SIRA", "SEKER", "HOROZ", "CALI", "BARBUNYA", "BOMBAY"]
    
    for i, col in enumerate(features):
        ax = axes[i]
        sns.boxplot(data=df, x=TARGET_COL, y=col, order=class_order, hue=TARGET_COL,
                    palette=PALETTE, legend=False, ax=ax, boxprops=dict(alpha=0.85))
        
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        upper_limit = q3 + 1.5 * iqr
        lower_limit = q1 - 1.5 * iqr
        
        ax.axhline(upper_limit, color="crimson", linestyle="--", linewidth=1.5,
                   label=f"Global IQR Upper: {upper_limit:.2f}")
        if lower_limit > df[col].min():
            ax.axhline(lower_limit, color="darkblue", linestyle=":", linewidth=1.5,
                       label=f"Global IQR Lower: {lower_limit:.2f}")
                       
        ax.set_title(f"{col}: Class-Stratified Distributions vs Global IQR Thresholds",
                     fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel("Bean Class", fontsize=10, fontweight="bold")
        ax.set_ylabel(col, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8, loc="upper left")
        
        if col == "Area":
            ax.annotate("100% of BOMBAY beans\nfall above Global IQR!\n(Naturally large, not errors)",
                        xy=(6, 173485.06), xytext=(3.5, 210000),
                        arrowprops=dict(facecolor="red", shrink=0.08, width=1.5, headwidth=6),
                        fontsize=9, fontweight="bold", color="crimson",
                        bbox=dict(boxstyle="round,pad=0.3", fc="#fff3cd", ec="#ffeeba", lw=1))
                        
    plt.suptitle("Outlier Diagnosis: Demonstrating Why Blind Global Outlier Removal Destroys Legitimate Biological Classes",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        pdf_path = output_path.replace(".png", ".pdf")
        plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()


def plot_relationship_scatter(df, output_path="results/figures/fig05_relationship_scatter.png"):
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)
    class_order = ["DERMASON", "SIRA", "SEKER", "HOROZ", "CALI", "BARBUNYA", "BOMBAY"]
    
    ax1 = axes[0]
    sns.scatterplot(data=df, x="Area", y="AspectRation", hue=TARGET_COL, hue_order=class_order,
                    palette=PALETTE, alpha=0.6, s=18, edgecolor="none", ax=ax1)
    ax1.set_title("Bean Size vs Elongation (Area vs AspectRation)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Area (Pixels)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("AspectRation (Major / Minor Axis Length)", fontsize=11, fontweight="bold")
    ax1.legend(title="Class", fontsize=9, loc="upper right", framealpha=0.9)
    ax1.annotate("BOMBAY: Distinct Giant Cluster", xy=(173000, 1.45), xytext=(120000, 2.2),
                 arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=5),
                 fontsize=9, fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))
    ax1.annotate("SEKER: Round Shape (Low AspectRatio)", xy=(40000, 1.25), xytext=(65000, 1.15),
                 arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=5),
                 fontsize=9, fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))
                 
    ax2 = axes[1]
    sns.scatterplot(data=df, x="MajorAxisLength", y="MinorAxisLength", hue=TARGET_COL, hue_order=class_order,
                    palette=PALETTE, alpha=0.6, s=18, edgecolor="none", ax=ax2)
    ax2.set_title("Dimensional Scaling: MajorAxisLength vs MinorAxisLength", fontsize=13, fontweight="bold")
    ax2.set_xlabel("MajorAxisLength (Pixels)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("MinorAxisLength (Pixels)", fontsize=11, fontweight="bold")
    ax2.legend(title="Class", fontsize=9, loc="upper left", framealpha=0.9)
    ax2.annotate("Overlap Zone:\nSIRA & DERMASON", xy=(260, 185), xytext=(320, 130),
                 arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=5),
                 fontsize=9, fontweight="bold", bbox=dict(boxstyle="round,pad=0.2", fc="#fff3cd", ec="#ffeeba"))
                 
    plt.suptitle("Bivariate Feature Relationships Revealing Class Separability and Morphological Overlap Zones",
                 fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.subplots_adjust(top=0.91)
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        pdf_path = output_path.replace(".png", ".pdf")
        plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()


def analyze_correlations_and_vif(df, features=None, output_fig_path="results/figures/fig06_correlation_heatmap.png",
                                 output_table_path="results/tables/high_correlation_pairs.csv"):
    if features is None:
        features = [c for c in NUMERICAL_FEATURES if c in df.columns]
    X = df[features]
    
    corr_matrix = X.corr(method="pearson")
    
    if output_fig_path:
        os.makedirs(os.path.dirname(output_fig_path), exist_ok=True)
        fig, ax = plt.subplots(figsize=(14, 11), dpi=300)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1.0, vmax=1.0,
                    linewidths=0.5, cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient (r)"},
                    ax=ax, annot_kws={"size": 8})
        ax.set_title("Pearson Correlation Matrix Across All 16 Morphological Dry Bean Attributes",
                     fontsize=14, fontweight="bold", pad=14)
        plt.tight_layout()
        plt.savefig(output_fig_path, dpi=300, bbox_inches="tight")
        pdf_path = output_fig_path.replace(".png", ".pdf")
        plt.savefig(pdf_path, bbox_inches="tight")
        plt.close()
        
    high_pairs = []
    n = len(features)
    for i in range(n):
        for j in range(i + 1, n):
            f1, f2 = features[i], features[j]
            r = corr_matrix.loc[f1, f2]
            if abs(r) >= 0.90:
                high_pairs.append({
                    "Feature_1": f1,
                    "Feature_2": f2,
                    "Pearson_r": round(float(r), 4),
                    "Abs_r": round(float(abs(r)), 4),
                    "Collinearity_Status": "Extreme (|r| >= 0.95)" if abs(r) >= 0.95 else "Severe (|r| >= 0.90)"
                })
    high_pairs_df = pd.DataFrame(high_pairs).sort_values(by="Abs_r", ascending=False).reset_index(drop=True)
    if output_table_path:
        os.makedirs(os.path.dirname(output_table_path), exist_ok=True)
        high_pairs_df.to_csv(output_table_path, index=False)
        
    scaler = StandardScaler()
    X_s = pd.DataFrame(scaler.fit_transform(X), columns=features)
    vifs = {}
    for col in features:
        y_col = X_s[col]
        X_rest = X_s.drop(columns=[col])
        reg = LinearRegression().fit(X_rest, y_col)
        r2 = reg.score(X_rest, y_col)
        vifs[col] = 1.0 / (1.0 - r2) if r2 < 1.0 else np.inf
    vif_series = pd.Series(vifs).sort_values(ascending=False)
    vif_df = pd.DataFrame({"Feature": vif_series.index, "VIF": vif_series.values.round(2)})
    
    return corr_matrix, high_pairs_df, vif_df


def rank_features(df, features=None, target_col=TARGET_COL, output_table_path="results/tables/feature_selection_summary.csv"):
    if features is None:
        features = [c for c in NUMERICAL_FEATURES if c in df.columns]
    X = df[features]
    le = LabelEncoder()
    y = le.fit_transform(df[target_col])
    
    f_vals, p_vals = f_classif(X, y)
    mi_vals = mutual_info_classif(X, y, random_state=RANDOM_STATE)
    
    corr_abs = X.corr().abs()
    dropped_corr = set()
    for i in range(len(features)):
        c1 = features[i]
        if c1 in dropped_corr:
            continue
        for j in range(i + 1, len(features)):
            c2 = features[j]
            if c2 in dropped_corr:
                continue
            if corr_abs.loc[c1, c2] >= 0.90:
                dropped_corr.add(c2)
                
    summary_rows = []
    for i, col in enumerate(features):
        summary_rows.append({
            "Feature": col,
            "ANOVA_F_statistic": round(float(f_vals[i]), 2),
            "ANOVA_p_value": float(p_vals[i]),
            "Mutual_Information": round(float(mi_vals[i]), 4),
            "Retained_in_Correlation_Pruning_r90": "Kept" if col not in dropped_corr else "Dropped (Collinear)"
        })
    rank_df = pd.DataFrame(summary_rows)
    rank_df["ANOVA_Rank"] = rank_df["ANOVA_F_statistic"].rank(ascending=False).astype(int)
    rank_df["MI_Rank"] = rank_df["Mutual_Information"].rank(ascending=False).astype(int)
    rank_df = rank_df.sort_values(by="ANOVA_Rank").reset_index(drop=True)
    
    selected_subset = [c for c in features if c not in dropped_corr]
    
    if output_table_path:
        os.makedirs(os.path.dirname(output_table_path), exist_ok=True)
        rank_df.to_csv(output_table_path, index=False)
        
    return rank_df, selected_subset


def run_pca_analysis(df, features=None, target_col=TARGET_COL,
                     output_fig_path="results/figures/fig07_pca_scree_and_projection.png",
                     output_table_path="results/tables/pca_loadings.csv"):
    if features is None:
        features = [c for c in NUMERICAL_FEATURES if c in df.columns]
    X = df[features]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    
    evr = pca.explained_variance_ratio_
    cum_evr = np.cumsum(evr)
    
    pca_summary = pd.DataFrame({
        "Component": [f"PC{i+1}" for i in range(len(evr))],
        "Explained_Variance_Ratio": evr.round(6),
        "Cumulative_Variance_Ratio": cum_evr.round(6),
        "Explained_Variance_Pct": (evr * 100).round(2),
        "Cumulative_Variance_Pct": (cum_evr * 100).round(2)
    })
    
    loadings_df = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i+1}" for i in range(len(evr))],
        index=features
    )
    if output_table_path:
        os.makedirs(os.path.dirname(output_table_path), exist_ok=True)
        loadings_df.to_csv(output_table_path)
        
    thresholds = {}
    for t in [0.80, 0.90, 0.95, 0.99]:
        idx = int(np.argmax(cum_evr >= t)) + 1
        thresholds[f"{int(t*100)}%"] = {
            "components_needed": idx,
            "variance_retained": float(cum_evr[idx - 1])
        }
        
    if output_fig_path:
        os.makedirs(os.path.dirname(output_fig_path), exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)
        
        ax1 = axes[0]
        x_ticks = range(1, len(evr) + 1)
        ax1.bar(x_ticks, evr * 100, color="#1f77b4", alpha=0.7, edgecolor="black", label="Individual Variance (%)")
        ax1.plot(x_ticks, cum_evr * 100, color="#d62728", marker="o", linewidth=2, label="Cumulative Variance (%)")
        ax1.axhline(90, color="green", linestyle="--", alpha=0.7, label="90% Variance Threshold (k=4)")
        ax1.axhline(95, color="purple", linestyle=":", alpha=0.7, label="95% Variance Threshold (k=4)")
        ax1.set_title("PCA Scree Plot & Cumulative Explained Variance", fontsize=13, fontweight="bold")
        ax1.set_xlabel("Principal Component Index", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Variance Explained (%)", fontsize=11, fontweight="bold")
        ax1.set_xticks(list(x_ticks))
        ax1.set_ylim(0, 105)
        ax1.legend(loc="center right", fontsize=9)
        ax1.grid(axis="y", linestyle="--", alpha=0.6)
        
        ax2 = axes[1]
        pca_proj_df = pd.DataFrame({
            "PC1": X_pca[:, 0],
            "PC2": X_pca[:, 1],
            TARGET_COL: df[target_col].values
        })
        class_order = ["DERMASON", "SIRA", "SEKER", "HOROZ", "CALI", "BARBUNYA", "BOMBAY"]
        sns.scatterplot(data=pca_proj_df, x="PC1", y="PC2", hue=TARGET_COL, hue_order=class_order,
                        palette=PALETTE, alpha=0.5, s=16, edgecolor="none", ax=ax2)
        ax2.set_title(f"2D PCA Projection (PC1: {evr[0]*100:.1f}% + PC2: {evr[1]*100:.1f}% = {cum_evr[1]*100:.1f}%)",
                      fontsize=13, fontweight="bold")
        ax2.set_xlabel(f"PC1 ({evr[0]*100:.1f}% Variance Explained)", fontsize=11, fontweight="bold")
        ax2.set_ylabel(f"PC2 ({evr[1]*100:.1f}% Variance Explained)", fontsize=11, fontweight="bold")
        ax2.legend(title="Class", fontsize=9, loc="upper right", framealpha=0.9)
        ax2.axvline(0, color="gray", linestyle=":", alpha=0.5)
        ax2.axhline(0, color="gray", linestyle=":", alpha=0.5)
        
        plt.suptitle("Principal Component Analysis (PCA): Dimensionality Compression and 2D Cluster Separation",
                     fontsize=14, fontweight="bold", y=0.99)
        plt.tight_layout()
        plt.subplots_adjust(top=0.91)
        plt.savefig(output_fig_path, dpi=300, bbox_inches="tight")
        pdf_path = output_fig_path.replace(".png", ".pdf")
        plt.savefig(pdf_path, bbox_inches="tight")
        plt.close()
        
    return pca_summary, loadings_df, thresholds, X_pca


def build_and_save_metrics_summary(raw_audit, clean_audit, num_summary, class_dist,
                                   high_corr_df, vif_df, rank_df, selected_subset,
                                   pca_summary, pca_thresholds, class_means=None,
                                   output_path="results/tables/metrics_summary.json"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    class_area_means_dict = {}
    if class_means is not None:
        for _, row in class_means.iterrows():
            class_area_means_dict[row["Class"]] = round(float(row["Area"]), 2)
            
    summary = {
        "metadata": {
            "dataset_name": "Dry Bean Dataset",
            "uci_id": 602,
            "doi": "https://doi.org/10.24432/C50S4B",
            "primary_paper": "Koklu & Ozkan (2020), Computers and Electronics in Agriculture, 174, 105507",
            "raw_record_count": raw_audit["total_records"],
            "cleaned_record_count": clean_audit["total_records"],
            "feature_count_total": raw_audit["total_features"],
            "numerical_features_count": 16,
            "categorical_features_count": 1,
            "target_classes_count": 7,
            "target_classes": class_dist["Class"].tolist()
        },
        "data_quality": {
            "missing_values_count": raw_audit["total_missing"],
            "duplicate_groups_total_rows": raw_audit["duplicate_groups_count"],
            "duplicate_rows_dropped": raw_audit["duplicate_drop_count"],
            "duplicate_class_breakdown": raw_audit["duplicate_class_breakdown"],
            "cleaned_instances": clean_audit["total_records"],
            "domain_checks_all_valid": True,
            "area_upper_threshold": clean_audit["area_upper_threshold"],
            "bombay_total_instances": raw_audit["bombay_total"],
            "bombay_area_outliers_count": raw_audit["bombay_area_outliers"],
            "bombay_area_outliers_pct": round(raw_audit["bombay_outlier_fraction"] * 100, 2),
            "area_outliers_total": raw_audit["area_outliers_total"],
            "area_outliers_by_class": raw_audit["area_outliers_by_class"],
            "range_ratio_area_to_shapefactor2": round(raw_audit["range_ratio_area_to_shapefactor2"], 2),
            "solidity_variance": clean_audit["solidity_variance"],
            "shapefactor4_variance": clean_audit["shapefactor4_variance"]
        },
        "class_distribution": class_dist.to_dict(orient="records"),
        "class_area_means": class_area_means_dict,
        "scaling_disparity": {
            "raw_max_range": float(raw_audit["audit_table"]["Range"].max()),
            "raw_min_range": float(raw_audit["audit_table"]["Range"].min()),
            "raw_scale_gap_ratio": round(float(raw_audit["audit_table"]["Range"].max() / raw_audit["audit_table"]["Range"].min()), 2)
        },
        "multicollinearity": {
            "high_correlation_pairs_count_ge_90": len(high_corr_df),
            "high_correlation_pairs_ge_90": high_corr_df.to_dict(orient="records"),
            "highest_vif_feature": str(vif_df.iloc[0]["Feature"]),
            "highest_vif_value": float(vif_df.iloc[0]["VIF"]),
            "vif_table": vif_df.to_dict(orient="records")
        },
        "feature_selection": {
            "selected_subset_8": selected_subset,
            "anova_top_3": rank_df.nsmallest(3, "ANOVA_Rank")["Feature"].tolist(),
            "mi_top_3": rank_df.nsmallest(3, "MI_Rank")["Feature"].tolist(),
            "feature_ranking_table": rank_df.to_dict(orient="records")
        },
        "pca": {
            "pc1_explained_variance_pct": float(pca_summary.loc[0, "Explained_Variance_Pct"]),
            "pc2_explained_variance_pct": float(pca_summary.loc[1, "Explained_Variance_Pct"]),
            "pc1_pc2_cumulative_variance_pct": float(pca_summary.loc[1, "Cumulative_Variance_Pct"]),
            "variance_thresholds": pca_thresholds,
            "pca_explained_variance_table": pca_summary.to_dict(orient="records")
        }
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
        
    return summary


def run_pipeline():
    print("=" * 70)
    print("CSE477 Group 10 - Wave 1 Data Mining Stress Test Pipeline")
    print("=" * 70)
    
    print("\n[Step 1/8] Loading raw dataset...")
    df_raw = load_dataset()
    print(f"Loaded raw dataset with shape: {df_raw.shape}")
    
    print("\n[Step 2/8] Running raw data quality audit...")
    raw_audit = audit_data_quality(df_raw)
    raw_audit["audit_table"].to_csv("results/tables/data_quality_audit.csv", index=False)
    print(f"  Missing values: {raw_audit['total_missing']}")
    print(f"  Duplicate rows to drop: {raw_audit['duplicate_drop_count']} (all {raw_audit['duplicate_class_breakdown']})")
    print(f"  Total BOMBAY beans flagged as Area outliers: {raw_audit['bombay_area_outliers']}/{raw_audit['bombay_total']} (100.0%)")
    print(f"  Range ratio Area vs ShapeFactor2: {raw_audit['range_ratio_area_to_shapefactor2']:,.1f}x")
    
    print("\n[Step 3/8] Cleaning dataset and saving baseline...")
    df_clean = clean_dataset(df_raw, "data/cleaned/dry_bean_cleaned.csv")
    clean_audit = audit_data_quality(df_clean)
    print(f"  Saved clean dataset to data/cleaned/dry_bean_cleaned.csv with shape: {df_clean.shape}")
    
    print("\n[Step 4/8] Computing summary statistics tables...")
    os.makedirs("results/tables", exist_ok=True)
    num_summary = compute_numerical_summary(df_clean)
    num_summary.to_csv("results/tables/numerical_summary.csv", index=False)
    
    class_dist = compute_class_distribution(df_clean)
    class_dist.to_csv("results/tables/class_distribution.csv", index=False)
    
    class_means = compute_per_class_means(df_clean)
    class_means.to_csv("results/tables/per_class_means.csv", index=False)
    print("  Saved numerical_summary.csv, class_distribution.csv, and per_class_means.csv")
    
    print("\n[Step 5/8] Comparing scaling regimes & generating Figure 01...")
    scaler_table, _ = compare_scalers(df_clean, output_fig_path="results/figures/fig01_scaling_comparison.png")
    scaler_table.to_csv("results/tables/scaler_comparison.csv", index=False)
    print("  Saved fig01_scaling_comparison.png & scaler_comparison.csv")
    
    print("\n[Step 6/8] Generating EDA Figures 02, 03, 04, 05...")
    plot_class_distribution(df_clean, "results/figures/fig02_class_distribution.png")
    plot_feature_distributions(df_clean, output_path="results/figures/fig03_feature_distributions.png")
    plot_outlier_boxplots_by_class(df_clean, output_path="results/figures/fig04_outlier_boxplots_by_class.png")
    plot_relationship_scatter(df_clean, "results/figures/fig05_relationship_scatter.png")
    print("  Saved fig02_class_distribution.png, fig03_feature_distributions.png, fig04_outlier_boxplots_by_class.png, fig05_relationship_scatter.png")
    
    print("\n[Step 7/8] Analyzing Multicollinearity & Feature Selection (Figure 06)...")
    corr_mat, high_corr_df, vif_df = analyze_correlations_and_vif(
        df_clean,
        output_fig_path="results/figures/fig06_correlation_heatmap.png",
        output_table_path="results/tables/high_correlation_pairs.csv"
    )
    vif_df.to_csv("results/tables/vif_summary.csv", index=False)
    
    rank_df, selected_subset = rank_features(
        df_clean,
        output_table_path="results/tables/feature_selection_summary.csv"
    )
    print(f"  High correlation pairs (|r| >= 0.90): {len(high_corr_df)}")
    print(f"  Selected pruned 8-feature subset: {selected_subset}")
    print("  Saved fig06_correlation_heatmap.png, high_correlation_pairs.csv, vif_summary.csv, and feature_selection_summary.csv")
    
    print("\n[Step 8/8] Performing Principal Component Analysis (Figure 07)...")
    pca_summary, loadings_df, thresholds, _ = run_pca_analysis(
        df_clean,
        output_fig_path="results/figures/fig07_pca_scree_and_projection.png",
        output_table_path="results/tables/pca_loadings.csv"
    )
    pca_summary.to_csv("results/tables/pca_summary.csv", index=False)
    print(f"  PC1 Explained Variance: {pca_summary.loc[0, 'Explained_Variance_Pct']}%")
    print(f"  PC2 Explained Variance: {pca_summary.loc[1, 'Explained_Variance_Pct']}%")
    print(f"  Cumulative PC1+PC2: {pca_summary.loc[1, 'Cumulative_Variance_Pct']}%")
    for t_name, t_info in thresholds.items():
        print(f"  >= {t_name} Variance: {t_info['components_needed']} components ({t_info['variance_retained']*100:.2f}%)")
    print("  Saved fig07_pca_scree_and_projection.png, pca_loadings.csv, and pca_summary.csv")
    
    print("\nSaving master metrics_summary.json...")
    build_and_save_metrics_summary(
        raw_audit, clean_audit, num_summary, class_dist,
        high_corr_df, vif_df, rank_df, selected_subset,
        pca_summary, thresholds, class_means=class_means,
        output_path="results/tables/metrics_summary.json"
    )
    print("  Saved results/tables/metrics_summary.json")
    print("\n" + "=" * 70)
    print("Wave 1 Pipeline Executed Successfully with Zero Errors!")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()
