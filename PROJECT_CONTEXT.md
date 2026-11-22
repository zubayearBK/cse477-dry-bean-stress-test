---

# CSE477 — Data Mining: Final Research Project Specification

| Field | Details |
| :--- | :--- |
| **University** | East West University — Department of Computer Science and Engineering |
| **Course** | CSE477 — Data Mining |
| **Faculty** | Sanzana Karim Lora, Lecturer, CSE |
| **Semester & Section** | Fall 2026 \| Section: 4 |
| **Assigned Group & Theme** | **Group 10 — "The Data Mining Stress Test"** (*How Much Can a Data-Mining Model Survive?*) |
| **Proposed Research Title** | *Surviving Degradation: A Systematic Robustness and Stress-Test Investigation of Clustering, Classification, and Similarity Models on Agricultural Computer Vision Data* |
| **Presenters** | **Syed Zubayear Bin Khaled** — `2023-2-60-107` <br> **Md. Arifur Rahman Shihab** — `2023-2-60-176` |
| **Primary Dataset** | UCI Dry Bean Dataset ($N = 13,611$ instances, $16$ numerical features, $7$ bean classes) |

---

## 1. Purpose of This Context Document

This document is the **single source of truth** for all AI coding agents, report-writing agents, and human contributors working on Group 10's CSE477 Data Mining project. Every Python script, Jupyter/Colab notebook (`notebooks/project_analysis.ipynb`), saved figure, summary table, presentation slide, and LaTeX report section **must strictly comply** with the specifications, experimental philosophy, data-crediting requirements, and rubric checkpoints defined herein.

### Core Philosophy (Official Guideline Golden Rule)
> **"Do not start with an algorithm. Start with a question. Do not ask only 'Which algorithm gives the highest score?' Instead ask: 'Under what conditions does this method reveal useful patterns, and why?'"**

Our mission as **Group 10 (The Data Mining Stress Test)** is to treat the clean Dry Bean dataset as a controlled scientific baseline ($100\%$ data quality) and systematically subject standard data-mining pipelines (similarity analysis, clustering, classification, and discretized association mining) to controlled data-quality degradations:
1. **Missing Value Stress** (MCAR injection at $5\%$, $10\%$, $20\%$, and $30\%$, evaluated across imputation strategies).
2. **Outlier & Noise Stress** (Extreme value corruption / Gaussian perturbation at $5\%$, $10\%$, and $20\%$ of records).
3. **Feature Reduction & Multicollinearity Stress** (Full $16$ features vs. correlation-pruned subset vs. Mutual Information top-$k$ vs. PCA at $95\%$, $90\%$, and $80\%$ retained variance).
4. **Normalization & Scaling Stress** (Unnormalized Raw Data vs. Min-Max Scaling vs. Z-Score Standardization vs. RobustScaler).

The flagship scientific deliverable is a suite of **Data-Mining Robustness Curves** quantifying exact performance degradation trajectories across algorithms and exposing *which methods fail gracefully, which collapse catastrophically, and why*.

---

## 2. Dataset Metadata & Proper Academic Crediting

> **Agent Requirement:** The full dataset attribution block below **must** appear at the top of `notebooks/project_analysis.ipynb`, `README.md`, and in Section 4 (*Dataset Description*) and *References* of the LaTeX report.

### 2.1 Official Citations
* **Dataset Repository Citation (UCI):**
  Dry Bean [Dataset]. (2020). UCI Machine Learning Repository. https://doi.org/10.24432/C50S4B.
* **Primary Research Paper Citation (Mandatory Acknowledgement per Original Authors):**
  Koklu, M., & Ozkan, I. A. (2020). Multiclass classification of dry beans using computer vision and machine learning techniques. *Computers and Electronics in Agriculture*, 174, 105507. https://doi.org/10.1016/j.compag.2020.105507
* **Original Authors & Provenance:**
  * **Murat Koklu** — Faculty of Technology, Selcuk University, Turkey (ORCID: `0000-0002-2737-2360`)
  * **Ilker Ali Ozkan** — Faculty of Technology, Selcuk University, Turkey (ORCID: `0000-0002-5715-1040`)
* **Official Repository Link:** https://archive.ics.uci.edu/dataset/602/dry+bean+dataset
* **Team Google Drive Mirror:** https://docs.google.com/spreadsheets/d/1AEydjVjItMCaLwIYIttbSQa0ZHGTV4yA/edit?usp=sharing&ouid=111969229142560713820&rtpof=true&sd=true

### 2.2 Dataset Summary Table (Guideline Stage 2 Compliance)

| Property | Specification |
| :--- | :--- |
| **Dataset Name** | Dry Bean Dataset |
| **Domain / Area** | Agricultural Computer Vision / CS & Engineering |
| **Data Object (1 Row)** | One individual segmented dry bean grain captured via high-resolution camera |
| **Number of Records ($N$)** | $13,611$ instances |
| **Number of Attributes ($d$)** | $17$ total ($16$ predictive numerical features + $1$ categorical target class) |
| **Numerical Attributes** | $16$ ($2$ discrete pixel counts: `Area`, `ConvexArea`; $14$ continuous real-valued geometric & shape metrics) |
| **Categorical Attributes** | $1$ (`Class` — Nominal, $7$ registered varieties) |
| **Target Classes ($C = 7$)** | `SEKER`, `BARBUNYA`, `BOMBAY`, `CALI`, `DERMASON`, `HOROZ`, `SIRA` *(verify exact casing programmatically from `.xlsx`)* |
| **Original Missing Values** | $0$ (Clean baseline—ideal for controlled missingness stress testing!) |
| **Known Quality Notes** | Check programmatically for exact duplicate rows (historically $\approx 68$ duplicates exist in UCI Dry Bean), extreme class imbalance (`BOMBAY` $\ll$ `DERMASON`), severe multicollinearity among size features, and naming discrepancy (`Dermosan` in `.txt` metadata vs. `DERMASON` in `.xlsx`). |
| **Dataset Limitations** | Controlled lab lighting/segmentation (lacks real-world field occlusion/dirt), no color/texture/spectral features (purely 2D silhouette geometry), severe multicollinearity among derived shape factors, and imbalanced class counts. |

### 2.3 Complete Attribute Dictionary & Data-Mining Types (Guideline Stage 3 Compliance)

| # | Attribute | Symbol | Type | Mathematical / Physical Meaning | Role & Collinearity Notes |
| :---: | :--- | :---: | :--- | :--- | :--- |
| 1 | `Area` | $A$ | Ratio (Discrete/Integer) | Pixel count within bean boundary | Core size; heavily collinear with `ConvexArea`, `EquivDiameter`, `Perimeter` |
| 2 | `Perimeter` | $P$ | Ratio (Continuous) | Length of bean circumference | Core size; scale $\sim 10^3$ |
| 3 | `MajorAxisLength` | $L$ | Ratio (Continuous) | Longest line drawable across bean | Primary elongation axis |
| 4 | `MinorAxisLength` | $l$ | Ratio (Continuous) | Longest line perpendicular to major axis | Secondary width axis |
| 5 | `AspectRation` | $K$ | Ratio (Continuous) | $K = L / l$ (note exact column spelling in `.xlsx`!) | Scale-invariant elongation ratio |
| 6 | `Eccentricity` | $E_c$ | Ratio (Continuous) | Eccentricity of ellipse with identical second moments | Bounded $[0, 1)$; strongly non-linearly linked to $K$ |
| 7 | `ConvexArea` | $C$ | Ratio (Discrete/Integer) | Pixels in smallest convex polygon containing bean | Nearly identical to `Area` ($r > 0.99$) |
| 8 | `EquivDiameter` | $E_d$ | Ratio (Continuous) | $E_d = \sqrt{4A / \pi}$ | Deterministic transform of `Area` |
| 9 | `Extent` | $E_x$ | Ratio (Continuous) | Ratio of pixels in bounding box to bean area | Orientation-sensitive bounding efficiency |
| 10 | `Solidity` | $S$ | Ratio (Continuous) | $S = A / C$ (convexity ratio) | Very low variance (clustered near $0.98 - 0.995$) |
| 11 | `roundness` | $R$ | Ratio (Continuous) | $R = \frac{4\pi A}{P^2}$ | Scale-invariant circularity measure |
| 12 | `Compactness` | $CO$ | Ratio (Continuous) | $CO = E_d / L$ | Measures circular compactness |
| 13 | `ShapeFactor1` | $SF_1$ | Ratio (Continuous) | Derived shape ratio ($L / A$) | Extreme small scale ($\sim 10^{-3}$)—vulnerable without normalization! |
| 14 | `ShapeFactor2` | $SF_2$ | Ratio (Continuous) | Derived shape ratio ($l / A$) | Extreme small scale ($\sim 10^{-3}$) |
| 15 | `ShapeFactor3` | $SF_3$ | Ratio (Continuous) | Derived shape ratio ($A / (\frac{\pi}{4} L^2) = CO^2$) | Nearly collinear with `Compactness` |
| 16 | `ShapeFactor4` | $SF_4$ | Ratio (Continuous) | Derived shape ratio involving minor/major axes | Very low variance (clustered near $0.99$) |
| 17 | `Class` | $y$ | Nominal (Categorical) | Bean variety label ($7$ classes) | Supervised target; strictly **excluded** from unsupervised clustering & distance features (prevent leakage!) |

> **Important Schema Check for Agents:** Always inspect `df.columns.tolist()` first. The official UCI Excel file uses exact spellings such as `AspectRation` (missing 'i') and lowercase `roundness`. Standardize or reference exact column names programmatically without hardcoding mismatched names.

---

## 3. Research Questions (Guideline Stages 1 & 19 Compliance)

### Primary Stress-Test Research Questions (Group 10 Specific)
* **SRQ1 (Missing Data Tolerance):** How rapidly do unsupervised clustering quality (Silhouette, ARI) and supervised classification performance (Macro-F1, Accuracy) degrade under $5\%$, $10\%$, $20\%$, and $30\%$ MCAR missingness, and how effectively do median versus KNN imputation buffer this collapse?
* **SRQ2 (Outlier & Noise Vulnerability):** To what extent do injected extreme outliers ($5\%$, $10\%$, $20\%$) distort centroid-based clustering ($k$-Means) and distance-based classification ($k$-NN) compared to density-based (DBSCAN), hierarchical, tree-based (Decision Tree), and probabilistic (Gaussian Naïve Bayes) methods?
* **SRQ3 (Scale & Normalization Sensitivity):** Because raw feature magnitudes span seven orders of magnitude (`Area` $\sim 10^5$ vs. `ShapeFactor2` $\sim 10^{-3}$), how drastically do Min-Max, Z-Score, and RobustScaler alter Euclidean vs. Manhattan nearest-neighbor neighborhoods, clustering geometry, and model robustness under outliers?
* **SRQ4 (Dimensionality & Collinearity Stress):** Given severe geometric multicollinearity ($r > 0.98$), does removing redundant features via correlation pruning or PCA improve robustness and efficiency, or does aggressive compression degrade discrimination among morphologically similar bean varieties (`SIRA`, `DERMASON`, `HOROZ`)?

### Mandatory CSE477 Final Research Questions (Must Be Explicitly Answered in Notebook & Report)
* **RQ1:** What are the major patterns present in the Dry Bean dataset?
* **RQ2:** Which geometric and shape attributes appear to be most informative?
* **RQ3:** What meaningful groups or classes can be identified, and where do morphological boundaries overlap?
* **RQ4:** Which data-mining method provides the most useful insight and highest structural resilience for this stress-test problem, and why?

---

## 4. Reproducible Repository Architecture

All agents must read from and write strictly to the following directory layout (100% compliant with CSE477 Section 22, extended for LaTeX reporting and presentations):

```text
cse477-group10-stress-test/
├── data/
│   ├── original/
│   │   ├── Dry_Bean_Dataset.xlsx          # Untouched raw dataset
│   │   └── Dry_Bean_Dataset.txt           # Original UCI metadata text
│   ├── cleaned/
│   │   ├── dry_bean_cleaned.csv           # Deduplicated, validated baseline dataset
│   │   └── dry_bean_discretized.csv       # Quantile-binned dataset for Association Mining
│   └── stress_snapshots/                  # Optional reproducible perturbed test snapshots (gitignored if large)
├── notebooks/
│   └── project_analysis.ipynb             # Complete executable master notebook (Colab + Local compatible)
├── src/
│   ├── __init__.py
│   ├── data_loader.py                     # Multi-source loader (Local -> GDrive -> UCI fallback)
│   ├── preprocessing.py                   # Cleaning, validation, scaling, feature selection
│   ├── stress_injector.py                 # Deterministic MCAR missingness, outlier, & noise generators
│   └── analysis.py                        # Complete CLI-executable reproduction script (Guideline Sec 22)
├── results/
│   ├── figures/                           # High-res 300 DPI PNG & PDF figures for LaTeX & slides
│   │   ├── fig01_class_distribution.png
│   │   ├── fig02_feature_distributions.png
│   │   ├── fig03_correlation_heatmap.png
│   │   ├── fig04_pca_scree_and_biplot.png
│   │   ├── fig05_distance_metric_agreement.png
│   │   ├── fig06_clustering_selection_and_comparison.png
│   │   ├── fig07_classification_confusion_matrices.png
│   │   ├── fig08_association_rules_lift.png
│   │   ├── fig09_stress_curve_missingness.png
│   │   ├── fig10_stress_curve_outliers.png
│   │   ├── fig11_stress_curve_normalization.png
│   │   └── fig12_stress_curve_feature_reduction.png
│   └── tables/                            # Machine-readable CSV + LaTeX (.tex) summary tables
│       ├── summary_statistics.csv
│       ├── data_quality_audit.csv
│       ├── clustering_comparison.csv
│       ├── classification_comparison.csv
│       ├── association_rules_top.csv
│       ├── stress_test_master_results.csv
│       └── metrics_summary.json           # Single verified JSON for LaTeX report grounding
├── report/
│   ├── latex/
│   │   ├── main.tex                       # Master LaTeX report source
│   │   ├── references.bib                 # BibTeX bibliography
│   │   └── sections/                      # Optional modular .tex chapters
│   └── project_report.pdf                 # Final compiled research PDF (Guideline Sec 22 & 30)
├── presentation/
│   └── group10_stress_test_slides.pdf     # 12-slide final presentation (Guideline Sec 24)
├── docs/
│   ├── PROJECT_CONTEXT.md                 # This master context document
│   ├── SETUP_AND_WORKFLOW_GUIDE.md        # Setup, GitHub, Colab, and Checkpoint playbook
│   └── CSE477_Final_Project_Guideline.docx
├── requirements.txt                       # Pinned Python dependencies
├── .gitignore
└── README.md                              # Reproducibility guide (Guideline Sec 22 & 30)
```

---

## 5. Complete Technical & Experimental Specification (Notebook & `src/analysis.py`)

### 5.1 Engineering Rules & Reproducibility Constraints
1. **Global Seed:** Fix `RANDOM_STATE = 42` everywhere (`np.random.seed(42)`, `random_state=42` in all splits, imputers, PCA, $k$-Means, Trees, and stress injectors).
2. **Environment-Agnostic Data Loading:** The notebook must run seamlessly on **both local machines and Google Colab**:
   * Priority 1: Read `../data/original/Dry_Bean_Dataset.xlsx` (or `./data/original/Dry_Bean_Dataset.xlsx`).
   * Priority 2: If missing (e.g., fresh Colab session), automatically download from the team's Google Drive link via `gdown` (`id=1AEydjVjItMCaLwIYIttbSQa0ZHGTV4yA`) into `data/original/Dry_Bean_Dataset.xlsx`.
   * Priority 3: Fallback to `ucimlrepo` (`fetch_ucirepo(id=602)`).
3. **Zero Data Leakage:**
   * All Train/Test splits (`80/20` stratified by `Class`) occur **before** fitting scalers, imputers, PCA, or feature selectors.
   * For **Stress Tests on Classification**, evaluate **two clearly stated regimes** (or justify the chosen regime rigorously):
     * *Corruption Regime A (Training + Test Corruption):* Simulates training and deploying on degraded sensor pipelines (both training and evaluation data suffer missingness/outliers, with imputer/scaler fit strictly on corrupted train split).
     * *Corruption Regime B (Test-Time / Inference Drift, optional contrast):* Clean model evaluated on corrupted test inputs. *(Primary robustness curves should use Regime A for end-to-end pipeline stress testing, and optionally include a focused Regime B comparison for extra analytical depth!)*
4. **Memory & Compute Safety (Local CPU Friendly):**
   * $N = 13,611$ runs effortlessly on local laptops for $k$-Means, DBSCAN, Decision Trees, Naïve Bayes, $k$-NN, PCA, and FP-Growth.
   * **Exception — Hierarchical Clustering & Full Pairwise Distance Matrices:** Computing a dense $13,611 \times 13,611$ distance matrix requires $\approx 1.5\text{ GB}$ RAM per matrix. Therefore:
     * For **Distance/Similarity Neighbor Agreement (Stage 8)**: Use a stratified representative sample ($n = 2,000$) or `sklearn.neighbors.NearestNeighbors` (KD-tree/ball-tree exact $k$-NN without materializing full $N \times N$ matrices).
     * For **Agglomerative Hierarchical Clustering (Stage 9)**: Run Ward / Average linkage on a stratified sample of $n = 3,500$ (preserving exact class proportions, `random_state=42`) to render clean dendrograms and compute Silhouette/ARI swiftly in $< 5$ seconds on any laptop, while explicitly documenting this computational justification!
5. **High-Quality Visualization Standard:**
   * Use clean matplotlib/seaborn academic styling (`dpi=300`, readable fonts $\ge 11\text{pt}$, colorblind-friendly palette like `viridis`/`tab10`/`Set2`, clear axis labels with units, descriptive titles, legends outside plots when crowded).
   * Save every key figure automatically to `results/figures/` in `.png` (300 DPI) **and** `.pdf` (vector for LaTeX!).
6. **Mandatory Markdown Interpretation:** Every code section producing a table or figure in `project_analysis.ipynb` **must** be followed immediately by an analytical Markdown cell interpreting the result (never leave raw outputs unexplained).

### 5.2 Exact 14-Section Notebook Blueprint (100% Mapped to Guideline Stages 1–13)

| Notebook Section | Guideline Stage | Rubric Marks | Required Analyses & Outputs |
| :--- | :---: | :---: | :--- |
| **0. Header, Group Metadata & Dataset Attribution** | Sec 22, 27 | — | Professionally formatted course/group table, full UCI + Koklu & Ozkan (2020) citations, environment check, smart data loader. |
| **1. Problem Definition & Research Questions** | Stage 1 | 8 | Clear stress-test problem statement, SRQ1–SRQ4 + Mandatory RQ1–RQ4, experimental framework diagram/summary. |
| **2. Dataset Selection & Data Understanding** | Stages 2–3 | 8 | Automated schema verification, attribute type classification table (Nominal/Discrete/Continuous), identifier/leakage audit, collinearity preview. |
| **3. Data Quality Audit** | Stage 4 | 12 *(shared)* | Check missing values, **exact duplicates** (report exact count & inspect), inconsistent class labels/casing, physical domain validity checks (e.g., $A > 0$, $0 < S \le 1$, $0 < R \le 1$, $0 \le E_c < 1$), IQR & Z-score outlier audit per feature, low-variance audit (`Solidity`, `ShapeFactor4`). Export `data_quality_audit.csv`. |
| **4. Data Preparation & Justification** | Stage 5 | 12 *(shared)* | Deduplication (save `data/cleaned/dry_bean_cleaned.csv`), label encoding (`LabelEncoder` saved cleanly), justification-driven scaling setup (`StandardScaler`, `MinMaxScaler`, `RobustScaler`), stratified `80/20` train-test split. Explicit *why* for every step. |
| **5. Exploratory Data Analysis (EDA)** | Stage 6 | 10 | Full numerical stats table (mean, median, std, min, max, skew, IQR $\rightarrow$ `summary_statistics.csv`); categorical frequency & proportion table (highlighting `DERMASON` dominance vs. `BOMBAY` rarity); **$\ge 4$ interpreted visualizations**: (a) Class distribution bar chart, (b) Boxplots/Violin distributions across classes, (c) Pairwise scatter/relationship plots (`Area` vs `AspectRation`, `Compactness` vs `Roundness`), (d) Correlation heatmap. |
| **6. Feature Selection & Dimensionality Reduction** | Stage 7 | 8 | (a) Multicollinearity correlation pruning ($|r| > 0.95$ audit), (b) Mutual Information & ANOVA F-score ranking (answering RQ2), (c) PCA analysis with Scree Plot, cumulative variance table ($80\%, 90\%, 95\%$ thresholds), 2D PCA projection colored by class, loadings interpretation, and explicit discussion of PCA interpretability trade-offs. |
| **7. Similarity & Distance Metric Analysis** | Stage 8 | 8 | Compare **Euclidean ($L_2$)**, **Manhattan ($L_1$)**, and **Cosine / Chebyshev** distances on $k=10$ nearest neighbors. Compute Jaccard overlap of top-$k$ neighbor sets AND top-$k$ same-class purity across: (i) Raw unnormalized vs. Z-score standardized data, and (ii) Clean vs. $10\%$ Outlier-injected data. Proves mathematically why unnormalized `Area` hijacks distance! |
| **8. Baseline Unsupervised Clustering** | Stage 9 | 12 | Compare **3 clustering algorithms** on Z-score scaled clean features (without `Class` label): <br>1. **$k$-Means** (Elbow inertia + Silhouette scan $k \in [2, 10]$; justify selected $k$ AND compare optimal geometric $k$ vs. domain $k=7$).<br>2. **Agglomerative Hierarchical Clustering** (Ward & Average linkage on stratified $n=3,500$, Dendrogram + Silhouette/ARI).<br>3. **DBSCAN** ($k$-distance elbow plot for $\varepsilon$ tuning, cluster count + noise/outlier ratio).<br>Profile cluster centroids, compare cross-algorithm agreement, identify which bean varieties merge vs. cleanly separate (e.g., `BOMBAY`). Export `clustering_comparison.csv`. |
| **9. Baseline Supervised Classification** | Stage 10 | 10 | Compare **4 classifiers** (covering all guideline recommendations + distance contrast): <br>1. **Decision Tree** (`DecisionTreeClassifier`, tuned max depth) <br>2. **Gaussian Naïve Bayes** (`GaussianNB`) <br>3. **Rule-Based Classifier** (Extracted human-readable `IF-THEN` rules via pruned Decision Tree / `RIPPER` via `wittgenstein` or `export_text` rule table with rule coverage & precision) <br>4. **$k$-Nearest Neighbors** (`KNeighborsClassifier`, Euclidean vs. Manhattan—crucial for stress testing!) <br>*(Optional 5th benchmark: Random Forest to show ensemble resilience).*<br>Report Accuracy, Macro/Weighted Precision, Recall, F1-score, and Confusion Matrices. Explicitly analyze **easiest class (`BOMBAY`) vs. hardest classes (`SIRA`/`DERMASON`/`BARBUNYA`) and why**. Export `classification_comparison.csv`. |
| **10. Association Pattern Mining (Discretized Morphology)** | Stage 11 | 7 | Select $6$–$8$ low-collinearity interpretable morphological features (`Area`, `AspectRation`, `Solidity`, `roundness`, `Compactness`, `Extent`), discretize via quantile binning (`Low`/`Med`/`High`), combine with `Class` items, and run **FP-Growth / Apriori** (`mlxtend`). Filter class-consequent and morphology-to-morphology rules by Support, Confidence, and Lift ($> 1.5$). Interpret whether rules are frequent, reliable, interesting, and practically useful, while noting discretization trade-offs. Export `association_rules_top.csv`. |
| **11. THE DATA MINING STRESS TEST (Core Group 10 Contribution)** | Theme Core + Stages 12–13 | 7+ *(boosts all sections)* | Systematic perturbation experiments producing **4 Master Robustness Curves**:<br>• **Experiment A — Missing Data Stress ($0\%, 5\%, 10\%, 20\%, 30\%$ MCAR):** Compare Median vs. KNN imputation across $k$-Means (Silhouette, ARI) and all 4 Classifiers (Macro-F1, Accuracy).<br>• **Experiment B — Outlier & Noise Stress ($0\%, 5\%, 10\%, 20\%$ corrupted rows with $5\sigma$–$10\sigma$ spikes):** Evaluate breakdown points of $k$-Means, DBSCAN, $k$-NN, Decision Tree, and Gaussian Naïve Bayes.<br>• **Experiment C — Normalization Stress:** Compare Raw vs. Min-Max vs. Z-Score vs. RobustScaler (both clean and under $10\%$ outliers!) across distance metrics, $k$-Means, and classifiers.<br>• **Experiment D — Feature Reduction Stress:** Compare Full $16$ features vs. Correlation-Pruned vs. MI Top-$6$ vs. PCA ($95\%, 90\%, 80\%$ variance).<br>Export `stress_test_master_results.csv` and `metrics_summary.json`. |
| **12. Comparative Analysis & Synthesis** | Stage 13 | 7 *(shared)* | Cross-method comparison tables (Algorithm vs. Preprocessing vs. Feature Set vs. Distance Metric) with explicit explanations of *why* results diverge (e.g., why Decision Trees survive unnormalized scales and outliers while $k$-Means and $k$-NN collapse; why Gaussian NB fails when outliers inflate variance $\sigma^2$). |
| **13. Five (5+) Interesting Findings** | Sec 18 | 5 | Exactly 5–6 high-impact findings strictly structured as **Observation $\rightarrow$ Evidence (exact table/figure/metric) $\rightarrow$ Interpretation (mathematical & domain rationale)**. |
| **14. Answers to RQ1–RQ4, Limitations & Conclusion** | Sec 19, 23.11 | — | Direct answers to RQ1–RQ4 and SRQ1–SRQ4, honest limitations, and structured takeaways for the final report and 12-slide presentation. |

---

## 6. Evaluation Rubric Alignment Checklist (100 Marks)

| Rubric Component | Marks | Guaranteed Coverage in This Project |
| :--- | :---: | :--- |
| Problem definition & research questions | 8 | Section 1: Stress-test framing, SRQ1–SRQ4 + Mandatory RQ1–RQ4 |
| Dataset selection & understanding | 8 | Sections 0 & 2: Full citations, 17-attribute taxonomy, data-object & leakage audit |
| Data cleaning & preprocessing | 12 | Sections 3 & 4: Duplicates, validity checks, outlier audit, justified scaling & imputation |
| Exploratory data analysis | 10 | Section 5: Complete stats table, imbalance analysis, $\ge 4$ interpreted visualizations |
| Feature selection / reduction | 8 | Section 6: Multicollinearity pruning, Mutual Information, PCA scree/loadings/limitations |
| Similarity / distance analysis | 8 | Section 7: Euclidean vs. Manhattan vs. Cosine neighbor agreement under clean/raw/noisy regimes |
| Clustering | 12 | Section 8 + 11: $k$-Means, Agglomerative Hierarchical, DBSCAN + robustness under stress |
| Classification | 10 | Section 9 + 11: Decision Tree, Naïve Bayes, Rule-Based, $k$-NN + per-class error analysis + stress curves |
| Association pattern mining / additional analysis | 7 | Section 10: Quantile-discretized morphological FP-Growth/Apriori rules with Support/Confidence/Lift + critique |
| Evaluation & comparative analysis | 7 | Sections 11 & 12: Multi-metric evaluation + cross-algorithm robustness matrices |
| Interesting findings & interpretation | 5 | Section 13: 5+ structured Observation–Evidence–Interpretation discoveries |
| Report quality | 3 | Professionally typeset LaTeX PDF (`report/project_report.pdf`) strictly following Sec 23 |
| **Total** | **100** | **100% Compliant** |

---

## 7. Comprehensive Academic Report Writing Instructions (LaTeX)

When generating `report/latex/main.tex` and `report/latex/references.bib` to compile `report/project_report.pdf`, the writing agent **must strictly follow every instruction below**.

### 7.1 Mandatory Report Structure (Exact Match to Guideline Section 23)
1. **Title Page:** University name (East West University, Dept. of CSE), Course (`CSE477: Data Mining`), Semester (`Fall 2026 | Section: 4`), Faculty (`Sanzana Karim Lora, Lecturer, CSE`), Research Title, Group 10 theme, and Presenters with IDs (`Syed Zubayear Bin Khaled — 2023-2-60-107`, `Md. Arifur Rahman Shihab — 2023-2-60-176`).
2. **Abstract (Strictly 150–250 words):** One crisp, self-contained paragraph covering: (i) research problem (model robustness under realistic data degradation), (ii) dataset ($13,611$ Dry Bean instances, $16$ geometric attributes, $7$ classes), (iii) methodology (baseline mining + 4-axis stress testing), (iv) headline quantitative findings (exact verified numbers from `metrics_summary.json`), and (v) main conclusion.
3. **1. Introduction:** Problem background, motivation (why real-world data pipelines degrade via sensor noise, missingness, and scaling drift), why comparative data mining matters beyond accuracy chasing, explicit SRQ1–SRQ4 & RQ1–RQ4, and concise bulleted project contributions.
4. **2. Dataset Description:** Complete academic attribution (UCI DOI + Koklu & Ozkan 2020), structured metadata table, complete 17-attribute taxonomy table with LaTeX formulas ($R = \frac{4\pi A}{P^2}$, $CO = \frac{E_d}{L}$, $K = \frac{L}{l}$), class balance summary, and critical dataset limitations.
5. **3. Data Preprocessing:** Findings from the Data Quality Audit (exact duplicate count, physical range checks, low-variance features), followed by *rationale-first* explanations of deduplication, encoding, stratified splitting, scaling choices, and feature selection.
6. **4. Exploratory Data Analysis:** Summary statistics table, class imbalance discussion, and embedded high-resolution figures (`fig01`–`fig03`) with dedicated analytical commentary beneath every figure.
7. **5. Methodology:** Formal mathematical overview of *only the techniques actually executed*:
   * Distance metrics ($L_2$, $L_1$, Cosine)
   * Dimensionality reduction (Correlation pruning, Mutual Information, PCA)
   * Clustering ($k$-Means, Agglomerative Ward/Average, DBSCAN)
   * Classification (Decision Tree, Gaussian Naïve Bayes, Rule-Based Classifier, $k$-NN)
   * Discretized Association Rule Mining (Support, Confidence, Lift formulas)
   * Stress-Test Experimental Protocol (MCAR missingness, $k\sigma$ outlier injection, normalization swap, feature reduction) and Evaluation Metrics (Silhouette, ARI, Macro-F1, Confusion Matrix).
8. **6. Experimental Results:** Structured presentation of baseline results (Similarity, Clustering, Classification, Association Rules) followed by the **Group 10 Stress-Test Results** featuring the four Robustness Curve figures (`fig09`–`fig12`) and exact degradation tables.
9. **7. Comparative Analysis:** Direct side-by-side synthesis answering *why* methods differ across (a) algorithms, (b) preprocessing/scaling choices, (c) feature sets, and (d) distance metrics.
10. **8. Interesting Findings:** Five (5) clearly numbered, publication-grade findings, each explicitly separated into **Observation**, **Evidence** (referencing exact tables/figures/metrics), and **Interpretation**.
11. **9. Limitations:** Critical reflection on dataset constraints (2D lab silhouettes without color/texture), MCAR vs. MNAR missingness assumptions, synthetic outlier modeling, hierarchical clustering subsampling ($n=3,500$), discretization artifacts in association mining, and generalizability.
12. **10. Conclusion & Future Work:** Concise synthesis explicitly answering RQ1–RQ4, summarizing actionable engineering lessons on model survival under stress, and outlining realistic future extensions (e.g., MNAR missingness, robust ensemble defenses, real-world noisy agricultural imagery).
13. **References:** BibTeX-managed citations covering the UCI Dry Bean dataset, Koklu & Ozkan (2020), Han/Kamber/Pei (*Data Mining: Concepts and Techniques*), Tan/Steinbach/Kumar (*Introduction to Data Mining*), Scikit-learn (Pedregosa et al., 2011), and `mlxtend` (Raschka, 2018).

### 7.2 Preferred Academic Style & LaTeX Formatting Rules
* **Tone & Voice:** Rigorous, objective, publication-grade academic prose (third-person or restrained academic "we"). Avoid conversational filler, hyperbole ("amazingly high", "super accurate"), and circular justifications (*never* write "we normalized because normalization is important").
* **Analytical Writing Pattern:** Follow the **Claim $\rightarrow$ Quantitative Evidence $\rightarrow$ Algorithmic/Domain Mechanism** triad in every analytical paragraph.
  * *Example (Good):* "Under unnormalized raw features, $k$-NN Macro-F1 dropped by $\Delta F_1 = -0.184$ compared to Z-score standardization (Table 5). Because `Area` spans $[20{,}420,\; 254{,}616]$ whereas `ShapeFactor2` spans $[0.0005,\; 0.0037]$, Euclidean distance collapses effectively onto a one-dimensional size ranking, erasing shape cues required to separate `SIRA` from `DERMASON`."
* **Zero-Fabrication Grounding Rule (CRITICAL):** Every single statistic, count, metric, rule lift, and delta reported in `main.tex` **must** be parsed directly from `results/metrics_summary.json` and `results/tables/*.csv` generated by executing `project_analysis.ipynb`. Never invent or approximate numbers.
* **LaTeX Technical Standards:**
  * Document class: `\documentclass[11pt,a4paper]{article}` with `geometry` (`margin=1in`), `booktabs`, `amsmath`, `amssymb`, `graphicx`, `subcaption`, `hyperref`, `microtype`, `float`, `xcolor`, and `natbib` (or `biblatex`).
  * Tables: Use `\toprule`, `\midrule`, `\bottomrule` (never vertical lines `|`). Bold the best/most resilient model in each row/column.
  * Figures: Include vector `.pdf` or 300 DPI `.png` plots from `../../results/figures/` with informative, self-contained captions (`\caption{...}`) and explicit in-text references (`Figure~\ref{fig:...}`).
  * Math: Format all variables, formulas, metrics, and Greek symbols cleanly in LaTeX math mode (e.g., $\varepsilon$, $\sigma$, $k$-Means, $F_1$, $R = \frac{4\pi A}{P^2}$).
  * No raw Python code blocks in the PDF report (keep code in `notebooks/` and `src/`).

---

## 8. Viva & Individual Understanding Prep (Guideline Section 26)

Both **Syed Zubayear Bin Khaled** and **Md. Arifur Rahman Shihab** should be prepared to answer any of these instructor questions directly from our experimental evidence:
1. *Why did you select the Dry Bean dataset for the Stress Test?* $\rightarrow$ Clean baseline ($0$ missing values), large enough ($N=13,611$) for stable statistical degradation curves, extreme natural scale disparity ($10^5$ vs. $10^{-3}$) making normalization stress dramatic, and severe natural multicollinearity ($r > 0.98$) making feature reduction stress meaningful.
2. *Why did you normalize the data, and why compare Z-Score vs. RobustScaler?* $\rightarrow$ `Area` dominates raw Euclidean distances; Z-Score standardizes via mean/std (ideal for clean data) whereas RobustScaler uses median/IQR, resisting the mean/variance distortion caused by $5\sigma$–$10\sigma$ injected outliers.
3. *Why is Macro-F1 lower (or more informative) than Accuracy here?* $\rightarrow$ Class imbalance (`DERMASON` $\approx 26\%$ vs. `BOMBAY` $\approx 3.8\%$); Macro-F1 weights every bean variety equally so collapse on minority/overlapping classes isn't masked by majority-class accuracy.
4. *Why did Decision Trees survive raw scaling and outliers better than $k$-Means, $k$-NN, and Gaussian Naïve Bayes?* $\rightarrow$ Axis-aligned rank-order splits are invariant to monotonic feature scaling and insensitive to extreme tails beyond split thresholds, whereas $k$-Means centroids, Euclidean $k$-NN distances, and Gaussian NB variance estimates ($\sigma^2$) are algebraically pulled by extreme values.
5. *What does Lift mean in your morphological association rules?* $\rightarrow$ $\text{Lift}(A \rightarrow B) = \frac{P(B|A)}{P(B)}$; $\text{Lift} > 1$ proves the morphological profile co-occurs with a bean variety more often than chance baseline class prevalence predicts.