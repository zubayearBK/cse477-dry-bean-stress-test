---

# CSE477 Group 10 — Setup, GitHub, Colab & Halfway Checkpoint Playbook

| Field | Details |
| :--- | :--- |
| **Course** | CSE477 — Data Mining (Fall 2026, Section X) — East West University |
| **Faculty** | Sanzana Karim Lora, Lecturer, CSE |
| **Group** | Group 10 — "The Data Mining Stress Test" |
| **Members** | Syed Zubayear Bin Khaled (`2023-2-60-107`) & Md. Arifur Rahman Shihab (`2023-2-60-176`) |

---

## 1. Can You Run Everything Locally, or Do You Need Google Colab?

### Short Answer: **Yes—100% of this project runs smoothly and quickly on your local laptop!**
Here is the exact computational reality for the Dry Bean dataset ($13,611 \text{ rows} \times 16 \text{ numeric features} \approx 1.5\text{ MB}$):

| Task | Local Laptop CPU (8–16 GB RAM) | Notes & Safeguards Built Into `PROJECT_CONTEXT.md` |
| :--- | :--- | :--- |
| Data Loading, Cleaning, EDA, Plots | $< 5$ seconds | Instantaneous in Pandas/Seaborn |
| PCA, Correlation, Mutual Info | $< 2$ seconds | Fast matrix ops on $13{,}611 \times 16$ |
| $k$-Means, DBSCAN, $k$-NN, Trees, Naïve Bayes | $5$–$25$ seconds total | Fast on CPU; no GPU required |
| **All 4 Stress-Test Sweeps (Full Robustness Curves)** | $\approx 1$–$3$ minutes total | Even with KNN Imputation + multiple seeds/levels |
| **Hierarchical Clustering & Pairwise Distances** | **Potential RAM Spike if naive** | Naive $13{,}611 \times 13{,}611$ matrix uses $\sim 1.5\text{ GB}$ RAM per call. Our context enforces stratified subsampling ($n=3,500$ for Ward dendrograms, $n=2,000$ or KD-tree `NearestNeighbors` for distance agreement) $\rightarrow$ runs in $< 5$ seconds! |
| FP-Growth / Apriori (Discretized) | $< 3$ seconds | Very fast on $6$–$8$ quantile-binned attributes |

### Why You Should Still Connect GitHub $\leftrightarrow$ Google Colab
Even though local execution is fastest for iterating with AI agents (Cursor, Claude Code, VS Code Copilot, Windsurf), connecting your repo to **Google Colab** gives you three major advantages:
1. **Zero-Friction Instructor Verification:** Your faculty (Sanzana Karim Lora) can click a single **"Open in Colab"** badge in your `README.md` and run every cell in the browser without installing Python locally.
2. **Easy Partner Collaboration:** Both you and Shihab can inspect/run the notebook identically regardless of Windows/Mac/Linux differences.
3. **Backup Cloud Compute:** If you ever expand stress tests to 20 Monte Carlo repeats per corruption level, Colab runs it cleanly in the cloud.

---

## 2. Step-by-Step Local Folder Setup

Open your terminal (PowerShell on Windows, or Bash/Zsh on macOS/Linux) in the directory where you keep university projects and run the following commands.

### Option A: macOS / Linux / Git Bash (Windows)
```bash
mkdir -p cse477-group10-stress-test/{data/{original,cleaned,stress_snapshots},notebooks,src,results/{figures,tables},report/latex/sections,presentation,docs}
cd cse477-group10-stress-test
```

### Option B: Windows PowerShell
```powershell
$root = "cse477-group10-stress-test"
$dirs = @(
  "$root/data/original",
  "$root/data/cleaned",
  "$root/data/stress_snapshots",
  "$root/notebooks",
  "$root/src",
  "$root/results/figures",
  "$root/results/tables",
  "$root/report/latex/sections",
  "$root/presentation",
  "$root/docs"
)
$dirs | ForEach-Object { New-Item -ItemType Directory -Force -Path $_ | Out-Null }
Set-Location $root
```

### Place Your Existing Files Immediately
Copy your files into these exact paths:
1. `Dry_Bean_Dataset.xlsx` $\rightarrow$ `data/original/Dry_Bean_Dataset.xlsx`
2. `Dry_Bean_Dataset.txt` $\rightarrow$ `data/original/Dry_Bean_Dataset.txt`
3. `CSE477_Final_Project_Guideline (1).docx` $\rightarrow$ `docs/CSE477_Final_Project_Guideline.docx`
4. `PROJECT_CONTEXT.md` $\rightarrow$ `docs/PROJECT_CONTEXT.md` *(and optionally keep a copy at root as `CLAUDE.md` or `.cursorrules` pointer so your coding agent reads it automatically!)*
5. `SETUP_AND_WORKFLOW_GUIDE.md` $\rightarrow$ `docs/SETUP_AND_WORKFLOW_GUIDE.md`

### Create Your Local Python Virtual Environment
```bash
python -m venv .venv

# Activate on Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# OR Activate on macOS / Linux:
source .venv/bin/activate
```

Create `requirements.txt` in the project root with these pinned core packages:
```text
numpy>=1.26.0
pandas>=2.1.0
openpyxl>=3.1.2
scipy>=1.11.0
scikit-learn>=1.4.0
matplotlib>=3.8.0
seaborn>=0.13.0
mlxtend>=0.23.0
gdown>=5.1.0
ucimlrepo>=0.0.7
jupyter>=1.0.0
ipykernel>=6.29.0
```

Install dependencies and register the kernel for Jupyter:
```bash
pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name=cse477-stress-test --display-name "Python (CSE477 Stress Test)"
```

---

## 3. GitHub Repository Setup & Best Practices

### Step 1: Create `.gitignore` in the Project Root
This keeps your repository clean, prevents pushing giant virtual environments or LaTeX build junk, while **keeping** your datasets, notebook, figures, tables, LaTeX source, and final PDFs safely tracked:

```gitignore
# Python environments & caches
.venv/
venv/
__pycache__/
*.py[cod]
.ipynb_checkpoints/

# Large optional scratch snapshots
data/stress_snapshots/

# LaTeX build artifacts (keep .tex, .bib, and final .pdf!)
report/latex/*.aux
report/latex/*.log
report/latex/*.out
report/latex/*.toc
report/latex/*.bbl
report/latex/*.blg
report/latex/*.fls
report/latex/*.fdb_latexmk
report/latex/*.synctex.gz

# OS & IDE files
.DS_Store
Thumbs.db
.vscode/settings.json
```

### Step 2: Initialize Git & Push to GitHub
1. Go to [github.com/new](https://github.com/new) and create a repository named `cse477-group10-stress-test` (do **not** initialize with README or `.gitignore` on GitHub so there are no merge conflicts).
2. Add your teammate **Md. Arifur Rahman Shihab** as a collaborator under **Settings $\rightarrow$ Collaborators**.
3. Run in your project root terminal:

```bash
git init
git branch -M main
git add .
git commit -m "chore: initialize CSE477 Group 10 Stress Test repository structure and dataset"
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/cse477-group10-stress-test.git
git push -u origin main
```

### Step 3: Clean Git Workflow for Two Teammates
* **Before starting any work session:** `git pull origin main`
* **Commit logically by milestone:**
  * `feat(eda): complete data quality audit, cleaning, and exploratory visualizations`
  * `feat(mining): add distance analysis, baseline clustering, classification, and FP-growth`
  * `feat(stress-test): implement 4-axis degradation sweeps and export robustness curves`
  * `docs(report): draft and compile LaTeX final project report`
* **Pro Tip for Notebooks:** Always clear unnecessary scratch outputs or run **Restart Kernel & Run All Cells** before committing `notebooks/project_analysis.ipynb` so GitHub renders clean, reproducible outputs and charts.

---

## 4. Connecting Your Notebook Between GitHub and Google Colab

You can edit locally with AI agents, push to GitHub, and run/present seamlessly on Google Colab with **two simple mechanisms**:

### A. Opening Your GitHub Notebook Directly in Colab
1. Add this badge to the top of your `README.md` (replace `<YOUR_GITHUB_USERNAME>`):
   ```markdown
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<YOUR_GITHUB_USERNAME>/cse477-group10-stress-test/blob/main/notebooks/project_analysis.ipynb)
   ```
2. Inside the **very first code cell** of `notebooks/project_analysis.ipynb`, have your agent include a smart Colab bootstrap snippet:
   ```python
   import os, sys
   IN_COLAB = 'google.colab' in sys.modules
   if IN_COLAB:
       # Install any packages not pre-installed in Colab
       !pip install -q openpyxl mlxtend gdown ucimlrepo
       # Clone repo if running standalone in Colab so src/ and data/ are available
       if not os.path.exists('cse477-group10-stress-test'):
           !git clone https://github.com/<YOUR_GITHUB_USERNAME>/cse477-group10-stress-test.git
       os.chdir('cse477-group10-stress-test')
   ```
   *(If your repo is **private** during development, the fallback data loader in `PROJECT_CONTEXT.md` still works automatically in Colab by downloading `Dry_Bean_Dataset.xlsx` directly from your public Google Drive link via `gdown`!)*

### B. Saving Edits Made in Colab Back to GitHub
If you ever tweak code directly inside Google Colab:
1. Click **File $\rightarrow$ Save a copy in GitHub**.
2. Authorize Colab with your GitHub account, choose `cse477-group10-stress-test`, branch `main`, and file path `notebooks/project_analysis.ipynb`.
3. Write a concise commit message and click **OK**.
4. Next time you open your laptop, run `git pull origin main` to sync your local folder!

---

## 5. Strategy for Tomorrow's Halfway Checkpoint vs. Final Submission

Since **tomorrow is your halfway checkpoint**, do not try to polish the LaTeX report before your core experiments and numbers are verified. Use this exact two-phase execution plan:

### Phase 1: Ready for Tomorrow's Halfway Checkpoint (Tonight's Goal)
By tomorrow, you want a clean GitHub repo, a runnable notebook, and verified outputs through baseline mining + initial stress-test proof-of-concept:
- [ ] Repository folder structure initialized and pushed to GitHub.
- [ ] **Sections 0–6 Complete:** Metadata/citations, Research Questions, Data Understanding, Data Quality Audit (duplicates/validity/outliers), Cleaning (`data/cleaned/dry_bean_cleaned.csv`), EDA stats + 4 clean figures, Feature Selection & PCA.
- [ ] **Sections 7–9 Complete (Baseline Mining):** Distance comparison ($L_2$ vs. $L_1$ vs. Cosine), Baseline Clustering ($k$-Means, Hierarchical, DBSCAN), and Baseline Classification (Decision Tree, Naïve Bayes, Rule-Based, $k$-NN) with confusion matrices.
- [ ] **Section 11 Preview (At Least 1–2 Stress Curves Ready!):** Missingness curve ($0\%$–$30\%$) and/or Normalization stress comparison—showing your instructor immediately that Group 10's core *Stress Test* mechanism is live and revealing real degradation patterns!

### Phase 2: Complete After Checkpoint (Final Delivery)
- [ ] Finish **Section 10** (Discretized Morphological Association Mining with FP-Growth) and **remaining Stress Curves** (Outliers + Feature Reduction).
- [ ] Generate `src/analysis.py` so the entire pipeline reproduces end-to-end from CLI (`python src/analysis.py`).
- [ ] Verify `results/metrics_summary.json` and all 12 high-DPI figures in `results/figures/`.
- [ ] Draft and compile `report/latex/main.tex` $\rightarrow$ `report/project_report.pdf` (using Overleaf or local TeX Live / MiKTeX).
- [ ] Build the 12-slide presentation (`presentation/group10_stress_test_slides.pdf`) following Guideline Section 24.
- [ ] Finalize `README.md`, run **Restart & Run All** on `notebooks/project_analysis.ipynb`, and commit final release to GitHub.

---

## 6. Copy-Paste Prompts to Drive Your AI Agents

Feed `docs/PROJECT_CONTEXT.md` to your coding agent (Cursor, Claude Code, Windsurf, Copilot, etc.) in **four disciplined waves** rather than one giant prompt. This prevents truncated notebooks and hallucinated metrics.

### Agent Prompt 1 — Build Modular `src/` + Notebook Sections 0 to 6 (EDA & Feature Reduction)
```text
Read docs/PROJECT_CONTEXT.md completely before writing any code.
First, create the modular helper modules in src/ (data_loader.py, preprocessing.py, stress_injector.py) following all engineering constraints (RANDOM_STATE=42, zero leakage, local+Colab+GDrive fallback loader, exact column-name inspection on data/original/Dry_Bean_Dataset.xlsx).
Second, create notebooks/project_analysis.ipynb starting with the formatted CSE477 Group 10 header and full UCI + Koklu & Ozkan (2020) dataset attribution, and implement Sections 0 through 6 (Problem Definition, Data Understanding, Data Quality Audit, Data Preparation, EDA, and Feature Selection/PCA).
Run the notebook to verify zero errors, export data/cleaned/dry_bean_cleaned.csv, results/tables/summary_statistics.csv, results/tables/data_quality_audit.csv, and save 300 DPI PNG+PDF figures (fig01 through fig04) into results/figures/. Write thorough analytical Markdown interpretations after every output cell.
```

### Agent Prompt 2 — Implement Notebook Sections 7 to 10 (Similarity, Clustering, Classification, Association)
```text
Using docs/PROJECT_CONTEXT.md and the verified cleaned dataset from Sections 0–6, append Sections 7, 8, 9, and 10 to notebooks/project_analysis.ipynb:
- Section 7: Euclidean vs. Manhattan vs. Cosine neighbor-agreement & purity analysis (clean vs. unnormalized vs. noisy).
- Section 8: Baseline Clustering comparing k-Means, Agglomerative Hierarchical (stratified n=3,500), and DBSCAN, with Elbow/Silhouette justification and centroid profiling.
- Section 9: Baseline Classification comparing Decision Tree, Gaussian Naive Bayes, Rule-Based Classifier (explicit extracted IF-THEN rules), and k-NN, reporting Accuracy, Macro/Weighted Precision/Recall/F1, Confusion Matrices, and per-class easiest-vs-hardest analysis.
- Section 10: Quantile-discretized morphological Association Pattern Mining via FP-Growth (mlxtend) with Support, Confidence, Lift, and methodological discussion.
Execute all cells, verify outputs, export CSV tables to results/tables/, and save fig05 through fig08 (PNG+PDF) to results/figures/. Add Markdown interpretations after every output.
```

### Agent Prompt 3 — Implement Section 11–14 (The 4-Axis Stress Test, Findings, RQs) & `src/analysis.py`
```text
Following docs/PROJECT_CONTEXT.md Sections 5.2 (Sections 11–14), implement the core Group 10 Data Mining Stress Test in notebooks/project_analysis.ipynb:
- Experiment A: Missingness MCAR (0%, 5%, 10%, 20%, 30%) comparing Median vs. KNN imputation across Clustering and Classification.
- Experiment B: Outlier Injection (0%, 5%, 10%, 20% at 5-10 sigma) across Clustering and Classification.
- Experiment C: Normalization Stress (Raw vs. MinMax vs. Z-Score vs. RobustScaler, clean & under 10% outliers).
- Experiment D: Feature Reduction Stress (Full 16 vs. Corr-Pruned vs. MI Top-6 vs. PCA 95%/90%/80%).
Generate the 4 Robustness Curve figures (fig09–fig12, 300 DPI PNG+PDF), export results/tables/stress_test_master_results.csv and results/tables/metrics_summary.json.
Then write Sections 12–14 in the notebook using ONLY exact verified numbers from metrics_summary.json: Comparative Analysis tables, 5+ structured Interesting Findings (Observation -> Evidence -> Interpretation), explicit answers to RQ1–RQ4 and SRQ1–SRQ4, and Limitations.
Finally, assemble src/analysis.py to reproduce the full pipeline from CLI and generate a clean, guideline-compliant README.md with an Open-in-Colab badge.
```

### Agent Prompt 4 — Generate the Publication-Grade LaTeX Report (After Execution Is Verified)
```text
Read docs/PROJECT_CONTEXT.md (specifically Section 7: Academic Report Writing Instructions) and inspect ALL verified outputs in results/tables/metrics_summary.json, results/tables/*.csv, and results/figures/*.
Generate report/latex/main.tex and report/latex/references.bib strictly matching Guideline Sections 23.1–23.13.
CRITICAL RULES:
1. Every number, metric, count, and delta in main.tex MUST match results/metrics_summary.json and results/tables/*.csv verbatim—zero invented values.
2. Include the formatted East West University / CSE477 / Group 10 title page and complete UCI + Koklu & Ozkan (2020) citations.
3. Embed our saved vector PDF/PNG figures from ../../results/figures/ with publication-grade captions and booktabs tables.
4. Follow the Claim -> Quantitative Evidence -> Mechanism academic writing rhythm throughout, and format all 5 Interesting Findings with Observation, Evidence, and Interpretation.
5. Compile main.tex to report/project_report.pdf (or confirm clean pdflatex/bibtex compilation with zero undefined references).
```