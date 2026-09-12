# Overview
Early-stage lung cancer is hard to diagnose because it is largely asymptomatic, and conventional diagnostic pipelines (imaging, biopsy) are invasive, costly, or come late in disease progression. Breathomics profiling volatile organic compounds (VOCs) in exhaled breath via gas chromatography–mass spectrometry (GC‑MS) offers a non‑invasive alternative, but most prior work relies on black‑box machine learning with little biological validation.

This project addresses that gap with a pipeline that:

Cleans and risk-stratifies breath VOC data into low‑risk, intermediate‑risk, and high‑risk groups.
Applies statistically guided feature selection (one‑way ANOVA + Benjamini–Hochberg FDR correction) to shrink ~11.5K raw ion features down to a compact, significant biomarker set.
Trains a TabNet deep learning model with Bayesian/TPE hyperparameter search (Optuna) inside a 5×3 nested group cross‑validation scheme, using SMOTE to address class imbalance.
Explains predictions with SHAP, validates biomarker robustness with permutation importance and cumulative feature ablation, and maps top VOC biomarkers to chemical classes implicated in cancer metabolism.

# Dataset
Source: Publicly available GC‑MS breathomics dataset from STAT540-UBC-2026/project-spruce.
Size: 3,264 samples × 11,502 features + metadata (age, sex, smoking status, sample date, sample type).
Sample types: Exhaled breath (EX), room air (RM), CASPER-filtered air (CP), standard/reference (RI).
Patient categories: Indeterminate Pulmonary Nodule (IPN), Control, Stable Nodule, Cancer, Cancer Possible — remapped into 3 risk groups (Low / Intermediate / High).

# Methodology Pipeline
- Raw GC-MS VOC data + metadata
- Risk-group label mapping (Low / Intermediate / High)
- Background correction (subtract RM from EX; drop CP + RI)
- Missing-value imputation (zero) + log-modulus transform
- Patient-wise 80:20 train/test split
- Near-zero-variance filtering + robust scaling (train only)
- Feature selection: one-way ANOVA + Benjamini-Hochberg FDR (p < 0.05)
- TabNet training with SMOTE, 5×3 nested group CV,
- Optuna (TPE + MedianPruner) hyperparameter search
- Held-out test evaluation (AUC, precision, recall, F1, specificity)
- Explainability: SHAP (global + local) → permutation importance
- cumulative feature ablation → biological/chemical annotation

# Model: TabNet
TabNet is a deep learning architecture for tabular data that uses sequential, learnable sparse attention (via Sparsemax) to select relevant features at each decision step, combining representation learning with built-in interpretability

# Explainability stack
SHAP (Shapley Additive exPlanations): global and local feature attribution.
Permutation feature importance: measures macro-AUC drop when a feature is shuffled.
Cumulative feature ablation: progressively retrains on top-ranked VOCs to find the minimal informative subset.
UMAP visualization: low-dimensional inspection of class separability and misclassification patterns.
Biological annotation: links top VOC ions to known chemical classes and cancer-associated metabolic pathways.

# Repository Structure
  README.md
  requirements.txt
  data
  notebook
  
# Disclaimer 
This repository is intended for academic and research purpose only. The models and results presented here have not been established as a clinically validated diagnostic system and should not be used for medical diagnosis or clinical decision making.
  
