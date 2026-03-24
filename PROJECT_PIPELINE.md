# Spam Detection Pipeline — Ethereum Token Transfer Data

**Project:** FinTech-540 | **Data:** 1,000-block window (~20M transfers) | **Status:** Stage 6 Complete — Final Report Remaining

---

## Overview

Classify Ethereum tokens as **spam (0)** or **legitimate (1)** using token-level behavioral features engineered from raw transfer events. Labels are derived from:
- **Legit (1):** Contract address in `token_labels.csv` (verified)
- **Spam (0):** Same symbol as a verified token but different contract address (symbol collision)
- **Unlabeled (NaN):** Unverified, no collision — excluded from supervised training

---

## Pipeline Stages

```
EDA                              ✅ Done  →  EDA.ipynb
   ↓
Stage 1: Feature Engineering & Preprocessing   ✅ Done  →  preprocessing.ipynb
   ↓
Stage 2: Train/Test Split & Class Imbalance    ✅ Done  (folded into Stage 1)
   ↓
Stage 3: Baseline Model Training               ✅ Done  →  modeling.ipynb
   ↓
Stage 4: Advanced Models & Hyperparameter Tuning  ✅ Done  →  modeling.ipynb
   ↓
Stage 5: Model Evaluation & Interpretation     ✅ Done  →  evaluation.ipynb
   ↓
Stage 6: Semi-Supervised Extension             ✅ Done  →  semi_supervised.ipynb
   ↓
Stage 7: Final Report & Deliverables           ⬜ Next
```

---

## Stage 1 — Feature Engineering & Preprocessing ✅ Complete

**Notebook:** `preprocessing.ipynb` | **Output:** `data/processed/`

| Item | Detail |
|---|---|
| Token transfers | 2,161,313 (filtered from 3.65M raw transfers) |
| Unique contracts | 12,356 |
| Labeled tokens | 3,606 (spam=2,007 / legit=1,599) |
| Features | 16 final (14 from EDA + `block_range`, `unique_values_count`, `zero_value_ratio`, `top1_sender_share`, `receiver_concentration`) |
| Leakage cols dropped | `symbol_collision`, `is_verified`, `asset` |
| Imputation | Median (0 nulls found in labeled set) |
| Transform | `log1p` on 10 skewed features |
| Outputs | `train/val/test.parquet` (scaled) + `_unscaled` variants + `scaler.joblib` |

---

## Stage 2 — Train/Test Split & Class Imbalance ✅ Complete

**Folded into `preprocessing.ipynb`.**

| Split | Size | Spam | Legit |
|---|---|---|---|
| Train | 2,524 | 1,405 (55.7%) | 1,119 (44.3%) |
| Val | 541 | 301 | 240 |
| Test | 541 | 301 | 240 |

- Stratified split preserves class ratios across all three sets
- Imbalance strategy: `class_weight='balanced'` (SMOTE not needed)
- Test set held out and untouched until Stage 5

---

## Stage 3 — Baseline Models ✅ Complete

**Notebook:** `modeling.ipynb` | **CV:** 5-fold stratified

| Model | CV F1-macro | CV ROC-AUC |
|---|---|---|
| Logistic Regression | 0.8192 ±0.019 | 0.8845 |
| Decision Tree | 0.8359 ±0.015 | 0.9023 |
| Gaussian Naive Bayes | 0.7923 ±0.019 | 0.8586 |

Logistic Regression established a strong linear baseline (0.819), showing that engineered features are largely linearly separable.

---

## Stage 4 — Advanced Models & Hyperparameter Tuning ✅ Complete

**Notebook:** `modeling.ipynb` | **Tuning:** RandomizedSearchCV, 50 iterations, 5-fold CV

**CV Results:**

| Model | CV F1-macro | CV ROC-AUC |
|---|---|---|
| XGBoost (default) | 0.8641 ±0.015 | 0.9369 |
| Random Forest | 0.8615 ±0.014 | 0.9346 |
| LightGBM (default) | 0.8567 ±0.012 | 0.9322 |

**After tuning:**

| Model | Best CV F1-macro |
|---|---|
| LightGBM (tuned) | 0.8711 |
| XGBoost (tuned) | 0.8687 |

**Validation set — final model selection:**

| Model | Val F1-macro | Val ROC-AUC | Val F1-spam | Val F1-legit |
|---|---|---|---|---|
| **LightGBM (tuned)** | **0.8548** | 0.9356 | 0.8752 | 0.8344 |
| Random Forest | 0.8528 | 0.9412 | 0.8738 | 0.8319 |
| XGBoost (tuned) | 0.8494 | 0.9370 | 0.8697 | 0.8291 |
| Logistic Regression | 0.8340 | 0.8832 | 0.8576 | 0.8103 |
| Decision Tree | 0.8113 | 0.9058 | 0.8382 | 0.7845 |
| Gaussian Naive Bayes | 0.7865 | 0.8530 | 0.8180 | 0.7549 |

**Selected model:** LightGBM (tuned) — saved to `models/best_model.joblib`

---

## Stage 5 — Model Evaluation & Interpretation ✅ Complete

**Notebook:** `evaluation.ipynb` | **Data:** held-out test set (541 tokens, never seen during training)

**Final test set results — LightGBM (tuned):**

| Metric | Score |
|---|---|
| F1-macro | 0.8838 |
| ROC-AUC | 0.9562 |
| Avg Precision | — |
| Spam precision / recall / F1 | 0.85 / 0.90 / 0.88 (approx from val) |
| Legit precision / recall / F1 | 0.86 / 0.81 / 0.83 (approx from val) |

**SHAP top features:** `n_unique_senders`, `n_transfers`, `n_unique_receivers`, `sender_receiver_ratio`, `top1_sender_share`, `n_distinct_blocks`

**Error analysis:**
- False negatives (spam that slips through): tokens with secondary organic activity post-airdrop, diluting the spam signal
- False positives (legit flagged as spam): niche/new tokens with sparse transfer history

**Threshold guidance:** Default 0.5 catches ~90% spam. Raise to 0.6–0.65 to reduce false alarms; lower to 0.35–0.40 to maximise recall.

---

## Stage 6 — Semi-Supervised Extension ✅ Complete

**Notebook:** `semi_supervised.ipynb` | **Unlabeled tokens:** 8,750

**Result: semi-supervised did not improve over supervised baseline.**

| Approach | Train size | F1-macro | Δ vs baseline |
|---|---|---|---|
| Supervised only (baseline) | 2,524 | **0.8838** | — |
| Pseudo-label (t=0.10) | 6,577 | 0.8520 | −0.032 |
| Pseudo-label (t=0.15) | 7,508 | 0.8648 | −0.019 |
| Pseudo-label (t=0.20) | 8,440 | 0.8667 | −0.017 |
| Self-Training (t=0.90) | 9,795 | 0.8557 | −0.028 |
| Self-Training (t=0.80) | 10,764 | 0.8611 | −0.023 |

**Why it failed:** (1) labeled set already representative; (2) 14:1 pseudo-label class skew; (3) unlabeled pseudo-spam are dormant contracts (different type from training spam); (4) 53% of unlabeled tokens are genuinely uncertain.

**Decision:** Retain supervised-only LightGBM as the final model.

---

## Stage 7 — Final Report & Deliverables ⬜ Next

### Report Checklist

- [ ] Problem framing and label construction methodology
- [ ] EDA summary (key features, distributions, correlations)
- [ ] Feature engineering decisions and justification
- [ ] Model comparison table (CV F1, AUC across all models)
- [ ] Best model test set performance with confusion matrix
- [ ] SHAP feature importance plot
- [ ] Error analysis: what types of tokens are hardest to classify?
- [ ] Semi-supervised extension results and conclusion
- [ ] Discussion: limitations, potential improvements, deployment considerations

### Supporting Documents Already Available

- [x] `feature_description.pdf` — full feature reference for all 16 features
- [x] `models/val_results.csv` — complete model comparison table
- [x] All notebooks with inline results and summary cells

---

## Quick Reference: Feature Leakage Checklist ✅

All confirmed excluded from feature matrix `X`:

- [x] `symbol_collision` — directly defines spam label
- [x] `is_verified` — directly defines legit label
- [x] `asset` (raw token symbol) — identifier, not a behavioral feature
- [x] `label` — target variable

---

## Recommended Next Action

```
Open report / presentation and write up findings.
All data, figures, and model outputs are available across:
  - EDA.ipynb              → data insights, feature correlations
  - preprocessing.ipynb    → feature engineering methodology
  - modeling.ipynb         → model comparison, best model summary
  - evaluation.ipynb       → final metrics, SHAP plots, error analysis
  - semi_supervised.ipynb  → semi-supervised results and analysis
  - feature_description.pdf → feature reference document
  - models/val_results.csv  → exportable comparison table
```

---

*Pipeline version 2.0 | Last updated 2026-03-23*
