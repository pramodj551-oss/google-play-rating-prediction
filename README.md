# 🤖 Google Play Store Rating Prediction Pipeline

A leak-free, end-to-end machine learning pipeline that predicts mobile app ratings on the Google Play Store using app metadata.

---

## 1. Problem Framing

- **Business Question:** Can we predict an app's overall user rating (`Rating`) using early metadata — category, install scale, file size, price tier, and content rating — available prior to or shortly after launch?
- **Problem Type:** Continuous regression (`Rating` ranges 1.0–5.0)

**Feature Split**

| Type | Features |
|---|---|
| **Target (y)** | `Rating` |
| **Categorical (X)** | `Category`, `Content Rating`, `Type` (Free/Paid) |
| **Numerical (X)** | `Reviews`, `Size` (KB), `Installs`, `Price` (USD) |

---

## 2. Feature Preparation & Leakage Prevention

1. **Split first:** 80/20 train/test split (`random_state=42`) performed **before** any feature engineering.
2. **Encoding:** `Category`, `Content Rating`, and `Type` encoded via **One-Hot Encoding** (`handle_unknown='ignore'`) — no intrinsic order.
3. **Scaling:** `Reviews`, `Size`, `Installs`, `Price` standardized via `StandardScaler`.
4. **Leakage control:**
   - Scaler and encoder fitted **only** on the training set.
   - Test set transformed using training-derived parameters only.
   - Preprocessing + model unified in a scikit-learn `Pipeline` with `ColumnTransformer`, so each cross-validation fold refits independently on its own training slice.

---

## 3. Model Performance

| Rank | Model | MAE | RMSE | R² | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 🥇 1 | **Random Forest Regressor** | **0.3173** | **0.4869** | **0.0987** | Best baseline |
| 2 | Linear Regression | 0.3575 | 0.5063 | 0.0257 | Baseline |
| 3 | Decision Tree Regressor | 0.3998 | 0.6486 | -0.5988 | Overfitting |

*Class rebalancing wasn't needed — this is a continuous regression task, not classification.*

---

## 4. Pipeline Validation & Hyperparameter Tuning

**5-Fold Cross-Validation (Random Forest)**
- Fold R² scores: `0.0429, 0.0803, 0.1525, 0.1250, 0.1960`
- Mean CV R²: **0.0841 (± 0.0075)**

**GridSearchCV**

| Parameter | Grid Searched | Best Value |
|---|---|---|
| `n_estimators` | 50, 100 | 100 |
| `max_depth` | 10, 20, None | 10 |
| `min_samples_split` | 2, 5 | 5 |

**Tuned Model — Holdout Test Set**
- R²: **0.1189**
- MAE: **0.3262**
- RMSE: **0.4815**

---

## 5. Repository Structure

```
google-play-rating-prediction/
├── data/
│   └── googleplaystore.csv     # Raw dataset
├── notebooks/
│   └── run_pipeline.ipynb      # Full pipeline notebook
├── requirements.txt             # Dependencies
└── README.md                    # Framing, leakage strategy, model results
```

---

## 6. Setup

```bash
git clone https://github.com/pramodj551-oss/google-play-rating-prediction.git
cd google-play-rating-prediction
pip install -r requirements.txt
```

---

## 📬 Contact

**Pramod Prakash Jadhav**
📧 pramodj551@gmail.com
🔗 [LinkedIn](https://www.linkedin.com/in/pramod-prakash-jadhav-42ba2281)
