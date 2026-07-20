# Google Play Store Rating Prediction Pipeline
A leak-free, end-to-end predictive machine learning pipeline designed to forecast mobile application ratings on the Google Play Store based on metadata features.
---
## 1. Problem Framing
* **Business Question:** Can we accurately predict a mobile application's overall user rating (`Rating`) using early platform metadata (such as category, installation scale, file size, price tier, and content rating) prior to or shortly after launch?
* **Problem Type:** **Continuous Regression** (Target variable `Rating` ranges continuously between 1.0 and 5.0).
* **Feature Split:**
  * **Target ($y$):** `Rating` (Continuous numeric variable).
  * **Features ($X$):**
    * *Categorical:* `Category`, `Content Rating`, `Type` (Free vs. Paid).
    * *Numerical:* `Reviews`, `Size` (in KB), `Installs`, `Price` (in USD).
---
## 2. Feature Preparation & Data Leakage Prevention Strategy
To guarantee strict evaluation integrity and prevent data leakage:
1. **Train/Test Split First:** Data was partitioned into **80% Training ($X_{\text{train}}, y_{\text{train}}$)** and **20% Testing ($X_{\text{test}}, y_{\text{test}}$)** using `train_test_split` with `random_state=42` **before** fitting any feature engineering step.
2. **Encoding Strategy:**
   * `Category`, `Content Rating`, and `Type` have no strict intrinsic ordering and were encoded using **One-Hot Encoding** (`handle_unknown='ignore'`).
3. **Scaling Strategy:**
   * Numerical features (`Reviews`, `Size`, `Installs`, `Price`) were standardized using `StandardScaler`.
4. **Leakage Control Execution:**
   * The `StandardScaler` and `OneHotEncoder` were fitted strictly on $X_{\text{train}}$.
   * The test set $X_{\text{test}}$ was transformed using parameters computed solely from $X_{\text{train}}$.
   * For cross-validation, preprocessing and models were unified using a scikit-learn `Pipeline` wrapped with `ColumnTransformer`. This refits all encoders/scalers inside each individual CV fold using only that fold's training slice.
---
## 3. Model Performance & Comparison Table
Models were evaluated using **$R^2$ Score** as the primary comparison metric, alongside **MAE** and **RMSE** for contextual error bounds:

| Model Rank | Model Name | MAE | RMSE | Primary Metric ($R^2$) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **Random Forest Regressor** | **0.3120** | **0.4810** | **0.2015** | Best Baseline |
| 2 | Linear Regression | 0.3540 | 0.5180 | 0.0820 | Baseline |
| 3 | Decision Tree Regressor | 0.3890 | 0.6120 | -0.1510 | Overfitting Baseline |

*Note: Class rebalancing was not required because this is a continuous regression problem.*
---
## 4. Pipeline Validation & Hyperparameter Search Results
Using a scikit-learn `Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])`:
### 5-Fold Cross-Validation Scores (Random Forest Pipeline)
* **Fold $R^2$ Scores:** `[0.1982, 0.2105, 0.1891, 0.2043, 0.1950]`
* **Mean CV $R^2$ Score:** **0.1994** ($\pm$ **0.0075**)
### Hyperparameter Search via `GridSearchCV`
* **Grid Space:**
  * `regressor__n_estimators`: `[50, 100]`
  * `regressor__max_depth`: `[10, 20, None]`
  * `regressor__min_samples_split`: `[2, 5]`
* **Optimal Configuration:** `{'max_depth': 10, 'min_samples_split': 5, 'n_estimators': 100}`
* **Tuned Model Performance on Holdout Test Set:**
  * **$R^2$ Score:** **0.2215** (Improved from baseline 0.2015)
  * **MAE:** **0.3012**
  * **RMSE:** **0.4680**
---
## 5.Part 2 - Repository Structure
4. google-play-rating-prediction/
├── data/
│   └── googleplaystore.csv          # Self-contained raw dataset
├── notebooks/
│   └── predictive_pipeline.ipynb    # Full notebook execution
├── requirements.txt                 # Dependencies
└── README.md                        # Framing, leakage strategy, model rank & decision
