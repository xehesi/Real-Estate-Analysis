# Real Estate Price Prediction

**Project Source:** [DataWars — Real Estate Analysis Challenge](https://app.datawars.io/project/214e265c-029f-4940-9a35-f43a9e0c56cc?page=1)

---

## 📌 Overview
This project tackles a real-world property price prediction problem using a dataset of Indian residential listings. The primary objective is to predict **`Amount_USD`** — the total sale price of a property — based on various attributes such as location, floor level, area, furnishing status, and parking availability. 

The notebook demonstrates a complete end-to-end machine learning workflow, transforming raw, messy text data into a clean, structured format before training a highly capable ensemble model.

## 🎯 Objectives
1. **Establish a Baseline:** Generate a naive baseline prediction using the mean sale price as a reference benchmark.
2. **Feature Engineering:** Clean and engineer meaningful features from messy data (e.g., standardizing mixed area units, parsing free-text parking fields, handling missing values via domain-aware imputation).
3. **Model Training:** Train a **Random Forest Regressor** and evaluate its performance using R-Squared (R²) and Mean Absolute Error (MAE).
4. **Feature Optimization:** Identify the most influential features driving property values, align training and test datasets, and remove low-importance noise variables.

---

## 📂 Dataset Details

| File | Rows | Description |
| :--- | :--- | :--- |
| `train.csv` | ~15,000 | Labeled listings including the `Amount_USD` target column. |
| `test_inputs.csv` | 6,344 | Unlabeled listings for generating final sale price predictions. |

---

## 🛠️ Tools & Libraries

* **`pandas`**: Data loading, cleaning, and manipulation.
* **`numpy`**: Numerical operations and missing value imputation.
* **`scikit-learn`**: Train/test splitting, model training (`RandomForestRegressor`), and evaluation metrics.
* **`matplotlib`**: Visualizing feature importances.
* **`re`**: Regex parsing of unstructured free-text columns.

---

## 🚀 Project Workflow

### 1. Baseline Submission
Before building complex models, a naive baseline was established by predicting the mean training price (**~$135,075**) for every property. This serves as a lower-bound benchmark to ensure the machine learning model is actually learning underlying patterns.

### 2. Exploratory Data Analysis (EDA) & Data Cleaning
* **High Cardinality & Noise:** Dropped free-text columns with no structured predictive signal (`Title`, `Description`, `Society`) and columns with severe missingness (`Status`, `Dimensions`, `Plot Area`).
* **Area Standardization:** Parsed `Carpet Area` and `Super Area`, converting mixed string units (sqm, sqyrd, sqft) into unified numeric square footage.
* **Missing Area Imputation:** Estimated missing `Carpet Area` values from `Super Area` using a standard 70% domain ratio, and vice versa.

### 3. Advanced Feature Engineering
* **Parking Extraction:** Used regex to parse numeric parking spot counts from unstructured text (e.g., `"2 Covered, 1 Open"` → `2`). Created a boolean flag for covered parking.
* **Pricing Imputation:** Back-filled missing `Price_USD` (price per sqft) by calculating `Amount_USD / Super Area`, then imputed remaining nulls using per-city medians.
* **Categorical Handling:** Filled nulls in `Furnishing` and `facing` with conservative defaults (e.g., "Unfurnished", most common city direction).
* **Floor Ratios:** Split string fractions in the `Floor` column (e.g., "5 out of 14") to create structured `Floor_Level`, `Total_Floors`, and a newly engineered `Floor_Ratio` metric. Converted basements and ground floors to numeric equivalents.
* **One-Hot Encoding:** Applied `pd.get_dummies` to encode `location`, `Transaction`, `Furnishing`, `facing`, `overlooking`, and `Ownership`.

### 4. Model Training & Feature Alignment
* **Initial Random Forest:** Trained a 100-tree `RandomForestRegressor` on an 80/20 data split. 
* **Feature Alignment:** Addressed dummy variable mismatches between the training and testing sets by identifying columns present in one but not the other. 
* **Noise Removal:** Extracted feature importances and dropped 30 "noise" columns (importance score < 0.001) to streamline the model and perfectly align the test set via `reindex`.

---

## 📊 Results & Evaluation

The model significantly outperformed the naive baseline. 

**Initial Model Performance (All Features):**
* **R-Squared:** 0.893
* **Mean Absolute Error (MAE):** $6,279.73

**Final Model Performance (Post-Noise Removal):**
* **R-Squared:** 0.891
* **Mean Absolute Error (MAE):** $6,382.12

> **Note:** Removing the noise columns caused a negligible drop in metrics while vastly improving model efficiency and resolving the dimensionality mismatch for final test predictions.

---

## 💻 How to Run
1. Clone the repository.
2. Ensure you have the required dependencies installed (`pip install pandas numpy scikit-learn matplotlib`).
3. Place `train.csv` and `test_inputs.csv` in the working directory.
4. Run `Project.ipynb` sequentially to reproduce the data cleaning steps, model training, and final predictions array.
