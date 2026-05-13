# 🚲 Bike Share Demand — Feature Engineering

> Predicting the total number of hourly bike-share rentals using engineered features and two regression models.

---

## 📌 Project Overview

This project explores how **feature engineering** impacts a model's ability to predict bike-share demand. Starting from raw hourly rental data, we extract meaningful time-based features, convert temperature units, and create a temperature variance column — then compare model performance before and after engineering.

---

## 📂 Dataset

| Property | Detail |
|---|---|
| **Source** | [Bike Sharing Dataset](https://docs.google.com/spreadsheets/d/e/2PACX-1vROUXPkYUkX-2W7JbJ0-oNKaXzpg4NtmU9IeWEY6yFKm32ZEJOpRh_soHD4BeIcuHjYik3SEoXmkgwj/pub?output=csv) |
| **Target** | `count` — total hourly rentals (casual + registered) |
| **Rows** | ~10,000 hourly observations |
| **Original features** | datetime, season, holiday, workingday, weather, temp, atemp, humidity, windspeed |

> `casual` and `registered` are dropped before modelling — they directly sum to `count` and would leak the target.

---

## 🔧 Feature Engineering Steps

| Step | Action | Reason |
|---|---|---|
| 1 | Drop `casual`, `registered` | Target leakage |
| 2 | Parse `datetime` → `month`, `day`, `hour` (object dtype) | Expose time-based demand patterns for OHE |
| 3 | Drop `datetime`, `season` | Redundant after extraction |
| 4 | Convert `temp` & `atemp` from °C → °F via lambda | Unit consistency |
| 5 | Create `temp_variance` = `temp` − `atemp` | Relative comfort signal |
| 6 | Drop `atemp` | Redundant after variance is computed |

---

## 🏗️ Modelling Pipeline

Both baseline and engineered pipelines follow the same structure:

```
ColumnTransformer
├── Numeric columns  → StandardScaler
└── Object columns   → OneHotEncoder (handle_unknown='ignore')
        ↓
    Estimator (Linear Regression or Random Forest)
```

- **Train / Test split:** 75% / 25%, `random_state=42`
- **Models:** `LinearRegression` and `RandomForestRegressor(n_estimators=100)`

---

## 📊 Results

### Baseline vs Engineered — Test R²

<img width="790" height="390" alt="download (1)" src="https://github.com/user-attachments/assets/376a9e38-7c28-4040-b306-f6791c8e0785" />


### Linear Regression

| Split | MAE | MSE | RMSE | R² |
|---|---|---|---|---|
| **Baseline — Training** | 114.263 | 22,742.727 | 150.807 | 0.307 |
| **Baseline — Test** | 112.072 | 22,260.976 | 149.201 | 0.322 |
| **Engineered — Training** | 78.544 | 11,889.931 | 109.041 | 0.638 |
| **Engineered — Test** | 80.337 | 12,282.482 | 110.826 | 0.626 |

### Random Forest

| Split | MAE | MSE | RMSE | R² |
|---|---|---|---|---|
| **Baseline — Training** | 32.190 | 2,162.726 | 46.505 | 0.934 |
| **Baseline — Test** | 84.843 | 15,385.076 | 124.037 | 0.531 |
| **Engineered — Training** | 18.282 | 777.804 | 27.889 | 0.976 |
| **Engineered — Test** | 48.368 | 5,142.217 | 71.709 | 0.843 |

### Summary — Test R² improvement

| Model | Baseline R² | Engineered R² | Δ R² |
|---|---|---|---|
| Linear Regression | 0.322 | 0.626 | **+0.304** |
| Random Forest | 0.531 | 0.843 | **+0.312** |

> ✅ Feature engineering improved both models significantly. Random Forest with engineered features achieved the best test R² of **0.843**.

---

## 💡 Key Findings

- **`hour` is the dominant feature.** Bike demand follows a clear morning/evening commute pattern. Encoding it categorically lets the model learn demand peaks at specific hours.
- **`month` and `day` add seasonality.** The model can now distinguish summer demand from winter, and weekday commuter patterns from weekend leisure rides.
- **Random Forest substantially outperforms Linear Regression** because the demand patterns are non-linear (two daily peaks, seasonal curves). Tree-based models handle this naturally.
- **The gap between training and test R² in Random Forest (0.976 vs 0.843) indicates some overfitting** — tuning `max_depth` or `min_samples_leaf` would help close this gap.

---

## 🚀 How to Run

1. Clone the repository
2. Open `Feature_Engineering.ipynb` in Jupyter or Google Colab
3. Run all cells in order — the dataset is loaded automatically from a public URL, no local file needed

```bash
git clone <your-repo-url>
cd <repo-folder>
jupyter notebook Feature_Engineering.ipynb
```

**Dependencies:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

---

## 🗂️ Repository Structure

```
├── Feature_Engineering.ipynb   # Main notebook
├── results_chart.png           # Baseline vs Engineered R² comparison chart
└── README.md                   # This file
```

---

## 👤 Author

Ali Abu Sohiban
