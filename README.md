# 📦 Flipkart Sales Forecasting — End-to-End ML Pipeline

A machine learning pipeline that compresses **~127 million rows** of Flipkart sales transaction data into a compact analytical dataset and trains multiple forecasting models to predict order volumes.

---
## 🗂️ Project Overview

This project tackles the challenge of working with massive e-commerce transactional data. The raw data spans **two datasets totalling ~127 million rows** of individual order records from Flipkart, covering product categories, sellers, price buckets, and order metrics (orders, units, GMV).

The goal is to:
1. Compress the data to a manageable size without losing business insights
2. Engineer meaningful time-series features
3. Train and compare multiple forecasting models to predict `orders`

---

## 📁 Dataset

| Dataset   | Raw Rows   | Columns | Memory   |
|-----------|------------|---------|----------|
| dataset1  | 56,118,563 | 17      | ~7.1 GB  |
| dataset2  | 71,731,038 | 17      | ~9.1 GB  |
| **Total** | **~127M**  |         | **~16 GB** |

### Key Columns

| Column | Description |
|--------|-------------|
| `cms_vertical` | Product vertical (e.g., ab_exerciser) |
| `price_bucket` | Price range category (e.g., 0–100, 500+) |
| `brand` | Product brand |
| `seller_id` / `seller_city` | Seller details |
| `analytic_business_unit` | Business unit (e.g., BGM, Home) |
| `analytic_super_category` | Super category (e.g., SportFitness) |
| `analytic_category` | Category (e.g., ExerciseAndFitness) |
| `analytic_vertical` | Vertical (e.g., AbExerciser) |
| `status` | Order status (DELIVERED, RETURNED) |
| `unit_creation_month` / `unit_creation_year` | Time of order |
| `orders` | Number of orders (target variable) |
| `units` | Number of units sold |
| `gmv` | Gross Merchandise Value (₹) |

---

## 🗜️ Data Reduction Pipeline

The compression from **~127 million → ~1.1 million rows** was achieved through the following steps:

### Step 1 — GroupBy Aggregation *(Primary Compression)*

Individual transaction rows are aggregated by 8 business-level dimensions, summing numeric metrics:

```python
columns_to_group_by = [
    'analytic_business_unit', 'analytic_super_category',
    'analytic_category', 'analytic_vertical', 'cms_vertical',
    'price_bucket', 'unit_creation_month', 'unit_creation_year'
]

df_reduced = df.groupby(columns_to_group_by)[['orders', 'units', 'gmv']].sum()
```

| Dataset  | Before       | After     | Reduction |
|----------|--------------|-----------|-----------|
| dataset1 | 56,118,563   | 460,693   | ~99.2%    |
| dataset2 | 71,731,038   | 652,073   | ~99.1%    |

Seller-level and brand-level granularity is dropped — only business-unit/category/time combinations are retained.

### Step 2 — Price Bucket Merging

7 granular price buckets are consolidated into 3 broader buckets:

```python
mapping = {
    '0-100': '0-300', '100-150': '0-300',
    '150-200': '0-300', '200-300': '0-300',
    '300-400': '300-500', '400-500': '300-500',
    '500+': '500+'
}
df['price_bucket'] = df['price_bucket'].replace(mapping)
```

### Step 3 — Concat & Re-aggregate

Both reduced datasets are concatenated and re-grouped to merge overlapping category/time combinations across the two sources.

### Step 4 — Datetime Construction

Month and year columns are combined into a proper datetime index for time-series operations:

```python
df['date'] = pd.to_datetime(
    df['unit_creation_year'].astype(int).astype(str) + '-' +
    df['unit_creation_month'].astype(int).astype(str) + '-01'
)
```

### Step 5 — NaN Removal

Rows with nulls (introduced by lag and rolling-average operations) are dropped.

**Final dataset: ~1.1 million rows, 11 columns, ~93 MB in memory.**

---

## 🔧 Feature Engineering

Time-series features are created per business category group (not global), preserving local trends:

### Lag Features (Previous Month)
```python
df['orders_lag_1'] = df.groupby([...])['orders'].shift(1)
df['units_lag_1']  = df.groupby([...])['units'].shift(1)
df['gmv_lag_1']    = df.groupby([...])['gmv'].shift(1)
```

### Moving Average Features (3-Month Rolling)
```python
df['orders_ma_3'] = df.groupby([...])['orders'].transform(
    lambda x: x.rolling(3, min_periods=1).mean()
)
```

> **Note:** Moving averages are used for feature engineering (model inputs), **not** for data compression.

### Final Feature Set

| Feature | Description |
|---------|-------------|
| `month` | Month extracted from date |
| `year` | Year extracted from date |
| `orders_lag_1` | Previous month's orders (per group) |
| `units_lag_1` | Previous month's units (per group) |
| `gmv_lag_1` | Previous month's GMV (per group) |
| `orders_ma_3` | 3-month rolling average of orders |
| `units_ma_3` | 3-month rolling average of units |
| `gmv_ma_3` | 3-month rolling average of GMV |

**Target variable:** `orders`

---

## 📊 Statistical Tests

Before modelling, three statistical checks are performed on the data:

| Test | Result | Interpretation |
|------|--------|----------------|
| **Shapiro-Wilk** (Normality) | Statistic = 0.1539 | Data is NOT normally distributed |
| **ADF Test** (Stationarity) | Data IS stationary | Safe to use for time-series modelling without differencing |
| **VIF** (Multicollinearity) | High across lag and MA features | Significant multicollinearity between lag and moving-average features |

> The high VIF values indicate that lag and moving-average features are highly correlated with each other. Tree-based models (Random Forest, XGBoost) handle this better than Linear Regression.

---

## 🤖 Models Trained

Five forecasting models are trained and compared:

| Model | Type | Notes |
|-------|------|-------|
| **Linear Regression** | Parametric | Baseline; affected by multicollinearity |
| **Random Forest** | Ensemble (tree-based) | Handles non-linearity and collinearity well |
| **XGBoost** | Gradient Boosting | Fast, regularised boosting |
| **LSTM** | Deep Learning (RNN) | Sequence model; trained on PCA-reduced features |
| **ARIMA** | Statistical time-series | Univariate; trained on `orders` series only |

All models use an **80/20 train-test split**. LSTM and ARIMA use **PCA-compressed features** (2 principal components) as input.

---

## 📈 Results

### Without PCA (Linear Regression & Random Forest)

| Model | RMSE | MAPE |
|-------|------|------|
| Linear Regression | 10,699.99 | 7.16 |
| Random Forest | 10,610.15 | 4.10 |

### With PCA (All Models)

| Model | RMSE | MAE | MAPE | SMAPE | R² Score |
|-------|------|-----|------|-------|----------|
| **Linear Regression** | **0.376** | 0.044 | 0.468 | 18.3 | **0.849** |
| Random Forest | 0.495 | 0.045 | 0.538 | 15.6 | 0.738 |
| XGBoost | 0.604 | 0.049 | 0.460 | 14.5 | 0.610 |
| ARIMA | 1.106 | 0.177 | 2.675 | 43.3 | -0.026 |
| LSTM | 2.223 | 0.254 | 1.076 | 183.4 | -3.141 |

> ✅ **Best model: Linear Regression** (after PCA) with R² = 0.849  
> ⚠️ LSTM and ARIMA underperform on this aggregated dataset — likely due to limited training epochs and univariate structure respectively.

### Direct ML on Raw Features (Final Evaluation)

| Model | RMSE | MAE | R² Score |
|-------|------|-----|----------|
| Random Forest | 12,298.63 | 1,364.56 | 0.95 |
| XGBoost | 29,017.61 | 2,001.61 | 0.70 |

> 🏆 **Random Forest achieves R² = 0.95 on the original feature space**, making it the strongest practical model for this dataset.

---





## 💡 Key Takeaways

- **GroupBy aggregation is the real compression technique** — not moving averages. It reduced 127M rows to ~1.1M rows (99%+ reduction) by collapsing individual transactions into business-level monthly summaries.
- **Moving averages serve as predictive features**, not compression tools.
- **Random Forest** is the strongest model for this structured, aggregated dataset.
- **LSTM underperforms** without sufficient sequence length and training depth.
- **High VIF** in lag/MA features is expected and manageable for tree-based models.

---



---
