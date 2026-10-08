# Demand Forecasting for Online Retail Using Machine Learning

Predicting future order demand from historical e-commerce sales data, and comparing Linear Regression (with PCA), Random Forest, XGBoost, LSTM and ARIMA.

## Business Goal

Help online retail platforms streamline supply chain management, minimise excess inventory and improve customer satisfaction.

**Aim:** Accurately predict future demand by training machine learning models on historical sales data.

## Dataset

Two large sales extracts (about 56M and 72M rows) from an online marketplace, covering 2020 to 2024. The data is proprietary and is not included in this repository.

| Column | Description |
|---|---|
| `analytic_business_unit`, `analytic_super_category`, `analytic_category`, `analytic_vertical`, `cms_vertical` | Product hierarchy |
| `price_bucket` | Price range (`0-300`, `300-500`, `500+`) |
| `units`, `gmv` | Units sold, gross merchandise value |
| `orders` | Number of orders (**target**) |

## What I Did

**1. Data collection and preprocessing** 
- Aggregated each raw file by product hierarchy(using groupby() method), reducing about 128M rows to about 1.1M.
- Merged the two files, grouped price buckets from 7 to 3, removed nulls and built a monthly `date` column.
- Final modelling dataset: about 570K records across product segments.

**2. Feature engineering**
- Lag features: previous month's `orders`, `units`, `gmv`.
- 3-month moving averages of `orders`, `units`, `gmv`, using previous months only to avoid leakage.
- All features are computed within each product segment, not globally.

**3. Exploratory analysis** 
- Checked missing values and duplicates, target skewness and the monthly order trend.
- Found heteroscedasticity (Breusch-Pagan test) and strong multicollinearity (VIF and correlation matrix).
- ADF Test performed before training time series based models; Results showed that data is Stationary.

**4. Train/test split**
- Chronological split: train on 2020 to 2023, test on 2024.
- Hyperparameter tuning was done in appropriate models to achieve optimal results.

**5. Modelling**

| Model | Approach |
|---|---|
| Linear Regression | Baseline on raw features, then with scaling and PCA to handle multicollinearity. Ridge regression also tested. |
| Random Forest | 100 trees, most important feature was found to be orders_ma_3 |
| XGBoost | Grid search over `max_depth` and `learning_rate`; best was depth 3, learning rate 0.10, 300 trees |
| LSTM | Sequence model with a 3-month window per product segment |
| ARIMA | Trained separately  on total monthly orders. ADF test showed stationarity, so d = 0. Order (1,0,1) was chosen via validation RMSE, AIC and BIC. |

## Results (Test Set: 2024)

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| **Linear Regression (PCA)** | **18,258.79** | 2,364.25 | **0.91** |
| Random Forest | 19,653.84 | **2,238.34** | 0.90 |
| LSTM | 24,118.21 | 3,991.56 | 0.85 |
| XGBoost | 33,661.77 | 2,697.07 | 0.70 |



**ARIMA** works on aggregated monthly orders, so its errors are on a different scale and cannot be compared with the table above. On the monthly totals it achieved MAPE 11.99% and SMAPE 13.36%.



**Takeaways**
- Linear Regression with PCA had the lowest RMSE and highest R². Random Forest had the lowest MAE.
- PCA improved the linear model by removing multicollinearity, and Ridge regularisation changed little, so overfitting was not a concern.
- The more complex models (XGBoost, LSTM) did not outperform simpler ones on this feature set.
- ARIMA captured the overall trend reasonably.
- Proper feature extraction and Hyperparameter tuning needs to be done according to each model to achieve the best performance scores.

## Technologies

Python, Pandas, NumPy, Scikit-learn, XGBoost, Statsmodels, TensorFlow/Keras, Matplotlib, Seaborn, Jupyter

